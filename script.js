/* ═══════════════════════════════════════════════════════════════════════════
   VoiceCoach – Frontend Script
   Recording · Assessment display · History · Progress charts
   Auth is handled by auth-check.js (PocketBase SDK) + login.html
═══════════════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────────────
let mediaRecorder  = null;
let audioChunks    = [];
let recordingTimer = null;
let recordSeconds  = 0;
const MAX_SECS     = 120;

let audioCtx       = null;
let analyser       = null;
let waveAnim       = null;
let charts         = {};          // keyed by canvas id

// ── DOM refs ──────────────────────────────────────────────────────────────────
const recordBtn    = document.getElementById('record-btn');
const timerEl      = document.getElementById('timer');
const statusEl     = document.getElementById('record-status');
const hintEl       = document.getElementById('record-hint');
const resultsEl    = document.getElementById('results');
const processingEl = document.getElementById('processing-overlay');
const waveCanvas   = document.getElementById('waveform');
const waveCtx      = waveCanvas.getContext('2d');
const micSvg       = recordBtn.querySelector('.mic-svg');
const stopSvg      = recordBtn.querySelector('.stop-svg');

// ═══════════════════════════════ TAB SWITCHING ════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`tab-${id}`).classList.remove('hidden');
    if (id === 'history')    loadHistory();
    if (id === 'progress')   loadProgress();
    if (id === 'assessment') window.initAssessment?.();
  });
});

// ═══════════════════════════════ WAVEFORM ════════════════════════════════════
function drawIdle() {
  const w = waveCanvas.width, h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);
  waveCtx.strokeStyle = '#2a2a50';
  waveCtx.lineWidth   = 2;
  waveCtx.beginPath();
  waveCtx.moveTo(0, h / 2);
  waveCtx.lineTo(w, h / 2);
  waveCtx.stroke();
}

function startWaveform(stream) {
  audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  audioCtx.createMediaStreamSource(stream).connect(analyser);

  const data = new Uint8Array(analyser.frequencyBinCount);

  function draw() {
    waveAnim = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    const w = waveCanvas.width, h = waveCanvas.height;
    waveCtx.clearRect(0, 0, w, h);
    waveCtx.lineWidth   = 2.5;
    waveCtx.strokeStyle = '#7c3aed';
    waveCtx.beginPath();
    const step = w / data.length;
    data.forEach((v, i) => {
      const y = (v / 128) * (h / 2);
      i === 0 ? waveCtx.moveTo(0, y) : waveCtx.lineTo(i * step, y);
    });
    waveCtx.stroke();
  }
  draw();
}

function stopWaveform() {
  cancelAnimationFrame(waveAnim);
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  drawIdle();
}

drawIdle(); // initial state

// ═══════════════════════════════ RECORDING ════════════════════════════════════
recordBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    setStatus('Microphone access denied. Please allow microphone access and reload.');
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMime() });

  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    stopWaveform();
    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
    submitAssessment(blob, recordSeconds);
  };

  mediaRecorder.start(200);
  recordSeconds = 0;

  // UI: recording state
  recordBtn.classList.add('is-recording');
  micSvg.classList.add('hidden');
  stopSvg.classList.remove('hidden');
  resultsEl.classList.add('hidden');
  hintEl.textContent = 'Recording… click to stop';
  setStatus('');

  startWaveform(stream);

  recordingTimer = setInterval(() => {
    recordSeconds++;
    timerEl.textContent = fmt(recordSeconds);
    if (recordSeconds >= MAX_SECS) stopRecording();
  }, 1000);
}

function stopRecording() {
  clearInterval(recordingTimer);
  if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
  setRecordIdle();
  setProcessing(true);
}

function setRecordIdle() {
  recordBtn.classList.remove('is-recording', 'is-processing');
  micSvg.classList.remove('hidden');
  stopSvg.classList.add('hidden');
  timerEl.textContent = '0:00';
  hintEl.textContent  = 'Click to start recording · up to 2 minutes';
}

function setProcessing(on) {
  processingEl.classList.toggle('hidden', !on);
  if (on) recordBtn.classList.add('is-processing');
  else    recordBtn.classList.remove('is-processing');
}

function getSupportedMime() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

// ═══════════════════════════════ ASSESSMENT ══════════════════════════════════
async function submitAssessment(blob, duration) {
  setStatus('Uploading…');

  const fd = new FormData();
  fd.append('audio', blob, 'recording.webm');
  fd.append('duration', String(duration));

  try {
    const res = await authFetch('/assess', { method: 'POST', body: fd });
    if (!res.ok) {
      if (res.status === 413) {
        throw new Error('Recording is too large for the server. Ask the admin to set client_max_body_size 50M in nginx.');
      }
      if (res.status === 504 || res.status === 502) {
        throw new Error('Server timed out processing the audio. Try a shorter recording.');
      }
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `Server error ${res.status}`);
    }
    const data = await res.json();
    displayResults(data);
    setStatus('Assessment complete ✓');
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    setProcessing(false);
  }
}

// ═══════════════════════════════ DISPLAY RESULTS ══════════════════════════════
function displayResults(d) {
  // Overall ring
  const pct       = d.overall / 100;
  const circ      = 2 * Math.PI * 52;          // r = 52
  const offset    = circ * (1 - pct);
  const ring      = document.getElementById('overall-ring');
  ring.style.strokeDasharray  = circ;
  ring.style.strokeDashoffset = offset;
  ring.className = 'ring-fg ' + cls(d.overall);

  const overallEl = document.getElementById('overall-score');
  overallEl.textContent = Math.round(d.overall);
  overallEl.className   = 'overall-num ' + cls(d.overall);

  // Metrics
  setMetric('pace',          d.pace.score,         `${d.pace.wpm.toFixed(0)} WPM`);
  setMetric('filler',        d.filler.score,
    d.filler.count
      ? `${d.filler.count} word${d.filler.count > 1 ? 's' : ''}: ${d.filler.words.slice(0,3).join(', ')}`
      : 'None detected');
  setMetric('pronunciation', d.pronunciation.score, scoreLabel(d.pronunciation.score));
  setMetric('grammar',       d.grammar.score,
    `${d.grammar.errors} issue${d.grammar.errors !== 1 ? 's' : ''}`);
  setMetric('clarity',       d.clarity.score,       scoreLabel(d.clarity.score));

  // Fluency — informational metric built from real pause-timing evidence;
  // not part of Overall (see DESIGN.md). May be absent on very old cached
  // responses, so guard rather than assume.
  if (d.fluency) {
    const pauseDetail = d.fluency.pause_data_available
      ? `${d.fluency.long_pause_count} long pause${d.fluency.long_pause_count !== 1 ? 's' : ''}`
      : 'No pause timing available';
    setMetric('fluency', d.fluency.score, pauseDetail);
  }

  // Low-evidence caveat (short recordings) — see app.py's `evidence` field.
  const evidenceEl = document.getElementById('evidence-notice');
  if (evidenceEl) {
    if (d.evidence && d.evidence.low_evidence) {
      evidenceEl.textContent = '⚠️ ' + d.evidence.reason;
      evidenceEl.classList.remove('hidden');
    } else {
      evidenceEl.classList.add('hidden');
    }
  }

  // Voice Archetype
  document.getElementById('archetype-emoji').textContent = d.archetype.emoji;
  document.getElementById('archetype-name').textContent  = d.archetype.archetype;
  document.getElementById('archetype-desc').textContent  = d.archetype.description;
  document.getElementById('archetype-traits').innerHTML  =
    d.archetype.traits.map(t => `<span class="trait-pill">${esc(t)}</span>`).join('');

  // Transcript
  const words = d.transcript.split(/\s+/).filter(Boolean).length;
  document.getElementById('transcript-text').textContent = d.transcript;
  document.getElementById('word-count-badge').textContent = `${words} words`;

  // Feedback
  document.getElementById('feedback-text').textContent = d.feedback;

  // Pronunciation & grammar detail panels
  displayPronunciationIssues(d.pronunciation.issues || []);
  displayGrammarIssues(d.grammar.issues || []);

  // Radar for this session
  renderRadar('result-radar', {
    pace: d.pace.score, filler: d.filler.score,
    pronunciation: d.pronunciation.score,
    grammar: d.grammar.score, clarity: d.clarity.score,
  });

  resultsEl.classList.remove('hidden');
  setTimeout(() => resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
}

function setMetric(name, score, detail, displayText) {
  const scoreEl  = document.getElementById(`${name}-score`);
  const barEl    = document.getElementById(`${name}-bar`);
  const detailEl = document.getElementById(`${name}-detail`);

  scoreEl.textContent = displayText !== undefined ? displayText : Math.round(score);
  scoreEl.className   = `metric-score ${cls(score)}`;
  barEl.style.width   = score + '%';
  barEl.className     = `progress-fill ${cls(score)}`;
  detailEl.textContent = detail;
}

function cls(s) { return s >= 80 ? 'good' : s >= 60 ? 'ok' : 'poor'; }

function scoreLabel(s) {
  return s >= 85 ? 'Excellent' : s >= 70 ? 'Good' : s >= 55 ? 'Fair' : 'Needs work';
}

// ═══════════════════════ PRONUNCIATION ISSUES ════════════════════════════════
function displayPronunciationIssues(issues) {
  const panel   = document.getElementById('pronun-panel');
  const listEl  = document.getElementById('pronun-list');
  const countEl = document.getElementById('pronun-count');

  if (!issues || issues.length === 0) {
    panel.classList.add('hidden');
    return;
  }

  countEl.textContent = issues.length;
  panel.classList.remove('hidden');

  listEl.innerHTML = issues.map((item, i) => {
    const conf     = item.confidence;          // 0-100
    const fillCls  = conf >= 75 ? 'good' : conf >= 55 ? 'ok' : '';
    return `
      <div class="pronun-item">
        <span class="word-pill">${esc(item.word)}</span>

        <div class="conf-bar-wrap" title="Clarity: ${conf}%">
          <span class="conf-label">Clarity ${conf}%</span>
          <div class="conf-track">
            <div class="conf-fill ${fillCls}" style="width:${conf}%"></div>
          </div>
        </div>

        <button class="play-btn" data-word="${esc(item.word)}" data-idx="${i}"
                onclick="speakWord(this, '${esc(item.word)}')">
          <span class="play-icon">▶</span> Hear it
        </button>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════ GRAMMAR ISSUES ══════════════════════════════════
function displayGrammarIssues(issues) {
  const panel   = document.getElementById('grammar-panel');
  const listEl  = document.getElementById('grammar-list');
  const countEl = document.getElementById('grammar-count');

  if (!issues || issues.length === 0) {
    panel.classList.add('hidden');
    return;
  }

  countEl.textContent = issues.length;
  panel.classList.remove('hidden');

  listEl.innerHTML = issues.map((item, i) => {
    // Highlight the wrong word inside context
    const ctxHtml = item.context
      ? esc(item.context).replace(
          esc(item.wrong),
          `<span class="ctx-wrong">${esc(item.wrong)}</span>`
        )
      : '';

    return `
      <div class="grammar-item">
        <div class="grammar-correction">
          <span class="wrong-text">${esc(item.wrong)}</span>
          <span class="arrow-icon">→</span>
          <span class="correct-text">${esc(item.correct)}</span>
        </div>
        <div class="grammar-meta">
          <span class="grammar-message">${esc(item.message)}</span>
          <button class="play-btn"
                  onclick="speakCorrection(this, '${esc(item.correct)}')">
            <span class="play-icon">▶</span> Hear correction
          </button>
        </div>
        ${ctxHtml ? `<small class="grammar-context">Context: "${ctxHtml}"</small>` : ''}
      </div>
    `;
  }).join('');
}

// ── Shared TTS helpers ────────────────────────────────────────────────────────
let _activeTtsBtn = null;

function _speak(text, rate, btn) {
  if (speechSynthesis.speaking) speechSynthesis.cancel();
  if (_activeTtsBtn) _activeTtsBtn.classList.remove('playing');

  const utt  = new SpeechSynthesisUtterance(text);
  utt.rate   = rate;
  utt.lang   = 'en-US';
  utt.onend  = () => { if (btn) btn.classList.remove('playing'); _activeTtsBtn = null; };
  utt.onerror = () => { if (btn) btn.classList.remove('playing'); _activeTtsBtn = null; };

  if (btn) { btn.classList.add('playing'); _activeTtsBtn = btn; }
  speechSynthesis.speak(utt);
}

window.speakWord        = (btn, word)       => _speak(word, 0.72, btn);
window.speakCorrection  = (btn, correction) => _speak(correction, 0.85, btn);

// ── TTS (browser Web Speech API – local, no server needed) ───────────────────
const speakBtn = document.getElementById('speak-btn');
speakBtn.addEventListener('click', () => {
  if (speechSynthesis.speaking) { speechSynthesis.cancel(); speakBtn.classList.remove('active'); return; }
  const text = document.getElementById('feedback-text').textContent;
  if (!text) return;
  _speak(text, 0.92, speakBtn);
});

// ═══════════════════════════════ HISTORY ════════════════════════════════════
async function loadHistory() {
  const listEl    = document.getElementById('history-list');
  const countEl   = document.getElementById('history-count');
  listEl.innerHTML = '<p class="placeholder">Loading…</p>';

  try {
    const rows = await authFetch('/history').then(r => r.json());
    countEl.textContent = `${rows.length} session${rows.length !== 1 ? 's' : ''}`;

    if (rows.length === 0) {
      listEl.innerHTML = '<p class="placeholder">No recordings yet — go to Record to get started.</p>';
      return;
    }

    listEl.innerHTML = rows.map(r => `
      <div class="history-card" onclick="toggleCard(this)">
        <div class="history-header">
          <div class="history-meta">
            <span class="history-date">${fmtTs(r.timestamp)}</span>
            <span class="history-duration">${fmt(Math.round(r.duration || 0))}</span>
            ${r.archetype ? `<span class="archetype-badge">${esc(r.archetype)}</span>` : ''}
          </div>
          <div class="history-scores">
            ${miniScore(r.overall,       'Overall')}
            ${miniScore(r.pace,          'Pace')}
            ${miniScore(r.filler,        'Filler')}
            ${miniScore(r.pronunciation, 'Pronun.')}
            ${miniScore(r.grammar,       'Grammar')}
            ${miniScore(r.clarity,       'Clarity')}
            ${r.vocabulary != null ? miniScore(r.vocabulary, 'Vocab.') : ''}
            ${r.cefr_score != null ? miniScore(r.cefr_score, 'CEFR', r.cefr_level) : ''}
          </div>
        </div>
        <div class="history-body hidden">
          <div class="history-detail">
            <strong>Speed:</strong> ${(r.wpm ?? 0).toFixed(0)} WPM &nbsp;·&nbsp;
            <strong>Fillers:</strong> ${r.filler_count ?? 0} &nbsp;·&nbsp;
            <strong>Grammar issues:</strong> ${r.grammar_errors ?? 0}
            ${r.cefr_level ? `&nbsp;·&nbsp;<strong>CEFR:</strong> ${esc(r.cefr_level)}` : ''}
          </div>
          <div class="history-transcript">
            <strong>Transcript</strong><br>
            <span class="transcript-text">${esc(r.transcript ?? '')}</span>
          </div>
          <div class="history-feedback">
            <strong>Feedback</strong><br>
            ${esc(r.feedback ?? '')}
          </div>
        </div>
      </div>
    `).join('');
  } catch (err) {
    listEl.innerHTML = `<p class="placeholder">Failed to load: ${err.message}</p>`;
  }
}

function miniScore(s, label, displayText) {
  s = Math.round(s ?? 0);
  return `<div class="mini-score ${cls(s)}"><span>${displayText ?? s}</span><small>${label}</small></div>`;
}

window.toggleCard = card => card.querySelector('.history-body').classList.toggle('hidden');

function fmtTs(ts) {
  if (!ts || ts.length < 8) return '—';
  return `${ts.slice(6,8)}/${ts.slice(4,6)}/${ts.slice(0,4)}  ${ts.slice(9,11)}:${ts.slice(11,13)}`;
}

// ═══════════════════════════════ PROGRESS ════════════════════════════════════
async function loadProgress() {
  try {
    const rows = await authFetch('/history').then(r => r.json());

    if (rows.length === 0) {
      document.getElementById('tab-progress').innerHTML =
        '<p class="placeholder">No sessions yet — record a few to see your progress!</p>';
      return;
    }

    const chrono = [...rows].reverse();  // oldest → newest
    const labels = chrono.map((_, i) => `#${i + 1}`);

    // Stats
    const avg  = rows.reduce((s, r) => s + r.overall, 0) / rows.length;
    const best = Math.max(...rows.map(r => r.overall));
    document.getElementById('stat-sessions').textContent    = rows.length;
    document.getElementById('stat-avg').textContent         = avg.toFixed(1);
    document.getElementById('stat-best').textContent        = best.toFixed(1);
    document.getElementById('stat-avg').className           = 'stat-num ' + cls(avg);
    document.getElementById('stat-best').className          = 'stat-num good';

    const impEl = document.getElementById('stat-improvement');
    if (rows.length >= 3) {
      const recent = rows.slice(0, 3).reduce((s, r) => s + r.overall, 0) / 3;
      const early  = rows.slice(-3).reduce((s, r)  => s + r.overall, 0) / 3;
      const diff   = recent - early;
      impEl.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(1);
      impEl.className   = 'stat-num ' + (diff >= 0 ? 'good' : 'poor');
    } else {
      impEl.textContent = '—';
      impEl.className   = 'stat-num';
    }

    // Radar – latest session (Progress shows all 7 metrics, unlike the immediate
    // Quick Assessment result screen, which only shows the original 5)
    const latest = rows[0];
    renderRadar('progress-radar', {
      pace: latest.pace, filler: latest.filler,
      pronunciation: latest.pronunciation,
      grammar: latest.grammar, clarity: latest.clarity,
      vocabulary: latest.vocabulary, cefr: latest.cefr_score,
    }, true);

    // Overall trend
    renderLine('overall-trend', labels, chrono.map(r => r.overall), 'Overall Score', '#7c3aed');

    // Per-metric lines
    const mCfg = [
      { key: 'pace',          color: '#06b6d4' },
      { key: 'filler',        color: '#f59e0b' },
      { key: 'pronunciation', color: '#10b981' },
      { key: 'grammar',       color: '#3b82f6' },
      { key: 'clarity',       color: '#8b5cf6' },
      { key: 'vocabulary',    color: '#ec4899' },
      { key: 'cefr_score',    color: '#14b8a6', canvasKey: 'cefr' },
    ];
    mCfg.forEach(m => renderLine(`chart-${m.canvasKey || m.key}`, labels, chrono.map(r => r[m.key]), null, m.color));

  } catch (err) {
    console.error('Progress error:', err);
  }
}

// ── Chart helpers ─────────────────────────────────────────────────────────────
function renderRadar(id, scores, extended = false) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  if (charts[id]) charts[id].destroy();

  const labels = ['Pace', 'Filler Words', 'Pronunciation', 'Grammar', 'Clarity'];
  const data   = [scores.pace, scores.filler, scores.pronunciation, scores.grammar, scores.clarity];
  if (extended) {
    labels.push('Vocabulary', 'CEFR');
    data.push(scores.vocabulary, scores.cefr);
  }

  charts[id] = new Chart(canvas.getContext('2d'), {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: 'rgba(124,58,237,.18)',
        borderColor:     '#7c3aed',
        pointBackgroundColor: '#7c3aed',
        pointBorderColor:     '#fff',
        borderWidth: 2,
      }],
    },
    options: {
      animation: { duration: 600 },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 20, color: '#8892a4', backdropColor: 'transparent', font: { size: 10 } },
          grid:        { color: 'rgba(255,255,255,.08)' },
          angleLines:  { color: 'rgba(255,255,255,.08)' },
          pointLabels: { color: '#e2e8f0', font: { size: 12 } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderLine(id, labels, values, label, color) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  if (charts[id]) charts[id].destroy();

  charts[id] = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: label || '',
        data: values,
        borderColor:     color,
        backgroundColor: color + '22',
        pointBackgroundColor: color,
        tension: 0.4, fill: true,
        borderWidth: 2, pointRadius: 4,
      }],
    },
    options: {
      animation: { duration: 500 },
      responsive: true,
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { color: '#8892a4', stepSize: 20 },
          grid:  { color: 'rgba(255,255,255,.06)' },
        },
        x: {
          ticks: { color: '#8892a4' },
          grid:  { color: 'rgba(255,255,255,.06)' },
        },
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0', boxWidth: 12 }, display: !!label },
      },
    },
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function fmt(s) {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function esc(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function setStatus(msg) {
  statusEl.textContent = msg;
}
