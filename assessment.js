/* ═══════════════════════════════════════════════════════════════════════════
   VoiceCoach – Guided English Assessment
   Sequence: Name → Picture Talk (3) → Listen & Repeat (3) →
             Describe & Compare (5) → Full Assessment → Report
   Self-contained: shares only visual language (CSS classes) with script.js,
   not any of its state, to keep the existing Record tab untouched.
═══════════════════════════════════════════════════════════════════════════ */

(function () {
  const root = () => document.getElementById('assessment-root');

  let inited   = false;
  let manifest = null;
  let userName = '';
  const results = [];   // collected /assess/stage responses, in wizard order

  window.initAssessment = async function () {
    if (inited) return;
    inited = true;
    root().innerHTML = '<p class="placeholder">Loading assessment…</p>';
    try {
      manifest = await fetch('/assessment/manifest').then(r => r.json());
    } catch (err) {
      root().innerHTML = `<p class="placeholder">Could not load assessment: ${err.message}</p>`;
      inited = false;
      return;
    }
    renderIntro();
  };

  // ── Utilities (namespaced locally — no reliance on script.js globals) ────
  function pickMime() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  }
  function escHtml(s) {
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function fmtTime(s) {
    s = Math.max(0, Math.round(s));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }
  function fmtTs(ts) {
    if (!ts || ts.length < 8) return '—';
    return `${ts.slice(6, 8)}/${ts.slice(4, 6)}/${ts.slice(0, 4)} ${ts.slice(9, 11)}:${ts.slice(11, 13)}`;
  }
  function cls(s) { return s >= 80 ? 'good' : s >= 60 ? 'ok' : 'poor'; }

  async function postStage(stageType, stageId, blob, duration) {
    const fd = new FormData();
    fd.append('stage_type', stageType);
    fd.append('stage_id', stageId || '');
    fd.append('audio', blob, 'clip.webm');
    fd.append('duration', String(duration));
    const res = await authFetch('/assess/stage', { method: 'POST', body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `Server error ${res.status}`);
    }
    return res.json();
  }

  /* Renders a record button that uploads to /assess/stage on stop (manual or
     auto at maxSecs) and resolves with the server's scoring response. */
  function renderRecorder(container, { stageType, stageId, maxSecs, autoStart = false, label = 'Click to record' }) {
    return new Promise((resolve, reject) => {
      container.innerHTML = `
        <div class="rec-widget">
          <div class="timer-wrap"><span class="timer-text" id="rw-timer">${fmtTime(maxSecs)}</span><span class="timer-max">remaining</span></div>
          <button class="record-btn" id="rw-btn" title="Start / stop recording">
            <span class="btn-ring"></span>
            <span class="btn-core">
              <svg class="mic-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/>
                <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
              <svg class="stop-svg hidden" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
            </span>
          </button>
          <p class="record-hint" id="rw-hint">${escHtml(label)}</p>
          <p class="record-status" id="rw-status"></p>
        </div>`;

      const btn     = container.querySelector('#rw-btn');
      const timerEl = container.querySelector('#rw-timer');
      const hintEl  = container.querySelector('#rw-hint');
      const statusEl = container.querySelector('#rw-status');
      const micSvg  = btn.querySelector('.mic-svg');
      const stopSvg = btn.querySelector('.stop-svg');

      let recorder, stream, chunks = [], seconds = 0, tickId = null, startTs = 0, stopped = false;

      async function start() {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch {
          statusEl.textContent = 'Microphone access denied. Please allow microphone access.';
          return;
        }
        chunks = [];
        recorder = new MediaRecorder(stream, { mimeType: pickMime() });
        recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = onStop;
        startTs = Date.now();
        recorder.start(200);
        btn.classList.add('is-recording');
        micSvg.classList.add('hidden'); stopSvg.classList.remove('hidden');
        hintEl.textContent = 'Recording… click to stop';
        tickId = setInterval(() => {
          seconds = (Date.now() - startTs) / 1000;
          timerEl.textContent = fmtTime(Math.max(0, maxSecs - seconds));
          if (seconds >= maxSecs) stop();
        }, 200);
      }

      function stop() {
        if (stopped) return;
        stopped = true;
        clearInterval(tickId);
        if (recorder && recorder.state === 'recording') recorder.stop();
      }

      async function onStop() {
        stream.getTracks().forEach(t => t.stop());
        btn.classList.remove('is-recording');
        btn.disabled = true;
        statusEl.textContent = 'Uploading…';
        const blob = new Blob(chunks, { type: recorder.mimeType });
        try {
          const r = await postStage(stageType, stageId, blob, seconds);
          statusEl.textContent = 'Done ✓';
          resolve(r);
        } catch (err) {
          statusEl.textContent = `Error: ${err.message}`;
          reject(err);
        }
      }

      btn.addEventListener('click', () => {
        if (!recorder || recorder.state !== 'recording') start(); else stop();
      });

      if (autoStart) start();
    });
  }

  /* A skippable prep countdown: shows a timer and a "record now" button.
     Resolves either when the user clicks the button, or when the timer hits 0 —
     at which point the caller starts recording automatically either way. */
  function prepCountdown(container, seconds) {
    return new Promise(resolve => {
      let remaining = seconds;
      let done = false;
      container.innerHTML = `
        <div class="rec-widget">
          <div class="timer-wrap"><span class="timer-text" id="prep-timer">${fmtTime(remaining)}</span><span class="timer-max">until recording starts</span></div>
          <button class="assess-btn" id="prep-skip">I'm Ready — Record Now</button>
        </div>`;
      const timerEl = container.querySelector('#prep-timer');
      const skipBtn = container.querySelector('#prep-skip');
      function finish() {
        if (done) return;
        done = true;
        clearInterval(id);
        container.innerHTML = '';
        resolve();
      }
      const id = setInterval(() => {
        remaining -= 1;
        timerEl.textContent = fmtTime(remaining);
        if (remaining <= 0) finish();
      }, 1000);
      skipBtn.addEventListener('click', finish);
    });
  }

  // ── Stepper header ────────────────────────────────────────────────────────
  const STEP_LABELS = ['Name', 'Picture Talk', 'Listen & Repeat', 'Describe & Compare', 'Full Assessment', 'Report'];
  function stepperHtml(activeIdx) {
    return `<div class="assess-stepper">${STEP_LABELS.map((l, i) => `
      <div class="assess-step ${i === activeIdx ? 'active' : ''} ${i < activeIdx ? 'done' : ''}">
        <span class="assess-step-dot">${i < activeIdx ? '✓' : i + 1}</span>
        <span class="assess-step-label">${l}</span>
      </div>`).join('')}</div>`;
  }

  // ── Screen: Intro ────────────────────────────────────────────────────────
  function renderIntro() {
    root().innerHTML = `
      <div class="assess-intro text-card">
        <h2>Comprehensive English Assessment</h2>
        <p class="record-hint" style="margin:10px 0 20px;">
          A guided, five-part speaking assessment: state your name, repeat sentences you see,
          repeat sentences you hear, describe and compare pictures, then a final free-speaking
          assessment. Allow about 12–15 minutes in a quiet spot with your microphone ready.
        </p>
        <button class="assess-btn" id="assess-start">Begin Assessment</button>
      </div>
      <div id="assess-recent"></div>`;
    root().querySelector('#assess-start').addEventListener('click', renderNameStep);
    loadRecent();
  }

  async function loadRecent() {
    const el = document.getElementById('assess-recent');
    if (!el) return;
    try {
      const rows = await authFetch('/assessment/history').then(r => r.json());
      if (!rows.length) return;
      el.innerHTML = `
        <div class="section-header" style="margin-top:26px;"><h2>Recent Assessments</h2></div>
        <div class="history-list">
          ${rows.map(r => `
            <div class="history-card" style="cursor:default;">
              <div class="history-header">
                <div class="history-meta">
                  <span class="history-date">${escHtml(r.name || 'Anonymous')}</span>
                  <span class="history-duration">${fmtTs(r.timestamp)}</span>
                </div>
                <div class="history-scores">
                  <div class="mini-score ${cls(r.overall_score)}"><span>${Math.round(r.overall_score)}</span><small>Overall</small></div>
                  ${r.picture_talk_score != null ? `<div class="mini-score ${cls(r.picture_talk_score)}"><span>${Math.round(r.picture_talk_score)}</span><small>Pic. Talk</small></div>` : ''}
                  ${r.media_repeat_score != null ? `<div class="mini-score ${cls(r.media_repeat_score)}"><span>${Math.round(r.media_repeat_score)}</span><small>Repeat</small></div>` : ''}
                  ${r.picture_describe_score != null ? `<div class="mini-score ${cls(r.picture_describe_score)}"><span>${Math.round(r.picture_describe_score)}</span><small>Describe</small></div>` : ''}
                </div>
              </div>
            </div>`).join('')}
        </div>`;
    } catch { /* recent list is a nice-to-have — fail silently */ }
  }

  // ── Screen: Step 1 — Name ───────────────────────────────────────────────
  async function renderNameStep() {
    root().innerHTML = `
      ${stepperHtml(0)}
      <div class="record-panel" style="margin-top:20px;">
        <h3>Greetings! Welcome to the English Assessment Test</h3>
        <p class="record-hint">Press record and You can say: "My name is …" or "I am _"</p>
        <div id="name-rec"></div>
      </div>`;
    let r;
    try {
      r = await renderRecorder(root().querySelector('#name-rec'), {
        stageType: 'name', stageId: '', maxSecs: 15, label: 'Click to record your name',
      });
    } catch { return; }
    userName = r.name || '';
    results.push({ stage_type: 'name', stage_id: '', transcript: r.transcript, name: userName });
    root().querySelector('#name-rec').insertAdjacentHTML('beforeend', `
      <p class="record-status" style="color:var(--good);margin-top:10px;">
        Nice to meet you${userName ? ', ' + escHtml(userName) : ''}! Let's continue.
      </p>
      <button class="assess-btn" id="name-continue">Continue</button>`);
    root().querySelector('#name-continue').addEventListener('click', () => renderPictureTalk(0));
  }

  // ── Screen: Step 2 — Picture Talk ───────────────────────────────────────
  async function renderPictureTalk(idx) {
    const items = manifest.picture_talk;
    if (idx >= items.length) return renderMediaRepeat(0);
    const item = items[idx];
    root().innerHTML = `
      ${stepperHtml(1)}
      <div class="record-panel" style="margin-top:20px;">
        <h3>Picture Talk (${idx + 1} of ${items.length})</h3>
        <p class="record-hint">Read the picture and say the sentence aloud — recording starts automatically
          and runs up to 15 seconds. Stop as soon as you're done to move on right away.</p>
        <img class="assess-pic" src="/assets/picture-talk/${item.image}" alt="Picture ${idx + 1}" />
        <div id="pt-rec"></div>
      </div>`;
    let r;
    try {
      r = await renderRecorder(root().querySelector('#pt-rec'), {
        stageType: 'picture_talk', stageId: item.id, maxSecs: 15, autoStart: true, label: 'Recording…',
      });
    } catch { return; }
    results.push(r);
    renderPictureTalk(idx + 1);
  }

  // ── Screen: Step 3 — Listen & Repeat ────────────────────────────────────
  async function renderMediaRepeat(idx) {
    const items = manifest.media_repeat;
    if (idx >= items.length) return renderPictureDescribe(0);
    const item = items[idx];
    root().innerHTML = `
      ${stepperHtml(2)}
      <div class="record-panel" style="margin-top:20px;">
        <h3>Listen &amp; Repeat (${idx + 1} of ${items.length})</h3>
        <p class="record-hint">Play the clip and repeat what you hear — recording starts automatically
          and runs up to 15 seconds. Stop as soon as you're done to move on right away.</p>
        <audio class="assess-audio" controls src="/assets/media-repeat/${item.audio}"></audio>
        <div id="mr-rec"></div>
      </div>`;
    let r;
    try {
      r = await renderRecorder(root().querySelector('#mr-rec'), {
        stageType: 'media_repeat', stageId: item.id, maxSecs: 15, autoStart: true, label: 'Recording…',
      });
    } catch { return; }
    results.push(r);
    renderMediaRepeat(idx + 1);
  }

  // ── Screen: Step 4 — Describe & Compare ─────────────────────────────────
  async function renderPictureDescribe(idx) {
    const items = manifest.picture_describe;
    if (idx >= items.length) return renderFinal();
    const item = items[idx];
    root().innerHTML = `
      ${stepperHtml(3)}
      <div class="record-panel" style="margin-top:20px;">
        <h3>${escHtml(item.title)} (${idx + 1} of ${items.length})</h3>
        <img class="assess-pic" src="/assets/picture-describe/${item.image}" alt="${escHtml(item.title)}" />
        <ul class="assess-prompts">${item.prompts.map(p => `<li>${escHtml(p)}</li>`).join('')}</ul>
        <div id="pd-prep"></div>
        <div id="pd-rec"></div>
      </div>`;
    await prepCountdown(root().querySelector('#pd-prep'), 30);
    let r;
    try {
      r = await renderRecorder(root().querySelector('#pd-rec'), {
        stageType: 'picture_describe', stageId: item.id, maxSecs: item.speak_secs,
        autoStart: true, label: 'Recording…',
      });
    } catch { return; }
    results.push(r);
    renderPictureDescribe(idx + 1);
  }

  // ── Screen: Step 5 — Final free assessment ──────────────────────────────
  async function renderFinal() {
    root().innerHTML = `
      ${stepperHtml(4)}
      <div class="record-panel" style="margin-top:20px;">
        <h3>Full English Assessment</h3>
        <p class="record-hint">Speak freely — introduce yourself, or talk about your day, for up to 3 minutes.
          This is the official assessment your overall score is based on.</p>
        <div id="final-prep"></div>
        <div id="final-rec"></div>
      </div>`;
    await prepCountdown(root().querySelector('#final-prep'), 30);
    let r;
    try {
      r = await renderRecorder(root().querySelector('#final-rec'), {
        stageType: 'final', stageId: '', maxSecs: 180, autoStart: true, label: 'Recording…',
      });
    } catch { return; }
    results.push(r);
    renderReport();
  }

  // ── Screen: Report ───────────────────────────────────────────────────────
  async function renderReport() {
    root().innerHTML = `${stepperHtml(5)}<p class="placeholder">Building your report…</p>`;
    let report;
    try {
      const res = await authFetch('/assessment/finalize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: userName, stages: results }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error ${res.status}`);
      report = await res.json();
    } catch (err) {
      root().innerHTML = `<p class="placeholder">Could not save your report: ${err.message}</p>`;
      return;
    }

    const f = report.final;
    const sec = report.sections;
    const circ = 2 * Math.PI * 52;
    const offset = circ * (1 - report.overall_score / 100);

    root().innerHTML = `
      ${stepperHtml(5)}
      <div class="results-wrap" style="margin-top:20px;">
        <div class="results-top">
          <div class="overall-card">
            <svg class="ring-svg" viewBox="0 0 120 120">
              <circle class="ring-bg" cx="60" cy="60" r="52"/>
              <circle class="ring-fg ${cls(report.overall_score)}" cx="60" cy="60" r="52"
                      stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
            </svg>
            <div class="overall-inner">
              <span class="overall-num ${cls(report.overall_score)}">${Math.round(report.overall_score)}</span>
              <span class="overall-label">Overall</span>
            </div>
          </div>
          <div class="result-radar-card">
            <h3 class="card-title">Assessment Summary</h3>
            <p class="record-hint">${escHtml(userName ? userName + ', here' : 'Here')} is your comprehensive
              English Assessment report, covering all five stages. The overall score reflects your final
              free-speaking assessment (Step 5); Vocabulary Coverage and CEFR Level draw on your Describe &amp;
              Compare and Full Assessment recordings together, plus your repeat-task accuracy.</p>
          </div>
        </div>

        <div class="archetype-card">
          <span class="archetype-emoji">${f.archetype.emoji}</span>
          <div class="archetype-body">
            <span class="archetype-eyebrow">Voice Archetype</span>
            <h3 class="archetype-name">${escHtml(f.archetype.archetype)}</h3>
            <p class="archetype-desc">${escHtml(f.archetype.description)}</p>
            <div class="archetype-traits">${f.archetype.traits.map(t => `<span class="trait-pill">${escHtml(t)}</span>`).join('')}</div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">⚡</span><span class="metric-name">Pace</span><span class="metric-score ${cls(f.pace.score)}">${Math.round(f.pace.score)}</span></div><p class="metric-detail">${f.pace.wpm} WPM</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">🔇</span><span class="metric-name">Filler Words</span><span class="metric-score ${cls(f.filler.score)}">${Math.round(f.filler.score)}</span></div><p class="metric-detail">${f.filler.count} used</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">🗣</span><span class="metric-name">Pronunciation</span><span class="metric-score ${cls(f.pronunciation.score)}">${Math.round(f.pronunciation.score)}</span></div></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">✏️</span><span class="metric-name">Grammar</span><span class="metric-score ${cls(f.grammar.score)}">${Math.round(f.grammar.score)}</span></div><p class="metric-detail">${f.grammar.errors} issues</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">💎</span><span class="metric-name">Clarity</span><span class="metric-score ${cls(f.clarity.score)}">${Math.round(f.clarity.score)}</span></div></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">📚</span><span class="metric-name">Vocabulary Coverage</span><span class="metric-score ${cls(report.vocabulary.score)}">${Math.round(report.vocabulary.score)}</span></div><p class="metric-detail">${report.vocabulary.unique_words} unique words · ${report.vocabulary.advanced_ratio.toFixed(0)}% advanced</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">🏅</span><span class="metric-name">CEFR Level</span><span class="metric-score ${cls(report.cefr.score)}">${escHtml(report.cefr.level)}</span></div><p class="metric-detail">${report.cefr.score.toFixed(0)}% composite</p></div>
        </div>

        <div class="section-header"><h2>Guided Stages</h2></div>
        <div class="metrics-grid">
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">🖼️</span><span class="metric-name">Picture Talk</span><span class="metric-score ${cls(sec.picture_talk.score)}">${sec.picture_talk.score != null ? Math.round(sec.picture_talk.score) : '—'}</span></div><p class="metric-detail">Sentence-repeat accuracy across ${sec.picture_talk.items.length} pictures</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">🎧</span><span class="metric-name">Listen &amp; Repeat</span><span class="metric-score ${cls(sec.media_repeat.score)}">${sec.media_repeat.score != null ? Math.round(sec.media_repeat.score) : '—'}</span></div><p class="metric-detail">Listening-repeat accuracy across ${sec.media_repeat.items.length} clips</p></div>
          <div class="metric-card"><div class="metric-header"><span class="metric-icon">💬</span><span class="metric-name">Describe &amp; Compare</span><span class="metric-score ${cls(sec.picture_describe.score)}">${sec.picture_describe.score != null ? Math.round(sec.picture_describe.score) : '—'}</span></div><p class="metric-detail">Free-speech average across ${sec.picture_describe.items.length} prompts</p></div>
        </div>

        <div class="text-card">
          <div class="text-card-header"><h3>Final Assessment Transcript</h3></div>
          <p class="transcript-body">${escHtml(f.transcript)}</p>
        </div>
        <div class="text-card">
          <div class="text-card-header"><h3>Personalised Feedback</h3></div>
          <p class="feedback-body">${escHtml(f.feedback)}</p>
        </div>

        <div class="report-actions">
          <button class="assess-btn" id="export-pdf">📄 Export PDF Report</button>
          <button class="assess-btn-secondary" id="restart-assess">Start New Assessment</button>
          <span class="assess-status" id="export-status"></span>
        </div>
      </div>`;

    root().querySelector('#export-pdf').addEventListener('click', () => {
      const statusEl = root().querySelector('#export-status');
      try {
        downloadPdfReport(report);
        statusEl.textContent = '';
      } catch (err) {
        statusEl.textContent = `Could not generate PDF: ${err.message}`;
      }
    });

    root().querySelector('#restart-assess').addEventListener('click', () => {
      results.length = 0;
      userName = '';
      renderIntro();
    });
  }

  // ── PDF export ──────────────────────────────────────────────────────────
  function downloadPdfReport(report) {
    if (!window.jspdf) throw new Error('PDF library failed to load — check your connection and try again.');
    const { jsPDF } = window.jspdf;
    const f   = report.final;
    const sec = report.sections;

    const doc    = new jsPDF({ unit: 'mm', format: 'a4' });
    const pageW  = doc.internal.pageSize.getWidth();
    const pageH  = doc.internal.pageSize.getHeight();
    const marginX = 18;
    const maxW    = pageW - marginX * 2;
    let y = 20;

    function ensureSpace(needed) {
      if (y + needed > pageH - 16) { doc.addPage(); y = 20; }
    }
    function heading(text) {
      ensureSpace(11);
      doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.setTextColor(30, 30, 30);
      doc.text(text, marginX, y);
      y += 7;
    }
    function field(label, value) {
      ensureSpace(6.5);
      doc.setFont('helvetica', 'bold'); doc.setFontSize(10); doc.setTextColor(90, 90, 90);
      doc.text(label, marginX, y);
      doc.setFont('helvetica', 'normal'); doc.setTextColor(40, 40, 40);
      doc.text(String(value), marginX + 58, y);
      y += 6.5;
    }
    function paragraph(text) {
      doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(60, 60, 60);
      const lines = doc.splitTextToSize(text || '—', maxW);
      lines.forEach(l => { ensureSpace(5.5); doc.text(l, marginX, y); y += 5.5; });
      y += 4;
    }
    function divider() {
      ensureSpace(7);
      doc.setDrawColor(225, 225, 225);
      doc.line(marginX, y, pageW - marginX, y);
      y += 8;
    }

    // Header
    doc.setFont('helvetica', 'bold'); doc.setFontSize(18); doc.setTextColor(224, 85, 48);
    doc.text('English Assessment Report', marginX, y);
    y += 9;
    doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(120, 120, 120);
    doc.text(`Name: ${userName || 'Anonymous'}`, marginX, y); y += 5.5;
    doc.text(`Date: ${fmtTs(report.timestamp)}`, marginX, y); y += 10;
    divider();

    // Overall score
    heading('Overall Score');
    doc.setFont('helvetica', 'bold'); doc.setFontSize(26); doc.setTextColor(224, 85, 48);
    doc.text(`${Math.round(report.overall_score)} / 100`, marginX, y + 2);
    y += 13;
    divider();

    // Voice Archetype
    heading('Voice Archetype');
    field('Archetype:', f.archetype.archetype);
    paragraph(f.archetype.description);
    paragraph(`Traits: ${f.archetype.traits.join(', ')}`);
    divider();

    // Core metrics
    heading('Full Assessment — Core Metrics');
    field('Pace:', `${Math.round(f.pace.score)} / 100  (${f.pace.wpm} WPM)`);
    field('Filler Words:', `${Math.round(f.filler.score)} / 100  (${f.filler.count} used)`);
    field('Pronunciation:', `${Math.round(f.pronunciation.score)} / 100`);
    field('Grammar:', `${Math.round(f.grammar.score)} / 100  (${f.grammar.errors} issues)`);
    field('Clarity:', `${Math.round(f.clarity.score)} / 100`);
    divider();

    // Vocabulary / CEFR (assessment-wide)
    heading('Vocabulary & CEFR (assessment-wide)');
    field('Vocabulary Coverage:',
      `${Math.round(report.vocabulary.score)} / 100  (${report.vocabulary.unique_words} unique words, ${report.vocabulary.advanced_ratio.toFixed(0)}% advanced)`);
    field('CEFR Level:', `${report.cefr.level}  (${report.cefr.score.toFixed(0)}% composite)`);
    divider();

    // Guided stages
    heading('Guided Stages');
    field('Picture Talk:', sec.picture_talk.score != null ? `${Math.round(sec.picture_talk.score)} / 100` : 'Not completed');
    field('Listen & Repeat:', sec.media_repeat.score != null ? `${Math.round(sec.media_repeat.score)} / 100` : 'Not completed');
    field('Describe & Compare:', sec.picture_describe.score != null ? `${Math.round(sec.picture_describe.score)} / 100` : 'Not completed');
    divider();

    // Transcript
    heading('Full Assessment Transcript');
    paragraph(f.transcript);
    divider();

    // Feedback
    heading('Personalised Feedback');
    paragraph(f.feedback);

    // Footer on every page
    const pageCount = doc.internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFont('helvetica', 'normal'); doc.setFontSize(8); doc.setTextColor(160, 160, 160);
      doc.text(`VoiceAssessment by Talking Labs — Page ${i} of ${pageCount}`, marginX, pageH - 8);
    }

    const safeName = (userName || 'report').trim().replace(/\s+/g, '-').toLowerCase().replace(/[^a-z0-9-]/g, '');
    doc.save(`english-assessment-${safeName}-${report.timestamp}.pdf`);
  }
})();
