// VoiceCoach Debug UI — standalone client for POST /debug/analyze-audio
// Records live from the microphone (MediaRecorder), then sends the recording
// through the SAME pipeline production's /assess uses (save_and_convert ->
// transcribe_wav -> score_free_speech). Renders only fields the backend
// actually returns. Never computes scores itself. No file upload — the
// recorded Blob is the only input this UI sends.

const ENDPOINT = "/debug/analyze-audio";
const PROVIDERS_ENDPOINT = "/pronunciation-providers";
const SAVED_RESULTS_ENDPOINT = "/tutor/debug-data";
const GUIDED_RESULTS_ENDPOINT = "/tutor/guided-data";

const els = {
  recordBtn: document.getElementById("recordBtn"),
  recStatus: document.getElementById("recStatus"),
  recTimer: document.getElementById("recTimer"),
  audioPreview: document.getElementById("audioPreview"),
  postRecordRow: document.getElementById("postRecordRow"),
  recordedDurationLine: document.getElementById("recordedDurationLine"),
  recorderError: document.getElementById("recorderError"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusLine: document.getElementById("statusLine"),
  results: document.getElementById("results"),
  analysisResults: document.getElementById("analysisResults"),
  emptyState: document.getElementById("emptyState"),
  learnerReport: document.getElementById("learnerReport"),
  waveform: document.getElementById("waveform"),
  pronProviderBox: document.getElementById("pronProviderBox"),
  sttProviderBox: document.getElementById("sttProviderBox"),
};

// ── Tab Switching ──────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
  
  if (tab === 'saved') {
    const password = document.getElementById('tutorPassword').value;
    if (password) {
      loadSavedResults();
    }
  }
  if (tab === 'guided') {
    const password = document.getElementById('guidedPassword').value;
    if (password) {
      loadGuidedAssessments();
    }
  }
}

// ── Saved Quick Assessments Viewer ──────────────────────────────────────────
let savedAssessments = [];

async function loadSavedResults() {
  const password = document.getElementById('tutorPassword').value.trim();
  const container = document.getElementById('savedResultsContainer');
  const errorEl = document.getElementById('passwordError');
  
  errorEl.style.display = 'none';
  
  if (!password) {
    container.innerHTML = '<div class="saved-loading">Please enter the password.</div>';
    return;
  }
  
  container.innerHTML = '<div class="saved-loading">Loading saved results...</div>';
  
  try {
    const res = await fetch(SAVED_RESULTS_ENDPOINT + '?password=' + encodeURIComponent(password));
    
    if (res.status === 401) {
      errorEl.style.display = 'block';
      container.innerHTML = '<div class="saved-loading">Enter the correct password to view results.</div>';
      return;
    }
    
    if (!res.ok) {
      throw new Error('Failed to load data');
    }
    
    const data = await res.json();
    savedAssessments = data;
    
    document.getElementById('savedBadge').textContent = data.length || 0;
    
    if (!data || data.length === 0) {
      container.innerHTML = `
        <div class="saved-loading" style="padding:30px;">
          <div style="font-size:32px;margin-bottom:12px;">📭</div>
          No saved debug assessments found.<br>
          Run some analyses in the <strong>Analysis Results</strong> tab first.
        </div>
      `;
      return;
    }
    
    renderSavedTable(data);
    
  } catch (err) {
    container.innerHTML = `
      <div class="saved-loading" style="color:var(--bad);">
        ❌ Error: ${err.message}
      </div>
    `;
  }
}

function renderSavedTable(data) {
  const container = document.getElementById('savedResultsContainer');
  
  const scores = data.map(r => r.overall_score || r.overall || 0).filter(s => s > 0);
  const total = data.length;
  const avg = scores.length ? (scores.reduce((a,b) => a + b, 0) / scores.length).toFixed(1) : '—';
  const best = scores.length ? Math.max(...scores).toFixed(1) : '—';
  const worst = scores.length ? Math.min(...scores).toFixed(1) : '—';
  
  const avgClass = avg !== '—' ? (avg >= 80 ? 'good' : avg >= 60 ? 'ok' : 'poor') : '';
  const bestClass = best !== '—' ? (best >= 80 ? 'good' : best >= 60 ? 'ok' : 'poor') : '';
  const worstClass = worst !== '—' ? (worst >= 80 ? 'good' : worst >= 60 ? 'ok' : 'poor') : '';
  
  let html = `
    <div class="saved-stats">
      <div class="saved-stat">
        <div class="num">${total}</div>
        <div class="label">Total</div>
      </div>
      <div class="saved-stat">
        <div class="num ${avgClass}">${avg}</div>
        <div class="label">Average</div>
      </div>
      <div class="saved-stat">
        <div class="num ${bestClass}">${best}</div>
        <div class="label">Best</div>
      </div>
      <div class="saved-stat">
        <div class="num ${worstClass}">${worst}</div>
        <div class="label">Worst</div>
      </div>
    </div>
    
    <div class="saved-header-row">
      <span class="title">📋 All Quick Assessments</span>
      <div class="actions">
        <button class="refresh-btn" onclick="loadSavedResults()">🔄 Refresh</button>
        <button class="export-btn" onclick="exportSavedCSV()">📥 Export CSV</button>
      </div>
    </div>
    
    <div class="saved-results-wrap">
      <table class="saved-results-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Timestamp</th>
            <th>Transcript</th>
            <th>Overall</th>
            <th>Pace</th>
            <th>WPM</th>
            <th>Filler</th>
            <th>Count</th>
            <th>Pronun.</th>
            <th>Grammar</th>
            <th>Errors</th>
            <th>Clarity</th>
            <th>Vocab.</th>
            <th>CEFR</th>
            <th>Archetype</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  data.forEach((r, index) => {
    const overall = r.overall_score || r.overall || 0;
    const pace = r.pace_score || r.pace?.score || 0;
    const wpm = r.pace_wpm || r.pace?.wpm || '—';
    const filler = r.filler_score || r.filler?.score || 0;
    const fillerCount = r.filler_count || r.filler?.count || '—';
    const pronunciation = r.pronunciation_score || r.pronunciation?.score || 0;
    const grammar = r.grammar_score || r.grammar?.score || 0;
    const errors = r.grammar_errors || r.grammar?.errors || '—';
    const clarity = r.clarity_score || r.clarity?.score || 0;
    const vocabulary = r.vocabulary_score || r.vocabulary?.score || 0;
    const cefr = r.cefr_level || r.cefr?.level || '—';
    const archetype = r.archetype || r.archetype?.archetype || '—';
    const transcript = r.transcript || '—';
    
    const overallClass = overall >= 80 ? 'good' : overall >= 60 ? 'ok' : 'poor';
    const paceClass = pace >= 80 ? 'good' : pace >= 60 ? 'ok' : 'poor';
    const fillerClass = filler >= 80 ? 'good' : filler >= 60 ? 'ok' : 'poor';
    const pronunClass = pronunciation >= 80 ? 'good' : pronunciation >= 60 ? 'ok' : 'poor';
    const grammarClass = grammar >= 80 ? 'good' : grammar >= 60 ? 'ok' : 'poor';
    const clarityClass = clarity >= 80 ? 'good' : clarity >= 60 ? 'ok' : 'poor';
    const vocabClass = vocabulary >= 80 ? 'good' : vocabulary >= 60 ? 'ok' : 'poor';
    
    const timestamp = r.timestamp || '—';
    const truncated = transcript.length > 50 ? transcript.substring(0, 50) + '...' : transcript;
    
    html += `
      <tr>
        <td style="color:var(--text-faint);font-size:10px;">#${r.id}</td>
        <td style="font-size:10px;color:var(--text-faint);">${timestamp}</td>
        <td class="transcript-cell" title="${transcript}">${truncated}</td>
        <td class="score-cell ${overallClass}">${overall}</td>
        <td class="score-cell ${paceClass}">${pace}</td>
        <td>${wpm}</td>
        <td class="score-cell ${fillerClass}">${filler}</td>
        <td>${fillerCount}</td>
        <td class="score-cell ${pronunClass}">${pronunciation}</td>
        <td class="score-cell ${grammarClass}">${grammar}</td>
        <td>${errors}</td>
        <td class="score-cell ${clarityClass}">${clarity}</td>
        <td class="score-cell ${vocabClass}">${vocabulary}</td>
        <td>${cefr}</td>
        <td>${archetype !== '—' ? `<span class="archetype-badge">${archetype}</span>` : '—'}</td>
        <td>
          <button class="view-btn" onclick="viewSavedDetail(${index})">📄</button>
        </td>
      </tr>
    `;
  });
  
  html += `
        </tbody>
      </table>
    </div>
  `;
  
  container.innerHTML = html;
}

function viewSavedDetail(index) {
  const r = savedAssessments[index];
  const modal = document.getElementById('detailModal');
  const body = document.getElementById('modalBody');
  
  document.getElementById('modalTitle').textContent = `Quick Assessment #${r.id} - ${r.timestamp || 'Unknown date'}`;
  
  const fields = [
    ['ID', r.id],
    ['Timestamp', r.timestamp || '—'],
    ['Transcript', r.transcript || '—'],
    ['Overall', r.overall_score || r.overall || '—'],
    ['Pace', r.pace_score || r.pace?.score || '—'],
    ['WPM', r.pace_wpm || r.pace?.wpm || '—'],
    ['Filler', r.filler_score || r.filler?.score || '—'],
    ['Filler Count', r.filler_count || r.filler?.count || '—'],
    ['Pronunciation', r.pronunciation_score || r.pronunciation?.score || '—'],
    ['Grammar', r.grammar_score || r.grammar?.score || '—'],
    ['Grammar Errors', r.grammar_errors || r.grammar?.errors || '—'],
    ['Clarity', r.clarity_score || r.clarity?.score || '—'],
    ['Vocabulary', r.vocabulary_score || r.vocabulary?.score || '—'],
    ['CEFR', r.cefr_level || r.cefr?.level || '—'],
    ['Archetype', r.archetype || r.archetype?.archetype || '—'],
    ['Feedback', r.feedback || '—'],
  ];
  
  body.innerHTML = fields.map(([label, value]) => `
    <div class="detail-row">
      <span class="label">${label}</span>
      <span class="value">${value}</span>
    </div>
  `).join('');
  
  if (r.full_result) {
    body.innerHTML += `
      <div class="detail-row" style="flex-direction:column;align-items:stretch;">
        <span class="label" style="width:100%;margin-bottom:4px;">Full JSON</span>
        <div class="value"><pre>${JSON.stringify(r.full_result, null, 2)}</pre></div>
      </div>
    `;
  }
  
  modal.classList.add('active');
}

function exportSavedCSV() {
  if (!savedAssessments || savedAssessments.length === 0) {
    alert('No data to export');
    return;
  }
  
  const headers = ['ID', 'Timestamp', 'Transcript', 'Overall', 'Pace', 'WPM', 'Filler', 'Filler Count',
    'Pronunciation', 'Grammar', 'Errors', 'Clarity', 'Vocabulary', 'CEFR', 'Archetype'];
  
  let csv = headers.join(',') + '\n';
  
  savedAssessments.forEach(r => {
    const row = [
      r.id,
      r.timestamp || '',
      (r.transcript || '').replace(/,/g, ' '),
      r.overall_score || r.overall || '',
      r.pace_score || r.pace?.score || '',
      r.pace_wpm || r.pace?.wpm || '',
      r.filler_score || r.filler?.score || '',
      r.filler_count || r.filler?.count || '',
      r.pronunciation_score || r.pronunciation?.score || '',
      r.grammar_score || r.grammar?.score || '',
      r.grammar_errors || r.grammar?.errors || '',
      r.clarity_score || r.clarity?.score || '',
      r.vocabulary_score || r.vocabulary?.score || '',
      r.cefr_level || r.cefr?.level || '',
      r.archetype || r.archetype?.archetype || ''
    ];
    csv += row.join(',') + '\n';
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `debug_assessments_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Guided Assessments Viewer with Teacher Reports ──────────────────────────
let guidedAssessments = [];

async function loadGuidedAssessments() {
  const password = document.getElementById('guidedPassword').value.trim();
  const container = document.getElementById('guidedResultsContainer');
  const errorEl = document.getElementById('guidedPasswordError');
  
  errorEl.style.display = 'none';
  
  if (!password) {
    container.innerHTML = '<div class="saved-loading">Please enter the password.</div>';
    return;
  }
  
  container.innerHTML = '<div class="saved-loading">Loading guided assessments...</div>';
  
  try {
    const res = await fetch(GUIDED_RESULTS_ENDPOINT + '?password=' + encodeURIComponent(password));
    
    if (res.status === 401) {
      errorEl.style.display = 'block';
      container.innerHTML = '<div class="saved-loading">Enter the correct password to view results.</div>';
      return;
    }
    
    if (!res.ok) {
      throw new Error('Failed to load data');
    }
    
    const data = await res.json();
    guidedAssessments = data;
    
    document.getElementById('guidedBadge').textContent = data.length || 0;
    
    if (!data || data.length === 0) {
      container.innerHTML = `
        <div class="saved-loading" style="padding:30px;">
          <div style="font-size:32px;margin-bottom:12px;">📭</div>
          No guided assessments found.<br>
          Complete a guided English Assessment in the main app first.
        </div>
      `;
      return;
    }
    
    renderGuidedTable(data);
    
  } catch (err) {
    container.innerHTML = `
      <div class="saved-loading" style="color:var(--bad);">
        ❌ Error: ${err.message}
      </div>
    `;
  }
}

function renderGuidedTable(data) {
  const container = document.getElementById('guidedResultsContainer');
  
  const scores = data.map(r => r.overall_score || 0).filter(s => s > 0);
  const total = data.length;
  const avg = scores.length ? (scores.reduce((a,b) => a + b, 0) / scores.length).toFixed(1) : '—';
  const best = scores.length ? Math.max(...scores).toFixed(1) : '—';
  const worst = scores.length ? Math.min(...scores).toFixed(1) : '—';
  
  const avgClass = avg !== '—' ? (avg >= 80 ? 'good' : avg >= 60 ? 'ok' : 'poor') : '';
  const bestClass = best !== '—' ? (best >= 80 ? 'good' : best >= 60 ? 'ok' : 'poor') : '';
  const worstClass = worst !== '—' ? (worst >= 80 ? 'good' : worst >= 60 ? 'ok' : 'poor') : '';
  
  let html = `
    <div class="saved-stats">
      <div class="saved-stat">
        <div class="num">${total}</div>
        <div class="label">Total</div>
      </div>
      <div class="saved-stat">
        <div class="num ${avgClass}">${avg}</div>
        <div class="label">Average</div>
      </div>
      <div class="saved-stat">
        <div class="num ${bestClass}">${best}</div>
        <div class="label">Best</div>
      </div>
      <div class="saved-stat">
        <div class="num ${worstClass}">${worst}</div>
        <div class="label">Worst</div>
      </div>
    </div>
    
    <div class="saved-header-row">
      <span class="title">📋 Guided English Assessments with Teacher Reports</span>
      <div class="actions">
        <button class="refresh-btn" onclick="loadGuidedAssessments()">🔄 Refresh</button>
        <button class="export-btn" onclick="exportGuidedCSV()">📥 Export CSV</button>
      </div>
    </div>
    
    <div class="saved-results-wrap">
      <table class="saved-results-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Timestamp</th>
            <th>Name</th>
            <th>Overall</th>
            <th>Pic Talk</th>
            <th>Repeat</th>
            <th>Describe</th>
            <th>Vocabulary</th>
            <th>CEFR</th>
            <th>Archetype</th>
            <th>Teacher Report</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  data.forEach((r, index) => {
    const hasReport = r.teacher_report !== null && r.teacher_report !== undefined;
    const reportStatus = hasReport ? '✅ Available' : '—';
    
    const timestamp = r.timestamp || '—';
    const name = r.name || 'Anonymous';
    const overall = r.overall_score || 0;
    const overallClass = overall >= 80 ? 'good' : overall >= 60 ? 'ok' : 'poor';
    
    html += `
      <tr>
        <td style="color:var(--text-faint);font-size:10px;">#${r.id}</td>
        <td style="font-size:10px;color:var(--text-faint);">${timestamp}</td>
        <td>${name}</td>
        <td class="score-cell ${overallClass}">${overall}</td>
        <td>${r.picture_talk_score || '—'}</td>
        <td>${r.media_repeat_score || '—'}</td>
        <td>${r.picture_describe_score || '—'}</td>
        <td>${r.vocabulary_score || '—'}</td>
        <td>${r.cefr_level || '—'}</td>
        <td>${r.archetype || '—'}</td>
        <td>${hasReport ? `<span style="color:var(--good);font-weight:600;">✅ Available</span>` : `<span style="color:var(--text-faint);">—</span>`}</td>
        <td>
          <button class="view-btn" onclick="viewGuidedDetail(${index})">📄</button>
          ${hasReport ? `<button class="report-btn" onclick="viewTeacherReport(${index})">📝 Report</button>` : ''}
        </td>
      </tr>
    `;
  });
  
  html += `
        </tbody>
      </table>
    </div>
  `;
  
  container.innerHTML = html;
}

function viewGuidedDetail(index) {
  const r = guidedAssessments[index];
  const modal = document.getElementById('detailModal');
  const body = document.getElementById('modalBody');
  
  document.getElementById('modalTitle').textContent = `Guided Assessment #${r.id} - ${r.name || 'Anonymous'}`;
  
  const fields = [
    ['ID', r.id],
    ['Timestamp', r.timestamp || '—'],
    ['Name', r.name || '—'],
    ['Overall Score', r.overall_score || '—'],
    ['Picture Talk', r.picture_talk_score || '—'],
    ['Listen & Repeat', r.media_repeat_score || '—'],
    ['Describe & Compare', r.picture_describe_score || '—'],
    ['Vocabulary', r.vocabulary_score || '—'],
    ['CEFR', r.cefr_level || '—'],
    ['Archetype', r.archetype || '—'],
  ];
  
  body.innerHTML = fields.map(([label, value]) => `
    <div class="detail-row">
      <span class="label">${label}</span>
      <span class="value">${value}</span>
    </div>
  `).join('');
  
  if (r.full_result) {
    body.innerHTML += `
      <div class="detail-row" style="flex-direction:column;align-items:stretch;">
        <span class="label" style="width:100%;margin-bottom:4px;">Full Result</span>
        <div class="value"><pre>${JSON.stringify(r.full_result, null, 2)}</pre></div>
      </div>
    `;
  }
  
  modal.classList.add('active');
}

function viewTeacherReport(index) {
  const r = guidedAssessments[index];

  // Shows the report inline in the Analysis tab (reusing the same
  // container/pipeline as a live recording's results) instead of a modal
  // popup — switches tabs once, to bring the learner to where the content
  // now lives, then renders in place; no dialog overlay involved.
  switchTab('analysis');
  els.emptyState.style.display = "none";
  els.analysisResults.innerHTML = "";
  const debugToggle = document.getElementById('debugDetailsToggle');
  if (debugToggle) debugToggle.open = true; // this view only ever renders into Debug Details, so expand it

  const report = r.teacher_report;
  const title = `📝 Teacher Report — ${r.name || 'Assessment #' + r.id}`;

  if (!report) {
    els.analysisResults.appendChild(
      card(`<h3>${esc(title)}</h3><div class="notice">${
        esc(r.teacher_report_detail || 'No teacher report available for this assessment.')
      }</div>`)
    );
    els.analysisResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }

  // Renders the ACTUAL Groq-generated report (same shape produced by
  // groq_provider.py: overview / growth_areas / vocabulary / repetitions /
  // advanced_grammar_used / performance_summary) via the shared builder —
  // this used to render a hardcoded strengths/areas_for_improvement/
  // specific_recommendations/teacher_notes shape that the backend never
  // actually sends, so real Groq remarks never showed up here.
  const html = `
    <div class="teacher-report">
      <div class="report-summary">
        <h4>📊 Summary</h4>
        <div class="metric-grid" style="grid-template-columns: repeat(3, 1fr);">
          <div class="metric">
            <div class="label">Overall Score</div>
            <div class="value">${r.overall_score ?? '—'}</div>
          </div>
          <div class="metric">
            <div class="label">CEFR Level</div>
            <div class="value">${r.cefr_level ?? '—'}</div>
          </div>
          <div class="metric">
            <div class="label">Archetype</div>
            <div class="value">${r.archetype ?? '—'}</div>
          </div>
        </div>
      </div>
      ${buildTeacherReportHtml(report)}
    </div>
  `;

  els.analysisResults.appendChild(card(`<h3>${esc(title)}</h3>${html}`));
  els.analysisResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function exportGuidedCSV() {
  if (!guidedAssessments || guidedAssessments.length === 0) {
    alert('No data to export');
    return;
  }
  
  const headers = ['ID', 'Timestamp', 'Name', 'Overall', 'Picture Talk', 'Listen & Repeat', 
    'Describe & Compare', 'Vocabulary', 'CEFR', 'Archetype', 'Teacher Report'];
  
  let csv = headers.join(',') + '\n';
  
  guidedAssessments.forEach(r => {
    const hasReport = r.teacher_report !== null && r.teacher_report !== undefined;
    const row = [
      r.id,
      r.timestamp || '',
      (r.name || '').replace(/,/g, ' '),
      r.overall_score || '',
      r.picture_talk_score || '',
      r.media_repeat_score || '',
      r.picture_describe_score || '',
      r.vocabulary_score || '',
      r.cefr_level || '',
      r.archetype || '',
      hasReport ? 'Yes' : 'No'
    ];
    csv += row.join(',') + '\n';
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `guided_assessments_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function closeModal() {
  document.getElementById('detailModal').classList.remove('active');
}

// Enter key on password inputs
document.getElementById('tutorPassword').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') loadSavedResults();
});
document.getElementById('guidedPassword').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') loadGuidedAssessments();
});

// Close modal on overlay click
document.getElementById('detailModal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// ── STT provider selector ──────────────────────────────────────────────────
async function initSttProviderSelector() {
  if (!els.sttProviderBox) return;
  try {
    const res = await fetch("/stt-providers");
    if (!res.ok) return;
    const data = await res.json();
    const statuses = data.providers || {};
    els.sttProviderBox.querySelectorAll(".provider-option").forEach((label) => {
      const name = label.dataset.provider;
      const input = label.querySelector("input");
      const badge = label.querySelector(".badge");
      const available = statuses[name] && statuses[name].available;
      if (!available) {
        input.disabled = true;
        label.classList.add("is-disabled");
        if (badge) {
          badge.textContent = "Not configured";
          badge.classList.add("unavailable");
        }
        if (input.checked) {
          const defaultInput = els.sttProviderBox.querySelector(
            `input[value="${data.default || "whisper"}"]`
          );
          if (defaultInput) defaultInput.checked = true;
        }
      }
    });
  } catch (err) {
    // Backend unreachable — leave at defaults
  }
}

function getSelectedSttProvider() {
  if (!els.sttProviderBox) return "whisper";
  const checked = els.sttProviderBox.querySelector('input[name="sttProvider"]:checked');
  return checked ? checked.value : "whisper";
}

initSttProviderSelector();

// ── Pronunciation provider selector ──────────────────────────────────────
async function initPronunciationProviderSelector() {
  if (!els.pronProviderBox) return;
  try {
    const res = await fetch(PROVIDERS_ENDPOINT);
    if (!res.ok) return;
    const data = await res.json();
    const statuses = data.providers || {};
    els.pronProviderBox.querySelectorAll(".provider-option").forEach((label) => {
      const name = label.dataset.provider;
      if (name === "gop") return;
      const input = label.querySelector("input");
      const badge = label.querySelector(".badge");
      const available = statuses[name] && statuses[name].available;
      if (!available) {
        input.disabled = true;
        label.classList.add("is-disabled");
        if (badge) {
          badge.textContent = "Not configured";
          badge.classList.add("unavailable");
        }
        if (input.checked) {
          const defaultInput = els.pronProviderBox.querySelector(
            `input[value="${data.default || "whisper_confidence"}"]`
          );
          if (defaultInput) defaultInput.checked = true;
        }
      }
    });
  } catch (err) {
    // Backend unreachable — leave at defaults
  }
}

function getSelectedPronunciationProvider() {
  if (!els.pronProviderBox) return "whisper_confidence";
  const checked = els.pronProviderBox.querySelector('input[name="pronunciationProvider"]:checked');
  return checked ? checked.value : "whisper_confidence";
}

initPronunciationProviderSelector();

// ─── Waveform ──────────────────────────────────────────────────────────────
const waveBars = els.waveform ? els.waveform.querySelectorAll(".bar") : [];
let waveHandle = null;

function animateWave() {
  waveBars.forEach((bar) => {
    bar.style.height = 6 + Math.random() * 28 + "px";
  });
}

function resetWave() {
  waveBars.forEach((bar) => {
    bar.style.height = "6px";
  });
  if (els.waveform) els.waveform.classList.remove("live");
}

function startWaveAnimation() {
  if (!els.waveform) return;
  els.waveform.classList.add("live");
  clearInterval(waveHandle);
  waveHandle = setInterval(animateWave, 120);
}

function stopWaveAnimation() {
  clearInterval(waveHandle);
  resetWave();
}

// ─── Recording state ──────────────────────────────────────────────────────
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recordedDurationSec = 0;
let recordedMimeType = "";
let recordingStartedAt = 0;
let timerHandle = null;
let isRecording = false;

const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
  "audio/mp4",
];

function pickSupportedMimeType() {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return null;
  }
  for (const type of CANDIDATE_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function showRecorderError(message) {
  els.recorderError.textContent = message;
  els.recorderError.style.display = "block";
}

function clearRecorderError() {
  els.recorderError.style.display = "none";
  els.recorderError.textContent = "";
}

function formatTimer(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function setRecordingUi(recording) {
  isRecording = recording;
  if (recording) {
    els.recordBtn.innerHTML = '<span class="dot"></span> Stop recording';
    els.recordBtn.classList.add("recording");
    els.recStatus.textContent = "Recording";
    els.recStatus.className = "rec-status live";
    els.postRecordRow.style.display = "none";
    els.audioPreview.style.display = "none";
    els.analyzeBtn.disabled = true;
    startWaveAnimation();
  } else {
    els.recordBtn.innerHTML = '<span class="dot"></span> Start recording';
    els.recordBtn.classList.remove("recording");
    stopWaveAnimation();
  }
}

async function startRecording() {
  clearRecorderError();
  setStatus("", "");
  recordedBlob = null;
  recordedChunks = [];
  els.postRecordRow.style.display = "none";
  els.audioPreview.style.display = "none";
  els.analyzeBtn.disabled = true;

  if (typeof navigator.mediaDevices === "undefined" || !navigator.mediaDevices.getUserMedia) {
    showRecorderError("This browser doesn't support microphone recording.");
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    showRecorderError("This browser doesn't support the MediaRecorder API.");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "SecurityError") {
      showRecorderError("Microphone permission was denied. Allow microphone access and try again.");
    } else if (err.name === "NotFoundError") {
      showRecorderError("No microphone was found. Connect a microphone and try again.");
    } else {
      showRecorderError(`Couldn't access the microphone: ${err.message || err.name}`);
    }
    return;
  }

  const mimeType = pickSupportedMimeType();
  if (mimeType === null) {
    showRecorderError("Can't safely pick a recording format in this browser.");
    stopStreamTracks();
    return;
  }
  recordedMimeType = mimeType;

  try {
    mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
  } catch (err) {
    showRecorderError(`Failed to start MediaRecorder: ${err.message || err}`);
    stopStreamTracks();
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onerror = (e) => {
    showRecorderError(`Recording failed: ${e.error ? e.error.message || e.error.name : "unknown error"}`);
    stopRecordingInternal(true);
  };

  mediaRecorder.onstop = () => {
    finalizeRecording();
  };

  recordedChunks = [];
  recordingStartedAt = performance.now();
  mediaRecorder.start();
  setRecordingUi(true);
  startTimer();
}

function startTimer() {
  clearInterval(timerHandle);
  timerHandle = setInterval(() => {
    const elapsed = (performance.now() - recordingStartedAt) / 1000;
    els.recTimer.textContent = formatTimer(elapsed);
  }, 200);
}

function stopStreamTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  mediaRecorder.stop();
}

function stopRecordingInternal(fromError) {
  clearInterval(timerHandle);
  setRecordingUi(false);
  stopStreamTracks();
  if (fromError) {
    els.recStatus.textContent = "Error";
    els.recStatus.className = "rec-status";
  }
}

function finalizeRecording() {
  clearInterval(timerHandle);
  const actualDurationSec = (performance.now() - recordingStartedAt) / 1000;
  setRecordingUi(false);
  stopStreamTracks();

  if (!recordedChunks.length) {
    els.recStatus.textContent = "Ready";
    els.recStatus.className = "rec-status";
    els.recTimer.textContent = "00:00";
    showRecorderError("Recording produced no audio data. Check your microphone and try again.");
    return;
  }

  recordedBlob = new Blob(recordedChunks, { type: recordedMimeType || "audio/webm" });

  if (recordedBlob.size === 0) {
    showRecorderError("Recording produced an empty file. Check your microphone and try again.");
    els.recStatus.textContent = "Ready";
    els.recStatus.className = "rec-status";
    els.recTimer.textContent = "00:00";
    return;
  }

  recordedDurationSec = actualDurationSec;

  const url = URL.createObjectURL(recordedBlob);
  els.audioPreview.src = url;
  els.audioPreview.style.display = "block";

  els.recStatus.textContent = "Recorded";
  els.recStatus.className = "rec-status done";
  els.recTimer.textContent = formatTimer(recordedDurationSec);
  els.recordedDurationLine.textContent = `${recordedDurationSec.toFixed(2)}s — ${formatBytes(recordedBlob.size)} (${recordedMimeType || "browser default"})`;
  els.postRecordRow.style.display = "flex";
  els.analyzeBtn.disabled = false;
}

els.recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

els.analyzeBtn.addEventListener("click", runAnalysis);

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

async function runAnalysis() {
  if (!recordedBlob || recordedBlob.size === 0) {
    setStatus("Record something first.", "error");
    return;
  }

  const duration = recordedDurationSec || 0;

  setLoading(true);
  setStatus("Uploading + analyzing (STT can take a while)...", "");
  clearResults();
  clearRecorderError();

  const ext = recordedMimeType.includes("ogg") ? "ogg" : recordedMimeType.includes("mp4") ? "m4a" : "webm";
  const form = new FormData();
  form.append("audio", recordedBlob, `debug-recording.${ext}`);
  form.append("duration", duration);
  form.append("pronunciation_provider", getSelectedPronunciationProvider());
  form.append("stt_provider", getSelectedSttProvider());

  const started = performance.now();
  let response, elapsedMs;

  try {
    response = await fetch(ENDPOINT, { method: "POST", body: form });
    elapsedMs = Math.round(performance.now() - started);
  } catch (networkErr) {
    elapsedMs = Math.round(performance.now() - started);
    setLoading(false);
    setStatus("Request failed", "error");
    renderNetworkError(networkErr, elapsedMs);
    return;
  }

  let body, parseError = null;
  const rawText = await response.text();
  try {
    body = rawText ? JSON.parse(rawText) : null;
  } catch (e) {
    parseError = e;
  }

  setLoading(false);

  if (!response.ok) {
    setStatus(`Request failed (${response.status})`, "error");
    renderHttpError(response.status, body, rawText, elapsedMs);
    return;
  }

  if (parseError || body === null) {
    setStatus("Invalid response", "error");
    renderInvalidResponse(rawText, parseError, elapsedMs);
    return;
  }

  setStatus(`Done in ${elapsedMs} ms`, "ok");
  renderResults(body, elapsedMs);
}

function setLoading(isLoading) {
  els.analyzeBtn.disabled = isLoading || !recordedBlob;
  els.recordBtn.disabled = isLoading;
  els.analyzeBtn.textContent = isLoading ? "Analyzing..." : "Analyze recording";
}

function setStatus(text, kind) {
  els.statusLine.textContent = text;
  els.statusLine.className = "status-line" + (kind ? " " + kind : "");
}

function clearResults() {
  els.analysisResults.innerHTML = "";
  if (els.learnerReport) els.learnerReport.innerHTML = "";
}

function card(html) {
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = html;
  return div;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined) return "—";
  if (typeof n !== "number") return esc(n);
  return n.toFixed(digits);
}

function metric(label, value) {
  return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
}

function renderNetworkError(err, elapsedMs) {
  els.analysisResults.appendChild(
    card(`
      <div class="error-box">
        <h3>Request failed</h3>
        <div>Type: <strong>Network error</strong></div>
        <pre>${esc(err.message)}</pre>
        <div class="perf-line" style="margin-top:8px;">Elapsed: ${elapsedMs} ms</div>
      </div>
    `)
  );
}

function renderHttpError(status, body, rawText, elapsedMs) {
  const detail = body ? JSON.stringify(body, null, 2) : rawText || "(empty response body)";
  els.analysisResults.appendChild(
    card(`
      <div class="error-box">
        <h3>Request failed</h3>
        <div>Type: <strong>HTTP error</strong></div>
        <div>Status: <strong>${status}</strong></div>
        <pre>${esc(detail)}</pre>
        <div class="perf-line" style="margin-top:8px;">Elapsed: ${elapsedMs} ms</div>
      </div>
    `)
  );
}

function renderInvalidResponse(rawText, parseError, elapsedMs) {
  els.analysisResults.appendChild(
    card(`
      <div class="error-box">
        <h3>Request failed</h3>
        <div>Type: <strong>Invalid response</strong></div>
        <pre>${esc(parseError ? parseError.message : "unknown parse error")}</pre>
        <div class="perf-line" style="margin-top:8px;">Elapsed: ${elapsedMs} ms</div>
        <pre>${esc(rawText).slice(0, 4000)}</pre>
      </div>
    `)
  );
}

// ─── Result rendering ──────────────────────────────────────────────────────
function renderResults(data, elapsedMs) {
  els.emptyState.style.display = "none";

  // Polished, PDF-referenced report — built from the exact same `data`
  // object the Debug Details cards below render from. No second request,
  // no recomputation: every number here is read straight off backendResult.
  if (els.learnerReport) els.learnerReport.innerHTML = buildLearnerReportHtml(data);

  renderPerf(elapsedMs, data.duration);
  renderTranscript(data.transcript);
  if (data.stt) renderStt(data.stt);
  if (data.vocabulary) renderVocabulary(data.vocabulary);
  if (data.grammar) renderGrammar(data.grammar, data.grammar_tool_available, data.grammar_context);
  renderFillers(data.filler, data.filler_occurrences);
  renderPronunciation(data.pronunciation);
  renderPacing(data.pace);
  renderAcoustic(data.clarity);
  renderOverall(data);
  renderTokenAnalysis(data.linguistic_analysis, data.languagetool_errors);
  renderWordTimings(data.word_timings);
  renderTeacherReport(data.teacher_report, data.teacher_report_detail);
  if (data.note) renderNote(data.note);
  renderRawJson(data);
}

// ─────────────────────────────────────────────────────────────────────────
// LEARNER REPORT — polished, PDF-referenced rendering of the SAME response
// object (`data`) the Debug Details cards below render from. Nothing here
// computes a score, a CEFR level, or an error count; every value is read
// straight off backendResult / backendResult.teacher_report. Where the PDF
// reference showed copy this backend doesn't produce (a per-CEFR-level
// narrative sentence, the standard CEFR level names), only the standard,
// candidate-independent CEFR taxonomy names are used as static labels
// (CEFR_LEVEL_NAMES below) — never anything that reads as feedback about
// this specific candidate. See the end-of-task report for what's still
// missing from the backend (a pacing-only explanation distinct from the
// combined fluency explanation).
// ─────────────────────────────────────────────────────────────────────────

const CEFR_LEVEL_NAMES = {
  A1: "Beginner", A2: "Elementary", B1: "Intermediate",
  B2: "Upper-Intermediate", C1: "Advanced", C2: "Proficiency",
};

function lrBandClass(band) {
  const b = (band || "").toLowerCase();
  return b === "high" ? "band-high" : b === "low" ? "band-low" : "band-medium";
}

function lrScoreClass(score) {
  if (score === undefined || score === null) return "";
  if (score >= 75) return "lr-good";
  if (score >= 50) return "lr-ok";
  return "lr-poor";
}

function lrSection(title, score, bodyHtml) {
  if (!bodyHtml) return "";
  return `<div class="lr-section">
    <div class="lr-section-head">
      <h3>${esc(title)}</h3>
      ${score !== undefined && score !== null ? `<div class="lr-score ${lrScoreClass(score)}">${esc(Math.round(score))}</div>` : ""}
    </div>
    <div class="lr-section-body">${bodyHtml}</div>
  </div>`;
}

function lrMetric(label, value) {
  if (value === undefined || value === null) return "";
  return `<div class="lr-metric"><span class="k">${esc(label)}</span><span class="v">${esc(value)}</span></div>`;
}

function lrFeedbackBlock(band, text) {
  if (!text) return "";
  return `<div class="lr-feedback">
    ${band ? `<span class="band ${lrBandClass(band)}">${esc(band)}</span>` : ""}
    <div>${esc(text)}</div>
  </div>`;
}

function lrRecommendation(text) {
  if (!text) return "";
  return `<div class="lr-rec"><span class="arrow">→</span><span>${esc(text)}</span></div>`;
}

function buildOverallResultHtml(data) {
  if (data.overall === undefined && !data.cefr) return "";
  const cefr = data.cefr || {};
  const levelName = CEFR_LEVEL_NAMES[cefr.level] || "";
  return `<div class="lr-overall">
    <div class="lr-overall-kicker">Your Overall Result</div>
    <div class="lr-overall-score">${data.overall !== undefined ? esc(Math.round(data.overall)) : "—"}<span class="lr-overall-max">/100</span></div>
    ${cefr.level ? `<div class="lr-overall-cefr">${esc(cefr.level)}</div>` : ""}
    ${levelName ? `<div class="lr-overall-cefr-label">${esc(levelName)}</div>` : ""}
  </div>`;
}

function buildGrammarSectionHtml(data, tr) {
  const g = data.grammar;
  if (!g) return "";
  const growth = (tr && tr.growth_areas) || {};
  const breakdown = Array.isArray(growth.grammar_breakdown) ? growth.grammar_breakdown : [];
  const accuracy = tr && tr.performance_summary && tr.performance_summary.accuracy;

  let evidenceHtml = "";
  if (breakdown.length) {
    // Existing backend-generated grammar feedback — one card per real,
    // validated grammar issue (see groq_provider.merge_report()).
    evidenceHtml = breakdown.map(item => `
      <div class="grammar-fix-card">
        ${item.you_said ? `<div class="gfc-said"><span class="tag">You said</span>${esc(item.you_said)}</div>` : ""}
        ${item.what_went_wrong || item.why_its_wrong ? `<div class="gfc-why">${[item.what_went_wrong, item.why_its_wrong].filter(Boolean).map(esc).join(" ")}</div>` : ""}
        ${item.correct_version ? `<div class="gfc-correct"><span class="tag">Say instead</span>${esc(item.correct_version)}</div>` : ""}
        ${item.how_to_avoid_next_time ? `<div class="gfc-tip"><span class="icon">💡</span><span>${esc(item.how_to_avoid_next_time)}</span></div>` : ""}
      </div>`).join("");
  } else if (Array.isArray(g.issues) && g.issues.length) {
    // No teacher_report (or malformed breakdown) — fall back to the raw,
    // still-backend-sourced grammar.issues rather than inventing prose.
    evidenceHtml = g.issues.map(iss => `
      <div class="issue-card">
        ${iss.wrong !== undefined ? `<div class="issue-row"><span class="k">Wrong:</span><span class="v-wrong">${esc(iss.wrong)}</span></div>` : ""}
        ${iss.correct !== undefined ? `<div class="issue-row"><span class="k">Correct:</span><span class="v-correct">${esc(iss.correct)}</span></div>` : ""}
        ${iss.learner_explanation ? `<div class="issue-row"><span class="k">Why:</span><span>${esc(iss.learner_explanation)}</span></div>` : ""}
      </div>`).join("");
  } else if (g.errors === 0 || growth.grammar_breakdown) {
    evidenceHtml = `<div class="lr-strength"><span>✓</span><span>${esc((tr && tr.overview && tr.overview.grammar_accuracy_summary) || "No grammar errors detected.")}</span></div>`;
  }

  const body = `
    <div class="lr-metric-row">
      ${lrMetric("Errors", g.errors)}
    </div>
    ${evidenceHtml}
    ${accuracy ? lrFeedbackBlock(accuracy.band, accuracy.teacher_explanation) : ""}
  `;
  return lrSection("Grammar", g.score, body);
}

function buildVocabularySectionHtml(data, tr) {
  const v = data.vocabulary;
  if (!v) return "";
  const trVocab = (tr && tr.vocabulary) || {};
  const repetitions = Array.isArray(tr && tr.repetitions) ? tr.repetitions : [];
  const overview = (tr && tr.overview) || {};

  const distribution = trVocab.vocabulary_distribution_by_level;
  let distHtml = "";
  if (distribution && typeof distribution === "object") {
    distHtml = `<div style="margin:12px 0;">` + Object.entries(distribution).map(([level, pct]) => `
      <div class="lr-dist-row">
        <span class="dist-label">${esc(level)}</span>
        <span class="dist-bar-wrap"><span class="dist-bar" style="width:${Math.max(0, Math.min(100, Number(pct) || 0))}%"></span></span>
        <span class="dist-pct">${esc(pct)}%</span>
      </div>`).join("") + `</div>`;
  }

  let wordsHtml = "";
  if (Array.isArray(trVocab.useful_higher_level_words_used) && trVocab.useful_higher_level_words_used.length) {
    wordsHtml = `<div style="margin-bottom:10px;">${trVocab.useful_higher_level_words_used.map(w => `<span class="lr-word-chip">${esc(w)}</span>`).join("")}</div>`;
  }

  let repsHtml = "";
  if (repetitions.length) {
    repsHtml = `<div style="margin-top:10px;"><strong style="font-size:12px;color:#8a97a0;text-transform:uppercase;letter-spacing:.04em;">Word repetitions</strong>` +
      repetitions.map(r => `<div class="repeat-row">
        <span class="word">${esc(r.word_or_phrase)}</span><span class="freq">used ${esc(r.frequency)}×</span>
        ${Array.isArray(r.better_alternatives) && r.better_alternatives.length ? `<span class="arrow">→</span>${r.better_alternatives.map(a => `<span class="alt">${esc(a)}</span>`).join("")}` : ""}
      </div>`).join("") + `</div>`;
  }

  const body = `
    <div class="lr-metric-row">
      ${lrMetric("Active vocabulary", trVocab.active_vocabulary_size ?? v.unique_words)}
      ${lrMetric("Vocabulary level", trVocab.cefr_level || data.cefr?.level)}
      ${lrMetric("Unique words", v.unique_words)}
      ${lrMetric("Total words", v.total_words)}
    </div>
    ${distHtml}
    ${wordsHtml}
    ${trVocab.active_vocabulary_note ? `<div class="lr-feedback">${esc(trVocab.active_vocabulary_note)}</div>` : (overview.strong_vocabulary_observations ? `<div class="lr-feedback">${esc(overview.strong_vocabulary_observations)}</div>` : "")}
    ${repsHtml}
    ${lrRecommendation(trVocab.suggestions_for_improving_vocabulary || (tr && tr.growth_areas && tr.growth_areas.vocabulary_improvements))}
  `;
  return lrSection("Vocabulary", v.score, body);
}

function buildPacingSectionHtml(data) {
  const pace = data.pace;
  if (!pace) return "";
  // Pacing is WPM specifically (see score_free_speech's pace scoring against
  // the backend's target WPM range) — deliberately NOT paired with the
  // Fluency teacher_explanation here, since that explanation covers
  // hesitations/pauses/flow too and would misrepresent itself as a
  // pacing-only assessment. See buildFluencySectionHtml for that content.
  // No recommendation line: the backend doesn't currently generate a
  // pacing-specific recommendation distinct from the Fluency one.
  const body = `
    <div class="lr-metric-row">
      ${lrMetric("Speaking rate (WPM)", pace.wpm)}
    </div>
  `;
  return lrSection("Pacing", pace.score, body);
}

function buildFluencySectionHtml(data, tr) {
  // Distinct backend field from `pace` — score_fluency() in app.py measures
  // continuity of delivery (pause frequency/length + filler/hesitation
  // rate), not speaking rate. See data.fluency, not data.pace.
  const fluency = data.fluency;
  if (!fluency) return "";
  const perf = tr && tr.performance_summary && tr.performance_summary.fluency;
  const body = `
    <div class="lr-metric-row">
      ${lrMetric("Hesitations", fluency.hesitation_count)}
      ${lrMetric("Unexpected pauses", fluency.long_pause_count)}
      ${lrMetric("Pause rate / min", fluency.pause_rate_per_min)}
    </div>
    ${fluency.pause_data_available === false ? `<div class="lr-gap-note">Pause detection wasn't available for this STT provider — this score falls back to the filler/hesitation signal alone.</div>` : ""}
    ${perf ? lrFeedbackBlock(perf.band, perf.teacher_explanation) : ""}
  `;
  return lrSection("Fluency", fluency.score, body);
}

function buildFillerSectionHtml(data, tr) {
  const filler = data.filler;
  if (!filler) return "";
  const growth = (tr && tr.growth_areas) || {};
  const f = growth.fillers;

  let feedbackHtml = "";
  if (f && typeof f === "object" && f.summary) {
    feedbackHtml = `<div class="lr-feedback">
      <div>${esc(f.summary)}</div>
      ${f.why_it_matters ? `<div style="margin-top:6px;color:#56636b;">${esc(f.why_it_matters)}</div>` : ""}
    </div>
    ${Array.isArray(f.how_to_reduce) && f.how_to_reduce.length ? f.how_to_reduce.map(t => lrRecommendation(t)).join("") : ""}`;
  } else if (typeof f === "string" && f) {
    feedbackHtml = `<div class="lr-feedback">${esc(f)}</div>`;
  }

  const body = `
    <div class="lr-metric-row">
      ${lrMetric("Filler count", filler.count)}
      ${lrMetric("Rate / min", filler.rate_per_min)}
    </div>
    ${Array.isArray(filler.words) && filler.words.length ? `<div style="margin-bottom:10px;">${filler.words.map(w => `<span class="lr-word-chip">${esc(w)}</span>`).join("")}</div>` : ""}
    ${feedbackHtml}
  `;
  return lrSection("Filler Words", filler.score, body);
}

function buildPronunciationSectionHtml(data, tr) {
  const p = data.pronunciation;
  if (!p) return "";
  const useOfEnglish = tr && tr.performance_summary && tr.performance_summary.use_of_english;

  let issuesHtml = "";
  if (Array.isArray(p.issues) && p.issues.length) {
    issuesHtml = `<div style="margin-bottom:10px;">${p.issues.map(i => `<span class="lr-word-chip">${esc(i.word)} · ${esc(i.confidence)}%</span>`).join("")}</div>`;
  }

  const availabilityNote = p.available === false
    ? `<div class="lr-gap-note">Pronunciation provider (${esc(p.requested_provider || "requested provider")}) was unavailable for this run — score falls back per backend logic.</div>`
    : "";

  const body = `
    ${issuesHtml}
    ${useOfEnglish ? lrFeedbackBlock(useOfEnglish.band, useOfEnglish.teacher_explanation) : ""}
    ${availabilityNote}
  `;
  return lrSection("Pronunciation", p.score, body);
}

function buildAdvancedGrammarSectionHtml(tr) {
  const list = Array.isArray(tr && tr.advanced_grammar_used) ? tr.advanced_grammar_used : [];
  if (!list.length) return ""; // per spec: show this section only when the backend actually provides evidence
  const body = list.map(a => `
    <div class="adv-grammar-card">
      <span class="badge">${esc(a.construction)}</span>
      <div class="quote">"${esc(a.quoted_example)}"</div>
      ${a.note ? `<div class="note">${esc(a.note)}</div>` : ""}
    </div>`).join("");
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Advanced Grammar</h3></div>
    <div class="lr-section-body">${body}</div>
  </div>`;
}

function buildTranscriptSectionHtml(data) {
  if (data.transcript === undefined) return "";
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Transcript</h3></div>
    <div class="lr-section-body"><div class="lr-transcript">${esc(data.transcript) || "<em>(empty)</em>"}</div></div>
  </div>`;
}

// ─────────────────────────────────────────────────────────────────────────
// Learner Assessment / Synthesized Feedback — pattern-aware synthesis layer.
//
// This is a presentation layer only: it groups/ranks/reuses evidence that
// already exists in `data` (grammar.issues, incl. their real `category`
// field from grammar_heuristics.py/grammar_pos_rules.py/languagetool_provider.py)
// and `data.teacher_report` (the existing Groq report). It never calls a new
// API, never computes a score, and never invents an error, quote, or
// recommendation — every string surfaced here is read verbatim from
// grammar_breakdown / growth_areas / vocabulary / performance_summary, the
// same fields the per-metric sections below already render. Categories,
// frequency counts, and priority tiers are the only things computed here,
// and that computation is a plain count/group-by over real evidence.
// ─────────────────────────────────────────────────────────────────────────

// Groups this response's grammar.issues by their real `category` field
// (e.g. "Preposition", "Subject-Verb Agreement", "Article") and aligns each
// issue, by index, with its corresponding grammar_breakdown entry — the
// backend guarantees grammar_breakdown has exactly one entry per issue, in
// the same order (see groq_provider.py's prompt rules), so this alignment
// never mismatches an explanation to the wrong issue.
function analyzeGrammarPatterns(data, tr) {
  const issues = Array.isArray(data.grammar && data.grammar.issues) ? data.grammar.issues : [];
  if (!issues.length) return [];
  const breakdown = Array.isArray(tr && tr.growth_areas && tr.growth_areas.grammar_breakdown)
    ? tr.growth_areas.grammar_breakdown : [];

  const byCategory = new Map();
  issues.forEach((iss, idx) => {
    const cat = (iss.category || "Other").trim();
    if (!byCategory.has(cat)) byCategory.set(cat, { category: cat, count: 0, examples: [], breakdownEntries: [] });
    const entry = byCategory.get(cat);
    entry.count += 1;
    entry.examples.push(iss);
    if (breakdown[idx]) entry.breakdownEntries.push(breakdown[idx]);
  });

  // Frequency → pattern/priority, per the same tiers the spec's own example
  // table uses: 1 occurrence = Isolated/Low, 2 = Occasional/Medium, 3+ = Recurring/High.
  return Array.from(byCategory.values()).map(e => {
    let pattern, priority;
    if (e.count >= 3) { pattern = "Recurring"; priority = "High"; }
    else if (e.count === 2) { pattern = "Occasional"; priority = "Medium"; }
    else { pattern = "Isolated"; priority = "Low"; }
    return { ...e, pattern, priority };
  }).sort((a, b) => b.count - a.count);
}

function buildGrammarProfileTableHtml(categories) {
  if (!categories.length) return "";
  const rows = categories.map(c => `
    <tr>
      <td>${esc(c.category)}</td>
      <td><span class="lr-pattern-pill lr-pattern-${esc(c.pattern.toLowerCase())}">${esc(c.pattern)}</span></td>
      <td>${c.count}</td>
      <td><span class="lr-priority-pill lr-priority-${esc(c.priority.toLowerCase())}">${esc(c.priority)}</span></td>
    </tr>`).join("");
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Grammar Profile</h3></div>
    <div class="lr-section-body">
      <table class="lr-profile-table">
        <thead><tr><th>Category</th><th>Pattern</th><th>Frequency</th><th>Priority</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

// Builds the 3-5 prioritized "what to work on" list, blending grammar
// categories (ranked by real frequency) with fluency/vocabulary/pronunciation
// signals — but ONLY when the underlying evidence for that signal actually
// indicates a problem (non-zero filler count with a real teacher_report
// summary, actual repeated words with a real suggestion, actual low-confidence
// pronunciation words alongside a non-High use-of-English band). Nothing here
// is shown without a concrete evidence field backing it.
function collectPriorityAreas(data, tr, grammarCategories) {
  const areas = [];

  const grammarCandidates = grammarCategories.filter(c => c.pattern !== "Isolated");
  const grammarPool = grammarCandidates.length ? grammarCandidates : grammarCategories;
  grammarPool.forEach(c => {
    const bd = c.breakdownEntries[0] || {};
    const raw = c.examples[0] || {};
    areas.push({
      title: c.category,
      priority: c.priority,
      count: c.count,
      evidence: bd.you_said || (raw.wrong !== undefined ? `"${raw.wrong}"` : null),
      why: bd.why_its_wrong || bd.what_went_wrong || raw.learner_explanation || null,
      practice: bd.how_to_avoid_next_time || null,
    });
  });

  const growth = (tr && tr.growth_areas) || {};
  const f = growth.fillers;
  const filler = data.filler;
  if (filler && filler.count > 0 && f && typeof f === "object" && f.summary) {
    areas.push({
      title: "Filler words",
      priority: (filler.rate_per_min || 0) >= 6 ? "High" : "Medium",
      count: filler.count,
      evidence: `Used ${filler.count} filler word${filler.count === 1 ? "" : "s"}${filler.rate_per_min ? ` (${filler.rate_per_min}/min)` : ""}.`,
      why: f.why_it_matters || null,
      practice: Array.isArray(f.how_to_reduce) && f.how_to_reduce.length ? f.how_to_reduce[0] : null,
    });
  }

  const trVocab = (tr && tr.vocabulary) || {};
  const reps = Array.isArray(tr && tr.repetitions) ? tr.repetitions : [];
  if (reps.length && trVocab.suggestions_for_improving_vocabulary) {
    areas.push({
      title: "Vocabulary variety",
      priority: reps.length >= 3 ? "High" : "Medium",
      count: reps.length,
      evidence: `Repeated word${reps.length === 1 ? "" : "s"}: ${reps.slice(0, 3).map(r => r.word_or_phrase).join(", ")}.`,
      why: trVocab.active_vocabulary_note || null,
      practice: trVocab.suggestions_for_improving_vocabulary,
    });
  }

  const p = data.pronunciation;
  const useOfEnglish = tr && tr.performance_summary && tr.performance_summary.use_of_english;
  if (p && Array.isArray(p.issues) && p.issues.length && useOfEnglish && (useOfEnglish.band || "").toLowerCase() !== "high") {
    areas.push({
      title: "Pronunciation clarity",
      priority: useOfEnglish.band || "Medium",
      count: p.issues.length,
      evidence: `Lower-confidence words: ${p.issues.slice(0, 3).map(i => i.word).join(", ")}.`,
      why: useOfEnglish.teacher_explanation || null,
      practice: null,
    });
  }

  const rank = { High: 0, Medium: 1, Low: 2 };
  areas.sort((a, b) => (rank[a.priority] ?? 1) - (rank[b.priority] ?? 1) || (b.count || 0) - (a.count || 0));
  return areas.slice(0, 5);
}

function buildPriorityAreasHtml(areas) {
  if (!areas.length) return "";
  const cards = areas.map((a, i) => `
    <div class="lr-priority-card">
      <div class="lr-priority-head">
        <span class="lr-priority-num">${i + 1}</span>
        <span class="lr-priority-title">${esc(a.title)}</span>
        ${a.priority ? `<span class="lr-priority-pill lr-priority-${esc(String(a.priority).toLowerCase())}">${esc(a.priority)}</span>` : ""}
      </div>
      ${a.evidence ? `<div class="lr-priority-row"><span class="k">Evidence</span><span class="v">${esc(a.evidence)}</span></div>` : ""}
      ${a.why ? `<div class="lr-priority-row"><span class="k">Why it matters</span><span class="v">${esc(a.why)}</span></div>` : ""}
      ${a.practice ? `<div class="lr-priority-row"><span class="k">Practice</span><span class="v">${esc(a.practice)}</span></div>` : ""}
    </div>`).join("");
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Priority Areas for Improvement</h3></div>
    <div class="lr-section-body">${cards}</div>
  </div>`;
}

// Strengths: only ever built from a High performance_summary band, a real
// overview observation string, or a real advanced_grammar_used entry — never
// generic praise. Returns [] (→ no section rendered) when none of those exist.
function collectStrengths(tr) {
  const strengths = [];
  const perf = (tr && tr.performance_summary) || {};
  const perfLabels = { accuracy: "Grammar accuracy", fluency: "Fluency", use_of_english: "Pronunciation" };
  Object.entries(perfLabels).forEach(([key, label]) => {
    const p = perf[key];
    if (p && (p.band || "").toLowerCase() === "high") {
      strengths.push(p.teacher_explanation ? `${label}: ${p.teacher_explanation}` : `Strong ${label.toLowerCase()}.`);
    }
  });
  const overview = (tr && tr.overview) || {};
  if (overview.strong_vocabulary_observations) strengths.push(overview.strong_vocabulary_observations);
  if (overview.strong_language_use_observations) strengths.push(overview.strong_language_use_observations);
  const adv = Array.isArray(tr && tr.advanced_grammar_used) ? tr.advanced_grammar_used : [];
  if (adv.length) {
    strengths.push(`Uses ${adv.length > 1 ? "advanced grammar constructions correctly" : "an advanced grammar construction correctly"} (e.g. ${adv[0].construction}: "${adv[0].quoted_example}").`);
  }
  return strengths;
}

function buildStrengthsHtml(strengths) {
  if (!strengths.length) return "";
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Strengths</h3></div>
    <div class="lr-section-body">${strengths.map(s => `<div class="lr-strength"><span>✓</span><span>${esc(s)}</span></div>`).join("")}</div>
  </div>`;
}

// Reuses the existing overview.overall_assessment narrative verbatim (the
// backend's own teacher-style summary) rather than generating a new one
// client-side — per spec, this must be "generated from the actual evidence",
// and that sentence already is.
function buildLearnerProfileSummaryHtml(tr) {
  const overview = (tr && tr.overview) || {};
  if (!overview.overall_assessment) return "";
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Overall Learner Profile</h3></div>
    <div class="lr-section-body"><div class="lr-profile-summary">${esc(overview.overall_assessment)}</div></div>
  </div>`;
}

// Final actionable plan — one line per priority area, reusing that area's
// own `practice` text (already sourced from real evidence above). No new
// text is generated here; areas without a practice tip are simply skipped.
function buildLearningPlanHtml(areas) {
  const steps = areas.filter(a => a.practice).map(a => ({ title: a.title, practice: a.practice }));
  if (!steps.length) return "";
  return `<div class="lr-section">
    <div class="lr-section-head"><h3>Recommended Practice Priorities</h3></div>
    <div class="lr-section-body">${steps.map((s, i) => `<div class="lr-plan-step"><span class="num">${i + 1}</span><span><strong>${esc(s.title)}.</strong> ${esc(s.practice)}</span></div>`).join("")}</div>
  </div>`;
}

function buildLearnerReportHtml(data) {
  const tr = data.teacher_report || null;
  const grammarCategories = analyzeGrammarPatterns(data, tr);
  const priorityAreas = collectPriorityAreas(data, tr, grammarCategories);
  const strengths = collectStrengths(tr);
  return `
    <div class="lr-masthead">VoiceCoach <span>Assessment Report</span></div>
    ${buildOverallResultHtml(data)}
    ${buildLearnerProfileSummaryHtml(tr)}
    ${buildStrengthsHtml(strengths)}
    ${buildPriorityAreasHtml(priorityAreas)}
    ${buildGrammarProfileTableHtml(grammarCategories)}
    ${buildGrammarSectionHtml(data, tr)}
    ${buildVocabularySectionHtml(data, tr)}
    ${buildPacingSectionHtml(data)}
    ${buildFluencySectionHtml(data, tr)}
    ${buildFillerSectionHtml(data, tr)}
    ${buildPronunciationSectionHtml(data, tr)}
    ${buildAdvancedGrammarSectionHtml(tr)}
    ${buildLearningPlanHtml(priorityAreas)}
    ${buildTranscriptSectionHtml(data)}
    ${!tr ? `<div class="notice warn" style="margin-top:8px;">AI-generated narrative feedback (Groq teacher report) wasn't available for this run${data.teacher_report_detail ? ": " + esc(data.teacher_report_detail) : "."} Scores and metrics above are still the real backend values — only the feedback/recommendation text is missing.</div>` : ""}
  `;
}

// Shared renderer for the actual Groq teacher_report shape produced by
// groq_provider.py (overview / growth_areas / vocabulary / repetitions /
// advanced_grammar_used / performance_summary). Used both by the single
// debug-assessment view (renderTeacherReport) and the guided-assessment
// modal (viewTeacherReport) so both surfaces show the real model output —
// nothing here invents fields the backend doesn't send.
function buildTeacherReportHtml(report) {
  const perf = report.performance_summary || {};
  const bandClass = (band) => {
    const b = (band || "").toLowerCase();
    return b === "high" ? "band-high" : b === "low" ? "band-low" : "band-medium";
  };
  const perfRow = (key, label) => {
    const p = perf[key];
    if (!p) return "";
    const band = p.band || p.rank || "";
    return `<div class="tr-band-card">
      <span class="tr-band-pill ${bandClass(band)}">${esc(band)}</span>
      <div class="tr-band-body">
        <div class="label">${esc(label)}${p.score !== undefined && p.score !== null ? ` — score ${esc(p.score)}` : ""}</div>
        ${p.teacher_explanation ? `<div class="explain">${esc(p.teacher_explanation)}</div>` : ""}
      </div>
    </div>`;
  };

  const overview = report.overview || {};
  const overviewFieldLabels = {
    grammar_accuracy_summary: "Grammar & accuracy",
    advanced_grammar_constructions_detected: "Advanced constructions",
    complex_sentence_usage: "Sentence complexity",
    strong_vocabulary_observations: "Vocabulary strengths",
    strong_language_use_observations: "Language use strengths",
  };
  const overviewHtml = overview.overall_assessment
    ? `<div class="tr-lede">
        ${esc(overview.overall_assessment)}
        <div class="tr-lede-extra">
          ${Object.entries(overviewFieldLabels)
            .filter(([k]) => overview[k])
            .map(([k, label]) => `<div class="row"><span class="k">${esc(label)}</span><span class="v">${esc(overview[k])}</span></div>`)
            .join("")}
        </div>
      </div>`
    : "";

  const growth = report.growth_areas || {};

  // Fillers is a structured object: {summary, why_it_matters, how_to_reduce[]}
  const fillers = growth.fillers;
  const fillersHtml = fillers && typeof fillers === "object" && fillers.summary
    ? `<div class="filler-card">
        <div class="summary">${esc(fillers.summary)}</div>
        ${fillers.why_it_matters ? `<div class="why">${esc(fillers.why_it_matters)}</div>` : ""}
        ${Array.isArray(fillers.how_to_reduce) && fillers.how_to_reduce.length
          ? `<ul class="filler-reduce-list">${fillers.how_to_reduce.map(t => `<li>${esc(t)}</li>`).join("")}</ul>`
          : ""}
      </div>`
    : (typeof fillers === "string" && fillers
        ? `<div class="filler-card"><div class="summary">${esc(fillers)}</div></div>`
        : `<div class="notice">No significant filler use detected.</div>`);

  // Grammar breakdown is an array of full 4-part explanations, one per real
  // grammar.issues entry — this is the evidence-based, learner-facing core
  // of the report, so it gets a dedicated "you said → why → say instead →
  // next time" layout rather than a generic key/value list.
  const grammarBreakdown = Array.isArray(growth.grammar_breakdown) ? growth.grammar_breakdown : [];
  const grammarHtml = grammarBreakdown.length
    ? grammarBreakdown.map(g => `
        <div class="grammar-fix-card">
          ${g.you_said ? `<div class="gfc-said"><span class="tag">You said</span>${esc(g.you_said)}</div>` : ""}
          ${g.what_went_wrong || g.why_its_wrong ? `<div class="gfc-why">${
            [g.what_went_wrong, g.why_its_wrong].filter(Boolean).map(esc).join(" ")
          }</div>` : ""}
          ${g.correct_version ? `<div class="gfc-correct"><span class="tag">Say instead</span>${esc(g.correct_version)}</div>` : ""}
          ${g.how_to_avoid_next_time ? `<div class="gfc-tip"><span class="icon">💡</span><span>${esc(g.how_to_avoid_next_time)}</span></div>` : ""}
        </div>`).join("")
    : `<div class="notice">No grammar issues detected.</div>`;

  const otherGrowthLabels = {
    linking_word_suggestions: "Linking word suggestions",
    vocabulary_improvements: "Vocabulary improvements",
    fluency_pacing_improvements: "Fluency & pacing",
    other_weaknesses: "Other observations",
  };
  const otherGrowthHtml = Object.entries(otherGrowthLabels)
    .filter(([k]) => growth[k])
    .map(([k, label]) => `<li><span class="label">${esc(label)}:</span> ${esc(growth[k])}</li>`)
    .join("");

  const repetitions = Array.isArray(report.repetitions) ? report.repetitions : [];
  const repsHtml = repetitions.length
    ? repetitions.map(r => `<div class="repeat-row">
        <span class="word">${esc(r.word_or_phrase)}</span>
        <span class="freq">used ${esc(r.frequency)}×</span>
        <span class="arrow">→</span>
        ${Array.isArray(r.better_alternatives) ? r.better_alternatives.map(a => `<span class="alt">${esc(a)}</span>`).join("") : ""}
      </div>`).join("")
    : `<div class="notice">No significant repetition detected.</div>`;

  const advGrammar = Array.isArray(report.advanced_grammar_used) ? report.advanced_grammar_used : [];
  const advGrammarHtml = advGrammar.length
    ? advGrammar.map(a => `<div class="adv-grammar-card">
        <span class="badge">${esc(a.construction)}</span>
        <div class="quote">"${esc(a.quoted_example)}"</div>
        ${a.note ? `<div class="note">${esc(a.note)}</div>` : ""}
      </div>`).join("")
    : `<div class="notice">None clearly evidenced in this sample.</div>`;

  const vocab = report.vocabulary || {};
  const vocabHtml = vocab.active_vocabulary_note || (Array.isArray(vocab.useful_higher_level_words_used) && vocab.useful_higher_level_words_used.length) || vocab.suggestions_for_improving_vocabulary
    ? `<div class="vocab-block">
        ${vocab.active_vocabulary_note ? `<div class="row"><span class="k">Where you stand</span>${esc(vocab.active_vocabulary_note)}</div>` : ""}
        ${Array.isArray(vocab.useful_higher_level_words_used) && vocab.useful_higher_level_words_used.length
          ? `<div class="row"><span class="k">Higher-level words used</span><div class="word-list">${vocab.useful_higher_level_words_used.map(w => `<span class="word-chip">${esc(w)}</span>`).join("")}</div></div>`
          : ""}
        ${vocab.suggestions_for_improving_vocabulary ? `<div class="row"><span class="k">Suggestions</span>${esc(vocab.suggestions_for_improving_vocabulary)}</div>` : ""}
      </div>`
    : "";

  return `
    ${overviewHtml}
    <div class="tr-band-row" style="margin-top:12px;">
      ${perfRow("accuracy", "Accuracy")}
      ${perfRow("fluency", "Fluency")}
      ${perfRow("use_of_english", "Use of English")}
    </div>

    <div class="tr-section-title">✏️ Grammar — what went wrong &amp; how to fix it</div>
    ${grammarHtml}

    <div class="tr-section-title">💬 Fillers</div>
    ${fillersHtml}

    <div class="tr-section-title">🔁 Repeated words</div>
    ${repsHtml}

    <div class="tr-section-title">📚 Vocabulary</div>
    ${vocabHtml || `<div class="notice">No vocabulary notes available.</div>`}

    <div class="tr-section-title">⭐ Advanced constructions used well</div>
    ${advGrammarHtml}

    ${otherGrowthHtml ? `<div class="tr-section-title">🎯 Other growth areas</div><ul class="tr-growth-list">${otherGrowthHtml}</ul>` : ""}
  `;
}

function renderTeacherReport(report, detail) {
  if (!report) {
    els.analysisResults.appendChild(
      card(`<h3>📝 Teacher Report (Groq)</h3><div class="notice">${
        esc(detail || "Not available — GROQ_API_KEY may be unset, or the request failed.")
      }</div>`)
    );
    return;
  }

  els.analysisResults.appendChild(
    card(`<h3>📝 Teacher Report (Groq)</h3>${buildTeacherReportHtml(report)}`)
  );
}

function renderPerf(elapsedMs, duration) {
  els.analysisResults.appendChild(
    card(`<h3>Performance</h3><div class="metric-grid">
      <div class="metric"><div class="label">API Response Time</div><div class="value">${elapsedMs} ms</div></div>
      ${duration !== undefined ? metric("Duration Sent (s)", duration) : ""}
    </div>`)
  );
}

function renderTranscript(transcript) {
  if (transcript === undefined) return;
  els.analysisResults.appendChild(
    card(`<h3>Transcript</h3><div class="notice" style="color:var(--text); border-style:solid;">${esc(transcript) || "<em>(empty)</em>"}</div>`)
  );
}

function renderStt(stt) {
  const mismatch = stt.requested_provider && stt.requested_provider !== stt.provider;
  els.analysisResults.appendChild(
    card(`<h3>Speech-to-Text</h3>
      <div class="notice ${mismatch ? "warn" : ""}">
        <strong>Engine used: ${esc(stt.provider)}</strong>${
          mismatch ? ` (requested <code>${esc(stt.requested_provider)}</code>, which was unavailable)` : ""
        }
        ${stt.detail ? `<div style="margin-top:6px;">${esc(stt.detail)}</div>` : ""}
      </div>`)
  );
}

function renderVocabulary(v) {
  const rows = [];
  if (v.score !== undefined) rows.push(metric("Score", v.score));
  if (v.diversity !== undefined) rows.push(metric("Diversity (MATTR)", v.diversity));
  if (v.sophistication !== undefined) rows.push(metric("Sophistication", v.sophistication));
  if (v.variety !== undefined) rows.push(metric("Variety (content MATTR)", v.variety));
  if (v.advanced_ratio !== undefined) rows.push(metric("Advanced Ratio", v.advanced_ratio + "%"));
  if (v.repetition_penalty !== undefined) rows.push(metric("Repetition Penalty", v.repetition_penalty + "%"));
  if (v.confidence !== undefined) rows.push(metric("Confidence", v.confidence));
  if (v.total_words !== undefined) rows.push(metric("Total Words", v.total_words));
  if (v.unique_words !== undefined) rows.push(metric("Unique Words", v.unique_words));

  let wordsHtml = "";
  if (Array.isArray(v.advanced_words) && v.advanced_words.length) {
    wordsHtml = `<div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin-top:6px;">Advanced Words</div>
      <div class="word-list">${v.advanced_words.map((w) => `<span class="word-chip">${esc(w)}</span>`).join("")}</div>`;
  }

  els.analysisResults.appendChild(
    card(`<h3>Vocabulary</h3><div class="metric-grid">${rows.join("")}</div>${wordsHtml}`)
  );
}

function renderGrammar(g, toolAvailable, grammarContext) {
  // g.issues is POST-VALIDATION: only candidates the contextual-validation
  // layer (grammar_context_validator.py — Groq, or its offline heuristic
  // fallback) judged "true_grammar_error" ever land here / count toward
  // g.errors. See grammarContext for everything that was filtered out and
  // why (written-English notes, style/register, spoken-usage asides,
  // non-errors) and the full candidate+judgment evidence trail.
  let issuesHtml = "";
  if (Array.isArray(g.issues) && g.issues.length) {
    issuesHtml = g.issues
      .map(
        (iss) => `
      <div class="issue-card">
        ${iss.wrong !== undefined ? `<div class="issue-row"><span class="k">Wrong:</span><span class="v-wrong">${esc(iss.wrong)}</span></div>` : ""}
        ${iss.correct !== undefined ? `<div class="issue-row"><span class="k">Correct:</span><span class="v-correct">${esc(iss.correct)}</span></div>` : ""}
        ${iss.learner_explanation ? `<div class="issue-row"><span class="k">What went wrong:</span><span>${esc(iss.learner_explanation)}</span></div>` : (iss.message !== undefined ? `<div class="issue-row"><span class="k">Message:</span><span>${esc(iss.message)}</span></div>` : "")}
        ${iss.context !== undefined ? `<div class="issue-row"><span class="k">Context:</span><span>${esc(iss.context)}</span></div>` : ""}
      </div>`
      )
      .join("");
  } else if (g.errors === 0) {
    issuesHtml = `<div class="notice">No grammar issues detected.</div>`;
  }

  const toolLine =
    toolAvailable === false
      ? `<div class="notice warn" style="margin-top:8px;"><strong>Note:</strong> the grammar tool (language_tool_python / Java) is unavailable on this backend instance — error count is a crude fallback, not real LanguageTool checking.</div>`
      : "";

  // ── Contextual validation summary + non-scoring notes ──────────────────
  let validationHtml = "";
  if (grammarContext) {
    const src = grammarContext.validation_source;
    const srcLabel = src === "llm_groq"
      ? `LLM (Groq${grammarContext.model ? `, ${esc(grammarContext.model)}` : ""}) contextual validation`
      : src === "heuristic_fallback"
        ? "Offline heuristic fallback (Groq unavailable)"
        : src === "no_candidates" ? "No candidates to validate" : esc(src || "unknown");

    const notes = Array.isArray(grammarContext.context_notes) ? grammarContext.context_notes : [];
    const classLabel = {
      written_only_issue: "Written-English only — not spoken grammar",
      style_or_register: "Style / register — not a grammar mistake",
      spoken_usage_issue: "Spoken-usage note — not a scoring error",
      not_an_error: "Not actually an error here",
    };
    const notesHtml = notes.length
      ? notes.map((n) => `
        <div class="issue-card" style="opacity:0.85;">
          <div class="issue-row"><span class="k">Flagged text:</span><span class="v-wrong">${esc(n.wrong || "")}</span></div>
          <div class="issue-row"><span class="k">Classification:</span><span>${esc(classLabel[n.classification] || n.classification)}</span></div>
          <div class="issue-row"><span class="k">Why it's not counted:</span><span>${esc(n.learner_explanation || "")}</span></div>
          ${n.written_note ? `<div class="issue-row"><span class="k">Written-English note:</span><span>${esc(n.written_note)}</span></div>` : ""}
        </div>`).join("")
      : `<div class="notice">No candidates were reclassified away from the grammar score.</div>`;

    const debugTrail = Array.isArray(grammarContext.debug_trail) ? grammarContext.debug_trail : [];
    const debugTrailHtml = debugTrail.length
      ? `<details style="margin-top:8px;">
           <summary style="cursor:pointer;color:var(--text-dim);">Full candidate → judgment evidence trail (${debugTrail.length})</summary>
           <pre style="max-height:320px;overflow:auto;">${esc(JSON.stringify(debugTrail, null, 2))}</pre>
         </details>`
      : "";

    validationHtml = `
      <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin-top:14px;">Contextual Validation</div>
      <div class="metric-grid">
        ${metric("Candidates evaluated", grammarContext.candidates_evaluated ?? 0)}
        ${metric("Reclassified away from score", grammarContext.reclassified_away_from_score ?? 0)}
      </div>
      <div class="notice" style="margin-top:6px;">${srcLabel}${grammarContext.detail ? ` — ${esc(grammarContext.detail)}` : ""}</div>
      <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin-top:10px;">Written-English / Style / Usage Notes (not scored)</div>
      ${notesHtml}
      ${debugTrailHtml}
    `;
  }

  els.analysisResults.appendChild(
    card(`<h3>Grammar</h3><div class="metric-grid">
        ${g.score !== undefined ? metric("Score", g.score) : ""}
        ${g.errors !== undefined ? metric("Errors", g.errors) : ""}
      </div>
      ${issuesHtml}
      ${toolLine}
      ${validationHtml}`)
  );
}

function renderOccurrenceTable(occurrences) {
  const rows = occurrences
    .map(
      (o) => `<tr>
        <td>${esc(o.word)}</td>
        <td>${fmt(o.start)}</td>
        <td>${fmt(o.end)}</td>
        <td>${esc(o.type)}</td>
        <td>${o.confidence !== null && o.confidence !== undefined ? o.confidence + "%" : "—"}</td>
        <td>${esc(o.reason)}</td>
      </tr>`
    )
    .join("");
  return `<table class="tokens">
      <thead><tr><th>Word</th><th>Start</th><th>End</th><th>Type</th><th>Confidence</th><th>Reason</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderFillers(filler, legacyOccurrences) {
  if (!filler) return;

  const summaryRows = [
    filler.score !== undefined ? metric("Score", filler.score) : "",
    filler.count !== undefined ? metric("Count (scored)", filler.count) : "",
  ].join("");

  let wordsHtml = "";
  if (Array.isArray(filler.words) && filler.words.length) {
    wordsHtml = `<div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin:10px 0 4px;">Aggregated (from scoring path)</div>
      <div class="word-list">${filler.words.map((w) => `<span class="word-chip">${esc(w)}</span>`).join("")}</div>`;
  }

  let occHtml = "";
  if (Array.isArray(filler.occurrences) && filler.occurrences.length) {
    occHtml = `
      <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin:12px 0 4px;">
        Detected Occurrences (authoritative — matches Score/Count above)
      </div>
      ${renderOccurrenceTable(filler.occurrences)}`;
  } else if (Array.isArray(filler.occurrences)) {
    occHtml = `<div class="notice" style="margin-top:10px;">No filler occurrences detected in the transcript.</div>`;
  }

  let legacyHtml = "";
  if (Array.isArray(legacyOccurrences)) {
    legacyHtml = `
      <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin:12px 0 4px;">
        Legacy Whisper-word-match cross-check (informational only)
      </div>
      ${legacyOccurrences.length
        ? renderOccurrenceTable(legacyOccurrences)
        : `<div class="notice">No matches (either none found, or this STT provider doesn't return word-level timing data).</div>`}`;
  }

  els.analysisResults.appendChild(
    card(`<h3>Filler Words</h3><div class="metric-grid">${summaryRows}</div>${wordsHtml}${occHtml}${legacyHtml}`)
  );
}

function renderPronunciation(p) {
  if (!p) return;
  const rows = [p.score !== undefined ? metric("Score", p.score) : ""].join("");

  let issuesHtml = "";
  if (Array.isArray(p.issues) && p.issues.length) {
    issuesHtml = `<table class="tokens">
      <thead><tr><th>Word</th><th>Confidence</th></tr></thead>
      <tbody>${p.issues.map((i) => `<tr><td>${esc(i.word)}</td><td>${i.confidence}%</td></tr>`).join("")}</tbody>
    </table>`;
  } else {
    issuesHtml = `<div class="notice">No low-confidence words flagged.</div>`;
  }

  let providerHtml = "";
  if (p.provider) {
    const mismatch = p.requested_provider && p.requested_provider !== p.provider;
    providerHtml = `
      <div class="notice ${mismatch ? "warn" : ""}" style="margin-top:10px;">
        <strong>Provider used: ${esc(p.provider)}</strong>${
          mismatch ? ` (requested <code>${esc(p.requested_provider)}</code>, which was unavailable)` : ""
        }
        ${p.detail ? `<div style="margin-top:6px;">${esc(p.detail)}</div>` : ""}
      </div>`;
  }

  els.analysisResults.appendChild(
    card(`<h3>Pronunciation</h3><div class="metric-grid">${rows}</div>
      ${providerHtml}
      <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin:8px 0 4px;">Flagged Words (capped at 8, from scoring path)</div>
      ${issuesHtml}`)
  );
}

function renderPacing(pace) {
  if (!pace) return;
  const rows = [
    pace.score !== undefined ? metric("Score", pace.score) : "",
    pace.wpm !== undefined ? metric("WPM", pace.wpm) : "",
  ].join("");
  els.analysisResults.appendChild(card(`<h3>Pacing</h3><div class="metric-grid">${rows}</div>`));
}

function renderAcoustic(clarity) {
  if (!clarity) return;
  const rows = [clarity.score !== undefined ? metric("Clarity Score", clarity.score) : ""].join("");
  els.analysisResults.appendChild(
    card(`<h3>Acoustic / Clarity</h3><div class="metric-grid">${rows}</div>
      <div class="notice">Clarity is a weighted blend of pace, filler, grammar, and pronunciation scores (see <code>score_clarity()</code>) — there is no separate acoustic-signal analysis in this backend.</div>`)
  );
}

function renderOverall(data) {
  const rows = [];
  if (data.overall !== undefined) rows.push(metric("Overall", data.overall));
  if (data.cefr) {
    rows.push(metric("CEFR Level", data.cefr.level));
    rows.push(metric("CEFR Score", data.cefr.score));
    if (data.cefr.avg_sentence_length !== undefined) rows.push(metric("Avg Sentence Length", data.cefr.avg_sentence_length));
  }
  if (data.archetype) rows.push(metric("Archetype", `${data.archetype.emoji || ""} ${data.archetype.archetype || ""}`.trim()));

  els.analysisResults.appendChild(card(`<h3>Overall Assessment</h3><div class="metric-grid">${rows.join("")}</div>
    ${data.feedback ? `<div class="notice" style="margin-top:10px; color:var(--text); border-style:solid;">${esc(data.feedback)}</div>` : ""}`));
}

function renderTokenAnalysis(linguisticAnalysis, ltErrors) {
  if (!linguisticAnalysis) {
    const reason =
      ltErrors && ltErrors.analyze
        ? `The /v2/analyze call failed: ${ltErrors.analyze}`
        : "No linguistic_analysis was returned for this request.";
    els.analysisResults.appendChild(
      card(`<h3>LanguageTool Token Analysis (POS / Lemma)</h3>
        <div class="notice warn">
          <strong>Unavailable for this request.</strong> ${esc(reason)}
          Grammar scoring is unaffected — it falls back independently.
        </div>`)
    );
    return;
  }

  const sentences = linguisticAnalysis.sentences || [];
  if (!sentences.length) {
    els.analysisResults.appendChild(
      card(`<h3>LanguageTool Token Analysis (POS / Lemma)</h3><div class="notice">No tokens returned.</div>`)
    );
    return;
  }

  const rowsHtml = sentences
    .map((sent, sIdx) => {
      const tokenRows = (sent.tokens || [])
        .map((tok) => {
          const alts = Array.isArray(tok.alternatives) && tok.alternatives.length
            ? `<div class="alt-list">Alternatives: ${tok.alternatives
                .map((a) => `<span class="word-chip">${esc(a.posTag)} — ${esc(a.partOfSpeech)}</span>`)
                .join(" ")}</div>`
            : "";
          return `<tr>
            <td>${esc(tok.text)}</td>
            <td>${esc(tok.lemma ?? "")}</td>
            <td>${esc(tok.partOfSpeech ?? "")}</td>
            <td><code>${esc(tok.posTag ?? "")}</code></td>
            <td>${tok.startOffset ?? ""}</td>
            <td>${tok.endOffset ?? ""}</td>
          </tr>${alts ? `<tr><td colspan="6" style="padding-top:0;">${alts}</td></tr>` : ""}`;
        })
        .join("");
      return `<div style="margin-bottom:12px;">
        <div class="label" style="font-size:11px;color:var(--text-dim);text-transform:uppercase;margin-bottom:4px;">
          Sentence ${sIdx + 1}: <span style="color:var(--text);text-transform:none;">${esc(sent.text)}</span>
        </div>
        <table class="tokens">
          <thead><tr><th>Text</th><th>Lemma</th><th>Part of Speech</th><th>POS Tag</th><th>Start</th><th>End</th></tr></thead>
          <tbody>${tokenRows}</tbody>
        </table>
      </div>`;
    })
    .join("");

  els.analysisResults.appendChild(
    card(`<h3>LanguageTool Token Analysis (POS / Lemma)</h3>
      <div class="notice" style="margin-bottom:10px;">
        From LanguageTool's <code>/v2/analyze</code> — token/lemma/POS data only.
        Does not indicate grammatical correctness (see Grammar above) or pronunciation quality.
      </div>
      ${rowsHtml}`)
  );
}

function renderWordTimings(wordTimings) {
  if (!wordTimings) return;
  const source = wordTimings.source;
  const sourceLabel = source ? source.charAt(0).toUpperCase() + source.slice(1) : "Unknown";

  if (!wordTimings.available || !Array.isArray(wordTimings.words) || !wordTimings.words.length) {
    const reason =
      source === "saaras"
        ? "Saaras's response only includes chunk/sentence-level timestamps, not word-level — this is expected, not an error."
        : "No word-level timing data returned.";
    els.analysisResults.appendChild(
      card(`<h3>Word Timings</h3>
        <div class="notice">
          <strong>Source: ${esc(sourceLabel)}</strong><br>${esc(reason)}
        </div>`)
    );
    return;
  }

  const rows = wordTimings.words
    .map(
      (w) => `<tr>
        <td>${esc(w.word)}</td>
        <td>${fmt(w.start)}</td>
        <td>${fmt(w.end)}</td>
        <td>${fmt(w.duration)}</td>
        <td>${w.probability !== null && w.probability !== undefined ? (w.probability * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`
    )
    .join("");

  els.analysisResults.appendChild(
    card(`<h3>Word Timings</h3>
      <div class="notice" style="margin-bottom:10px;"><strong>Source: ${esc(sourceLabel)}</strong> — every word from this provider's word-level timing output.</div>
      <table class="tokens">
        <thead><tr><th>Word</th><th>Start</th><th>End</th><th>Duration</th><th>Probability</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`)
  );
}

function renderNote(note) {
  els.analysisResults.appendChild(card(`<h3>Backend Note</h3><div class="notice">${esc(note)}</div>`));
}

function renderRawJson(data) {
  const jsonText = JSON.stringify(data, null, 2);
  const wrapper = document.createElement("div");
  wrapper.className = "card raw-json";
  wrapper.innerHTML = `
    <details>
      <summary>▼ Raw API Response <button class="copy-btn" type="button">Copy JSON</button></summary>
      <pre>${esc(jsonText)}</pre>
    </details>
  `;
  wrapper.querySelector(".copy-btn").addEventListener("click", (e) => {
    e.preventDefault();
    navigator.clipboard.writeText(jsonText).then(() => {
      const btn = e.target;
      const original = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = original), 1200);
    });
  });
  els.analysisResults.appendChild(wrapper);
}