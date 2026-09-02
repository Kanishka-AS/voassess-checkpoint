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
  emptyState: document.getElementById("emptyState"),
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
  const modal = document.getElementById('detailModal');
  const body = document.getElementById('modalBody');
  
  document.getElementById('modalTitle').textContent = `📝 Teacher Report - ${r.name || 'Assessment #' + r.id}`;
  
  const report = r.teacher_report;
  
  if (!report) {
    body.innerHTML = '<div class="notice">No teacher report available for this assessment.</div>';
    modal.classList.add('active');
    return;
  }
  
  let html = `
    <div class="teacher-report">
      <div class="report-summary">
        <h4>📊 Summary</h4>
        <div class="metric-grid" style="grid-template-columns: repeat(3, 1fr);">
          <div class="metric">
            <div class="label">Overall Score</div>
            <div class="value">${report.overall_score || r.overall_score || '—'}</div>
          </div>
          <div class="metric">
            <div class="label">CEFR Level</div>
            <div class="value">${report.cefr_level || r.cefr_level || '—'}</div>
          </div>
          <div class="metric">
            <div class="label">Archetype</div>
            <div class="value">${report.archetype || r.archetype || '—'}</div>
          </div>
        </div>
      </div>
  `;
  
  if (report.strengths && report.strengths.length) {
    html += `
      <div style="margin:12px 0;">
        <h4>💪 Strengths</h4>
        <ul style="list-style:none;padding:0;margin:0;">
          ${report.strengths.map(s => `<li style="padding:4px 0;border-bottom:1px solid var(--border);">✅ ${s}</li>`).join('')}
        </ul>
      </div>
    `;
  }
  
  if (report.areas_for_improvement && report.areas_for_improvement.length) {
    html += `
      <div style="margin:12px 0;">
        <h4>📈 Areas for Improvement</h4>
        <ul style="list-style:none;padding:0;margin:0;">
          ${report.areas_for_improvement.map(s => `<li style="padding:4px 0;border-bottom:1px solid var(--border);">📌 ${s}</li>`).join('')}
        </ul>
      </div>
    `;
  }
  
  if (report.specific_recommendations && report.specific_recommendations.length) {
    html += `
      <div style="margin:12px 0;">
        <h4>🎯 Specific Recommendations</h4>
        <ul style="list-style:none;padding:0;margin:0;">
          ${report.specific_recommendations.map(s => `<li style="padding:4px 0;border-bottom:1px solid var(--border);">💡 ${s}</li>`).join('')}
        </ul>
      </div>
    `;
  }
  
  if (report.teacher_notes) {
    html += `
      <div style="margin:12px 0;">
        <h4>📝 Teacher Notes</h4>
        <div class="notice">${report.teacher_notes}</div>
      </div>
    `;
  }
  
  html += `
    </div>
  `;
  
  body.innerHTML = html;
  modal.classList.add('active');
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
  els.results.innerHTML = "";
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
  els.results.appendChild(
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
  els.results.appendChild(
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
  els.results.appendChild(
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

  renderPerf(elapsedMs, data.duration);
  renderTranscript(data.transcript);
  if (data.stt) renderStt(data.stt);
  if (data.vocabulary) renderVocabulary(data.vocabulary);
  if (data.grammar) renderGrammar(data.grammar, data.grammar_tool_available);
  renderFillers(data.filler, data.filler_occurrences);
  renderPronunciation(data.pronunciation);
  renderPacing(data.pace);
  renderAcoustic(data.clarity);
  renderOverall(data);
  renderTokenAnalysis(data.linguistic_analysis, data.languagetool_errors);
  renderWordTimings(data.word_timings);
  if (data.note) renderNote(data.note);
  renderRawJson(data);
}

function renderPerf(elapsedMs, duration) {
  els.results.appendChild(
    card(`<h3>Performance</h3><div class="metric-grid">
      <div class="metric"><div class="label">API Response Time</div><div class="value">${elapsedMs} ms</div></div>
      ${duration !== undefined ? metric("Duration Sent (s)", duration) : ""}
    </div>`)
  );
}

function renderTranscript(transcript) {
  if (transcript === undefined) return;
  els.results.appendChild(
    card(`<h3>Transcript</h3><div class="notice" style="color:var(--text); border-style:solid;">${esc(transcript) || "<em>(empty)</em>"}</div>`)
  );
}

function renderStt(stt) {
  const mismatch = stt.requested_provider && stt.requested_provider !== stt.provider;
  els.results.appendChild(
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

  els.results.appendChild(
    card(`<h3>Vocabulary</h3><div class="metric-grid">${rows.join("")}</div>${wordsHtml}`)
  );
}

function renderGrammar(g, toolAvailable) {
  let issuesHtml = "";
  if (Array.isArray(g.issues) && g.issues.length) {
    issuesHtml = g.issues
      .map(
        (iss) => `
      <div class="issue-card">
        ${iss.wrong !== undefined ? `<div class="issue-row"><span class="k">Wrong:</span><span class="v-wrong">${esc(iss.wrong)}</span></div>` : ""}
        ${iss.correct !== undefined ? `<div class="issue-row"><span class="k">Correct:</span><span class="v-correct">${esc(iss.correct)}</span></div>` : ""}
        ${iss.message !== undefined ? `<div class="issue-row"><span class="k">Message:</span><span>${esc(iss.message)}</span></div>` : ""}
        ${iss.context !== undefined ? `<div class="issue-row"><span class="k">Context:</span><span>${esc(iss.context)}</span></div>` : ""}
      </div>`
      )
      .join("");
  } else if (g.errors === 0) {
    issuesHtml = `<div class="notice">No grammar issues detected.</div>";
  }

  const toolLine =
    toolAvailable === false
      ? `<div class="notice warn" style="margin-top:8px;"><strong>Note:</strong> the grammar tool (language_tool_python / Java) is unavailable on this backend instance — error count is a crude fallback, not real LanguageTool checking.</div>`
      : "";

  els.results.appendChild(
    card(`<h3>Grammar</h3><div class="metric-grid">
        ${g.score !== undefined ? metric("Score", g.score) : ""}
        ${g.errors !== undefined ? metric("Errors", g.errors) : ""}
      </div>
      ${issuesHtml}
      ${toolLine}`)
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

  els.results.appendChild(
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

  els.results.appendChild(
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
  els.results.appendChild(card(`<h3>Pacing</h3><div class="metric-grid">${rows}</div>`));
}

function renderAcoustic(clarity) {
  if (!clarity) return;
  const rows = [clarity.score !== undefined ? metric("Clarity Score", clarity.score) : ""].join("");
  els.results.appendChild(
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

  els.results.appendChild(card(`<h3>Overall Assessment</h3><div class="metric-grid">${rows.join("")}</div>
    ${data.feedback ? `<div class="notice" style="margin-top:10px; color:var(--text); border-style:solid;">${esc(data.feedback)}</div>` : ""}`));
}

function renderTokenAnalysis(linguisticAnalysis, ltErrors) {
  if (!linguisticAnalysis) {
    const reason =
      ltErrors && ltErrors.analyze
        ? `The /v2/analyze call failed: ${ltErrors.analyze}`
        : "No linguistic_analysis was returned for this request.";
    els.results.appendChild(
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
    els.results.appendChild(
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

  els.results.appendChild(
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
    els.results.appendChild(
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

  els.results.appendChild(
    card(`<h3>Word Timings</h3>
      <div class="notice" style="margin-bottom:10px;"><strong>Source: ${esc(sourceLabel)}</strong> — every word from this provider's word-level timing output.</div>
      <table class="tokens">
        <thead><tr><th>Word</th><th>Start</th><th>End</th><th>Duration</th><th>Probability</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`)
  );
}

function renderNote(note) {
  els.results.appendChild(card(`<h3>Backend Note</h3><div class="notice">${esc(note)}</div>`));
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
  els.results.appendChild(wrapper);
}