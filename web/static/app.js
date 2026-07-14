/* ====================================================
   app.js — Valuation RAG Chatbot (Flask + multi-LLM)
   ==================================================== */

const $ = id => document.getElementById(id);
const safe = v => String(v ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

let isSending   = false;
let typingEl    = null;
let providers   = [];
let activeProvider = null;

// ── Provider & model helpers ───────────────────────
function baseUrl() {
  if (!activeProvider) return '';
  if (activeProvider.id === 'custom') return ($('settingsCustomBaseUrl')?.value || '').trim();
  return activeProvider.base_url;
}

function apiKey()   { return ($('settingsApiKey')?.value || '').trim(); }

function apiModel() {
  const sel = $('settingsModelSelect').value;
  if (sel === '__custom__') return ($('settingsModelCustom')?.value || '').trim();
  return sel || ($('settingsModelCustom')?.value || '').trim() || '';
}

// ── Persist / restore settings ─────────────────────
function loadSettings() {
  const pid   = localStorage.getItem('rag_provider_id') || 'openai';
  const key   = localStorage.getItem('rag_api_key') || '';
  const model = localStorage.getItem('rag_model') || '';
  const curl  = localStorage.getItem('rag_custom_url') || '';

  if (key) {
    $('settingsApiKey').value = key;
    $('settingsRememberKey').checked = true;
  }
  if (curl) {
    $('settingsCustomBaseUrl').value = curl;
  }

  return { pid, model };
}

function saveSettings() {
  const remember = $('settingsRememberKey').checked;
  const keyValue = (apiKey() || $('settingsApiKey').value.trim()).trim();
  const selectedModel = $('settingsModelSelect').value;
  const customModel = $('settingsModelCustom').value.trim();
  const modelValue = selectedModel === '__custom__' ? customModel : (selectedModel || customModel || '');

  if (remember && keyValue) localStorage.setItem('rag_api_key', keyValue);
  else localStorage.removeItem('rag_api_key');
  localStorage.setItem('rag_model', modelValue);
  localStorage.setItem('rag_custom_url', $('settingsCustomBaseUrl').value.trim());
  if (activeProvider) localStorage.setItem('rag_provider_id', activeProvider.id);
}

function setStatus(txt) { $('statusText').textContent = txt; }

async function readJson(res) {
  const d = await res.json();
  if (!res.ok) throw new Error(d.error || 'Request failed');
  return d;
}

// ── API key dot ────────────────────────────────────
function setKeyDot(state) {
  ['apiKeyDot', 'settingsApiKeyDot'].forEach(id => {
    const el = $(id);
    if (el) el.className = 'api-key-dot' + (state ? ' ' + state : '');
  });
}

// ── Load providers from backend ────────────────────
async function loadProviders() {
  try {
    const data = await fetch('/api/providers').then(r => r.json());
    providers = data.providers || [];
    const { pid, model } = loadSettings();
    renderProviderGrid(pid, model);
  } catch(e) {
    [$('providerGrid'), $('settingsProviderGrid')].filter(Boolean).forEach(grid => {
      grid.innerHTML = '<div style="font-size:11.5px;color:#f87171;grid-column:1/-1">Không thể tải providers</div>';
    });
  }
}

function renderProviderGrid(activePid, savedModel) {
  const grids = [$('providerGrid'), $('settingsProviderGrid')].filter(Boolean);
  grids.forEach(grid => {
    grid.innerHTML = providers.map(p => `
      <button class="provider-btn${p.id === activePid ? ' active' : ''}"
              data-id="${safe(p.id)}"
              style="--provider-color:${safe(p.color)}">
        <span class="provider-dot" style="background:${safe(p.color)}"></span>
        ${safe(p.label)}
      </button>`).join('');

    grid.querySelectorAll('.provider-btn').forEach(btn => {
      btn.addEventListener('click', () => selectProvider(btn.dataset.id, ''));
    });
  });

  selectProvider(activePid, savedModel);
}

function selectProvider(pid, preferredModel) {
  const p = providers.find(x => x.id === pid) || providers[0];
  if (!p) return;
  activeProvider = p;

  const grids = [$('providerGrid'), $('settingsProviderGrid')].filter(Boolean);
  grids.forEach(grid => {
    grid.querySelectorAll('.provider-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.id === p.id);
      btn.style.setProperty('--provider-color', p.color);
    });
  });

  const customRow = $('settingsCustomUrlRow');
  if (customRow) customRow.style.display = p.id === 'custom' ? '' : 'none';

  const modelSelects = [$('settingsModelSelect')].filter(Boolean);
  const customInputs = [$('settingsModelCustom')].filter(Boolean);
  const keyInputs = [$('settingsApiKey')].filter(Boolean);

  modelSelects.forEach(sel => {
    sel.innerHTML = p.models.map(m =>
      `<option value="${safe(m)}">${safe(m)}</option>`
    ).join('') + '<option value="__custom__">Nhập tên khác…</option>';
  });

  const target = preferredModel || p.models[0] || '';
  const useCustom = !!target && !p.models.includes(target);
  const selectedValue = useCustom ? '__custom__' : (p.models.includes(target) ? target : (p.models[0] || ''));

  modelSelects.forEach(sel => {
    sel.value = selectedValue;
  });
  customInputs.forEach(input => {
    input.value = useCustom ? target : '';
    input.style.display = useCustom ? '' : 'none';
  });

  keyInputs.forEach(input => {
    input.placeholder = p.key_placeholder || 'API key…';
  });

  setKeyDot('');
}

function syncSettingsToModal() {
  $('settingsApiKey').value = apiKey();
  $('settingsRememberKey').checked = $('settingsRememberKey').checked;
}

function syncSettingsFromModal() {
  $('settingsApiKey').value = $('settingsApiKey').value.trim();
  $('settingsRememberKey').checked = $('settingsRememberKey').checked;
}

$('settingsModelSelect').addEventListener('change', () => {
  $('settingsModelCustom').style.display = $('settingsModelSelect').value === '__custom__' ? '' : 'none';
});

// ── Library ────────────────────────────────────────
async function loadLibrary() {
  try {
    const data = await fetch('/api/library').then(readJson);
    $('libCount').textContent = data.documents;
    const docs = data.recent_documents || [];
    $('sidebarDocs').innerHTML = docs.length
      ? docs.map(d => `
          <div class="lib-item" data-id="${d.id}">
            <span class="lib-dot lib-dot--${d.source_type === 'book' ? 'book' : 'report'}"></span>
            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${safe(d.title)}</span>
            <span style="flex-shrink:0;font-size:10.5px;color:var(--text-muted)">${d.page_count}tr</span>
            <button class="lib-del" onclick="deleteBook(${d.id})" title="Xoá tài liệu">✕</button>
          </div>`).join('')
      : '<p style="font-size:11.5px;color:var(--text-muted);padding:0 4px">Chưa có tài liệu</p>';
  } catch(e) {
    $('sidebarDocs').innerHTML = `<p style="font-size:11.5px;color:#f87171;padding:0 4px">${safe(e.message)}</p>`;
  }
}

async function deleteBook(id) {
  if (!confirm('Xoá tài liệu này?')) return;
  try {
    const res = await fetch(`/api/books/${id}`, { method: 'DELETE' }).then(readJson);
    if (res.error) throw new Error(res.error);
    loadLibrary();
  } catch(e) {
    alert('Lỗi: ' + e.message);
  }
}

// ── Welcome screen ─────────────────────────────────
function setWelcome(visible) {
  $('welcomeScreen').style.display = visible ? 'flex' : 'none';
}

// ── Message rendering ──────────────────────────────
function appendMessage(role, html, sources = []) {
  setWelcome(false);
  const now = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const wrap = document.createElement('div');
  wrap.className = `message message--${role}`;

  const av = document.createElement('div');
  av.className = 'message-avatar';
  av.textContent = role === 'user' ? 'U' : 'AI';

  const cont = document.createElement('div');
  cont.className = 'message-content';

  const bub = document.createElement('div');
  bub.className = 'message-bubble';
  bub.innerHTML = html;

  // Copy button
  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-btn';
  copyBtn.setAttribute('aria-label', 'Copy message');
  const textOnly = html.replace(/<[^>]*>/g, '');
  copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(textOnly).then(() => {
      const orig = copyBtn.innerHTML;
      copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
      copyBtn.classList.add('copied');
      setTimeout(() => { copyBtn.innerHTML = orig; copyBtn.classList.remove('copied'); }, 1500);
    });
  });

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = now + (activeProvider ? ` · ${activeProvider.label}` : '');

  const bwrap = document.createElement('div');
  bwrap.className = 'message-bubble-wrapper';
  bwrap.appendChild(bub);
  bwrap.appendChild(copyBtn);
  cont.appendChild(bwrap);

  if (role === 'bot' && sources.length) {
    const src = document.createElement('div');
    src.className = 'message-sources';
    sources.forEach(s => {
      const t = document.createElement('span');
      t.className = 'source-tag';
      t.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="3" x2="9" y2="21"/></svg> ${safe(s)}`;
      src.appendChild(t);
    });
    cont.appendChild(src);
  }

  cont.appendChild(meta);
  wrap.appendChild(av);
  wrap.appendChild(cont);
  $('messagesList').appendChild(wrap);
  scrollBottom();
}

function appendUserMessage(text) {
  appendMessage('user', safe(text).replace(/\n/g,'<br>'));
}
function appendBotText(text, sources = []) {
  appendMessage('bot', safe(text).replace(/\n/g,'<br>'), sources);
}
function appendBotHtml(html, sources = []) {
  appendMessage('bot', html, sources);
}

function scrollBottom() {
  const body = $('chatBody');
  body.scrollTo({top: body.scrollHeight, behavior: 'smooth'});
}

// ── Typing indicator ───────────────────────────────
function showTyping() {
  if (typingEl) return;
  typingEl = document.createElement('div');
  typingEl.className = 'typing-indicator';

  const av = document.createElement('div');
  av.className = 'message-avatar';
  av.style.cssText = 'width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;margin-top:4px;background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff';
  av.textContent = 'AI';

  const bub = document.createElement('div');
  bub.className = 'typing-bubble';
  for (let i=0;i<3;i++) {
    const d = document.createElement('span');
    d.className = 'typing-dot';
    bub.appendChild(d);
  }

  typingEl.appendChild(av);
  typingEl.appendChild(bub);
  $('typingAnchor').appendChild(typingEl);
  scrollBottom();
}

function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

// ── Send question ──────────────────────────────────
async function sendQuestion(question) {
  if (!question || isSending) return;
  appendUserMessage(question);
  isSending = true; updateSendBtn();
  showTyping();
  saveSettings();

  try {
    setStatus(`Đang truy xuất · ${activeProvider?.label || ''}…`);
    const result = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        api_key:  apiKey(),
        model:    apiModel(),
        base_url: baseUrl(),
      })
    }).then(readJson);

    hideTyping();
    const sources = (result.citations || []).map(c => c.title || 'Nguồn');
    appendBotText(result.answer, sources);
    setStatus(result.mode === 'llm' ? 'Câu trả lời LLM ✓' : 'Câu trả lời bằng chứng ✓');
  } catch(err) {
    hideTyping();
    appendBotText('Lỗi: ' + err.message);
    setStatus('Lỗi');
  }

  isSending = false; updateSendBtn();
}

// ── Upload book ────────────────────────────────────
async function submitBook() {
  const file = $('bookFile').files[0];
  if (!file) { alert('Vui lòng chọn tệp sách.'); return; }
  closeModal('bookModal');
  appendBotHtml(`<span style="color:var(--text-muted)">⏳ Đang thêm sách <strong>${safe(file.name)}</strong> vào RAG…</span>`);
  setStatus('Đang thêm sách…');

  const fd = new FormData();
  fd.append('book_file', file);
  fd.append('book_title', $('bookTitle').value.trim() || file.name.replace(/\.[^.]+$/,''));

  try {
    const result = await fetch('/api/upload-book', {method:'POST', body:fd}).then(readJson);
    const msg = result.inserted
      ? `✅ Đã thêm sách "${safe(result.title)}" — ${result.page_count} trang, ${result.chunk_count} đoạn.`
      : `ℹ️ Sách "${safe(result.title)}" đã có trong thư viện (${result.chunk_count} đoạn).`;
    appendBotText(msg);
    setStatus('Đã thêm sách ✓');
    await loadLibrary();
  } catch(err) {
    appendBotText('Lỗi khi thêm sách: ' + err.message);
    setStatus('Lỗi');
  }
}

// ── Analyze report ─────────────────────────────────
async function submitReport() {
  const file = $('reportFile').files[0];
  if (!file) { alert('Vui lòng chọn file PDF báo cáo.'); return; }
  const company = $('reportCompany').value.trim() || 'Công ty';
  const ticker  = $('reportTicker').value.trim()  || 'N/A';
  closeModal('reportModal');

  appendBotHtml(`<span style="color:var(--text-muted)">⏳ Đang phân tích báo cáo của <strong>${safe(company)} (${safe(ticker)})</strong> qua <strong>${safe(activeProvider?.label||'LLM')}</strong>…</span>`);
  setStatus('Đang phân tích…');
  saveSettings();

  const fd = new FormData();
  fd.append('report_file', file);
  fd.append('company', company);
  fd.append('ticker', ticker);
  fd.append('api_key',  apiKey());
  fd.append('model',    apiModel());
  fd.append('base_url', baseUrl());

  try {
    const r = await fetch('/api/analyze-report', {method:'POST', body:fd}).then(readJson);
    renderAnalysis(r, company, ticker);
    setStatus('Phân tích hoàn tất ✓');
    await loadLibrary();
  } catch(err) {
    appendBotText('Lỗi khi phân tích: ' + err.message);
    setStatus('Lỗi');
  }
}

function renderAnalysis(r, company, ticker) {
  const s = r.scores || {};
  const adv = r.competitive_advantage || {};
  const sit = r.company_situation || {};
  const actions = r.growth_actions || [];
  const lrep = r.llm_report;
  const reason = r.reasoned_analysis || {};

  const scoreHtml = [
    ['Tổng Thể', s.overall_growth_score],
    ['Lợi Thế', s.moat_sustainability],
    ['Tăng Trưởng', s.growth_capacity],
    ['Thực Thi', s.execution_quality],
    ['Tài Chính', s.financial_resilience],
  ].map(([l,v]) => `<span class="score-chip">${safe(l)}: <strong>${v}/100</strong></span>`).join('');

  let commentaryHtml = '';
  const comm = reason.score_commentary || {};
  if (comm.moat || comm.growth || comm.execution || comm.financial || comm.risk) {
    const commLabels = [
      ['moat','Lợi Thế Cạnh Tranh','#a78bfa'],
      ['growth','Tăng Trưởng','#34d399'],
      ['execution','Thực Thi','#fbbf24'],
      ['financial','Tài Chính','#60a5fa'],
      ['risk','Rủi Ro','#f87171'],
    ];
    commentaryHtml = commLabels
      .filter(([k]) => comm[k])
      .map(([k,l,c]) => `<div style="margin-bottom:8px"><span style="font-size:11px;font-weight:600;color:${c}">${l}</span><p style="margin:2px 0 0;font-size:12.5px;color:var(--text-secondary);line-height:1.5">${safe(comm[k])}</p></div>`)
      .join('');
    if (commentaryHtml) {
      commentaryHtml = `<div class="result-section"><div class="result-section-title">Giải Thích Chi Tiết</div>${commentaryHtml}</div>`;
    }
  }

  const swotColor = {Điểm_Mạnh:'#34d399',Điểm_Yếu:'#f87171',Cơ_Hội:'#818cf8',Thách_Thức:'#fbbf24'};
  const rswot = reason.reasoned_swot || {};
  const swotData = [
    ['Điểm_Mạnh','Điểm Mạnh', rswot.strengths || sit.strengths],
    ['Điểm_Yếu','Điểm Yếu', rswot.weaknesses || sit.weaknesses],
    ['Cơ_Hội','Cơ Hội', rswot.opportunities || sit.opportunities],
    ['Thách_Thức','Thách Thức', rswot.threats || sit.threats],
  ];
  const swotHtml = swotData.map(([key,label,items]) => `
    <div class="result-section">
      <div class="result-section-title" style="color:${swotColor[key]}">${label}</div>
      <ul>${(items||[]).map(i=>`<li>${safe(i)}</li>`).join('')||'<li>—</li>'}</ul>
    </div>`).join('');

  let keyHtml = '';
  if (reason.key_considerations) {
    keyHtml = `<div class="result-section"><div class="result-section-title">Lưu Ý Cho Nhà Phân Tích</div><p style="margin:4px 0 0;font-size:12.5px;color:var(--text-secondary);line-height:1.6">${safe(reason.key_considerations)}</p></div>`;
  }

  const actHtml = actions.length
    ? '<ol style="margin:4px 0 0 16px">' + actions.map(a=>`<li>${safe(a)}</li>`).join('') + '</ol>'
    : '<p style="color:var(--text-muted)">Không có hành động</p>';

  let repHtml = '';
  if (lrep && lrep.sections) {
    const sec = lrep.sections;
    const titles = [
      ['qualitative_report','Báo Cáo Định Tính'],
      ['quantitative_report','Báo Cáo Định Lượng'],
      ['valuation_method_rules','Quy Tắc Định Giá'],
      ['excel_model_format','Mô Hình Excel'],
      ['recommended_next_steps','Bước Tiếp Theo'],
    ];
    repHtml = `<div class="result-section"><div class="result-section-title">Báo Cáo KB (${safe(lrep.mode_label||'')})</div>`
      + titles.map(([k,l])=> sec[k]
        ? `<div style="margin-bottom:10px"><span style="font-size:11.5px;font-weight:600;color:var(--text-primary)">${l}</span><p style="margin:4px 0 0;font-size:12.5px;color:var(--text-secondary);line-height:1.6">${safe(sec[k])}</p></div>`
        : '').join('')
      + '</div>';
  }

  appendBotHtml(`
    <div class="result-card">
      <h4>📊 Phân Tích: ${safe(company)} (${safe(ticker)})</h4>
      <div class="score-row">${scoreHtml}</div>
      <div class="result-section">
        <div class="result-section-title">Lợi Thế Cạnh Tranh</div>
        <p><strong>${safe(adv.rating||'')}</strong> — ${safe(adv.assessment||'')}</p>
      </div>
      ${commentaryHtml}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px">${swotHtml}</div>
      ${keyHtml}
      <div class="result-section" style="margin-top:12px">
        <div class="result-section-title">Hành Động Tăng Trưởng</div>
        ${actHtml}
      </div>
      ${repHtml}
    </div>`);
  appendBotText(`Phân tích ${company} hoàn tất. Bạn có thể hỏi tôi bất kỳ câu hỏi nào về báo cáo này.`);
}

// ── Test key ───────────────────────────────────────
async function testKey() {
  saveSettings();
  if (!apiKey()) { alert('Vui lòng dán API key trước.'); return; }
  setStatus('Đang kiểm tra…'); setKeyDot('');
  try {
    const r = await fetch('/api/test-key', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({api_key: apiKey(), model: apiModel(), base_url: baseUrl()})
    }).then(r => r.json());
    setKeyDot(r.ok ? 'ok' : 'err');
    setStatus(r.ok ? `${activeProvider?.label||'API'} ✓` : 'Kết nối thất bại');
    alert(r.message);
  } catch(e) {
    setKeyDot('err'); setStatus('Lỗi'); alert(e.message);
  }
}

// ── Input helpers ──────────────────────────────────
function autoResize() {
  const t = $('userInput');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 180) + 'px';
}
function updateSendBtn() {
  $('sendButton').disabled = !$('userInput').value.trim() || isSending;
}

// ── Modal helpers ──────────────────────────────────
function openModal(id)  { $(id).classList.add('open'); }
function closeModal(id) { $(id).classList.remove('open'); }

// ── Sidebar ────────────────────────────────────────
function openSidebar() {
  $('sidebar').classList.add('open');
  $('sidebarOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeSidebar() {
  $('sidebar').classList.remove('open');
  $('sidebarOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

function clearChat() {
  $('messagesList').innerHTML = '';
  setWelcome(true);
}

// ── Event wiring ───────────────────────────────────
$('sendButton').addEventListener('click', () => {
  const q = $('userInput').value.trim();
  if (!q) return;
  $('userInput').value = '';
  $('userInput').style.height = 'auto';
  updateSendBtn();
  sendQuestion(q);
});

$('userInput').addEventListener('input', () => { autoResize(); updateSendBtn(); });
$('userInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('sendButton').click(); }
});

document.querySelectorAll('.starter-card').forEach(c => {
  c.addEventListener('click', () => {
    const p = c.dataset.prompt;
    if (p) { $('userInput').value = p; autoResize(); updateSendBtn(); $('sendButton').click(); }
  });
});

$('menuBtn').addEventListener('click', openSidebar);
$('sidebarCloseBtn').addEventListener('click', closeSidebar);
$('sidebarOverlay').addEventListener('click', closeSidebar);

$('openSettingsBtn').addEventListener('click', () => { syncSettingsToModal(); openModal('settingsModal'); });
$('headerOpenSettings').addEventListener('click', () => { syncSettingsToModal(); openModal('settingsModal'); });
$('openBookModal').addEventListener('click', () => { closeSidebar(); openModal('bookModal'); });
$('headerUploadBook').addEventListener('click', () => openModal('bookModal'));
$('openReportModal').addEventListener('click', () => { closeSidebar(); openModal('reportModal'); });
$('headerAnalyzeReport').addEventListener('click', () => openModal('reportModal'));
$('clearChatBtn').addEventListener('click', () => { closeSidebar(); clearChat(); });
$('settingsTestBtn').addEventListener('click', async () => { syncSettingsFromModal(); saveSettings(); await testKey(); });
$('settingsSaveBtn').addEventListener('click', () => { syncSettingsFromModal(); saveSettings(); closeModal('settingsModal'); setStatus('Cài đặt LLM đã lưu ✓'); });
$('refreshLibBtn').addEventListener('click', () => loadLibrary().catch(console.error));
$('settingsApiKey').addEventListener('change', () => setKeyDot(''));

$('bookModalClose').addEventListener('click', () => closeModal('bookModal'));
$('bookSubmit').addEventListener('click', submitBook);
$('reportModalClose').addEventListener('click', () => closeModal('reportModal'));
$('reportSubmit').addEventListener('click', submitReport);
$('settingsModalClose').addEventListener('click', () => closeModal('settingsModal'));

$('bookModal').addEventListener('click', e => { if (e.target === $('bookModal')) closeModal('bookModal'); });
$('reportModal').addEventListener('click', e => { if (e.target === $('reportModal')) closeModal('reportModal'); });
$('settingsModal').addEventListener('click', e => { if (e.target === $('settingsModal')) closeModal('settingsModal'); });

// ── Init ───────────────────────────────────────────
updateSendBtn();
setWelcome(true);
loadProviders();
loadLibrary().catch(console.error);
