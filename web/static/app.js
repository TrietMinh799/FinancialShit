/* ====================================================
   app.js - Valuation RAG Chatbot (Flask + multi-LLM)
   ==================================================== */

const $ = id => document.getElementById(id);
const safe = v => String(v ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function renderMarkdown(text) {
  if (!text) return '';

  // Escape HTML first to prevent XSS
  text = safe(text);

  // Fenced code blocks — protect from other formatting
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

  // Tables — consecutive pipe-delimited lines
  text = text.replace(/((?:^\|.+\|\n?)+)/gm, (match) => {
    const lines = match.trim().split('\n').filter(l => l.trim());
    if (lines.length < 2) return match;
    const hasSep = /^\|[-:| ]+\|$/.test(lines[1].trim());
    let tbl = '<table><thead><tr>';
    lines[0].split('|').filter(c => c.trim()).forEach(c => {
      tbl += `<th>${inlineMd(c.trim())}</th>`;
    });
    tbl += '</tr></thead>';
    if (hasSep && lines.length > 2 || !hasSep && lines.length > 1) {
      tbl += '<tbody>';
      for (let i = hasSep ? 2 : 1; i < lines.length; i++) {
        const cells = lines[i].split('|').filter(c => c.trim());
        if (cells.length) {
          tbl += '<tr>';
          cells.forEach(c => tbl += `<td>${inlineMd(c.trim())}</td>`);
          tbl += '</tr>';
        }
      }
      tbl += '</tbody>';
    }
    return tbl + '</table>';
  });

  // Citation references [N] — convert to styled badges BEFORE block processing
  // Match [N] where N is 1-3 digits, but not inside a markdown link [N](url)
  text = text.replace(/\[(\d{1,3})\](?!\()/g,
    '<span class="citation-ref">$1</span>');

  // Clean up snippet truncation noise: "word ..." → "word…"
  // and "... word" → "… word"
  text = text.replace(/\.{3}\s/g, '… ').replace(/\s\.{3}/g, '…');

  // Clean up raw <br> tags that leaked from the LLM output (e.g. from
  // chunk text that contained <br>).  These are now literal "&lt;br&gt;"
  // after safe() escaped them, so convert them to actual line breaks.
  text = text.replace(/&lt;br\s*\/?&gt;/gi, '\n');

  // Split into blocks by blank lines
  return text.split(/\n{2,}/).map(block => {
    block = block.trim();
    if (!block) return '';

    // Already an HTML block — pass through
    if (block[0] === '<') return block;

    const lines = block.split('\n');

    // Heading — may be followed by other content without a blank line,
    // so extract just the first line as the heading and process the rest.
    if (/^#{1,6}\s/.test(lines[0])) {
      const level = lines[0].indexOf(' ');
      const heading = `<h${level}>${inlineMd(lines[0].slice(level + 1))}</h${level}>`;
      const rest = lines.slice(1).join('\n').trim();
      if (!rest) return heading;
      // Process the remaining lines as a separate block
      return heading + '\n' + _renderBlock(rest);
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(block)) return '<hr>';

    // Unordered list
    if (lines.every(l => /^[-*+]\s/.test(l))) {
      return '<ul>' + lines.map(l => {
        const m = l.match(/^[-*+]\s+(.*)/);
        return m ? `<li>${inlineMd(m[1])}</li>` : '';
      }).join('') + '</ul>';
    }

    // Ordered list
    if (lines.every(l => /^\d+\.\s/.test(l))) {
      return '<ol>' + lines.map(l => {
        const m = l.match(/^\d+\.\s+(.*)/);
        return m ? `<li>${inlineMd(m[1])}</li>` : '';
      }).join('') + '</ol>';
    }

    // Paragraph
    return '<p>' + lines.map((l, i) =>
      i > 0 ? `<br>${inlineMd(l)}` : inlineMd(l)
    ).join('') + '</p>';
  }).filter(Boolean).join('\n');
}

function _renderBlock(block) {
  block = block.trim();
  if (!block) return '';
  if (block[0] === '<') return block;
  const lines = block.split('\n');

  if (/^#{1,6}\s/.test(lines[0])) {
    const level = lines[0].indexOf(' ');
    const heading = `<h${level}>${inlineMd(lines[0].slice(level + 1))}</h${level}>`;
    const rest = lines.slice(1).join('\n').trim();
    if (!rest) return heading;
    return heading + '\n' + _renderBlock(rest);
  }
  if (/^[-*_]{3,}$/.test(block)) return '<hr>';
  if (lines.every(l => /^[-*+]\s/.test(l))) {
    return '<ul>' + lines.map(l => {
      const m = l.match(/^[-*+]\s+(.*)/);
      return m ? `<li>${inlineMd(m[1])}</li>` : '';
    }).join('') + '</ul>';
  }
  if (lines.every(l => /^\d+\.\s/.test(l))) {
    return '<ol>' + lines.map(l => {
      const m = l.match(/^\d+\.\s+(.*)/);
      return m ? `<li>${inlineMd(m[1])}</li>` : '';
    }).join('') + '</ol>';
  }
  return '<p>' + lines.map((l, i) =>
    i > 0 ? `<br>${inlineMd(l)}` : inlineMd(l)
  ).join('') + '</p>';
}

function inlineMd(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

let isSending   = false;
let _abortCtrl  = null;
let providers   = [];
let activeProvider = null;
let conversation = [];            // {role, content} history for LLM context

// -- Provider & model helpers -----------------------
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

// -- Persist / restore settings ---------------------
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

// -- API key dot ------------------------------------
function setKeyDot(state) {
  ['apiKeyDot', 'settingsApiKeyDot'].forEach(id => {
    const el = $(id);
    if (el) el.className = 'api-key-dot' + (state ? ' ' + state : '');
  });
}

// -- Load providers from backend --------------------
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
    ).join('') + '<option value="__custom__">Nhập tên khác...</option>';
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
    input.placeholder = p.key_placeholder || 'API key...';
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

// -- Library ----------------------------------------
async function loadLibrary() {
  try {
    const data = await fetch('/api/library').then(readJson);
    $('libCount').textContent = data.documents;
    const docs = data.recent_documents || [];
    $('sidebarDocs').innerHTML = docs.length
      ? docs.map(d => {
          const typeLabel = d.source_type === 'annual_report' ? 'Báo cáo' : 'Sách';
          const toggleLabel = d.source_type === 'annual_report' ? 'Sách' : 'Báo cáo';
          const toggleIcon = d.source_type === 'annual_report' ? '📖' : '📊';
          return `
          <div class="lib-item" data-id="${d.id}">
            <span class="lib-dot lib-dot--${d.source_type === 'book' ? 'book' : 'report'}"></span>
            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${safe(d.title)} (${typeLabel})">${safe(d.title)}</span>
            <span style="flex-shrink:0;font-size:10.5px;color:var(--text-muted)">${d.page_count}tr · ${d.chunk_count} đoạn</span>
            <button class="lib-del" data-action="chunks" title="Xem các đoạn">⊞</button>
            <button class="lib-del" data-action="reclassify" data-target="${d.source_type === 'annual_report' ? 'book' : 'annual_report'}" title="Đổi thành ${toggleLabel}">${toggleIcon}</button>
            <button class="lib-del" data-action="delete" title="Xóa tài liệu">×</button>
          </div>`;
        }).join('')
      : '<p style="font-size:11.5px;color:var(--text-muted);padding:0 4px">Chưa có tài liệu</p>';
  } catch(e) {
    $('sidebarDocs').innerHTML = `<p style="font-size:11.5px;color:#f87171;padding:0 4px">${safe(e.message)}</p>`;
  }
}

// Event delegation for library action buttons
$('sidebarDocs').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const item = btn.closest('[data-id]');
  if (!item) return;
  const id = Number(item.dataset.id);
  const action = btn.dataset.action;
  if (action === 'chunks') viewChunks(id);
  else if (action === 'delete') deleteBook(id);
  else if (action === 'reclassify') reclassifyBook(id, btn.dataset.target);
});

async function deleteBook(id) {
  if (!confirm('Xóa tài liệu này?')) return;
  try {
    const res = await fetch(`/api/books/${id}`, {
      method: 'DELETE',
      headers: {'X-Requested-With': 'XMLHttpRequest'}
    }).then(readJson);
    if (res.error) throw new Error(res.error);
    loadLibrary();
  } catch(e) {
    alert('Lỗi: ' + e.message);
  }
}

async function reclassifyBook(id, newType) {
  const label = newType === 'annual_report' ? 'báo cáo thường niên' : 'sách';
  if (!confirm(`Đổi tài liệu thành "${label}" và re-index?`)) return;
  try {
    const res = await fetch(`/api/books/${id}/reclassify`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source_type: newType})
    }).then(readJson);
    if (res.error) throw new Error(res.error);
    if (res.changed) {
      appendBotText(`[OK] Đã đổi thành "${label}" và re-index xong.`);
    } else {
      appendBotText(`[i] Tài liệu đã là "${label}" rồi.`);
    }
    loadLibrary();
  } catch(e) {
    alert('Lỗi: ' + e.message);
  }
}

async function viewChunks(id) {
  try {
    const data = await fetch(`/api/books/${id}/chunks?limit=100`).then(readJson);
    if (data.error) throw new Error(data.error);
    const html = data.chunks.map(c =>
      `<div style="margin-bottom:10px;padding:8px 10px;background:var(--bg-elevated);border-radius:8px;font-size:12px;line-height:1.5">
        <div style="display:flex;gap:8px;margin-bottom:4px;color:var(--text-muted);font-size:10.5px">
          <span>Đoạn ${c.chunk_index + 1}</span>
          <span>Trang ${c.page_start}${c.page_end !== c.page_start ? '-' + c.page_end : ''}</span>
          <span>${c.char_count} ký tự</span>
        </div>
        <div style="color:var(--text-secondary)">${safe(c.text_preview)}</div>
      </div>`
    ).join('');
    const overlay = document.createElement('div');
    overlay.className = 'modal-backdrop open';
    overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };
    overlay.innerHTML = `<div class="modal" style="width:min(680px,94vw);max-height:80vh;overflow-y:auto">
      <p class="modal-title">${safe(data.title)} — ${data.total_chunks} đoạn</p>
      ${html || '<p style="color:var(--text-muted)">Không có đoạn nào.</p>'}
      <div class="modal-actions"><button class="btn-cancel" onclick="this.closest('.modal-backdrop').remove()">Đóng</button></div>
    </div>`;
    document.body.appendChild(overlay);
  } catch(e) {
    alert('Lỗi: ' + e.message);
  }
}

// -- Welcome screen ---------------------------------
function setWelcome(visible) {
  $('welcomeScreen').style.display = visible ? 'flex' : 'none';
}

// -- Message rendering ------------------------------
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
  body.scrollTo({top: body.scrollHeight});
}

// -- Send question ----------------------------------
async function sendQuestion(question) {
  if (!question || isSending) return;

  // Cancel any previous in-flight request
  if (_abortCtrl) _abortCtrl.abort();
  const abortCtrl = new AbortController();
  _abortCtrl = abortCtrl;

  isSending = true;
  updateSendBtn();

  conversation.push({role: "user", content: question});
  // Keep last 20 messages to avoid overflowing the LLM context window.
  if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
  appendUserMessage(question);

  // Create a streaming bot message bubble
  const botMsg = document.createElement('div');
  botMsg.className = 'message message--bot';
  const av = document.createElement('div');
  av.className = 'message-avatar';
  av.textContent = 'AI';
  const cont = document.createElement('div');
  cont.className = 'message-content';
  const bub = document.createElement('div');
  bub.className = 'message-bubble';
  bub.innerHTML = '<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
  botMsg.appendChild(av);
  const bwrap = document.createElement('div');
  bwrap.className = 'message-bubble-wrapper';
  bwrap.appendChild(bub);
  cont.appendChild(bwrap);
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = '[]'; // placeholder, update later
  cont.appendChild(meta);
  botMsg.appendChild(cont);
  
  $('messagesList').appendChild(botMsg);
  scrollBottom();
  
  let full_text = '';
  let lastRenderLen = 0;
  let renderTimer = null;
  let statusText = 'Đang truy xuất · ' + (activeProvider?.label || '') + '...';
  setStatus(statusText);

  function scheduleRender() {
    if (renderTimer) return;
    renderTimer = setTimeout(() => {
      renderTimer = null;
      // Only render new text since last render
      if (full_text.length > lastRenderLen) {
        bub.innerHTML = renderMarkdown(full_text);
        lastRenderLen = full_text.length;
      }
    }, 80);
  }
  function flushRender() {
    if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
    if (full_text.length > lastRenderLen) {
      bub.innerHTML = renderMarkdown(full_text);
      lastRenderLen = full_text.length;
    }
  }
  
  try {
    // Try streaming endpoint
    const res = await fetch('/api/ask/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        messages: conversation.slice(0, -1),
        api_key:  apiKey(),
        model:    apiModel(),
        base_url: baseUrl(),
      }),
      signal: abortCtrl.signal,
    });
    
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    if (!res.body) {
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      if (data.answer) {
        full_text = data.answer;
        bub.innerHTML = renderMarkdown(full_text);
        meta.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + (activeProvider ? ` · ${activeProvider.label}` : '') + ` · ${full_text.length} ký tự`;
        if (data.citations?.length) {
          const src = document.createElement('div');
          src.className = 'message-sources';
          data.citations.forEach(c => {
            const t = document.createElement('span');
            t.className = 'source-tag';
            t.textContent = c.title || 'Nguồn';
            src.appendChild(t);
          });
          cont.appendChild(src);
        }
        setStatus('Done');
      }
      conversation.push({role: "assistant", content: full_text});
      if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
    } else {
      let reader;
      try {
        reader = res.body.getReader();
      } catch {
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (data.answer) full_text = data.answer;
        bub.innerHTML = renderMarkdown(full_text);
        meta.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + (activeProvider ? ` · ${activeProvider.label}` : '') + ` · ${full_text.length} ký tự`;
        conversation.push({role: "assistant", content: full_text});
        if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
        return;
      }
      const decoder = new TextDecoder();
      let partial = '';  // holds an incomplete line from previous chunk
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const text = partial + decoder.decode(value, {stream: true});
        const lastNewline = text.lastIndexOf('\n');
        if (lastNewline === -1) { partial = text; continue; }
        partial = text.slice(lastNewline + 1);
        for (const line of text.slice(0, lastNewline).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = JSON.parse(line.slice(6));
          if (data.status) {
            setStatus(data.status);
          } else if (data.token) {
            full_text += data.token;
            scheduleRender();
            meta.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + (activeProvider ? ` · ${activeProvider.label}` : '');
          } else if (data.done) {
            flushRender();
            const sources = (data.citations || []).map(c => c.title || 'Nguồn');
            const cnt = full_text.length;
            meta.textContent = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) + (activeProvider ? ` · ${activeProvider.label}` : '') + ` · ${cnt} ký tự`;
            if (sources.length) {
              const src = document.createElement('div');
              src.className = 'message-sources';
              sources.forEach(s => {
                const t = document.createElement('span');
                t.className = 'source-tag';
                t.textContent = s;
                src.appendChild(t);
              });
              cont.appendChild(src);
            }
            if (activeProvider) {
              setStatus(data.mode === 'llm' ? 'Câu trả lời LLM xong' : 'Câu trả lời bằng chứng xong');
            }
            conversation.push({role: "assistant", content: full_text});
            if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
            return;
          } else if (data.error) {
            throw new Error(data.error);
          }
        }
      }
      // Flush any remaining text
      flushRender();
    }
  } catch(err) {
    if (err.name === 'AbortError') return;
    botMsg.classList.add('error');
    bub.textContent = 'Lỗi: ' + err.message;
    meta.textContent = '';
    conversation.push({role: "assistant", content: "Lỗi: " + err.message});
    if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
    setStatus('Lỗi');
  } finally {
    isSending = false;
    _abortCtrl = null;
    updateSendBtn();
  }
}

// -- Upload book ------------------------------------
async function submitBook() {
  const file = $('bookFile').files[0];
  if (!file) { alert('Vui lòng chọn tệp sách.'); return; }
  closeModal('bookModal');
  const sourceType = $('bookSourceType').value || 'book';
  const label = sourceType === 'annual_report' ? 'báo cáo thường niên' : 'sách';
  appendBotHtml(`<span style="color:var(--text-muted)">Đang thêm ${label} <strong>${safe(file.name)}</strong> vào RAG...</span>`);
  setStatus(`Đang thêm ${label}...`);

  const fd = new FormData();
  fd.append('book_file', file);
  fd.append('book_title', $('bookTitle').value.trim() || file.name.replace(/\.[^.]+$/,''));
  fd.append('source_type', sourceType);

  try {
    const result = await fetch('/api/upload-book', {
      method: 'POST',
      headers: {'X-Requested-With': 'XMLHttpRequest'},
      body: fd
    }).then(readJson);
    const msg = result.inserted
      ? `[OK] Đã thêm "${safe(result.title)}" (${result.source_type}) - ${result.page_count} trang, ${result.chunk_count} đoạn.`
      : `[i] "${safe(result.title)}" đã có trong thư viện (${result.chunk_count} đoạn).`;
    appendBotText(msg);
    setStatus('Đã thêm xong');
    await loadLibrary();
  } catch(err) {
    appendBotText('Lỗi khi thêm: ' + err.message);
    setStatus('Lỗi');
  }
}

// -- Analyze report ---------------------------------
async function submitReport() {
  const file = $('reportFile').files[0];
  if (!file) { alert('Vui lòng chọn file PDF báo cáo.'); return; }
  const company = $('reportCompany').value.trim() || 'Công ty';
  const ticker  = $('reportTicker').value.trim()  || 'N/A';
  closeModal('reportModal');

  appendBotHtml(`<span style="color:var(--text-muted)">Đang phân tích báo cáo của <strong>${safe(company)} (${safe(ticker)})</strong> qua <strong>${safe(activeProvider?.label||'LLM')}</strong>...</span>`);
  setStatus('Đang phân tích...');
  saveSettings();

  const fd = new FormData();
  fd.append('report_file', file);
  fd.append('company', company);
  fd.append('ticker', ticker);
  fd.append('api_key',  apiKey());
  fd.append('model',    apiModel());
  fd.append('base_url', baseUrl());

  try {
    const r = await fetch('/api/analyze-report', {
      method: 'POST',
      headers: {'X-Requested-With': 'XMLHttpRequest'},
      body: fd
    }).then(readJson);
    renderAnalysis(r, company, ticker);
    setStatus('Phân tích hoàn thành');
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

  const swotColor = {'Điểm_Mạnh':'#34d399','Điểm_Yếu':'#f87171','Cơ_Hội':'#818cf8','Thách_Thức':'#fbbf24'};
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
      <ul>${(items||[]).map(i=>`<li>${safe(i)}</li>`).join('')||'<li>-</li>'}</ul>
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
      <h4>• Phân Tích: ${safe(company)} (${safe(ticker)})</h4>
      <div class="score-row">${scoreHtml}</div>
      <div class="result-section">
        <div class="result-section-title">Lợi Thế Cạnh Tranh</div>
        <p><strong>${safe(adv.rating||'')}</strong> - ${safe(adv.assessment||'')}</p>
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
  appendBotText(`Phân tích ${company} hoàn thành. Bạn có thể hỏi bất kỳ câu hỏi nào về báo cáo này.`);
}

// -- Test key ---------------------------------------
async function testKey() {
  saveSettings();
  if (!apiKey()) { alert('Vui lòng dán API key trước.'); return; }
  setStatus('Đang kiểm tra...'); setKeyDot('');
  try {
    const r = await fetch('/api/test-key', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({api_key: apiKey(), model: apiModel(), base_url: baseUrl()})
    }).then(r => r.json());
    setKeyDot(r.ok ? 'ok' : 'err');
    setStatus(r.ok ? `${activeProvider?.label||'API'} xong` : 'Kết nối thất bại');
    alert(r.message);
  } catch(e) {
    setKeyDot('err'); setStatus('Lỗi'); alert(e.message);
  }
}

// -- Input helpers ----------------------------------
function autoResize() {
  const t = $('userInput');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 180) + 'px';
}
function updateSendBtn() {
  $('sendButton').disabled = !$('userInput').value.trim() || isSending;
}

// -- Modal helpers ----------------------------------
function openModal(id)  { $(id).classList.add('open'); }
function closeModal(id) { $(id).classList.remove('open'); }

// -- Sidebar ----------------------------------------
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
  conversation = [];
  setWelcome(true);
}

// -- Event wiring -----------------------------------
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
$('settingsSaveBtn').addEventListener('click', () => { syncSettingsFromModal(); saveSettings(); closeModal('settingsModal'); setStatus('Cài đặt LLM đã lưu xong'); });
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

// -- Init -------------------------------------------
updateSendBtn();
setWelcome(true);
loadProviders();
loadLibrary().catch(console.error);
