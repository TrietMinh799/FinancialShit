/* =========================================================
   chat.js — RAG Chatbot UI Interactions
   ========================================================= */

(function () {
  // ── DOM References ──────────────────────────────────────
  const sendBtn        = document.getElementById('sendButton');
  const userInput      = document.getElementById('userInput');
  const messagesList   = document.getElementById('messagesList');
  const chatMessages   = document.getElementById('chatMessages');
  const welcomeScreen  = document.getElementById('welcomeScreen');
  const newChatBtn     = document.getElementById('newChatBtn');
  const clearChatBtn   = document.getElementById('clearChatBtn');
  const menuBtn        = document.getElementById('menuBtn');
  const sidebar        = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebarOverlay');
  const sidebarCloseBtn= document.getElementById('sidebarCloseBtn');
  const starterCards   = document.querySelectorAll('.starter-card');
  const recentItems    = document.querySelectorAll('.recent-item');

  let isTyping = false;

  // ── Utilities ────────────────────────────────────────────

  /** Escape HTML to prevent XSS */
  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** Format time as HH:MM */
  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  /** Auto-resize textarea */
  function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
  }

  /** Scroll to bottom of chat */
  function scrollToBottom(behavior = 'smooth') {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior });
  }

  /** Show/hide welcome screen */
  function setWelcomeVisible(visible) {
    welcomeScreen.style.display = visible ? 'flex' : 'none';
  }

  /** Toggle send button state */
  function updateSendBtn() {
    const hasText = userInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isTyping;
  }

  // ── Message Rendering ────────────────────────────────────

  /**
   * Build a message element.
   * @param {'user'|'bot'} role
   * @param {string} text
   * @param {string[]} [sources]
   */
  function createMessageEl(role, text, sources = []) {
    const now = formatTime(new Date());

    const wrapper = document.createElement('div');
    wrapper.className = `message message--${role}`;

    const avatarEl = document.createElement('div');
    avatarEl.className = 'message-avatar';
    avatarEl.textContent = role === 'user' ? 'U' : 'AI';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    const bubbleEl = document.createElement('div');
    bubbleEl.className = 'message-bubble';
    // Allow simple line-break rendering
    bubbleEl.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');

    const metaEl = document.createElement('div');
    metaEl.className = 'message-meta';
    metaEl.textContent = now;

    contentEl.appendChild(bubbleEl);

    // Source tags (bot only)
    if (role === 'bot' && sources.length > 0) {
      const sourcesEl = document.createElement('div');
      sourcesEl.className = 'message-sources';
      sources.forEach(src => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        tag.innerHTML = `
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="3" y1="9" x2="21" y2="9"/>
            <line x1="9" y1="3" x2="9" y2="21"/>
          </svg>
          ${escapeHtml(src)}`;
        sourcesEl.appendChild(tag);
      });
      contentEl.appendChild(sourcesEl);
    }

    contentEl.appendChild(metaEl);
    wrapper.appendChild(avatarEl);
    wrapper.appendChild(contentEl);

    return wrapper;
  }

  /** Append a message to the list */
  function appendMessage(role, text, sources = []) {
    setWelcomeVisible(false);
    const el = createMessageEl(role, text, sources);
    messagesList.appendChild(el);
    scrollToBottom();
  }

  // ── Typing Indicator ─────────────────────────────────────

  let typingEl = null;

  function showTyping() {
    if (typingEl) return;
    typingEl = document.createElement('div');
    typingEl.className = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar message--bot';
    // Reuse bot avatar style
    avatar.style.cssText = `
      width:32px;height:32px;border-radius:50%;display:flex;
      align-items:center;justify-content:center;font-size:13px;
      font-weight:700;flex-shrink:0;margin-top:4px;
      background:linear-gradient(135deg,#6366f1,#a855f7);
      color:#fff;box-shadow:0 0 24px rgba(99,102,241,0.25);
    `;
    avatar.textContent = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('span');
      dot.className = 'typing-dot';
      bubble.appendChild(dot);
    }

    typingEl.appendChild(avatar);
    typingEl.appendChild(bubble);

    // Insert after messages-list in chatMessages
    chatMessages.appendChild(typingEl);
    scrollToBottom();
  }

  function hideTyping() {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  // ── Mock Bot Responses ────────────────────────────────────
  // Simulated responses that showcase a RAG bot's capability.
  const botResponses = [
    {
      keywords: ['health', 'overall', 'summary', 'summarize'],
      text: `Based on the three financial statements for SSI BMC (May 2026), here is an overall summary:\n\n📊 Revenue performance appears solid with consistent income generation across reporting periods.\n\n💼 The balance sheet reflects a manageable debt-to-equity structure, with assets adequately covering liabilities.\n\n💵 Cash flow from operations is positive, suggesting the business is generating real cash from its core activities.\n\nWould you like a deeper dive into any specific statement?`,
      sources: ['Balance Sheet', 'Cash Flow', 'Income Statement'],
    },
    {
      keywords: ['revenue', 'net income', 'income'],
      text: `From the Income Statement (May 2026):\n\n• Total Revenue: pulled from SSI_BMC Income Statement\n• Cost of Goods Sold (COGS): deducted from revenue to arrive at Gross Profit\n• Operating Expenses: general and administrative costs\n• Net Income: bottom-line profit after all deductions and taxes\n\nNote: Exact figures will be extracted once the backend is connected to the document parser. The structure above reflects typical income statement layout found in your uploaded file.`,
      sources: ['Income Statement'],
    },
    {
      keywords: ['cash', 'flow'],
      text: `The Cash Flow Statement (May 2026) for SSI BMC is structured into three sections:\n\n🔵 Operating Activities — cash generated from day-to-day business operations.\n🟡 Investing Activities — capital expenditures, asset purchases/sales.\n🟢 Financing Activities — debt repayments, equity issuances, dividends.\n\nPositive operating cash flow is a healthy sign. Net cash position changes are summarized at the bottom of the statement.`,
      sources: ['Cash Flow'],
    },
    {
      keywords: ['balance', 'assets', 'liabilities', 'equity'],
      text: `From the Balance Sheet (May 2026):\n\n• Total Assets = Current Assets + Non-Current Assets\n• Total Liabilities = Short-term + Long-term obligations\n• Shareholders' Equity = Total Assets − Total Liabilities\n\nThe accounting equation holds: Assets = Liabilities + Equity.\n\nFull numeric values will be parsed once the Excel document is processed by the RAG pipeline.`,
      sources: ['Balance Sheet'],
    },
  ];

  const defaultResponse = {
    text: `I've reviewed the uploaded financial statements for SSI BMC (May 2026). Could you clarify what you'd like to know?\n\nHere are some things I can help with:\n• Revenue and profitability metrics\n• Cash flow analysis\n• Balance sheet breakdown\n• Year-over-year comparisons\n• Financial ratios (liquidity, leverage, profitability)`,
    sources: ['Balance Sheet', 'Cash Flow', 'Income Statement'],
  };

  function getBotResponse(userText) {
    const lower = userText.toLowerCase();
    for (const resp of botResponses) {
      if (resp.keywords.some(k => lower.includes(k))) {
        return resp;
      }
    }
    return defaultResponse;
  }

  // ── Send Logic ────────────────────────────────────────────

  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isTyping) return;

    // Append user message
    appendMessage('user', text);

    // Clear & reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    updateSendBtn();

    // Show typing
    isTyping = true;
    updateSendBtn();
    showTyping();

    // Simulate network delay (0.8 – 2s)
    const delay = 800 + Math.random() * 1200;
    await sleep(delay);

    hideTyping();
    const { text: botText, sources } = getBotResponse(text);
    appendMessage('bot', botText, sources);

    isTyping = false;
    updateSendBtn();
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ── Clear / New Chat ──────────────────────────────────────

  function clearChat() {
    messagesList.innerHTML = '';
    setWelcomeVisible(true);
    scrollToBottom('instant');
  }

  // ── Sidebar (Mobile) ─────────────────────────────────────

  function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
    document.body.style.overflow = '';
  }

  // ── Event Listeners ───────────────────────────────────────

  // Input events
  userInput.addEventListener('input', () => {
    autoResize();
    updateSendBtn();
  });

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener('click', sendMessage);

  // Clear / new chat
  clearChatBtn.addEventListener('click', clearChat);
  newChatBtn.addEventListener('click', clearChat);

  // Mobile sidebar
  menuBtn.addEventListener('click', openSidebar);
  sidebarCloseBtn.addEventListener('click', closeSidebar);
  sidebarOverlay.addEventListener('click', closeSidebar);

  // Starter cards
  starterCards.forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.dataset.prompt;
      if (prompt) {
        userInput.value = prompt;
        autoResize();
        updateSendBtn();
        sendMessage();
      }
    });
  });

  // Recent items
  recentItems.forEach(item => {
    item.addEventListener('click', () => {
      userInput.value = item.textContent.trim();
      autoResize();
      updateSendBtn();
      userInput.focus();
      closeSidebar();
    });
  });

  // ── Init ──────────────────────────────────────────────────
  updateSendBtn();
  setWelcomeVisible(true);

})();
