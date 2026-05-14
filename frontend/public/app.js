const CHARS_PER_PAGE = 1150;
const REQUEST_TIMEOUT_MS = 30000;

const state = {
  books: [],
  personas: [],
  characterCandidates: [],
  activeBook: null,
  activeBookDetail: null,
  activeChapter: 1,
  activeParagraphIndex: null,
  activeChunkId: null,
  activePageIndex: 0,
  assistantMode: "persona",
  personaId: "persona_lu_xun",
  activeCharacterName: "",
  activeCharacterProfile: null,
  personaConversation: [],
  characterConversation: [],
  inlineBubblesByChunk: {},
  sessionId: `sess_${Date.now()}`,
  requestCounter: 0,
  chapterEnteredAt: Date.now(),
  readingProgress: {
    book_id: "",
    chapter_id: 1,
    section_id: "sec-1",
    paragraph_id: "",
    token_offset: 0,
    scroll_offset: 0,
    dwell_seconds: 0,
    updated_at: "",
  },
  selectionContext: {
    book_id: "",
    selection_id: "",
    selected_text: "",
    left_context: "",
    right_context: "",
    anchor: {
      chapter_id: 1,
      section_id: "sec-1",
      paragraph_id: "",
    },
  },
};

async function fetchJSON(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(url, { ...options, signal: options.signal || controller.signal });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`请求超时（${REQUEST_TIMEOUT_MS / 1000}s）`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}

function setPendingState(active, label = "idle") {
  const indicator = document.getElementById("pending-indicator");
  const pendingLabel = document.getElementById("pending-label");
  if (!indicator || !pendingLabel) {
    return;
  }
  pendingLabel.textContent = label;
  indicator.classList.toggle("is-active", active);
}

function setButtonLoading(buttonId, isLoading, loadingText = "") {
  const button = document.getElementById(buttonId);
  if (!button) {
    return;
  }
  if (!button.dataset.defaultText) {
    button.dataset.defaultText = button.textContent;
  }
  button.disabled = isLoading;
  button.textContent = isLoading ? loadingText : button.dataset.defaultText;
}

function nextRequestId(prefix) {
  state.requestCounter += 1;
  return `${prefix}_${state.requestCounter}`;
}

function getPersonaById(personaId) {
  return state.personas.find((persona) => persona.persona_id === personaId) || state.personas[0] || null;
}

function previewText(text, fallback = "暂无内容") {
  return text && text.trim() ? text.trim() : fallback;
}

function getCurrentPassages() {
  if (!state.activeBookDetail) {
    return [];
  }
  return state.activeBookDetail.chapters[String(state.activeChapter)] || state.activeBookDetail.chapters[state.activeChapter] || [];
}

function getFirstReadableChapter() {
  if (!state.activeBookDetail) {
    return 1;
  }
  for (let chapter = 1; chapter <= state.activeBookDetail.chapter_count; chapter += 1) {
    const passages = state.activeBookDetail.chapters[String(chapter)] || state.activeBookDetail.chapters[chapter] || [];
    if (passages.length) {
      return chapter;
    }
  }
  return 1;
}

function getCurrentPages() {
  const passages = getCurrentPassages();
  if (!passages.length) {
    return [];
  }
  const pages = [];
  let current = [];
  let size = 0;
  passages.forEach((passage, index) => {
    const nextSize = (passage.text || "").length + 80;
    if (current.length && size + nextSize > CHARS_PER_PAGE) {
      pages.push(current);
      current = [];
      size = 0;
    }
    current.push({ ...passage, _index: index });
    size += nextSize;
  });
  if (current.length) {
    pages.push(current);
  }
  return pages;
}

function getCurrentPageItems() {
  const pages = getCurrentPages();
  if (!pages.length) {
    return [];
  }
  state.activePageIndex = Math.max(0, Math.min(state.activePageIndex, pages.length - 1));
  return pages[state.activePageIndex];
}

function currentConversation() {
  return state.assistantMode === "persona" ? state.personaConversation : state.characterConversation;
}

function pushConversation(role, content) {
  const target = state.assistantMode === "persona" ? state.personaConversation : state.characterConversation;
  target.push({ role, content });
}

function resetSelection() {
  state.selectionContext = {
    book_id: state.activeBook || "",
    selection_id: "",
    selected_text: "",
    left_context: "",
    right_context: "",
    anchor: {
      chapter_id: state.activeChapter,
      section_id: `sec-${state.activeChapter}`,
      paragraph_id: "",
    },
  };
}

function updateProgressFromSelection(passage) {
  const pages = getCurrentPages();
  const scrollOffset = pages.length ? Number(((state.activePageIndex + 1) / pages.length).toFixed(2)) : 0;
  state.readingProgress = {
    book_id: state.activeBook || "",
    chapter_id: state.activeChapter,
    section_id: `sec-${state.activeChapter}`,
    paragraph_id: String(passage.paragraph_index ?? ""),
    token_offset: passage.text ? passage.text.length : 0,
    scroll_offset: scrollOffset,
    dwell_seconds: Math.max(1, Math.floor((Date.now() - state.chapterEnteredAt) / 1000)),
    updated_at: new Date().toISOString(),
  };
}

function buildSelectionFromPassage(passage, index, passages) {
  const prev = passages[index - 1];
  const next = passages[index + 1];
  state.selectionContext = {
    book_id: state.activeBook || "",
    selection_id: `sel_${passage.chunk_id || index + 1}`,
    selected_text: passage.text,
    left_context: prev ? prev.text : "",
    right_context: next ? next.text : "",
    anchor: {
      chapter_id: state.activeChapter,
      section_id: `sec-${state.activeChapter}`,
      paragraph_id: String(passage.paragraph_index ?? index + 1),
    },
  };
}

function renderPersonaDetails() {
  const persona = getPersonaById(state.personaId);
  if (!persona) {
    return;
  }
  document.getElementById("persona-type-badge").textContent = persona.source_type;
  document.getElementById("persona-name").textContent = persona.name;
  document.getElementById("persona-citation").textContent = persona.citation;
  const traits = document.getElementById("persona-traits");
  traits.innerHTML = "";
  [...persona.style_traits, ...persona.reasoning_style].slice(0, 6).forEach((item) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = item;
    traits.appendChild(pill);
  });
}

function renderCharacterCandidates() {
  const select = document.getElementById("character-select");
  select.innerHTML = "";
  const emptyOption = document.createElement("option");
  emptyOption.value = "";
  emptyOption.textContent = state.characterCandidates.length ? "请选择角色候选" : "暂无角色候选";
  select.appendChild(emptyOption);

  state.characterCandidates.forEach((candidate) => {
    const option = document.createElement("option");
    option.value = candidate.character_name;
    option.textContent = `${candidate.character_name} · ${candidate.mention_count} 次`;
    select.appendChild(option);
  });

  if (state.activeCharacterName) {
    select.value = state.activeCharacterName;
  }
}

function renderCharacterProfile() {
  const container = document.getElementById("character-profile-card");
  container.innerHTML = "";
  if (!state.activeCharacterProfile) {
    container.innerHTML = '<p class="muted">打开书籍后可选择角色候选，或手动输入角色名来生成角色画像。</p>';
    return;
  }

  const profile = state.activeCharacterProfile;
  const traitRow = document.createElement("div");
  traitRow.className = "pill-row";
  (profile.core_traits || []).forEach((trait) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = trait;
    traitRow.appendChild(pill);
  });

  const relationList = document.createElement("ul");
  relationList.className = "plain-list relationship-list";
  (profile.relationships || []).forEach((relation) => {
    const li = document.createElement("li");
    li.textContent = `${relation.target}：${relation.description}`;
    relationList.appendChild(li);
  });

  container.innerHTML = `
    <h4 class="character-name">${profile.character_name}</h4>
    <p class="muted">${profile.summary}</p>
    <p class="label">核心张力</p>
    <p class="signature-tension">${profile.signature_tension || "当前证据尚不足以提炼更强张力。"}</p>
    <p class="label">当前已读范围</p>
    <p class="muted">${profile.current_scope}</p>
    <p class="label">模型</p>
    <p class="muted">${profile.model_name}</p>
  `;
  container.appendChild(traitRow);
  if (relationList.children.length) {
    const heading = document.createElement("p");
    heading.className = "label";
    heading.textContent = "关系网络";
    container.appendChild(heading);
    container.appendChild(relationList);
  }
}

function renderBooks() {
  const list = document.getElementById("book-list");
  list.innerHTML = "";
  document.getElementById("book-count").textContent = `${state.books.length} 本`;
  state.books.forEach((book) => {
    const item = document.createElement("li");
    item.className = "book-item";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `book-button ${state.activeBook === book.book_id ? "is-active" : ""}`;
    button.innerHTML = `
      <span class="book-title">${book.title}</span>
      <span class="book-meta">${book.book_id}</span>
    `;
    button.addEventListener("click", () => openBook(book.book_id));
    item.appendChild(button);
    list.appendChild(item);
  });
}

function renderReaderHeader() {
  if (!state.activeBookDetail) {
    document.getElementById("book-title").textContent = "请选择一本书开始阅读";
    document.getElementById("book-subtitle").textContent = "页面会跟踪章节、段落与翻页位置，并同步构造 reading_progress。";
    document.getElementById("progress-text").textContent = "尚未加载书籍。";
    document.getElementById("hero-chapter").textContent = "-";
    document.getElementById("hero-paragraph").textContent = "-";
    document.getElementById("hero-dwell").textContent = "0s";
    return;
  }
  const pages = getCurrentPages();
  document.getElementById("book-title").textContent = state.activeBookDetail.title;
  document.getElementById("book-subtitle").textContent = `book_id: ${state.activeBookDetail.book_id}，共 ${state.activeBookDetail.chapter_count} 章。`;
  document.getElementById("progress-text").textContent =
    `当前位于第 ${state.readingProgress.chapter_id} 章 / 第 ${state.activePageIndex + 1} 页 / 段落 ${state.readingProgress.paragraph_id || "-"}，全章共 ${pages.length || 0} 页。`;
  document.getElementById("hero-chapter").textContent = `第 ${state.activeChapter} 章`;
  document.getElementById("hero-paragraph").textContent = state.activeParagraphIndex === null ? "-" : `P${state.activeParagraphIndex}`;
  document.getElementById("hero-dwell").textContent = `${state.readingProgress.dwell_seconds || 0}s`;
}

function renderChapterNav() {
  const container = document.getElementById("chapter-nav");
  container.innerHTML = "";
  if (!state.activeBookDetail) {
    container.innerHTML = '<p class="muted">书籍载入后会显示章节目录。</p>';
    return;
  }
  for (let chapter = 1; chapter <= state.activeBookDetail.chapter_count; chapter += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chapter-button ${chapter === state.activeChapter ? "is-active" : ""}`;
    button.textContent = `第 ${chapter} 章`;
    button.addEventListener("click", () => setActiveChapter(chapter));
    container.appendChild(button);
  }
  document.getElementById("toc-progress").textContent = `已读至第 ${state.activeChapter} 章`;
}

function renderChapterSelects() {
  const chapterSelect = document.getElementById("chapter-select");
  const paragraphSelect = document.getElementById("paragraph-jump");
  chapterSelect.innerHTML = "";
  paragraphSelect.innerHTML = "";
  if (!state.activeBookDetail) {
    return;
  }
  for (let chapter = 1; chapter <= state.activeBookDetail.chapter_count; chapter += 1) {
    const option = document.createElement("option");
    option.value = String(chapter);
    option.textContent = `第 ${chapter} 章`;
    chapterSelect.appendChild(option);
  }
  chapterSelect.value = String(state.activeChapter);

  getCurrentPassages().forEach((passage, index) => {
    const option = document.createElement("option");
    const paragraphIndex = passage.paragraph_index ?? index + 1;
    option.value = String(paragraphIndex);
    option.textContent = `段落 ${paragraphIndex}`;
    paragraphSelect.appendChild(option);
  });
  if (state.activeParagraphIndex !== null) {
    paragraphSelect.value = String(state.activeParagraphIndex);
  }
}

function renderSelectionPreview() {
  document.getElementById("highlight-preview").textContent = previewText(
    state.selectionContext.selected_text,
    "点击正文中的段落后，这里会显示当前选中的阅读上下文。"
  );
}

function renderAssistantStatus() {
  const node = document.getElementById("assistant-status");
  if (state.assistantMode === "persona") {
    const persona = getPersonaById(state.personaId);
    node.textContent = persona ? `当前使用 ${persona.name} 作为名家导读。` : "当前使用名家导读模式。";
  } else if (state.activeCharacterProfile) {
    node.textContent = `当前使用角色 ${state.activeCharacterProfile.character_name} 进行陪读。`;
  } else if (state.activeCharacterName) {
    node.textContent = `当前准备生成角色 ${state.activeCharacterName} 的陪读视角。`;
  } else {
    node.textContent = "当前使用角色陪读模式，请先选择或生成一个角色。";
  }
}

function renderAssistantMode() {
  document.getElementById("persona-mode-btn").classList.toggle("mode-chip-active", state.assistantMode === "persona");
  document.getElementById("character-mode-btn").classList.toggle("mode-chip-active", state.assistantMode === "character");
  renderAssistantStatus();
  renderChatHistory();
}

function renderChatHistory() {
  const historyNode = document.getElementById("chat-history");
  historyNode.innerHTML = "";
  const conversation = currentConversation();
  if (!conversation.length) {
    historyNode.innerHTML = '<p class="muted">这里会连续显示聊天记录，新的回答会自动接在后面。</p>';
    return;
  }
  conversation.forEach((turn) => {
    const item = document.createElement("article");
    item.className = `chat-message chat-message-${turn.role}`;
    const role = turn.role === "user" ? "你" : state.assistantMode === "persona"
      ? getPersonaById(state.personaId)?.name || "导读 agent"
      : state.activeCharacterProfile?.character_name || state.activeCharacterName || "角色 agent";
    item.innerHTML = `
      <div class="chat-role">${role}</div>
      <div class="chat-content">${turn.content.replace(/\n/g, "<br />")}</div>
    `;
    historyNode.appendChild(item);
  });
  historyNode.scrollTop = historyNode.scrollHeight;
}

function updatePageIndicator() {
  const pages = getCurrentPages();
  const total = pages.length;
  const current = total ? state.activePageIndex + 1 : 0;
  document.getElementById("page-indicator").textContent = total ? `${current} / ${total}` : "- / -";
  document.getElementById("prev-page-btn").disabled = current <= 1;
  document.getElementById("next-page-btn").disabled = total === 0 || current >= total;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function createInlineBubbleMarkup(text, chunkId) {
  const bubbles = (state.inlineBubblesByChunk[chunkId] || [])
    .map((bubble) => ({ ...bubble, index: text.indexOf(bubble.anchor_text) }))
    .filter((bubble) => bubble.index >= 0)
    .sort((left, right) => left.index - right.index);

  if (!bubbles.length) {
    return escapeHtml(text);
  }

  let cursor = 0;
  let markup = "";
  bubbles.forEach((bubble) => {
    const start = text.indexOf(bubble.anchor_text, cursor);
    if (start < cursor || start < 0) {
      return;
    }
    const end = start + bubble.anchor_text.length;
    markup += escapeHtml(text.slice(cursor, start));
    markup += `
      <span class="inline-bubble" data-bubble-id="${bubble.bubble_id}">
        <button
          class="inline-bubble-anchor"
          type="button"
          data-bubble-id="${bubble.bubble_id}"
          aria-label="${escapeHtml(bubble.label)}"
        >${escapeHtml(bubble.anchor_text)}</button>
        <span class="inline-bubble-tip" data-bubble-id="${bubble.bubble_id}">
          <strong>${escapeHtml(bubble.label)}</strong>${escapeHtml(bubble.comment)}
        </span>
      </span>
    `;
    cursor = end;
  });
  markup += escapeHtml(text.slice(cursor));
  return markup;
}

function wireInlineBubbleToggles() {
  document.querySelectorAll(".inline-bubble-anchor").forEach((node) => {
    node.addEventListener("click", (event) => {
      const bubbleId = event.currentTarget.dataset.bubbleId;
      document.querySelectorAll(".inline-bubble-tip.is-open").forEach((tip) => {
        if (tip.dataset.bubbleId !== bubbleId) {
          tip.classList.remove("is-open");
        }
      });
      const target = document.querySelector(`.inline-bubble-tip[data-bubble-id="${bubbleId}"]`);
      if (target) {
        target.classList.toggle("is-open");
      }
      event.stopPropagation();
    });
  });
}

function renderPassages() {
  const container = document.getElementById("passage-list");
  const pageItems = getCurrentPageItems();
  container.innerHTML = "";
  updatePageIndicator();

  if (!pageItems.length) {
    container.innerHTML = '<p class="muted">当前章节还没有可显示的内容。</p>';
    return;
  }

  const page = document.createElement("article");
  page.className = "reading-page";
  page.innerHTML = `
    <header class="reading-page-header">
      <span>第 ${state.activeChapter} 章</span>
      <span>第 ${state.activePageIndex + 1} 页</span>
    </header>
  `;

  pageItems.forEach((passage, index) => {
    const paragraphIndex = passage.paragraph_index ?? passage._index + 1;
    const wrapper = document.createElement("article");
    wrapper.className = `reading-paragraph ${paragraphIndex === state.activeParagraphIndex ? "is-selected" : ""}`;
    wrapper.dataset.paragraphIndex = String(paragraphIndex);
    wrapper.innerHTML = `
      <span class="paragraph-marker">${paragraphIndex}</span>
      <div class="reading-paragraph-text">${createInlineBubbleMarkup(passage.text, passage.chunk_id)}</div>
    `;
    wrapper.addEventListener("click", () => selectPassage(passage, index, pageItems));
    page.appendChild(wrapper);
  });

  container.appendChild(page);
  wireInlineBubbleToggles();
}

function selectPassage(passage, index, passages) {
  state.activeParagraphIndex = passage.paragraph_index ?? passage._index + 1;
  state.activeChunkId = passage.chunk_id || null;
  updateProgressFromSelection(passage);
  buildSelectionFromPassage(passage, index, passages);
  renderSelectionPreview();
  renderReaderHeader();
  renderPassages();
}

function setPage(pageIndex) {
  const pages = getCurrentPages();
  if (!pages.length) {
    state.activePageIndex = 0;
    renderPassages();
    return;
  }
  state.activePageIndex = Math.max(0, Math.min(pageIndex, pages.length - 1));
  const currentItems = getCurrentPageItems();
  const firstVisible = currentItems[0];
  if (firstVisible) {
    state.activeParagraphIndex = firstVisible.paragraph_index ?? firstVisible._index + 1;
    updateProgressFromSelection(firstVisible);
  }
  renderReaderHeader();
  renderPassages();
  fetchInlineBubbles().catch((error) => {
    console.error(error);
  });
}

async function fetchInlineBubbles() {
  if (!state.activeBook) {
    return;
  }
  const pageItems = getCurrentPageItems();
  if (!pageItems.length) {
    state.inlineBubblesByChunk = {};
    renderPassages();
    return;
  }
  const payload = {
    book_id: state.activeBook,
    current_chapter: state.activeChapter,
    visible_chunk_ids: pageItems.map((item) => item.chunk_id),
    persona_id: state.personaId,
    assistant_mode: state.assistantMode,
    character_name: state.activeCharacterName,
    max_bubbles: 3,
  };
  try {
    const bubbles = await fetchJSON(`/api/books/${state.activeBook}/inline-bubbles`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const map = {};
    bubbles.forEach((bubble) => {
      if (!map[bubble.chunk_id]) {
        map[bubble.chunk_id] = [];
      }
      map[bubble.chunk_id].push(bubble);
    });
    state.inlineBubblesByChunk = map;
    renderPassages();
  } catch (error) {
    console.error("bubble generation failed", error);
  }
}

async function loadCharacterCandidates() {
  if (!state.activeBook) {
    state.characterCandidates = [];
    renderCharacterCandidates();
    return;
  }
  setButtonLoading("character-generate-btn", true, "加载候选中...");
  try {
    state.characterCandidates = await fetchJSON(
      `/api/books/${state.activeBook}/characters?current_chapter=${state.activeChapter}&limit=12`
    );
    renderCharacterCandidates();
  } catch (error) {
    state.characterCandidates = [];
    renderCharacterCandidates();
    document.getElementById("character-profile-card").innerHTML = `<p class="muted">角色候选加载失败：${error.message}</p>`;
  } finally {
    setButtonLoading("character-generate-btn", false);
  }
}

async function generateCharacterProfile() {
  if (!state.activeBook) {
    return;
  }
  const typedName = document.getElementById("character-input").value.trim();
  const selectedName = document.getElementById("character-select").value.trim();
  const characterName = typedName || selectedName;
  if (!characterName) {
    document.getElementById("character-profile-card").innerHTML = "<p class=\"muted\">请先选择或输入一个角色名。</p>";
    return;
  }
  state.activeCharacterName = characterName;
  renderAssistantStatus();
  setButtonLoading("character-generate-btn", true, "生成画像中...");
  setPendingState(true, "正在生成角色画像");
  try {
    const profile = await fetchJSON(`/api/books/${state.activeBook}/characters/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        book_id: state.activeBook,
        character_name: characterName,
        current_chapter: state.activeChapter,
      }),
    });
    state.activeCharacterProfile = profile;
    document.getElementById("character-input").value = profile.character_name;
    renderCharacterProfile();
    renderAssistantStatus();
    if (state.assistantMode === "character") {
      await fetchInlineBubbles();
    }
  } catch (error) {
    document.getElementById("character-profile-card").innerHTML = `<p class="muted">角色画像生成失败：${error.message}</p>`;
  } finally {
    setPendingState(false, "idle");
    setButtonLoading("character-generate-btn", false);
  }
}

async function setActiveChapter(chapter) {
  state.activeChapter = Number(chapter);
  state.activePageIndex = 0;
  state.chapterEnteredAt = Date.now();
  state.activeChunkId = null;
  state.activeCharacterProfile = null;
  resetSelection();
  const passages = getCurrentPassages();
  const first = passages[0] || null;
  state.activeParagraphIndex = first ? first.paragraph_index ?? 1 : null;
  state.readingProgress = {
    book_id: state.activeBook || "",
    chapter_id: state.activeChapter,
    section_id: `sec-${state.activeChapter}`,
    paragraph_id: first ? String(first.paragraph_index ?? 1) : "",
    token_offset: first?.text ? first.text.length : 0,
    scroll_offset: 0,
    dwell_seconds: 0,
    updated_at: new Date().toISOString(),
  };
  renderChapterNav();
  renderChapterSelects();
  renderReaderHeader();
  renderSelectionPreview();
  renderPassages();
  renderCharacterProfile();
  await loadCharacterCandidates();
  await fetchInlineBubbles();
}

async function loadPersonas() {
  state.personas = await fetchJSON("/api/personas");
  const select = document.getElementById("persona-select");
  select.innerHTML = state.personas.map((persona) => `<option value="${persona.persona_id}">${persona.name}</option>`).join("");
  const preferred =
    state.personas.find((persona) => persona.persona_id === state.personaId) ||
    state.personas.find((persona) => persona.persona_id !== "neutral") ||
    state.personas[0] ||
    null;
  if (preferred) {
    state.personaId = preferred.persona_id;
    select.value = state.personaId;
  }
  select.addEventListener("change", async (event) => {
    state.personaId = event.target.value;
    renderPersonaDetails();
    renderAssistantStatus();
    if (state.assistantMode === "persona") {
      await fetchInlineBubbles();
    }
  });
  renderPersonaDetails();
}

async function loadBooks() {
  state.books = await fetchJSON("/api/books");
  renderBooks();
}

async function openBook(bookId) {
  state.activeBook = bookId;
  state.activeBookDetail = await fetchJSON(`/api/books/${bookId}`);
  state.personaConversation = [];
  state.characterConversation = [];
  state.activeCharacterName = "";
  state.activeCharacterProfile = null;
  renderBooks();
  renderChatHistory();
  await setActiveChapter(getFirstReadableChapter());
}

async function uploadBook(event) {
  event.preventDefault();
  const input = document.getElementById("file-input");
  if (!input.files[0]) {
    return;
  }
  setPendingState(true, "正在导入书籍");
  try {
    const payload = new FormData();
    payload.append("file", input.files[0]);
    const uploaded = await fetchJSON("/api/upload", { method: "POST", body: payload });
    await loadBooks();
    await openBook(uploaded.book_id);
    input.value = "";
  } catch (error) {
    pushConversation("assistant", `导入失败：${error.message}`);
    renderChatHistory();
  } finally {
    setPendingState(false, "idle");
  }
}

function renderComposerQuestion(text = "") {
  document.getElementById("question-input").value = text;
}

async function askAssistant() {
  if (!state.activeBook) {
    return;
  }
  const question = document.getElementById("question-input").value.trim();
  if (!question) {
    return;
  }

  const history = currentConversation().slice(-8);
  pushConversation("user", question);
  renderChatHistory();
  renderComposerQuestion("");
  setButtonLoading("ask-btn", true, "发送中...");
  setPendingState(true, state.assistantMode === "persona" ? "名家导读正在回答" : "角色陪读正在回答");

  try {
    let answer = "";
    if (state.assistantMode === "persona") {
      const response = await fetchJSON("/api/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          book_id: state.activeBook,
          question,
          highlight_text: state.selectionContext.selected_text,
          current_chapter: state.activeChapter,
          persona_id: state.personaId,
          conversation_history: history,
        }),
      });
      answer = response.answer;
    } else {
      if (!state.activeCharacterName) {
        throw new Error("请先生成一个角色画像，再进入角色陪读对话。");
      }
      const response = await fetchJSON(`/api/books/${state.activeBook}/characters/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          book_id: state.activeBook,
          character_name: state.activeCharacterName,
          question,
          current_chapter: state.activeChapter,
          conversation_history: history,
        }),
      });
      answer = response.answer;
      state.activeCharacterProfile = response.profile;
      renderCharacterProfile();
    }
    pushConversation("assistant", answer);
    renderChatHistory();
  } catch (error) {
    pushConversation("assistant", `当前请求失败：${error.message}`);
    renderChatHistory();
  } finally {
    setButtonLoading("ask-btn", false);
    setPendingState(false, "idle");
  }
}

async function summarizeChapter() {
  if (!state.activeBook) {
    return;
  }
  setButtonLoading("summary-btn", true, "总结生成中...");
  setPendingState(true, "正在生成章节总结");
  try {
    const response = await fetchJSON("/api/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        book_id: state.activeBook,
        current_chapter: state.activeChapter,
        persona_id: state.personaId,
      }),
    });
    state.assistantMode = "persona";
    renderAssistantMode();
    pushConversation("assistant", response.summary);
    renderChatHistory();
  } catch (error) {
    pushConversation("assistant", `章节总结失败：${error.message}`);
    renderChatHistory();
  } finally {
    setButtonLoading("summary-btn", false);
    setPendingState(false, "idle");
  }
}

function clearConversation() {
  if (state.assistantMode === "persona") {
    state.personaConversation = [];
  } else {
    state.characterConversation = [];
  }
  renderChatHistory();
}

function setAssistantMode(mode) {
  state.assistantMode = mode;
  renderAssistantMode();
  fetchInlineBubbles().catch((error) => console.error(error));
}

function wireEvents() {
  document.getElementById("upload-form").addEventListener("submit", uploadBook);
  document.getElementById("ask-btn").addEventListener("click", askAssistant);
  document.getElementById("summary-btn").addEventListener("click", summarizeChapter);
  document.getElementById("clear-chat-btn").addEventListener("click", clearConversation);
  document.getElementById("persona-mode-btn").addEventListener("click", () => setAssistantMode("persona"));
  document.getElementById("character-mode-btn").addEventListener("click", () => setAssistantMode("character"));
  document.getElementById("character-select").addEventListener("change", (event) => {
    state.activeCharacterName = event.target.value.trim();
    document.getElementById("character-input").value = state.activeCharacterName;
    renderAssistantStatus();
  });
  document.getElementById("character-generate-btn").addEventListener("click", generateCharacterProfile);
  document.getElementById("chapter-select").addEventListener("change", async (event) => {
    await setActiveChapter(Number(event.target.value));
  });
  document.getElementById("paragraph-jump").addEventListener("change", (event) => {
    const targetValue = event.target.value;
    const pageItems = getCurrentPageItems();
    const passage = getCurrentPassages().find(
      (item, index) => String(item.paragraph_index ?? index + 1) === targetValue
    );
    if (!passage) {
      return;
    }
    const pageIndex = getCurrentPages().findIndex((page) =>
      page.some((item) => String(item.paragraph_index ?? item._index + 1) === targetValue)
    );
    if (pageIndex >= 0) {
      state.activePageIndex = pageIndex;
    }
    selectPassage(passage, 0, pageItems.length ? pageItems : getCurrentPassages());
  });
  document.getElementById("prev-page-btn").addEventListener("click", () => setPage(state.activePageIndex - 1));
  document.getElementById("next-page-btn").addEventListener("click", () => setPage(state.activePageIndex + 1));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".inline-bubble")) {
      document.querySelectorAll(".inline-bubble-tip.is-open").forEach((tip) => tip.classList.remove("is-open"));
    }
  });
}

async function bootstrap() {
  wireEvents();
  renderReaderHeader();
  renderSelectionPreview();
  renderCharacterProfile();
  renderAssistantMode();
  await loadPersonas();
  await loadBooks();
  if (state.books[0]) {
    await openBook(state.books[0].book_id);
  }
}

bootstrap().catch((error) => {
  console.error(error);
});
