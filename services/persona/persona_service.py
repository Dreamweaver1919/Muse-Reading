from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.config import PERSONA_KB_DIR, ROOT_DIR
from backend.models import (
    ChatMessage,
    PersonaAgentConfig,
    PersonaAgentStatus,
    PersonaCatalogSummary,
    PersonaKnowledgeBundle,
    PersonaProfile,
    PersonaPromptPreview,
    PersonaPromptPreviewRequest,
    PersonaPromptTraits,
    PersonaRAGHit,
    PersonaRAGQueryRequest,
)
from services.persona.model_client import invoke_openai_compatible_chat, invoke_openai_compatible_messages


class PersonaAgentConfigurationError(RuntimeError):
    """Raised when a persona agent is missing runtime configuration."""


class PersonaAgentInvocationError(RuntimeError):
    """Raised when a persona agent call fails upstream."""


def _default_prompt_traits() -> PersonaPromptTraits:
    return PersonaPromptTraits(
        system_role="中文阅读陪伴助手",
        opening_instruction="先基于用户当前可见正文回答，再用稳定、清晰、克制的中文组织表达。",
        tone_keywords=["清晰", "克制", "贴近文本"],
        reasoning_steps=[
            "先确认当前可见文本里发生了什么。",
            "再解释这段文字最重要的关系、情绪或主题。",
            "避免跳到未来剧情，也不要脱离文本空谈。",
        ],
        forbidden_patterns=[
            "编造未出现在当前可见范围的信息",
            "直接剧透后文情节",
            "只模仿名家口气却不给出文本依据",
        ],
        response_policies=[
            "回答必须引用当前可见正文，不得越过阅读进度。",
            "如果证据不足，要明确说明当前文本还不足以支持更强判断。",
            "名家风格只能改变解读角度，不能改变事实边界。",
        ],
    )


AGENT_CONFIGS: dict[str, PersonaAgentConfig] = {
    "neutral": PersonaAgentConfig(
        agent_id="neutral",
        persona_id="neutral",
        display_name="中性导读",
        language="zh-CN",
        api_key_env_var="MUSE_NEUTRAL_API_KEY",
        base_url_env_var="MUSE_NEUTRAL_BASE_URL",
        model_name_env_var="MUSE_NEUTRAL_MODEL_NAME",
        default_model_name="",
        prompt_traits=_default_prompt_traits(),
    ),
    "lu-xun": PersonaAgentConfig(
        agent_id="lu-xun",
        persona_id="persona_lu_xun",
        display_name="鲁迅",
        language="zh-CN",
        api_key_env_var="LU_XUN_API_KEY",
        base_url_env_var="LU_XUN_BASE_URL",
        model_name_env_var="LU_XUN_MODEL_NAME",
        default_model_name="",
        persona_pack_path="data/processed/personas/persona_lu_xun__v002.json",
        catalog_path="data/raw/persona_sources/catalog_lu_xun__v001.json",
        prompt_traits=PersonaPromptTraits(
            system_role="鲁迅风格的中文文学导读者",
            opening_instruction="像一位冷静、锋利、克制的中文导读者那样说话，从细节入手，指出麻木、压力、习惯背后的结构问题。",
            tone_keywords=["锋利", "克制", "诊断式", "不粉饰"],
            reasoning_steps=[
                "先抓住当前段落里最具体的动作、意象或措辞。",
                "再指出它折射出的心理麻木、关系压力或社会规训。",
                "如果文本证据不足，就停在当前可见范围，不夸大结论。",
            ],
            forbidden_patterns=[
                "空洞赞美",
                "脱离文本的宏大说教",
                "越过阅读进度推断未来情节",
            ],
            response_policies=[
                "保持中文表达，不模仿文言，不堆砌名句。",
                "结论要比情绪更靠近证据。",
                "即使语气锋利，也不能捏造书里没有的信息。",
            ],
        ),
    ),
    "mark-twain": PersonaAgentConfig(
        agent_id="mark-twain",
        persona_id="persona_mark_twain",
        display_name="马克·吐温",
        language="zh-CN",
        api_key_env_var="MARK_TWAIN_API_KEY",
        base_url_env_var="MARK_TWAIN_BASE_URL",
        model_name_env_var="MARK_TWAIN_MODEL_NAME",
        default_model_name="",
        persona_pack_path="data/processed/personas/persona_mark_twain__v001.json",
        catalog_path="data/raw/persona_sources/catalog_mark_twain__v001.json",
        prompt_traits=PersonaPromptTraits(
            system_role="马克·吐温风格的中文文学导读者",
            opening_instruction="用中文表达机智、讽刺和人情观察，但始终把玩笑建立在文本证据上，而不是空转。",
            tone_keywords=["机智", "讽刺", "贴近人情", "不失分寸"],
            reasoning_steps=[
                "先解释当前段落发生了什么。",
                "再指出其中可笑、可疑或自相矛盾的地方。",
                "最后落回人物处境和文本意义，而不是只抖机灵。",
            ],
            forbidden_patterns=[
                "把阅读回答写成段子",
                "用讽刺掩盖缺乏证据",
                "借风格之名扩写未来剧情",
            ],
            response_policies=[
                "语气可以俏皮，但判断必须扎根在当前可见文本里。",
                "优先揭示人物的虚伪、天真或社会惯性。",
                "如果问题超出已读范围，要直说并收回到当下段落。",
            ],
        ),
    ),
    "zhang-ailing": PersonaAgentConfig(
        agent_id="zhang-ailing",
        persona_id="persona_zhang_ailing",
        display_name="张爱玲",
        language="zh-CN",
        api_key_env_var="ZHANG_AILING_API_KEY",
        base_url_env_var="ZHANG_AILING_BASE_URL",
        model_name_env_var="ZHANG_AILING_MODEL_NAME",
        default_model_name="",
        persona_pack_path="data/processed/personas/persona_zhang_ailing__v001.json",
        catalog_path="data/raw/persona_sources/catalog_zhang_ailing__v001.json",
        prompt_traits=PersonaPromptTraits(
            system_role="张爱玲风格的中文文学导读者",
            opening_instruction="用细密、冷静、克制的中文观察人物关系，从衣着、动作、停顿、空气感这些小地方读出情感和权力。",
            tone_keywords=["细密", "冷静", "克制", "关系敏感"],
            reasoning_steps=[
                "先看当前段落里的细节和气氛。",
                "再解释人物关系里隐含的欲望、体面或压迫。",
                "保持判断轻一点、准一点，不把后文命运提前带进来。",
            ],
            forbidden_patterns=[
                "过度抒情",
                "只谈氛围不谈证据",
                "把未发生的结局当成已经显露的事实",
            ],
            response_policies=[
                "优先分析关系、姿态、气氛，而不是抽象空话。",
                "允许有一点冷意，但不做华丽堆砌。",
                "所有解读都必须能回到当前可见文本的具体细节。",
            ],
        ),
    ),
}

PERSONA_ALIASES = {
    "neutral": "neutral",
    "lu-xun": "lu-xun",
    "persona_lu_xun": "lu-xun",
    "mark-twain": "mark-twain",
    "persona_mark_twain": "mark-twain",
    "eileen-chang": "zhang-ailing",
    "zhang-ailing": "zhang-ailing",
    "persona_zhang_ailing": "zhang-ailing",
}


def _resolve_repo_path(relative_path: str) -> Path | None:
    if not relative_path:
        return None
    return ROOT_DIR / relative_path


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_catalog_summary(catalog: dict[str, Any]) -> PersonaCatalogSummary:
    counts = catalog.get("counts", {})
    return PersonaCatalogSummary(
        total_sources=int(counts.get("total_sources", 0)),
        works=int(counts.get("works", 0)),
        voice_sources=int(counts.get("voice_sources", 0)),
        biography_and_critical=int(counts.get("biography_and_critical", 0)),
    )


def _source_type_from_pack(pack: dict[str, Any]) -> str:
    persona_type = pack.get("persona_type")
    if persona_type == "author":
        return "literary_master"
    if persona_type == "character":
        return "book_character"
    return "neutral"


def _citation_from_pack(pack: dict[str, Any], config: PersonaAgentConfig) -> str:
    source_layer = pack.get("source_layer", [])
    citations = [item.get("citation", "") for item in source_layer[:3] if item.get("citation")]
    if citations:
        return " | ".join(citations)
    if config.persona_pack_path:
        return f"Persona pack: {config.persona_pack_path}"
    return "Project MVP default persona"


def _profile_from_bundle(config: PersonaAgentConfig, pack: dict[str, Any]) -> PersonaProfile:
    if not pack:
        return PersonaProfile(
            persona_id=config.persona_id,
            name=config.display_name,
            source_type="neutral",
            style_traits=config.prompt_traits.tone_keywords,
            reasoning_style=config.prompt_traits.reasoning_steps,
            citation="Project MVP default persona",
            prompt_scaffold=config.prompt_traits.response_policies,
        )

    style_layer = pack.get("style_layer", {})
    return PersonaProfile(
        persona_id=pack.get("persona_id", config.persona_id),
        name=config.display_name,
        source_type=_source_type_from_pack(pack),
        style_traits=style_layer.get("tone_keywords", []) or config.prompt_traits.tone_keywords,
        reasoning_style=style_layer.get("reasoning_steps", []) or config.prompt_traits.reasoning_steps,
        citation=_citation_from_pack(pack, config),
        prompt_scaffold=config.prompt_traits.response_policies,
    )


def _canonical_agent_id(persona_id: str) -> str:
    return PERSONA_ALIASES.get(persona_id, "neutral")


def _tokenize(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", value.lower()) if token]


def _score_snippet(query: str, row: dict[str, Any]) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    retrieval_text = str(row.get("retrieval_text") or row.get("text") or "").lower()
    title = str(row.get("title") or "").lower()
    matched = 0.0
    for token in query_tokens:
        if token in retrieval_text:
            matched += 1.0
        elif token in title:
            matched += 0.5
    weight = float(row.get("retrieval_weight", 1.0))
    return matched * weight


def _format_categories(categories: list[str]) -> str:
    if not categories:
        return "works / voice_sources / biography_and_critical / persona_pack"
    return ", ".join(categories)


@lru_cache(maxsize=1)
def _load_knowledge_bundles() -> dict[str, PersonaKnowledgeBundle]:
    bundles: dict[str, PersonaKnowledgeBundle] = {}
    for agent_id, config in AGENT_CONFIGS.items():
        persona_pack = _read_json(_resolve_repo_path(config.persona_pack_path))
        catalog = _read_json(_resolve_repo_path(config.catalog_path))
        bundles[agent_id] = PersonaKnowledgeBundle(
            config=config,
            profile=_profile_from_bundle(config, persona_pack),
            catalog_summary=_coerce_catalog_summary(catalog),
            persona_pack=persona_pack,
            catalog=catalog,
        )
    return bundles


@lru_cache(maxsize=1)
def _load_persona_kb_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for bundle in _load_knowledge_bundles().values():
        persona_id = bundle.config.persona_id
        kb_dir = PERSONA_KB_DIR / persona_id
        manifest = _read_json(kb_dir / "manifest.json")
        snippets = _read_jsonl(kb_dir / "retrieval_snippets.jsonl")
        documents = _read_jsonl(kb_dir / "documents.jsonl")
        index[persona_id] = {
            "manifest": manifest,
            "snippets": snippets,
            "documents": documents,
            "kb_dir": str(kb_dir),
        }
    return index


def list_personas() -> list[PersonaProfile]:
    return [bundle.profile for bundle in _load_knowledge_bundles().values()]


def list_persona_agents() -> list[PersonaAgentStatus]:
    statuses: list[PersonaAgentStatus] = []
    for agent_id, bundle in _load_knowledge_bundles().items():
        config = bundle.config
        resolved_base_url = os.getenv(config.base_url_env_var, config.default_base_url)
        resolved_model_name = os.getenv(config.model_name_env_var, config.default_model_name)
        statuses.append(
            PersonaAgentStatus(
                agent_id=agent_id,
                persona_id=config.persona_id,
                display_name=config.display_name,
                language=config.language,
                api_key_env_var=config.api_key_env_var,
                base_url_env_var=config.base_url_env_var,
                model_name_env_var=config.model_name_env_var,
                resolved_base_url=resolved_base_url,
                resolved_model_name=resolved_model_name,
                has_api_key=bool(os.getenv(config.api_key_env_var)),
                persona_pack_path=config.persona_pack_path,
                catalog_path=config.catalog_path,
                catalog_summary=bundle.catalog_summary,
                prompt_traits=config.prompt_traits,
            )
        )
    return statuses


def get_persona(persona_id: str) -> PersonaProfile:
    return _load_knowledge_bundles()[_canonical_agent_id(persona_id)].profile


def get_persona_agent(persona_id: str) -> PersonaAgentStatus:
    agent_id = _canonical_agent_id(persona_id)
    for status in list_persona_agents():
        if status.agent_id == agent_id:
            return status
    return list_persona_agents()[0]


def get_persona_knowledge_bundle(persona_id: str) -> PersonaKnowledgeBundle:
    return _load_knowledge_bundles()[_canonical_agent_id(persona_id)]


def get_persona_kb_manifest(persona_id: str) -> dict[str, Any]:
    bundle = get_persona_knowledge_bundle(persona_id)
    return _load_persona_kb_index().get(bundle.config.persona_id, {}).get("manifest", {})


def ensure_persona_assets(persona_id: str) -> None:
    bundle = get_persona_knowledge_bundle(persona_id)
    if bundle.config.agent_id == "neutral":
        return
    kb = _load_persona_kb_index().get(bundle.config.persona_id, {})
    missing = []
    if not bundle.persona_pack:
        missing.append("persona_pack")
    if not bundle.catalog:
        missing.append("catalog")
    if not kb.get("manifest"):
        missing.append("kb_manifest")
    if not kb.get("snippets"):
        missing.append("kb_snippets")
    if missing:
        missing_joined = ", ".join(missing)
        raise PersonaAgentConfigurationError(
            f"persona agent `{bundle.config.agent_id}` is missing required local assets: {missing_joined}"
        )


def retrieve_persona_snippets(persona_id: str, request: PersonaRAGQueryRequest) -> list[PersonaRAGHit]:
    bundle = get_persona_knowledge_bundle(persona_id)
    kb = _load_persona_kb_index().get(bundle.config.persona_id, {})
    rows = list(kb.get("snippets", []))
    categories = set(request.categories)
    if categories:
        rows = [row for row in rows if row.get("source_category") in categories]

    scored: list[PersonaRAGHit] = []
    for row in rows:
        score = _score_snippet(request.query, row)
        if score <= 0:
            continue
        scored.append(
            PersonaRAGHit(
                snippet_id=str(row.get("snippet_id", "")),
                title=str(row.get("title", "")),
                source_category=str(row.get("source_category", "")),
                snippet_type=str(row.get("snippet_type", "")),
                text=str(row.get("text", "")),
                score=score,
                retrieval_weight=float(row.get("retrieval_weight", 1.0)),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: request.top_k]


def resolve_persona_runtime(persona_id: str) -> tuple[PersonaAgentConfig, str, str, str]:
    bundle = get_persona_knowledge_bundle(persona_id)
    config = bundle.config
    api_key = os.getenv(config.api_key_env_var, "").strip()
    base_url = os.getenv(config.base_url_env_var, config.default_base_url).strip()
    model_name = os.getenv(config.model_name_env_var, config.default_model_name).strip()

    missing = []
    if not api_key:
        missing.append(config.api_key_env_var)
    if not base_url:
        missing.append(config.base_url_env_var)
    if not model_name:
        missing.append(config.model_name_env_var)
    if missing:
        missing_joined = ", ".join(missing)
        raise PersonaAgentConfigurationError(
            f"persona agent `{config.agent_id}` is not fully configured. Missing: {missing_joined}"
        )
    return config, api_key, base_url, model_name


def build_persona_system_prompt(persona_id: str, task: str) -> str:
    bundle = get_persona_knowledge_bundle(persona_id)
    traits = bundle.config.prompt_traits
    profile = bundle.profile
    pack = bundle.persona_pack
    constraints = pack.get("constraints", {})
    max_response_length = constraints.get("max_response_length", 320)
    representative_views = pack.get("fact_layer", {}).get("representative_views", [])
    core_positions = pack.get("stance_layer", {}).get("core_positions", [])

    sections = [
        f"你是 {bundle.config.display_name} 风格的中文阅读陪伴 agent。",
        f"角色定位：{traits.system_role}",
        f"开场原则：{traits.opening_instruction}",
        f"语言要求：全程使用中文，语气特征包括 {', '.join(profile.style_traits or traits.tone_keywords)}。",
        "推理步骤：",
        *[f"{index}. {step}" for index, step in enumerate(profile.reasoning_style or traits.reasoning_steps, start=1)],
        "禁止事项：",
        *[f"- {item}" for item in traits.forbidden_patterns],
        "回答策略：",
        *[f"- {item}" for item in traits.response_policies],
        "事实边界：只能使用用户当前可见的正文上下文和提供的人设资料，不能泄露未来剧情，不能编造文本中不存在的事实。",
        f"输出长度：默认控制在 {max_response_length} 字左右，必要时更短。",
    ]
    if representative_views:
        sections.extend(["代表性看法：", *[f"- {item}" for item in representative_views[:3]]])
    if core_positions:
        sections.extend(["解读立场：", *[f"- {item}" for item in core_positions[:3]]])
    if task == "summary":
        sections.append("当前任务是章节总结，请把重点放在已读章节中的人物关系、情绪走向、叙事变化和可见主题。")
    else:
        sections.append("当前任务是高亮问答，请直接回应用户问题，并显式依赖给定正文证据。")
    return "\n".join(sections)


def build_persona_prompt_preview(persona_id: str, request: PersonaPromptPreviewRequest) -> PersonaPromptPreview:
    bundle = get_persona_knowledge_bundle(persona_id)
    status = get_persona_agent(persona_id)
    query = " ".join(part for part in [request.question, request.book_context] if part.strip())
    hits = retrieve_persona_snippets(
        persona_id,
        PersonaRAGQueryRequest(query=query, top_k=request.top_k, categories=request.categories),
    )
    persona_context = "\n".join(
        [f"- {hit.source_category}/{hit.snippet_type}: {hit.text}" for hit in hits]
    )
    return PersonaPromptPreview(
        persona_id=bundle.config.persona_id,
        display_name=status.display_name,
        model_name=status.resolved_model_name,
        base_url=status.resolved_base_url,
        has_api_key=status.has_api_key,
        system_prompt=build_persona_system_prompt(persona_id, task="qa"),
        persona_context=persona_context,
        retrieved_hits=hits,
    )


def build_persona_user_prompt(
    *,
    persona_id: str,
    task: str,
    book_title: str,
    question: str,
    visible_contexts: list[str],
    current_chapter: int,
    highlight_text: str = "",
    persona_hits: list[PersonaRAGHit] | None = None,
    conversation_history: list[ChatMessage] | None = None,
) -> str:
    persona_hits = persona_hits or []
    conversation_history = conversation_history or []
    context_block = "\n\n".join(
        [f"[正文证据 {index}]\n{text}" for index, text in enumerate(visible_contexts, start=1)]
    )
    persona_block = "\n".join(
        [
            f"- {hit.source_category}/{hit.snippet_type} | {hit.title}: {hit.text}"
            for hit in persona_hits
        ]
    )
    task_instruction = (
        "请基于这些证据写一段章节总结。"
        if task == "summary"
        else "请直接回答用户问题。"
    )
    pieces = [
        f"书名：{book_title}",
        f"当前已读章节上限：第 {current_chapter} 章",
        f"任务类型：{task}",
        task_instruction,
    ]
    if question.strip():
        pieces.append(f"用户问题：{question.strip()}")
    if highlight_text.strip():
        pieces.append(f"用户高亮：{highlight_text.strip()}")
    if conversation_history:
        history_block = "\n".join(
            [f"{'用户' if turn.role == 'user' else '助手'}：{turn.content}" for turn in conversation_history[-6:]]
        )
        pieces.extend(["已有对话记录：", history_block])
    pieces.extend(
        [
            "当前可见正文：",
            context_block or "无",
            f"可引用的人设资料类别：{_format_categories([hit.source_category for hit in persona_hits])}",
            "相关名家资料：",
            persona_block or "无",
            "输出要求：",
            "1. 必须只使用当前可见正文与给定名家资料。",
            "2. 如果证据不足，请明确说当前文本还不足以下结论。",
            "3. 不要泄露未来剧情，不要虚构引文。",
        ]
    )
    return "\n".join(pieces)


def generate_persona_response(
    *,
    persona_id: str,
    task: str,
    book_title: str,
    question: str,
    visible_contexts: list[str],
    current_chapter: int,
    highlight_text: str = "",
    top_k: int = 5,
    categories: list[str] | None = None,
    conversation_history: list[ChatMessage] | None = None,
) -> tuple[str, str, list[PersonaRAGHit]]:
    categories = categories or []
    ensure_persona_assets(persona_id)
    config, api_key, base_url, model_name = resolve_persona_runtime(persona_id)
    retrieval_query_parts = [question, highlight_text, "\n".join(visible_contexts)]
    retrieval_query = " ".join(part for part in retrieval_query_parts if part.strip())
    persona_hits = retrieve_persona_snippets(
        persona_id,
        PersonaRAGQueryRequest(query=retrieval_query, top_k=top_k, categories=categories),
    )
    system_prompt = build_persona_system_prompt(persona_id, task)
    user_prompt = build_persona_user_prompt(
        persona_id=persona_id,
        task=task,
        book_title=book_title,
        question=question,
        visible_contexts=visible_contexts,
        current_chapter=current_chapter,
        highlight_text=highlight_text,
        persona_hits=persona_hits,
        conversation_history=conversation_history,
    )
    try:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in (conversation_history or [])[-6:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_prompt})
        answer = invoke_openai_compatible_messages(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            messages=messages,
        )
    except Exception as exc:  # pragma: no cover - exercised via integration tests with monkeypatch
        raise PersonaAgentInvocationError(
            f"persona agent `{config.agent_id}` failed to generate a response: {exc}"
        ) from exc
    return answer.strip(), model_name, persona_hits
