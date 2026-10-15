from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from FYP2.services.roman_urdu_service import contains_urdu_script

logger = logging.getLogger(__name__)

GREETING_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "salam",
    "assalam",
    "assalam o alaikum",
    "assalamu alaikum",
    "aoa",
    "kaisa hai",
    "kaise ho",
    "kese ho",
    "kia hal hai",
    "kya hal hai",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
)


class AdvisorState(TypedDict, total=False):
    query: str
    language: str  # "english" or "roman_urdu"
    detected_language: str  # "en", "ur", etc.
    emotion: str
    student_id: str
    session_id: int
    db_conn: Any
    intent: str  # "greeting" | "irrelevant" | "academic"
    profile: Optional[Dict[str, Any]]
    student_context: Optional[Dict[str, Any]]
    knowledge: str
    prompt: str
    model: str
    response: Optional[str]
    error: Optional[str]


def classify_intent_node(state: AdvisorState) -> dict[str, Any]:
    """Classify the user intent into 'greeting', 'irrelevant', or 'academic'."""
    from FYP2.services.voice_service import is_irrelevant_query

    query = (state.get("query") or "").strip()
    normalized = query.casefold()

    # 1. Check for irrelevant / off-topic queries first
    if is_irrelevant_query(normalized):
        return {"intent": "irrelevant"}

    # 2. Check for greetings
    words = re.findall(r"[a-z]+", normalized)
    is_greeting = False
    if words and len(words) <= 5:
        if any(term in normalized for term in GREETING_PATTERNS):
            is_greeting = True
    elif normalized in {"hi", "hello", "hey", "salam", "assalam", "assalam o alaikum"}:
        is_greeting = True

    if is_greeting:
        return {"intent": "greeting"}

    # 3. All other questions are treated as academic
    return {"intent": "academic"}


def route_intent(state: AdvisorState) -> str:
    """Conditional edge selector based on the classified intent."""
    return state.get("intent", "academic")


def handle_greeting_node(state: AdvisorState) -> dict[str, Any]:
    """Generate a direct greeting reply customized for the student."""
    from FYP2.services.voice_service import get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    first_name = ""
    if profile:
        full_name = profile.get("full_name") or profile.get("name") or ""
        first_name = full_name.split()[0] if full_name else ""

    name_suffix = f" {first_name}" if first_name else ""
    language = state.get("language", "english")

    if language == "roman_urdu":
        reply = (
            f"Walekum Assalam{name_suffix}! Main aap ka university academic advisor hoon. "
            "Aaj main aap ki kya madad kar sakta hoon?"
        )
    else:
        reply = (
            f"Hello{name_suffix}! I am your university academic advisor. "
            "How can I help you today with your courses, GPA, or university policies?"
        )

    return {"response": reply, "profile": profile}


def handle_irrelevant_node(state: AdvisorState) -> dict[str, Any]:
    """Provide a polite refusal guardrail for non-academic queries."""
    language = state.get("language", "english")

    if language == "roman_urdu":
        reply = (
            "Main aap ka university academic advisor hoon. Main sirf university policies, "
            "courses, CGPA, aur academic mamlaat mein aap ki madad kar sakta hoon. "
            "Meherbani karke apni parhai ya university se mutaliq sawal poochein."
        )
    else:
        reply = (
            "I am your university academic advisor. I can only assist with academic matters, "
            "such as courses, CGPA, exams, and university policies. Please ask a university or academic-related question."
        )

    return {"response": reply}


def retrieve_knowledge_node(state: AdvisorState) -> dict[str, Any]:
    """Retrieve university policy knowledge and student DB context, then prepare prompt."""
    from FYP2.services.voice_service import (
        OLLAMA_MODEL_ENGLISH,
        OLLAMA_MODEL_ROMAN_URDU,
        build_prompt,
        build_student_context,
        resolve_ollama_model,
        retrieve_knowledge,
    )

    query = state.get("query", "")
    session_id = state.get("session_id", 0)
    student_id = state.get("student_id", "")
    db_conn = state.get("db_conn")
    language = state.get("language", "english")
    emotion = state.get("emotion", "Neutral")

    # 1. Fetch DB records into student context (without any hardcoded DB answers)
    if db_conn and student_id:
        context = build_student_context(query, session_id, student_id, db_conn)
    else:
        context = {"profile": state.get("profile") or {}}

    # 2. Retrieve matching university rules from knowledge base
    knowledge = retrieve_knowledge(query)

    # 3. Build prompt for LLM
    prompt = build_prompt(query, language, emotion, context, knowledge)
    preferred_model = (
        OLLAMA_MODEL_ROMAN_URDU if language == "roman_urdu" else OLLAMA_MODEL_ENGLISH
    )
    model = resolve_ollama_model(
        preferred_model,
        fallback="qwen2.5:7b-instruct",
    )

    return {
        "student_context": context,
        "knowledge": knowledge,
        "prompt": prompt,
        "model": model,
    }


def generate_response_node(state: AdvisorState) -> dict[str, Any]:
    """Invoke the LLM model to generate the final response for academic queries."""
    from FYP2.services.voice_service import stream_llm_reply

    prompt = state.get("prompt", "")
    model = state.get("model", "gemma2:2b")
    language = state.get("language", "english")

    chunks: list[str] = []
    script_violation = False
    try:
        for token in stream_llm_reply(prompt, model):
            chunks.append(token)
            if language == "roman_urdu" and contains_urdu_script(token):
                script_violation = True

        reply = "".join(chunks).strip()
        if language == "roman_urdu":
            # Strip English bracket translations
            reply = re.sub(r"\s*\([A-Za-z0-9\s,.'’?!]+\)", "", reply).strip()

        if script_violation or (
            language == "roman_urdu" and contains_urdu_script(reply)
        ):
            reply = (
                "Maazrat, main jawab Roman Urdu mein tayar nahi kar saka. "
                "Meherbani karke dobara koshish karein."
            )
    except Exception:
        logger.exception("LLM generation node failed in LangGraph")
        reply = (
            "AI model abhi available nahi hai, thori dair baad dobara koshish karein."
            if language == "roman_urdu"
            else "The AI model is unavailable. Please try again shortly."
        )

    if not reply:
        reply = (
            "Main is sawal ka jawab abhi tayar nahi kar saka."
            if language == "roman_urdu"
            else "I could not prepare an answer to that question."
        )

    return {"response": reply}


def create_advisor_graph():
    """Build and compile the LangGraph StateGraph for Academic Advising."""
    workflow = StateGraph(AdvisorState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_irrelevant", handle_irrelevant_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_response", generate_response_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "greeting": "handle_greeting",
            "irrelevant": "handle_irrelevant",
            "academic": "retrieve_knowledge",
        },
    )
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_irrelevant", END)
    workflow.add_edge("retrieve_knowledge", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


advisor_graph = create_advisor_graph()
