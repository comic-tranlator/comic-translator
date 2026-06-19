from __future__ import annotations

import os

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

# ---------------------------------------------------------------------------
# Tools available to the agent
# ---------------------------------------------------------------------------


@tool
def translate_texts(
    texts: list[str], source_language: str, target_language: str
) -> list[str]:
    """
    Translate a batch of text strings extracted from an image (e.g. manga bubbles).
    Preserves order. Returns one translated string per input string.
    Keeps empty strings as empty strings.
    """
    return texts


@tool
def detect_language(texts: list[str]) -> str:
    """
    Detect the dominant language present in a list of text strings.
    Returns an ISO-639-1 language code (e.g. 'ja', 'zh', 'ko', 'en').
    """
    return "unknown"  # placeholder; agent reasons over this


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert manga/comic translation agent.

Your job:
1. Receive a list of OCR-extracted text strings from a single page (bubbles, signs, sfx).
2. Optionally detect the source language if not provided.
3. Translate every non-empty string into the requested target language.
4. Preserve original meaning, tone, and manga-style nuances.
5. Return ONLY the translated strings as a JSON array in the same order as the input.
   Do NOT add any extra keys, explanations, or markdown — just the JSON array.

Rules:
- Keep empty strings as empty strings ("").
- Keep sound effects (SFX) culturally adapted (e.g. "ドキドキ" → "thump thump").
- Preserve ALL-CAPS emphasis where present.
- Never merge or split strings — one output per input.
"""


class TranslationAgent:
    """
    LangChain agent backed by Mistral that translates OCR texts for one page.

    Usage
    -----
    agent = TranslationAgent(target_language="English", mistral_api_key="...")
    translated: list[str] = agent(["こんにちは", "世界"])
    """

    def __init__(
        self,
        mistral_api_key: str | None = None,
        target_language: str = "russian",
        source_language: str | None = None,
        model: str = "mistral-large-latest",
        temperature: float = 0.1,
    ) -> None:
        self.target_language = target_language
        self.source_language = source_language

        api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Mistral API key. Pass mistral_api_key or set MISTRAL_API_KEY."
            )

        llm = ChatMistralAI(
            model_name=model,
            temperature=temperature,
            api_key=api_key,
            disable_streaming="tool_calling",
            streaming=False,
        )

        tools = [translate_texts, detect_language]

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
        )

    def __call__(self, texts: list[str]) -> list[str]:
        """Translate *texts* and return a same-length list of translated strings."""
        if not texts:
            return []

        result = self._run_agent(texts)
        return result

    def _build_user_message(self, texts: list[str]) -> str:
        src = self.source_language or "auto-detect"
        lines = "\n".join(f"{i}: {repr(t)}" for i, t in enumerate(texts))
        return (
            f"Source language: {src}\n"
            f"Target language: {self.target_language}\n\n"
            f"Texts to translate (index: text):\n{lines}\n\n"
            "Return a JSON array of translated strings, one per index."
        )

    def _run_agent(self, texts: list[str]) -> list[str]:
        import json

        response = self.executor.invoke({"input": self._build_user_message(texts)})
        raw: str = response.get("output", "[]")

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            translated = json.loads(raw)
            if isinstance(translated, list) and len(translated) == len(texts):
                return [str(t) for t in translated]
        except (json.JSONDecodeError, ValueError):
            pass

        return texts
