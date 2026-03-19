"""System prompt and LLM interaction for language feedback."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import jsonschema
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

logger = logging.getLogger(__name__)

# Load response schema once at module load
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
_RESPONSE_SCHEMA = json.loads((_SCHEMA_DIR / "response.schema.json").read_text())

# Few-shot examples (abbreviated for prompt length)
_FEW_SHOT_EXAMPLE_IN = (
    "Target language: Spanish\nNative language: English\n"
    "Sentence: Yo soy fue al mercado ayer."
)
_FEW_SHOT_EXAMPLE_OUT = """{
  "corrected_sentence": "Yo fui al mercado ayer.",
  "is_correct": false,
  "errors": [{"original": "soy fue", "correction": "fui", "error_type": "conjugation", "explanation": "You mixed two verb forms. Use 'fui' (I went)."}],
  "difficulty": "A2"
}"""

SYSTEM_PROMPT = f"""\
You are a language-learning assistant. A student is practicing writing in their \
target language. Your job is to analyze their sentence, find errors, and provide \
helpful feedback.

RULES:
1. If the sentence is already correct, return is_correct=true, an empty errors \
array, and set corrected_sentence to the original sentence exactly.
2. For each error, identify the original text, provide the correction, classify \
the error type, and explain the error in the learner's NATIVE language so they \
can understand.
3. Error types must be one of: grammar, spelling, word_choice, punctuation, \
word_order, missing_word, extra_word, conjugation, gender_agreement, \
number_agreement, tone_register, other. Never invent new categories.
    - Use grammar for particle/article/case issues.
    - Use conjugation for tense/verb-form issues.
    - If none fit, use other.
4. Difficulty must be exactly one of: A1, A2, B1, B2, C1, C2 (CEFR level based on \
sentence complexity, NOT on whether it has errors).
5. The corrected_sentence should be the minimal correction -- preserve the \
learner's original meaning and style as much as possible.
6. Use a supportive, confidence-building tone. Explain what to improve (1-2 sentences) and briefly reinforce what the learner did well when possible.
7. Output ONLY valid JSON with no other text. No markdown, no code fences.

Example input:
{_FEW_SHOT_EXAMPLE_IN}

Example output:
{_FEW_SHOT_EXAMPLE_OUT}

Respond with valid JSON matching this exact schema:
{{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {{
      "original": "string",
      "correction": "string",
      "error_type": "string (one of the allowed types above)",
      "explanation": "string (in native language only)"
    }}
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}}
"""


class FeedbackUnavailableError(Exception):
    """Raised when feedback cannot be produced (timeout, retries exhausted, schema failure)."""

    pass


def _cache_key(request: FeedbackRequest) -> tuple[str, str, str]:
    return (
        request.sentence.strip(),
        request.target_language.strip().lower(),
        request.native_language.strip().lower(),
    )


# In-memory cache: key -> FeedbackResponse
_feedback_cache: dict[tuple[str, str, str], FeedbackResponse] = {}
_cache_lock = asyncio.Lock()


def _cache_enabled() -> bool:
    return os.getenv("DISABLE_CACHE", "").lower() not in ("1", "true", "yes")


def _validate_response_schema(data: dict) -> None:
    """Validate parsed LLM response against the response JSON schema. Raises jsonschema.ValidationError if invalid."""
    jsonschema.validate(instance=data, schema=_RESPONSE_SCHEMA)


# Timeout/retry config with an end-to-end budget to stay safely under 30s.
_END_TO_END_TIMEOUT_SEC = 28.0
_PER_ATTEMPT_TIMEOUT_SEC = 10.0
_MAX_RETRIES = 1
_BACKOFF_BASE_SEC = 0.6
_MIN_REMAINING_FOR_CALL_SEC = 1.5


def _remaining_time(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


async def _call_llm(
    client: AsyncOpenAI,
    user_message: str,
    deadline: float,
    max_retries: int = _MAX_RETRIES,
) -> tuple[dict, int, int]:
    """Call OpenAI with retry on transient errors. Returns (parsed_data, prompt_tokens, completion_tokens)."""
    last_error: Exception | None = None
    model_name = os.environ.get("DEFAULT_FEEDBACK_MODEL", "gpt-4o-mini")
    for attempt in range(max_retries + 1):
        try:
            remaining = _remaining_time(deadline)
            if remaining <= _MIN_REMAINING_FOR_CALL_SEC:
                raise FeedbackUnavailableError("Timeout budget exhausted before LLM call")

            attempt_timeout = min(_PER_ATTEMPT_TIMEOUT_SEC, max(0.5, remaining - 0.5))
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=attempt_timeout,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return data, prompt_tokens, completion_tokens
        except APIError as e:
            last_error = e
            if attempt < max_retries:
                delay = _BACKOFF_BASE_SEC * (2**attempt)
                remaining = _remaining_time(deadline)
                safe_delay = min(delay, max(0.0, remaining - _MIN_REMAINING_FOR_CALL_SEC))
                if safe_delay <= 0:
                    break
                logger.warning(
                    "LLM call failed (attempt %s), retrying in %.2fs: %s",
                    attempt + 1,
                    safe_delay,
                    e,
                )
                await asyncio.sleep(safe_delay)
            else:
                break
    raise FeedbackUnavailableError("LLM request failed after retries") from last_error


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    deadline = time.monotonic() + _END_TO_END_TIMEOUT_SEC

    key = _cache_key(request)
    if _cache_enabled():
        async with _cache_lock:
            if key in _feedback_cache:
                cached = _feedback_cache[key]
                logger.info(
                    "token_usage request_id=feedback prompt_tokens=0 completion_tokens=0 cache_hit=true",
                )
                return cached

    client = AsyncOpenAI()
    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    data, prompt_tokens, completion_tokens = await _call_llm(client, user_message, deadline)

    try:
        _validate_response_schema(data)
    except jsonschema.ValidationError:
        # Retry once with a fix prompt
        fix_message = (
            "Your previous response was invalid: it did not match the required JSON schema. "
            "Return ONLY valid JSON with keys: corrected_sentence, is_correct, errors (array of objects with original, correction, error_type, explanation), difficulty (one of A1,A2,B1,B2,C1,C2). "
            "error_type must be one of: grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other. "
            "Do NOT invent categories like particle/article/case/tense; map particle/article/case to grammar and tense/verb-form to conjugation. "
            "If unsure, use other."
        )
        data, prompt_tokens, completion_tokens = await _call_llm(
            client,
            fix_message + "\n\nOriginal request:\n" + user_message,
            deadline,
            max_retries=0,
        )
        try:
            _validate_response_schema(data)
        except jsonschema.ValidationError as e:
            logger.warning("Schema validation failed after retry: %s", e)
            raise FeedbackUnavailableError("Response did not match schema") from e

    logger.info(
        "token_usage request_id=feedback prompt_tokens=%s completion_tokens=%s",
        prompt_tokens,
        completion_tokens,
    )
    response = FeedbackResponse(**data)
    if _cache_enabled():
        async with _cache_lock:
            _feedback_cache[key] = response
    return response
