"""System prompt and LLM interaction for language feedback."""

import asyncio
import json
import logging
import os
from pathlib import Path

import jsonschema
from openai import APIError, AsyncOpenAI


from app.models import FeedbackRequest, FeedbackResponse

from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv()

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
number_agreement, tone_register, other.
4. Difficulty must be exactly one of: A1, A2, B1, B2, C1, C2 (CEFR level based on \
sentence complexity, NOT on whether it has errors).
5. The corrected_sentence should be the minimal correction -- preserve the \
learner's original meaning and style as much as possible.
6. Explanations should be concise (1–2 sentences), friendly, and educational.
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


# Retry config: stay under 30s total (25s timeout, 2 retries with 1s + 2s backoff)
_LLM_TIMEOUT = 25.0
_MAX_RETRIES = 2
_BACKOFF_BASE_SEC = 1.0


async def _call_llm(client: AsyncOpenAI, user_message: str) -> tuple[dict, int, int]:
    """Call OpenAI with retry on transient errors. Returns (parsed_data, prompt_tokens, completion_tokens)."""
    last_error: Exception | None = None
    model_name =  os.environ.get('DEFAULT_FEEDBACK_MODEL')
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=_LLM_TIMEOUT,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return data, prompt_tokens, completion_tokens
        except APIError as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning("LLM call failed (attempt %s), retrying in %ss: %s", attempt + 1, delay, e)
                await asyncio.sleep(delay)
            else:
                break
    raise FeedbackUnavailableError("LLM request failed after retries") from last_error


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
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

    data, prompt_tokens, completion_tokens = await _call_llm(client, user_message)

    try:
        _validate_response_schema(data)
    except jsonschema.ValidationError:
        # Retry once with a fix prompt
        fix_message = (
            "Your previous response was invalid: it did not match the required JSON schema. "
            "Return ONLY valid JSON with keys: corrected_sentence, is_correct, errors (array of objects with original, correction, error_type, explanation), difficulty (one of A1,A2,B1,B2,C1,C2). "
            "error_type must be one of: grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other."
        )
        data, prompt_tokens, completion_tokens = await _call_llm(client, fix_message + "\n\nOriginal request:\n" + user_message)
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
