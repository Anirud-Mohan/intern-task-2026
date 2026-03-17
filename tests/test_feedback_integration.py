"""Integration tests -- require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import jsonschema
import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

load_dotenv()
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set -- skipping integration tests",
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
RESPONSE_SCHEMA = json.loads((SCHEMA_DIR / "response.schema.json").read_text())

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.mark.asyncio
async def test_spanish_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_correct_german():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_french_gender_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_japanese_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_examples_conform_to_schema_and_invariants():
    """All example requests produce responses that conform to response schema and basic invariants."""
    examples = json.loads((EXAMPLES_DIR / "sample_inputs.json").read_text())
    for example in examples:
        req_data = example["request"]
        request = FeedbackRequest(**req_data)
        result = await get_feedback(request)
        # Schema compliance
        data = result.model_dump()
        jsonschema.validate(instance=data, schema=RESPONSE_SCHEMA)
        # Invariants
        if result.is_correct:
            assert result.errors == []
        assert result.difficulty in VALID_DIFFICULTIES
        for err in result.errors:
            assert err.error_type in VALID_ERROR_TYPES
