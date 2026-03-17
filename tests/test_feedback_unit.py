"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest


def _mock_completion(response_data: dict, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


@pytest.mark.asyncio
async def test_feedback_with_errors():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_retry_on_failure_then_succeed():
    """Retry logic: first call fails with APIError, second succeeds."""
    from openai import APIError

    mock_response = {
        "corrected_sentence": "Hello world.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    call_count = 0

    async def create_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIError("Timeout", request=MagicMock(), body={})
        return _mock_completion(mock_response)

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
        request = FeedbackRequest(
            sentence="Hello world.",
            target_language="English",
            native_language="Spanish",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.corrected_sentence == "Hello world."
    assert call_count == 2


@pytest.mark.asyncio
async def test_cache_returns_same_response_and_skips_llm():
    """Cache: two identical requests return same response; second does not call LLM."""
    with patch.dict("os.environ", {"DISABLE_CACHE": ""}, clear=False):
        # Clear module cache so test is isolated
        import app.feedback as feedback_module
        feedback_module._feedback_cache.clear()

        mock_response = {
            "corrected_sentence": "Cached.",
            "is_correct": False,
            "errors": [{"original": "x", "correction": "y", "error_type": "grammar", "explanation": "E"}],
            "difficulty": "A1",
        }
        create_mock = AsyncMock(return_value=_mock_completion(mock_response))

        with patch("app.feedback.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = create_mock
            request = FeedbackRequest(
                sentence="Cached.",
                target_language="English",
                native_language="French",
            )
            result1 = await get_feedback(request)
            result2 = await get_feedback(request)

        assert result1.corrected_sentence == result2.corrected_sentence == "Cached."
        assert create_mock.call_count == 1


@pytest.mark.asyncio
async def test_schema_validation_failure_raises_after_retry():
    """Schema validation: invalid enum (e.g. difficulty Z9) triggers retry then FeedbackUnavailableError if still invalid."""
    from app.feedback import FeedbackUnavailableError

    invalid_response = {
        "corrected_sentence": "Ok",
        "is_correct": True,
        "errors": [],
        "difficulty": "Z9",
    }
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(invalid_response)
        )
        request = FeedbackRequest(
            sentence="Ok",
            target_language="English",
            native_language="Spanish",
        )
        with pytest.raises(FeedbackUnavailableError):
            await get_feedback(request)
