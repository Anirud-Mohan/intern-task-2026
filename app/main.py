"""FastAPI application -- language feedback endpoint."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.feedback import FeedbackUnavailableError, get_feedback
from app.guardrails import check_input_allowed
from app.models import FeedbackRequest, FeedbackResponse, MAX_SENTENCE_LENGTH

load_dotenv()

app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    # Guardrails: reject empty, too long, or off-topic (NeMo Guardrails + structural checks)
    if not request.sentence.strip():
        raise HTTPException(status_code=400, detail="Sentence cannot be empty")
    if len(request.sentence) > MAX_SENTENCE_LENGTH:
        raise HTTPException(status_code=400, detail="Sentence too long")

    # First-layer guardrails via NeMo (topic + prompt-injection filter)
    allowed = await check_input_allowed(
        request.sentence.strip(),
        request.target_language.strip(),
        request.native_language.strip(),
    )
    if not allowed:
        raise HTTPException(
            status_code=400,
            detail="Request is not a valid language feedback request",
        )

    try:
        return await get_feedback(request)
    except FeedbackUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Feedback temporarily unavailable",
        ) from None
