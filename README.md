# Pangea Chat: Gen AI Intern Task (Summer 2026)

## Overview

This project implements a **production-oriented language feedback API** for the Pangea Chat intern task.

- **`POST /feedback`**: Given a learner’s `sentence`, `target_language`, and `native_language`, returns:
  - `corrected_sentence`
  - `is_correct` (boolean)
  - `errors[]` (with `original`, `correction`, `error_type`, `explanation`)
  - `difficulty` (CEFR `A1`–`C2`)
- **`GET /health`**: Simple 200 health check.

The implementation focuses on **robustness (guardrails, schema validation, retries)** and **efficiency (caching, token usage awareness)** while keeping the code simple and readable.

---

## Design decisions

### Guardrails (NeMo Guardrails + structural checks)

- **Structural limits** in `app/models.py`:
  - `MAX_SENTENCE_LENGTH = 1000`, `MAX_LANGUAGE_LENGTH = 50` to control cost and abuse.
- **First-layer topic / injection filter** in `app/guardrails.py`:
  - Uses **NeMo Guardrails** with `guardrails/config.yml` to classify requests as **ALLOW / REJECT**.
  - The payload (`sentence`, `target_language`, `native_language`) is passed as JSON; NeMo is instructed to return only `ALLOW` or `REJECT`.
  - If guardrails fail unexpectedly, the check **fails open** (allows the request) so the API remains usable.
- **FastAPI wiring** in `app/main.py`:
  - Rejects empty or too-long sentences with `400`.
  - Calls `check_input_allowed(...)`; if it returns `False`, responds with `400` and a clear message:
    - `"Request is not a valid language feedback request"`.

### Prompt + schema validation

- **Prompt** in `app/feedback.py`:
  - System prompt instructs the model to:
    - Use only allowed `error_type` values.
    - Use only CEFR levels `A1`–`C2` for `difficulty`.
    - Preserve learner voice and do **minimal corrections**.
    - Write explanations in the learner’s **native language**.
  - Includes a **few-shot example** (Spanish conjugation error).
  - Uses OpenAI `gpt-4o-mini` with `response_format={"type": "json_object"}` and `temperature=0.2`.
- **Schema validation**:
  - On every response, parses JSON and validates against `schema/response.schema.json` using `jsonschema`.
  - If validation fails once, sends a **repair prompt** asking the model to fix the JSON to match the schema.
  - If it still fails, raises `FeedbackUnavailableError`, which is mapped to `503` in `app/main.py`.

### Retry logic + timeout

- Implemented in `_call_llm` in `app/feedback.py`:
  - Uses OpenAI’s async client with `timeout=25.0` seconds.
  - Retries on `APIError` with **exponential backoff** (1s, 2s) for a total of 3 attempts (initial + 2 retries).
  - If all attempts fail, raises `FeedbackUnavailableError`.
- This keeps the **overall latency safely under 30 seconds** per `/feedback` request while handling transient issues.

### Caching

- In-memory cache in `app/feedback.py`:
  - Keyed by `(sentence.strip(), target_language.lower().strip(), native_language.lower().strip())`.
  - Uses an `asyncio.Lock` around a module-level dict to avoid races.
  - Cache hits:
    - Return the previous `FeedbackResponse` immediately.
    - Log `prompt_tokens=0, completion_tokens=0` with `cache_hit=true`.
  - `DISABLE_CACHE=1` environment variable can disable caching for benchmarking or tests.
- In production, this approach would be replaced with Redis or another shared cache, but the **interface is already cache-friendly**.

### Token usage logging + benchmarking

- **Logging** in `app/feedback.py`:
  - After each LLM call, logs:
    - `token_usage request_id=feedback prompt_tokens=... completion_tokens=...`
  - Cache hits log `prompt_tokens=0 completion_tokens=0 cache_hit=true`.
- **Benchmark script** in `scripts/benchmark_tokens.py`:
  - Runs a fixed set of requests `N` times (`BENCHMARK_N`) and reports:
    - Average latency, min, max.
  - Token usage is visible in the server logs, not in the API response body.

### LLM-as-a-judge evaluation

- **Examples** in `examples/sample_inputs.json`:
  - 8+ cases across Spanish, French, Japanese, German, Portuguese, Korean, Russian, and Chinese.
  - Each includes `request` and `expected_response`.
- **Judge script** in `scripts/eval_with_judge.py`:
  - For each example:
    - Calls `get_feedback(...)` to get the actual API response.
    - Calls `gpt-4o-mini` as a **judge** with a rubric:
      - `correction_score` (1–5): did it fix the sentence correctly?
      - `analysis_score` (1–5): are error types and explanations reasonable?
    - Prints per-example scores and average scores across the set.
- This separates **evaluation** from the main API and demonstrates how the system could be monitored and tuned over time.

### Error handling and status codes

- **400 Bad Request**: guardrails / validation failures:
  - Empty sentence.
  - Sentence too long.
  - NeMo Guardrails reject the request as not valid language feedback.
- **503 Service Unavailable**:
  - `FeedbackUnavailableError` raised by `app/feedback.py` when:
    - LLM fails after all retries.
    - Response cannot be coerced into a schema-conforming JSON object even after the repair prompt.
- **FastAPI/Pydantic** `422` validation errors remain for malformed request bodies.

---

## Running the project

### 1. Local setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/pangeachat/intern-task-2026.git
cd intern-task-2026

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=... (required for both the API and NeMo Guardrails)

uvicorn app.main:app --reload
```

Test the endpoints:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### 2. Docker

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

docker compose up --build
```

The API is available at `http://localhost:8000`.

### 3. Tests

```bash
# Unit + schema + guardrails tests (no external calls)
pytest tests/test_feedback_unit.py tests/test_schema.py tests/test_guardrails.py -v

# Integration tests (real LLM calls; requires OPENAI_API_KEY)
pytest tests/test_feedback_integration.py -v
```

### 4. Benchmarks and evaluation

**Token/latency benchmark:**

```bash
python -m scripts.benchmark_tokens
```

**LLM-as-a-judge evaluation:**

```bash
python -m scripts.eval_with_judge
```

Both scripts assume `OPENAI_API_KEY` is set.

---

## Notes on implementation choices

- **Model choice**: `gpt-4o-mini` for a good balance of quality and cost.
- **Frameworks**:
  - FastAPI for the HTTP API.
  - NeMo Guardrails for input topic/safety filtering.
  - OpenAI async client for LLM calls.
- **Cost-awareness**:
  - Caching + explicit token logging.
  - Benchmark scripts for latency and rough token usage.
- **Extensibility**:
  - Guardrails are layered (structural + NeMo), so swapping the main model or adding more safety checks is localized.
  - Evaluation is decoupled via the LLM-as-judge script.

---

## Commit message suggestions

You can summarize this round of work with a single, concise commit message such as:

- **Option 1:** `feat: add guarded feedback API with NeMo, caching, and eval tooling`
- **Option 2:** `feat: productionize feedback API (guardrails, retries, caching, tests, README)`
