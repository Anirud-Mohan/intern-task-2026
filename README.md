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
    - Use an encouraging, learner-friendly tone.
    - Avoid inventing new error categories (e.g., map particle/case/article issues to `grammar`).
  - Includes a **few-shot example** (Spanish conjugation error).
  - Uses OpenAI chat completions with `response_format={"type": "json_object"}`.
- **Schema validation**:
  - On every response, parses JSON and validates against `schema/response.schema.json` using `jsonschema`.
  - If validation fails once, sends a **repair prompt** asking the model to fix the JSON to match the schema.
  - If it still fails, raises `FeedbackUnavailableError`, which is mapped to `503` in `app/main.py`.

### Retry logic + timeout

- Implemented in `_call_llm` in `app/feedback.py`:
  - Uses an **end-to-end deadline budget** (28s) per request.
  - Uses bounded per-attempt timeout with deadline-aware retry/backoff.
  - Uses no extra retries on schema-repair call to avoid runaway latency.
  - If budget is exhausted or retries fail, raises `FeedbackUnavailableError`.
- This keeps failures fast and prevents retry behavior from growing unbounded.

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
    - Calls a configurable judge model (`JUDGE_MODEL` env var; used as `gpt-5-mini` during evaluation) with a rubric:
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


**LLM-as-a-judge evaluation:**

```bash
python -m scripts.eval_with_judge
```

Both scripts assume `OPENAI_API_KEY` is set.

---

## Model choice and cost analysis

### Evaluation methodology

Three candidate models were evaluated using `scripts/eval_with_judge.py` over 8 multilingual examples (Spanish, French, Japanese, German, Portuguese, Korean, Russian, Chinese), run twice each to reduce variance. The judge scored each response 1–5 on:
- **Correction accuracy**: did the model fix the sentence correctly?
- **Analysis quality**: are the error types and explanations reasonable?

### Results

| Model | Correction (avg) | Analysis (avg) | Combined |
|---|---|---|---|
| `gpt-4o-mini` (run 1) | 3.75 | 2.88 | 3.31 |
| `gpt-4o-mini` (run 2) | **4.25** | 3.50 | **3.88** |
| `gpt-4.1-nano` (run 1) | **4.25** | 3.50 | **3.88** |
| `gpt-4.1-nano` (run 2) | 4.12 | 3.25 | 3.69 |
| `gpt-5.4-nano` (run 1) | 4.12 | **3.62** | 3.87 |
| `gpt-5.4-nano` (run 2) | **4.25** | 3.38 | 3.82 |

All three models scored within **0.2 combined points** of each other. The two consistent weak spots across every model are:
- **Portuguese** (misses the `gostar + de` preposition error): a prompt engineering gap, not model-specific.
- **Chinese measure word**: `我有书。` is borderline acceptable in some contexts; all models pass it as correct.

### Cost comparison (per 1,000 uncached requests, ~750 prompt + ~250 completion tokens)

| Model | $/1M input | $/1M output | Cost / 1k reqs |
|---|---|---|---|
| `gpt-4o-mini` | $0.15 | $0.60 | ~$0.21 |
| `gpt-4.1-nano` | $0.10 | $0.40 | ~$0.14 |
| `gpt-5.4-nano` | $0.10 | $0.40 | ~$0.14 |

With caching enabled, repeated requests cost $0.00 in LLM tokens regardless of model.

### Decision: `gpt-4o-mini`

`gpt-4o-mini` is selected as the production model because:

1. **Score parity**: its best run matches the highest combined score of any model tested (3.88). There is no meaningful quality gap that justifies switching models.
2. **Stability**: `gpt-4o-mini` is a stable, publicly documented model with well-established pricing. Newer models carry more risk of API or pricing changes before a production deployment.
3. **Already cost-effective**: at ~$0.21/1,000 uncached requests, cost is already very low. The marginal saving from a cheaper model ($0.07/1k) does not outweigh the reliability advantage.
4. **Overridable**: the model is not hardcoded; it reads from the `DEFAULT_FEEDBACK_MODEL` env var, so upgrading to a stronger model is a one-line config change.

---

## Notes on implementation choices

- **Frameworks**: FastAPI, NeMo Guardrails, OpenAI async client.
- **Cost-awareness**: in-memory caching (cache hits use 0 tokens), server-side token logging, benchmark script.
- **Extensibility**: guardrails are layered (structural + NeMo) and the model is env-configurable, so both safety policy and model upgrades are localized changes.

## Future scaling roadmap

If this API were to be productionized for larger traffic, these are my next steps:

1. **Distributed cache**: Replace in-process cache with Redis and TTL-based invalidation so repeated requests are shared across replicas.
2. **Async job + fallback path**: Add a queue-based processing mode for spikes and return a polling token for long-running requests.
3. **Latency SLO controls**: Add explicit p95/p99 tracking, adaptive retry budgets, and circuit-breaking for upstream model instability.
4. **Cost controls**: Add model-routing (cheap model first, upgrade to stronger model on uncertainty), and monthly budget guardrails.
5. **Evaluation pipeline**: Persist judge outputs and regression baselines per model/prompt version to prevent silent quality drift.
6. **Observability**: Add structured logs, request IDs, and dashboards for schema-failure rate, retry rate, and per-language quality trends.

## Architecture

The following diagram shows the high-level components of the system and how requests are processed.

![Architecture diagram](Architecture.svg)

