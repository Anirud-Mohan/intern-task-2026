import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv

from app.feedback import get_feedback
from app.models import FeedbackRequest

load_dotenv()
EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "examples" / "sample_inputs.json"


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a language feedback API.
You will see:
- the original learner request (sentence, target_language, native_language)
- the API's actual response
- the expected 'ideal' response (from a human designer)

Score how good the API's response is on a scale of 1–5 for:
1) Correction accuracy (did it fix the sentence correctly?)
2) Error analysis quality (are error types and explanations reasonable?)

Return ONLY JSON like:
{"correction_score": 1-5, "analysis_score": 1-5, "comments": "short explanation"}.
"""


async def judge_example(client: AsyncOpenAI, example: dict, actual: dict) -> dict:
    payload = {
        "request": example["request"],
        "expected_response": example["expected_response"],
        "actual_response": actual,
    }
    resp = await client.chat.completions.create(
        model=os.environ.get('JUDGE_MODEL'),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


async def main() -> None:
    client = AsyncOpenAI()
    examples = json.loads(EXAMPLES_PATH.read_text())
    scores: list[dict] = []

    for example in examples:
        req = FeedbackRequest(**example["request"])
        result = await get_feedback(req)
        actual = result.model_dump()
        judgment = await judge_example(client, example, actual)
        scores.append(judgment)
        print(f"- {example['description']}: {judgment}")

    if scores:
        avg_corr = sum(s["correction_score"] for s in scores) / len(scores)
        avg_an = sum(s["analysis_score"] for s in scores) / len(scores)
        print(f"\\nAverage correction_score: {avg_corr:.2f}")
        print(f"Average analysis_score:   {avg_an:.2f}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set for judge.")
    asyncio.run(main())

