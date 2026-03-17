"""Optional benchmark: run N requests and report latency. Token usage is logged server-side; check server logs for token_usage lines."""
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root for app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.feedback import get_feedback
from app.models import FeedbackRequest


async def main() -> None:
    n = int(os.getenv("BENCHMARK_N", "5"))
    samples = [
        FeedbackRequest(sentence="Yo soy fue al mercado ayer.", target_language="Spanish", native_language="English"),
        FeedbackRequest(sentence="Ich habe gestern einen Film gesehen.", target_language="German", native_language="English"),
    ]
    latencies = []
    for i in range(n):
        req = samples[i % len(samples)]
        start = time.monotonic()
        await get_feedback(req)
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
    avg = sum(latencies) / len(latencies)
    print(f"Ran {n} requests, avg latency: {avg:.2f}s (min={min(latencies):.2f}s, max={max(latencies):.2f}s)")
    print("Token usage is logged server-side (token_usage prompt_tokens=... completion_tokens=...).")


if __name__ == "__main__":
    asyncio.run(main())
