"""Guardrails using NeMo Guardrails as first-layer filter.

This module checks whether an incoming request is a valid language-feedback request
or should be rejected as off-topic / prompt injection.
"""

import json
import logging
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "guardrails"

_config = RailsConfig.from_path(str(_CONFIG_DIR))
_llm_rails = LLMRails(_config)


async def check_input_allowed(sentence: str, target_language: str, native_language: str) -> bool:
    """Return True if NeMo Guardrails says this is a valid language-feedback request.

    The request is passed as a JSON object to NeMo. NeMo is instructed (in config.yml)
    to respond with exactly ALLOW or REJECT.

    If guardrails fail for any reason or return an unexpected output, we fail open
    (return True) so the API remains usable.
    """
    payload = {
        "sentence": sentence,
        "target_language": target_language,
        "native_language": native_language,
    }
    try:
        message = await _llm_rails.generate_async(
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        )
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
        else:
            content = str(message).strip()

        upper = content.upper()
        if "REJECT" in upper:
            return False
        if "ALLOW" in upper:
            return True

        logger.warning("NeMo guardrails returned unexpected output: %r", content)
        # If NeMo didn't clearly emit ALLOW/REJECT, treat it as non-compliant
        # instead of allowing the request to pass.
        return False
    except Exception as exc:
        logger.warning("NeMo guardrails failed, allowing request by default: %s", exc)
        return True

