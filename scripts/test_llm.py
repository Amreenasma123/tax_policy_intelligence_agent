"""Smoke-test the configured hosted LLM using synthetic, non-sensitive text."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taxguard.llm_provider import OpenAICompatibleProvider, LLMProviderError


def main():
    provider = OpenAICompatibleProvider()
    if not provider.configured:
        print("LLM not configured. Set TAXGUARD_LLM_BASE_URL, TAXGUARD_LLM_API_KEY and TAXGUARD_LLM_MODEL.")
        return 2
    # Do not print the API key. Show only the non-secret configuration.
    print(f"Endpoint: {provider.base_url.rstrip('/')}/chat/completions")
    print(f"Model: {provider.model}")
    print(f"Temperature: {provider.temperature}; max completion tokens: {provider.max_tokens}")
    context = "[chunk_id=DEMO-001 document_id=DEMO-US-TRAVEL version=1 section=Records]\n" \
              "Fictional demo policy: business travel claims must include a dated receipt and business purpose."
    try:
        output = provider.answer(
            "What records are required for a business travel claim?", "US", "2025-08-01", context
        )
        print("\nLLM RESPONSE:\n")
        print(output)
        print("\nLLM connection test: PASS (review grounding/citations separately).")
        return 0
    except LLMProviderError as exc:
        print(f"\nLLM connection test: FAIL\n{exc}")
        return 2
    except Exception as exc:
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
