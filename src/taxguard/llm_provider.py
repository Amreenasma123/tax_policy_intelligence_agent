"""Optional OpenAI-compatible hosted LLM adapter (e.g. Groq).

Use only approved providers and synthetic/demo data unless data-processing terms
have been reviewed. Never print or log API keys.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class LLMProviderError(RuntimeError):
    """Actionable, key-safe error returned by an LLM provider call."""


def _read_provider_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        message = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
        if not message and isinstance(payload, dict):
            message = payload.get("message")
        return str(message or raw[:1200] or exc.reason)
    except Exception:
        return str(exc.reason or "No error details returned")


class OpenAICompatibleProvider:
    def __init__(self, base_url=None, api_key=None, model=None, temperature=None, max_tokens=None):
        self.base_url = (base_url if base_url is not None else os.getenv("TAXGUARD_LLM_BASE_URL", "")).strip().rstrip("/")
        self.api_key = (api_key if api_key is not None else os.getenv("TAXGUARD_LLM_API_KEY", "")).strip()
        self.model = (model if model is not None else os.getenv("TAXGUARD_LLM_MODEL", "")).strip()
        self.temperature = float(temperature if temperature is not None else os.getenv("TAXGUARD_LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(max_tokens if max_tokens is not None else os.getenv("TAXGUARD_LLM_MAX_TOKENS", "800"))
        self.timeout = int(os.getenv("TAXGUARD_LLM_TIMEOUT_SECONDS", "45"))

    @property
    def configured(self):
        return bool(self.base_url and self.api_key and self.model)

    def _endpoint(self) -> str:
        if not self.configured:
            raise LLMProviderError(
                "LLM is not configured. Set TAXGUARD_LLM_BASE_URL, TAXGUARD_LLM_API_KEY and TAXGUARD_LLM_MODEL."
            )
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise LLMProviderError(
                f"Invalid TAXGUARD_LLM_BASE_URL: {self.base_url!r}. Expected a URL such as "
                "https://api.groq.com/openai/v1"
            )
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise LLMProviderError("Use HTTPS for hosted LLM endpoints; HTTP is allowed only for localhost.")
        # Accept either the API base URL or a full chat/completions URL.
        suffix = "/chat/completions"
        if self.base_url.endswith(suffix):
            return self.base_url
        return self.base_url + suffix

    def answer(self, question, country_code, as_of_date, retrieved_context):
        endpoint = self._endpoint()
        if self.temperature < 0 or self.temperature > 2:
            raise LLMProviderError("TAXGUARD_LLM_TEMPERATURE must be between 0 and 2.")
        if self.max_tokens < 1:
            raise LLMProviderError("TAXGUARD_LLM_MAX_TOKENS must be at least 1.")

        # Resolve the prompt relative to this package, not the shell's current directory.
        root = Path(__file__).resolve().parents[2]
        prompt_path = root / "prompts" / "tax_policy_answer.md"
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
            prompt = template.format(
                question=question, country_code=country_code, as_of_date=as_of_date,
                retrieved_context=retrieved_context,
            )
        else:
            # Safe fallback so a CLI launched from another working directory still works.
            prompt = (
                "Answer using only the supplied synthetic policy evidence. Do not invent tax rules. "
                "Return a JSON object with answer, citations (chunk_id/document_id), and abstention_reason. "
                "Clearly state this is a demo and not tax advice.\n\n"
                f"Jurisdiction: {country_code}\nAs-of date: {as_of_date}\nQuestion: {question}\n\n"
                f"Evidence:\n{retrieved_context}"
            )

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "Follow the grounded policy answer rules. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "TaxGuard-AI-Demo/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _read_provider_error(exc)
            if exc.code in (401, 403):
                hint = "Check that the key is valid, copied fully, active, and permitted to use this model. Do not paste the key into chat or logs."
            elif exc.code == 404:
                hint = "Check the model ID and base URL. The base URL should normally end in /openai/v1, not /chat/completions."
            elif exc.code == 429:
                hint = "The provider rate limit or quota may have been reached; check the provider console and retry later."
            else:
                hint = "Check the provider API docs, model availability, and network connectivity."
            raise LLMProviderError(f"HTTP {exc.code} from LLM provider: {detail}\n{hint}") from None
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"Could not connect to LLM provider: {exc.reason}. Check network/proxy and endpoint URL.") from None
        except TimeoutError:
            raise LLMProviderError(f"LLM request timed out after {self.timeout}s. Increase TAXGUARD_LLM_TIMEOUT_SECONDS or retry.") from None
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected LLM response format ({type(exc).__name__}); check provider compatibility.") from None

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMProviderError("Provider returned HTTP success but no choices[0].message.content. Check model/API compatibility.") from None
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("Provider returned an empty model response. Try again or inspect provider status/model availability.")
        return content.strip()
