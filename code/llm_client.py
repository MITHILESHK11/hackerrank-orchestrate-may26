from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests


class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash", timeout_seconds: int = 45) -> None:
        self.model = os.getenv("GEMINI_MODEL", model).strip() or model
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _endpoint(self) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            self._endpoint(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        candidates = body.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        merged = "\n".join(texts).strip()
        if not merged:
            raise RuntimeError("Gemini candidate text is empty.")
        return merged

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in response.")
        return json.loads(match.group(0))

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        base_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 700,
            },
        }
        if response_schema:
            base_payload["generationConfig"]["responseJsonSchema"] = response_schema

        last_error: Exception | None = None
        for _ in range(2):
            try:
                body = self._post(base_payload)
                raw = self._extract_text(body)
                return self._extract_json(raw)
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        fallback_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 700},
        }
        try:
            body = self._post(fallback_payload)
            raw = self._extract_text(body)
            return self._extract_json(raw)
        except Exception as exc:
            if last_error is not None:
                raise RuntimeError(f"{last_error}; fallback failed: {exc}") from exc
            raise
