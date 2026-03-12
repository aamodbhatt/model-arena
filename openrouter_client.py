"""OpenRouter API client for ModelArena."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv


class OpenRouterClient:
    def __init__(self, timeout: int = 90) -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.timeout = timeout
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 700,
    ) -> str:
        content, _usage = self.chat_with_usage(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content

    def chat_with_usage(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 700,
    ) -> Tuple[str, Dict[str, Any]]:
        data = self._send_chat_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        return self._normalize_content(content), usage

    def probe_model(self, model: str, timeout_seconds: int = 30) -> Tuple[bool, str]:
        """Cheap availability check for a model ID."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with: OK"}],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/modelarena/local",
            "X-Title": "ModelArena",
        }
        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=min(self.timeout, timeout_seconds),
            )
        except requests.RequestException as exc:
            return False, f"Request failed: {exc}"

        if response.status_code >= 400:
            details = response.text.strip()[:280]
            return False, f"{response.status_code}: {details}"

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return False, "No choices returned"
        except ValueError:
            return False, "Invalid JSON response"

        return True, ""

    def _send_chat_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/modelarena/local",
            "X-Title": "ModelArena",
        }

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            details = response.text.strip()[:500]
            raise RuntimeError(f"OpenRouter error {response.status_code}: {details}")

        return response.json()

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            return "\n".join(part.strip() for part in text_parts if part.strip())
        return str(content).strip()

    @staticmethod
    def build_vision_user_content(prompt: str, image_path: str) -> List[Dict[str, Any]]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        image_data_url = f"data:{mime_type};base64,{encoded}"

        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
