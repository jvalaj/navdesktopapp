import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_LONG_EDGE = 1568
MAX_IMAGE_PIXELS = 1_150_000
MAX_IMAGE_BYTES = 4_800_000
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _normalized_provider(value: Optional[str]) -> str:
    provider = (value or "anthropic").strip().lower()
    if provider in {"openai", "openai_compatible", "local"}:
        return "openai"
    if provider in {"gemini", "google", "google_genai"}:
        return "gemini"
    if provider in {"zai", "z.ai", "glm"}:
        return "zai"
    return "anthropic"


def _scaled_image_size(width: int, height: int) -> tuple[int, int]:
    long_edge = max(width, height)
    total_pixels = max(1, width * height)
    long_edge_scale = MAX_IMAGE_LONG_EDGE / float(max(1, long_edge))
    pixel_scale = (MAX_IMAGE_PIXELS / float(total_pixels)) ** 0.5
    scale = min(1.0, long_edge_scale, pixel_scale)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _encode_pil_image(img: Image.Image) -> Dict[str, str]:
    media_type = "image/png"
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()

    if len(raw) > MAX_IMAGE_BYTES:
        media_type = "image/jpeg"
        for quality in (90, 80, 70, 60, 50):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
            if len(raw) <= MAX_IMAGE_BYTES:
                break

    return {
        "media_type": media_type,
        "data": base64.b64encode(raw).decode("utf-8"),
    }


def encode_image_from_path(path: str) -> Optional[Dict[str, str]]:
    if not path:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            new_w, new_h = _scaled_image_size(*img.size)
            if (new_w, new_h) != img.size:
                img = img.resize((new_w, new_h), Image.LANCZOS)
            return _encode_pil_image(img)
    except Exception:
        return None


def encode_image_from_bgr(img_bgr: np.ndarray) -> Optional[Dict[str, str]]:
    if img_bgr is None or getattr(img_bgr, "size", 0) <= 0:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        new_w, new_h = _scaled_image_size(*img.size)
        if (new_w, new_h) != img.size:
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return _encode_pil_image(img)
    except Exception:
        return None


def _first_env(*keys: str) -> str:
    for key in keys:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def _parse_openai_style_message_content(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj.strip()
    if isinstance(content_obj, list):
        out: List[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    out.append(str(text))
        return "\n".join(out).strip()
    return str(content_obj).strip()


class ModelClient:
    def __init__(
        self,
        provider: Optional[str],
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
    ):
        self.provider = _normalized_provider(provider)
        self.model = (model or "").strip()
        self.timeout_s = float(timeout_s)
        self.base_url = (base_url or "").strip()

        if self.provider == "anthropic":
            if not api_key:
                api_key = _first_env("claudekey", "ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Missing Anthropic API key.")
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key, timeout=self.timeout_s)
            self.api_key = api_key
            return

        if self.provider == "openai":
            if not api_key:
                api_key = _first_env("OPENAI_API_KEY", "MODEL_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI API key.")
            if not self.base_url:
                self.base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()

            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_s}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            self.api_key = api_key
            return

        if self.provider == "gemini":
            if not api_key:
                api_key = _first_env("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Missing Gemini API key.")
            if not self.base_url:
                self.base_url = (os.environ.get("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL).strip()
            self._client = None
            self.api_key = api_key
            return

        if not api_key:
            api_key = _first_env("ZAI_API_KEY")
        if not api_key:
            raise ValueError("Missing Z.ai API key.")
        if not self.base_url:
            self.base_url = (os.environ.get("ZAI_BASE_URL") or DEFAULT_ZAI_BASE_URL).strip()
        self._client = None
        self.api_key = api_key

    def _openai_user_content(self, text_blocks: List[str], images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for text in text_blocks:
            if text:
                content.append({"type": "input_text", "text": str(text)})
        for img in images:
            data_url = f"data:{img['media_type']};base64,{img['data']}"
            content.append({"type": "input_image", "image_url": data_url})
        return content

    def _zai_user_content(self, text_blocks: List[str], images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for text in text_blocks:
            if text:
                content.append({"type": "text", "text": str(text)})
        for img in images:
            data_url = f"data:{img['media_type']};base64,{img['data']}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        return content

    def _gemini_http_complete(
        self,
        system_prompt: str,
        text_blocks: List[str],
        images: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        parts: List[Dict[str, Any]] = []
        prompt_text = "\n\n".join([system_prompt] + [t for t in text_blocks if t]).strip()
        if prompt_text:
            parts.append({"text": prompt_text})
        for img in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": img["media_type"],
                        "data": img["data"],
                    }
                }
            )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        model_ref = urllib.parse.quote(self.model, safe="")
        endpoint = (
            self.base_url.rstrip("/")
            + f"/models/{model_ref}:generateContent?key={urllib.parse.quote(self.api_key, safe='')}"
        )
        req = urllib.request.Request(
            endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail[:500]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except Exception as exc:
            raise RuntimeError("Gemini API returned non-JSON response.") from exc

        candidates = parsed.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini API response contained no candidates.")
        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        out = [str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(out).strip()

    def _zai_complete(
        self,
        system_prompt: str,
        text_blocks: List[str],
        images: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        user_content = self._zai_user_content(text_blocks, images)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept-Language": "en-US,en",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"Z.ai API HTTP {exc.code}: {detail[:500]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Z.ai API request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except Exception as exc:
            raise RuntimeError("Z.ai API returned non-JSON response.") from exc

        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError("Z.ai API response contained no choices.")
        message = (choices[0] or {}).get("message") or {}
        return _parse_openai_style_message_content(message.get("content", ""))

    def complete(
        self,
        system_prompt: str,
        text_blocks: List[str],
        images: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        images = images or []
        if self.provider == "anthropic":
            content: List[Dict[str, Any]] = []
            for text in text_blocks:
                if text:
                    content.append({"type": "text", "text": str(text)})
            for img in images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["media_type"],
                            "data": img["data"],
                        },
                    }
                )
            response = self._client.messages.create(
                model=self.model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            chunks: List[str] = []
            for block in getattr(response, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(str(text))
            return "\n".join(chunks).strip()

        if self.provider == "openai":
            input_payload: List[Dict[str, Any]] = []
            if system_prompt:
                input_payload.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                )
            input_payload.append(
                {
                    "role": "user",
                    "content": self._openai_user_content(text_blocks, images),
                }
            )
            response = self._client.responses.create(
                model=self.model,
                input=input_payload,
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return str(output_text).strip()
            parsed_output = getattr(response, "output", None) or []
            chunks: List[str] = []
            for item in parsed_output:
                content = getattr(item, "content", None) or []
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks).strip()

        if self.provider == "gemini":
            return self._gemini_http_complete(
                system_prompt=system_prompt,
                text_blocks=text_blocks,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return self._zai_complete(
            system_prompt=system_prompt,
            text_blocks=text_blocks,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
