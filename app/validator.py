"""
app/validator.py
────────────────
Pre-validation layer using NVIDIA NIM Vision-Language Model.

Uses microsoft/phi-3.5-vision-instruct via the NVIDIA NIM chat completions
endpoint to determine whether an uploaded image is a valid histopathology /
microscopic medical image before running the cancer detection pipeline.

API endpoint  : https://integrate.api.nvidia.com/v1/chat/completions
Model         : microsoft/phi-3.5-vision-instruct
Auth          : Bearer <nvapi-...>  (stored in .streamlit/secrets.toml)
"""

import base64
import io
import logging

import requests
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ── NVIDIA NIM settings ───────────────────────────────────────────────────────
_NIM_URL  = "https://integrate.api.nvidia.com/v1/chat/completions"
_MODEL    = "microsoft/phi-3.5-vision-instruct"
_TIMEOUT  = 20   # seconds

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a medical image classifier. "
    "Your only job is to decide whether the given image is a histopathology "
    "or microscopic medical image (tissue slide, cell staining, pathology patch, "
    "biopsy image) or not. "
    "Reply ONLY with a single word: VALID or INVALID. No explanation."
)


def _pil_to_b64(image: PILImage.Image) -> str:
    """Encode a PIL Image as a base64 JPEG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def validate_image(image: PILImage.Image, api_key: str | None = None) -> dict:
    """
    Validate whether *image* is a histopathology / medical image using
    NVIDIA NIM (microsoft/phi-3.5-vision-instruct).

    Args:
        image:   PIL Image to check.
        api_key: NVIDIA NIM API key (nvapi-...).
                 If missing or empty, validation is skipped (fail-open).

    Returns:
        {
            "is_valid":   bool   — True if medical/histopathology, False otherwise
            "label":      str    — human-readable classification label
            "confidence": float  — 1.0 if definitive, 0.0 if skipped
            "skipped":    bool   — True when API was not called
        }
    """
    # ── Fail-open if no key ───────────────────────────────────────────────────
    if not api_key:
        logger.warning("AI_API_KEY not set — skipping image validation.")
        return {
            "is_valid":   True,
            "label":      "Validation skipped (no API key)",
            "confidence": 0.0,
            "skipped":    True,
        }

    try:
        b64_image = _pil_to_b64(image)

        payload = {
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this image carefully. "
                                "Is it a histopathology tissue slide, "
                                "stained microscopy image, or medical pathology patch? "
                                "Reply ONLY: VALID or INVALID."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        resp = requests.post(_NIM_URL, headers=headers, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()

        raw_reply = (
            resp.json()["choices"][0]["message"]["content"]
            .strip()
            .upper()
        )

        logger.info("NIM CLIP validation raw reply: %r", raw_reply)

        is_valid = "VALID" in raw_reply and "INVALID" not in raw_reply

        return {
            "is_valid":   is_valid,
            "label":      (
                "Histopathology / medical image"
                if is_valid
                else "Non-medical or invalid image"
            ),
            "confidence": 1.0,
            "skipped":    False,
        }

    except requests.exceptions.Timeout:
        logger.warning("NVIDIA NIM VLM timed out — proceeding without validation.")
        return {
            "is_valid":   True,
            "label":      "Validation unavailable (timeout)",
            "confidence": 0.0,
            "skipped":    True,
        }

    except requests.exceptions.HTTPError as exc:
        logger.warning("NVIDIA NIM VLM HTTP error %s — proceeding.", exc)
        return {
            "is_valid":   True,
            "label":      f"Validation unavailable (HTTP {exc.response.status_code})",
            "confidence": 0.0,
            "skipped":    True,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("Image validation failed (%s) — proceeding.", exc)
        return {
            "is_valid":   True,
            "label":      f"Validation unavailable ({type(exc).__name__})",
            "confidence": 0.0,
            "skipped":    True,
        }
