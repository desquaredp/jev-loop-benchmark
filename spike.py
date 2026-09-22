"""Tiny JEV HTTP client used by the routing policy."""

import json
import urllib.request

from config import jev_api_key, load_settings


_SETTINGS = load_settings()
MODEL = _SETTINGS.jev_model
JEV_URL = _SETTINGS.jev_url
PRICE_PER_MILLION = _SETTINGS.jev_input_price


def post_jev(body, key=None):
    key = key or jev_api_key()
    if not key:
        raise RuntimeError("JEV_API_KEY is missing")
    request = urllib.request.Request(
        JEV_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)
