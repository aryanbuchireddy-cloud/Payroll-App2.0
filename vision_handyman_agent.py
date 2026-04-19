"""
vision_handyman_agent.py
------------------------
General-purpose Gemini vision helper for flaky browser automation.

Design goals:
- Try normal Playwright first.
- Only call Gemini when selectors or page state are flaky.
- Be reusable across SalonData and Heartland.
- Keep actions simple: click, wait, scroll.

Environment:
- GEMINI_API_KEY=...   (free Gemini API key)
- optional: GEMINI_MODEL=gemini-2.0-flash
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from typing import Any, Dict, Optional

import PIL.Image
import google.generativeai as genai

try:
    import streamlit as st
except Exception:
    st = None


def _get_secret_or_env(name: str, default: str = "") -> str:
    try:
        if st is not None:
            v = st.secrets.get(name, None)
            if v not in (None, ""):
                return str(v).strip()
    except Exception:
        pass
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY", "")
GEMINI_MODEL = _get_secret_or_env("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def _get_model() -> genai.GenerativeModel:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to Streamlit secrets or set it as an environment variable."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


class BrowserHandymanAgent:
    def __init__(self) -> None:
        self._model = _get_model()

    async def ask(self, page, task: str) -> Dict[str, Any]:
        raw_png = await page.screenshot(full_page=False)
        img = PIL.Image.open(io.BytesIO(raw_png))

        try:
            vp = await page.evaluate("({w: window.innerWidth, h: window.innerHeight})")
            w, h = int(vp["w"]), int(vp["h"])
        except Exception:
            w, h = 1280, 800

        prompt = (
            "You are a browser automation helper for business web apps.\n"
            f"Viewport size: {w}x{h}.\n\n"
            f"Task: {task}\n\n"
            "Return ONLY raw JSON with this schema:\n"
            "{"
            '"status":"loading|ready|done|error",'
            '"action":"click|wait|scroll_down|none",'
            '"x":null,'
            '"y":null,'
            '"element":"short label",'
            '"reasoning":"one short sentence"'
            "}\n"
            "Rules:\n"
            "- Use action=click only when the target is clearly visible.\n"
            "- Use action=scroll_down if the target seems below the fold.\n"
            "- Use action=wait if the page is still loading, blank, processing, or disabled.\n"
            "- Never output markdown fences.\n"
        )

        try:
            response = self._model.generate_content([prompt, img])
            text = getattr(response, "text", "") or ""
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Gemini did not return a JSON object")
            return {
                "status": str(data.get("status") or "error"),
                "action": str(data.get("action") or "wait"),
                "x": data.get("x"),
                "y": data.get("y"),
                "element": str(data.get("element") or ""),
                "reasoning": str(data.get("reasoning") or ""),
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "wait",
                "x": None,
                "y": None,
                "element": "",
                "reasoning": f"Gemini call failed: {e}",
            }

    async def wait_until_ready(
        self,
        page,
        *,
        task: str,
        timeout_sec: float = 30.0,
        poll_sec: float = 2.0,
    ) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            result = await self.ask(page, task)
            status = str(result.get("status") or "loading")
            reason = str(result.get("reasoning") or "")
            print(f"👁️ wait status={status} reason={reason}")
            if status in ("ready", "done"):
                return True
            await asyncio.sleep(poll_sec)
        return False

    async def smart_click(
        self,
        page,
        *,
        task: str,
        selector: Optional[str] = None,
        timeout_ms: int = 2500,
        max_steps: int = 10,
        step_wait_ms: int = 1800,
    ) -> bool:
        if selector:
            try:
                loc = page.locator(selector).first
                await loc.wait_for(state="visible", timeout=timeout_ms)
                await loc.scroll_into_view_if_needed()
                await loc.click()
                return True
            except Exception:
                pass

        print(f"👁️ Handyman click task: {task}")
        for step in range(max_steps):
            result = await self.ask(page, task)
            status = str(result.get("status") or "")
            action = str(result.get("action") or "")
            x = result.get("x")
            y = result.get("y")
            element = str(result.get("element") or "")
            reasoning = str(result.get("reasoning") or "")
            print(
                f"   step {step+1}/{max_steps} | status={status} action={action} "
                f"coords=({x},{y}) element={element} reason={reasoning}"
            )

            if status == "done":
                return True

            if action == "click" and x is not None and y is not None:
                try:
                    await page.mouse.click(int(x), int(y))
                    await page.wait_for_timeout(1200)
                    return True
                except Exception:
                    pass

            if action == "scroll_down":
                try:
                    await page.mouse.wheel(0, 750)
                except Exception:
                    pass
                await page.wait_for_timeout(900)
                continue

            await page.wait_for_timeout(step_wait_ms)

        return False

    async def smart_fill(
        self,
        page,
        *,
        value: str,
        task: str,
        selector: Optional[str] = None,
        timeout_ms: int = 2500,
        max_steps: int = 6,
    ) -> bool:
        if selector:
            try:
                loc = page.locator(selector).first
                await loc.wait_for(state="visible", timeout=timeout_ms)
                await loc.click()
                try:
                    await loc.fill(value)
                except Exception:
                    await loc.evaluate(
                        """(el, value) => {
                            el.focus();
                            el.value = value;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        }""",
                        value,
                    )
                return True
            except Exception:
                pass

        clicked = await self.smart_click(page, task=task, max_steps=max_steps)
        if not clicked:
            return False

        try:
            await page.keyboard.press("Control+A")
        except Exception:
            pass
        try:
            await page.keyboard.press("Backspace")
        except Exception:
            pass
        await page.keyboard.type(value, delay=70)
        await page.wait_for_timeout(300)
        return True

    async def capture_download(
        self,
        page,
        *,
        click_task: str,
        save_path: str,
        selector: Optional[str] = None,
        max_steps: int = 12,
        download_wait_sec: float = 45.0,
    ) -> Optional[str]:
        downloaded = []

        async def _on_download(dl):
            try:
                fname = dl.suggested_filename
            except Exception:
                fname = "unknown"
            print(f"📥 Handyman download intercepted: {fname}")
            await dl.save_as(save_path)
            downloaded.append(save_path)

        page.on("download", _on_download)

        clicked = await self.smart_click(
            page,
            task=click_task,
            selector=selector,
            max_steps=max_steps,
        )
        if not clicked:
            return None

        deadline = time.time() + download_wait_sec
        while time.time() < deadline:
            if downloaded:
                return downloaded[0]
            await asyncio.sleep(1.0)
        return None
