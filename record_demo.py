"""
Record a demo video of the NeuroQA Copilot app (~40 seconds).
Uses Playwright for browser automation + imageio for MP4 encoding.
"""

import time
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
import imageio.v3 as iio

FRAMES = []
FPS = 5
RES = (1408, 896)


def hold(page, seconds=3.0, pause_before=1.5):
    """Capture current page state and hold for given seconds."""
    time.sleep(pause_before)
    png = page.screenshot(full_page=False)
    img = Image.open(__import__("io").BytesIO(png)).convert("RGB").resize(RES, Image.LANCZOS)
    n = int(seconds * FPS)
    for _ in range(n):
        FRAMES.append(np.array(img))


def smooth_scroll(page, from_y, to_y, steps=10, hold_sec=2.0):
    """Smooth scroll with intermediate frames."""
    for i in range(steps + 1):
        y = from_y + (to_y - from_y) * i / steps
        page.evaluate(f"window.scrollTo(0, {y})")
        time.sleep(0.15)
    hold(page, hold_sec, pause_before=0.3)


def click_tab(page, text, wait=3.0):
    """Click a nav button and wait for content to load."""
    buttons = page.locator(f'button:has-text("{text}")')
    if buttons.count() > 0:
        buttons.first.click()
        time.sleep(wait)


def main():
    url = "http://localhost:8501"
    output = "screenshots/neuroqa_demo.mp4"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        print("Loading app...")
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(5)

        # ── Scene 1: Patient Queue (default) ──
        print("  Scene 1: Patient Queue")
        hold(page, 4.0)
        smooth_scroll(page, 0, 400, hold_sec=3.0)

        # ── Scene 2: Select a patient and click Review ──
        print("  Scene 2: Patient Review")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        click_tab(page, "Patient Review", wait=3.0)
        hold(page, 4.0)
        smooth_scroll(page, 0, 400, hold_sec=3.0)
        smooth_scroll(page, 400, 900, hold_sec=3.0)

        # ── Scene 3: Group Statistics ──
        print("  Scene 3: Group Statistics")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        click_tab(page, "Group Statistics", wait=3.0)
        hold(page, 3.0)
        smooth_scroll(page, 0, 500, hold_sec=3.0)
        smooth_scroll(page, 500, 1000, hold_sec=3.0)
        smooth_scroll(page, 1000, 1500, hold_sec=3.0)
        smooth_scroll(page, 1500, 2200, hold_sec=2.5)

        # ── Scene 4: Clinical Reference ──
        print("  Scene 4: Clinical Reference")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        click_tab(page, "Clinical Reference", wait=2.0)
        hold(page, 4.0)

        # ── Scene 5: Back to Patient Queue (closes the loop) ──
        print("  Scene 5: Back to Patient Queue")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        click_tab(page, "Patient Queue", wait=2.0)
        hold(page, 3.0)

        browser.close()

    # ── Encode video ──
    duration = len(FRAMES) / FPS
    print(f"\nEncoding {len(FRAMES)} frames ({duration:.0f}s at {FPS}fps)...")
    iio.imwrite(output, FRAMES, fps=FPS, codec="libx264",
                output_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"])
    import os
    size = os.path.getsize(output)
    print(f"Done! {output} ({size/1024/1024:.1f} MB, {duration:.0f}s)")


if __name__ == "__main__":
    main()
