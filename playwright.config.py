import os
from playwright.async_api import async_playwright

async def ensure_browser_installed():
    async with async_playwright() as p:
        await p.chromium.launch()