import asyncio
import random
import logging
import time
import re
from typing import Optional, Dict, List, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import json
from datetime import datetime, timedelta
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linkedin_scraper_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LinkedIn Company Post Timestamp API",
    description="API to fetch the timestamp of the last post from LinkedIn company pages",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkedInScraperAPI:
    def __init__(self, headless: bool = True, slow_mo: int = 100):
        self._browser_checked = False
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = 45000  # Increased timeout
        self.max_retries = 3
        
        # More diverse user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
        ]
        
        # Updated and more robust selectors
        self.timestamp_selectors = [
            '[data-test-id="main-feed-activity-card_entity-lockup"] time',
            '.base-main-feed-card__entity-lockup time',
            'span.px-0\\.25 time',
            'span.leading-\\[1\\.33333\\] time',
            '.base-main-feed-card__entity-lockup .flex-col time',
            '.base-main-feed-card__entity-lockup .text-color-text-low-emphasis time',
            '.feed-shared-actor__sub-description time',
            '.update-components-actor__meta time',
            '.feed-shared-update-v2__description time',
            '.feed-shared-actor__meta time',
            '.org-recent-activity-card time',
            '.org-recent-activity time',
            '.activity-card time',
            '[data-test-id="main-feed-activity-card"] time',
            '.share-update-card time',
            '.feed-shared-update-v2 time',
            '.org-page-recent-updates time',
            '.update-components-text time',
            'time[datetime]',
            '.time-ago',
            '.posted-time',
            '.update-time',
            '[data-time]',
            '.feed-shared-text time',
            '.feed-shared-header time',
            '.activity-card__meta time'
        ]

        # Enhanced patterns for text-based timestamps
        self.time_patterns = [
            r'(\d+)\s*(?:second|seconds|sec|s)\b',
            r'(\d+)\s*(?:month|months|mo)\b',
            r'(\d+)\s*(?:minute|minutes|min|m)\b', 
            r'(\d+)\s*(?:hour|hours|hr|h)\b',
            r'(\d+)\s*(?:day|days|d)\b',
            r'(\d+)\s*(?:week|weeks|w)\b',
            r'(\d+)\s*(?:year|years|y)\b',
            r'(?:just now|now)',
            r'(?:yesterday)',
            r'(?:today)',
            r'(?:last week)',
            r'(?:last month)'
        ]

    async def _ensure_browser(self):
            if not self._browser_checked:
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        await browser.close()
                    self._browser_checked = True
                except Exception as e:
                    raise RuntimeError(f"Browser setup failed: {e}")

    async def scrape_company(self, company_url: str) -> Optional[str]:
        await self._ensure_browser()
        """Scrape a single company URL"""
        company_name = self._extract_company_name(company_url)
        if not company_name:
            raise ValueError(f"Invalid LinkedIn company URL: {company_url}")
            
        for attempt in range(self.max_retries):
            logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries} for company: {company_name}")
            
            async with async_playwright() as p:
                browser = await self._launch_browser(p)
                context = await self._create_context(browser)
                page = await context.new_page()
                
                try:
                    # Try different URL formats
                    url = f"https://www.linkedin.com/company/{company_name}/"
                    logger.info(f"🌐 Trying URL: {url}")
                    
                    # Navigate with better error handling
                    await self._navigate_with_retry(page, url)
                    
                    # Enhanced popup handling
                    await self._handle_all_popups_comprehensive(page)
                    
                    # Better content loading
                    await self._load_content_strategically(page)
                    
                    # Comprehensive timestamp search
                    timestamp = await self._find_timestamp_advanced(page, company_name, attempt)
                    
                    if timestamp:
                        logger.info(f"✅ SUCCESS: Found timestamp for {company_name}: {timestamp}")
                        return self._standardize_timestamp(timestamp)
                    else:
                        logger.warning(f"⚠️ No timestamp found for {company_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {company_name}: {str(e)}")
                    continue
                    
                finally:
                    await browser.close()
            
            if attempt < self.max_retries - 1:
                delay = random.uniform(3, 8)
                logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
        
        logger.error(f"❌ FAILED: Could not get timestamp for {company_name} after all attempts")
        return None

    def _extract_company_name(self, url: str) -> Optional[str]:
        """Extract company name from LinkedIn URL"""
        patterns = [
            r"linkedin\.com/company/([^/]+)",
            r"linkedin\.com/company/([^/]+)/posts",
            r"linkedin\.com/company/([^/]+)/about",
            r"linkedin\.com/company/([^/]+)/feed"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def _launch_browser(self, playwright) -> Browser:
        """Launch browser with anti-detection measures"""
        logger.info("🚀 Launching browser...")
        
        return await playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--window-size=1920,1080',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',  # Speed up loading
                '--disable-javascript-harmony-shipping',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-restore-session-state',
                '--disable-ipc-flooding-protection'
            ]
        )

    async def _create_context(self, browser: Browser) -> BrowserContext:
        """Create context with realistic browser fingerprint"""
        user_agent = random.choice(self.user_agents)
        logger.info(f"🔧 Setting up context with User-Agent: {user_agent[:50]}...")
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
        )
        
        # Block heavy resources but allow essential ones
        await context.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,mp4,mp3,pdf}", lambda route: route.abort())
        await context.route("**/analytics**", lambda route: route.abort())
        await context.route("**/tracking**", lambda route: route.abort())
        await context.route("**/ads**", lambda route: route.abort())
        
        return context

    async def _navigate_with_retry(self, page: Page, url: str):
        """Navigate with better error handling and retries"""
        logger.info(f"🌐 Navigating to: {url}")
        
        # Add realistic delay
        await asyncio.sleep(random.uniform(2, 4))
        
        try:
            # Set referrer to look more natural
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.timeout,
                referer="https://www.google.com/"
            )
            
            # Wait for page to stabilize
            logger.info("⏳ Waiting for page to stabilize...")
            await page.wait_for_timeout(4000)
            
            # Check if page loaded correctly
            title = await page.title()
            logger.info(f"📄 Page title: {title}")
            
            # Check for error indicators
            if "error" in title.lower() or "not found" in title.lower():
                raise Exception(f"Page error detected: {title}")
                
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise

    async def _handle_all_popups_comprehensive(self, page: Page):
        """Comprehensive popup handling with detailed logging"""
        logger.info("🔍 Checking for popups and overlays...")
        
        # Wait for potential popups
        await page.wait_for_timeout(3000)
        
        popup_attempts = 0
        max_popup_attempts = 10
        
        popup_selectors = [
            'button[aria-label="Dismiss"]',
            'button[data-test-modal-close-btn]',
            'button[aria-label="Close"]',
            '.artdeco-modal__dismiss',
            '.sign-in-modal__dismiss-btn',
            '.cookie-consent__accept-button',
            '.contextual-sign-in-modal__modal-dismiss',
            '.join-now-modal__dismiss-btn',
            '.sign-in-modal__dismiss-btn',
            '.app-aware-link__dismiss',
            '.download-app-upsell__dismiss',
            '.overlay__dismiss',
            '.modal-overlay__dismiss',
            '[data-test-id="modal-close-btn"]'
        ]
        
        while popup_attempts < max_popup_attempts:
            popup_found = False
            
            for selector in popup_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        if element and await element.is_visible():
                            logger.info(f"🪟 Found and closing popup: {selector}")
                            await element.click()
                            await page.wait_for_timeout(2000)
                            popup_found = True
                            break
                    
                    if popup_found:
                        break
                        
                except Exception as e:
                    logger.debug(f"Popup handler error for {selector}: {e}")
                    continue
            
            if not popup_found:
                break
                
            popup_attempts += 1
            await page.wait_for_timeout(1000)
        
        logger.info(f"✅ Popup handling completed. Attempts made: {popup_attempts}")

    async def _load_content_strategically(self, page: Page):
        """Strategic content loading with detailed progress"""
        logger.info("📦 Loading content strategically...")
        
        try:
            # Wait for main content
            logger.info("⏳ Waiting for main content area...")
            await page.wait_for_selector('main, .application-outlet, body', timeout=15000)
            
            # Progressive scrolling to trigger lazy loading
            logger.info("📜 Progressive scrolling to load content...")
            scroll_steps = 5
            for i in range(scroll_steps):
                scroll_position = (i + 1) * 300
                await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                logger.info(f"📍 Scrolled to position: {scroll_position}")
                await page.wait_for_timeout(2000)
            
            # Scroll back to top
            logger.info("🔝 Scrolling back to top...")
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(3000)
            
            # Try to navigate to posts section
            await self._navigate_to_posts_section(page)
            
        except Exception as e:
            logger.warning(f"Content loading issues: {e}")

    async def _navigate_to_posts_section(self, page: Page):
        """Navigate to posts section with multiple strategies"""
        logger.info("🎯 Attempting to navigate to posts section...")
        
        posts_strategies = [
            'a[href*="/posts/"]',
            'button[data-control-name="page_posts"]',
            '.org-page-navigation__item[href*="posts"]',
            '[data-test-id="posts-tab"]',
            '.org-page-navigation-item--posts',
            'a[href$="/posts/"]',
            'a:has-text("Posts")',
            'button:has-text("Posts")',
            '.org-page-navigation a',
            '.artdeco-tabs__tab'
        ]
        
        for strategy in posts_strategies:
            try:
                elements = await page.query_selector_all(strategy)
                for element in elements:
                    if element and await element.is_visible():
                        text = await element.inner_text()
                        href = await element.get_attribute('href')
                        
                        logger.info(f"🔍 Found navigation element: '{text}' | href: {href}")
                        
                        if 'post' in text.lower() or (href and 'post' in href):
                            logger.info(f"🎯 Clicking on posts navigation: {strategy}")
                            await element.click()
                            await page.wait_for_timeout(4000)
                            return True
                            
            except Exception as e:
                logger.debug(f"Posts navigation error for {strategy}: {e}")
                continue
        
        logger.info("📝 No posts navigation found, continuing with current page...")
        return False

    async def _find_timestamp_advanced(self, page: Page, company_name: str, attempt: int) -> Optional[str]:
        """Advanced timestamp finding with comprehensive logging"""
        logger.info("🔍 Starting comprehensive timestamp search...")
        
        # Strategy 1: CSS Selectors
        logger.info("🎯 Strategy 1: CSS Selector search...")
        timestamp = await self._find_by_enhanced_selectors(page)
        if timestamp:
            logger.info(f"✅ Found via CSS selectors: {timestamp}")
            return timestamp
        
        # Strategy 2: Text Pattern Matching
        logger.info("🎯 Strategy 2: Text pattern matching...")
        timestamp = await self._find_by_enhanced_patterns(page)
        if timestamp:
            logger.info(f"✅ Found via text patterns: {timestamp}")
            return timestamp
        
        # Strategy 3: JavaScript Deep Search
        logger.info("🎯 Strategy 3: JavaScript deep search...")
        timestamp = await self._find_by_javascript_advanced(page)
        if timestamp:
            logger.info(f"✅ Found via JavaScript: {timestamp}")
            return timestamp
        
        # Strategy 4: DOM Tree Walking
        logger.info("🎯 Strategy 4: DOM tree walking...")
        timestamp = await self._find_by_dom_walking(page)
        if timestamp:
            logger.info(f"✅ Found via DOM walking: {timestamp}")
            return timestamp
        
        # Strategy 5: Content Area Analysis
        logger.info("🎯 Strategy 5: Content area analysis...")
        timestamp = await self._find_in_content_areas_advanced(page)
        if timestamp:
            logger.info(f"✅ Found via content analysis: {timestamp}")
            return timestamp
        
        logger.warning("❌ No timestamp found with any strategy")
        return None

    async def _find_by_enhanced_selectors(self, page: Page) -> Optional[str]:
        """Enhanced CSS selector search with detailed logging"""
        logger.info(f"🔍 Searching with {len(self.timestamp_selectors)} CSS selectors...")
        
        for i, selector in enumerate(self.timestamp_selectors):
            try:
                elements = await page.query_selector_all(selector)
                logger.info(f"📍 Selector {i+1}/{len(self.timestamp_selectors)}: {selector} -> {len(elements)} elements")
                
                for j, element in enumerate(elements):
                    if await element.is_visible():
                        # Try datetime attribute
                        datetime_attr = await element.get_attribute('datetime')
                        if datetime_attr:
                            logger.info(f"✅ Found datetime attribute: {datetime_attr}")
                            return datetime_attr
                        
                        # Try text content
                        text = await element.inner_text()
                        if text and self._is_valid_timestamp_text(text):
                            logger.info(f"✅ Found timestamp text: {text}")
                            return text.strip()
                        
                        logger.debug(f"Element {j+1} text: '{text}'")
                        
            except Exception as e:
                logger.debug(f"Selector error {selector}: {e}")
                continue
        
        logger.info("❌ No timestamps found via CSS selectors")
        return None

    async def _find_by_enhanced_patterns(self, page: Page) -> Optional[str]:
        """Enhanced text pattern matching"""
        logger.info("🔍 Searching page text for timestamp patterns...")
        
        try:
            # Get page text
            page_text = await page.inner_text('body')
            logger.info(f"📄 Page text length: {len(page_text)} characters")
            
            # Search for patterns
            for i, pattern in enumerate(self.time_patterns):
                # Use finditer to get match objects with start/end positions
                # and the full matched string.
                for match_obj in re.finditer(pattern, page_text, re.IGNORECASE):
                    # The full matched string is match_obj.group(0)
                    full_match = match_obj.group(0).lower()
                    
                    # Check for specific patterns and return formatted output
                    if 'second' in full_match or 'sec' in full_match or 's' in full_match:
                        return f"{match_obj.group(1)}s"
                    elif 'month' in full_match or 'mo' in full_match:
                        return f"{match_obj.group(1)}mo"
                    elif 'minute' in full_match or 'min' in full_match or 'm' in full_match:
                        return f"{match_obj.group(1)}m"
                    elif 'hour' in full_match or 'hr' in full_match or 'h' in full_match:
                        return f"{match_obj.group(1)}h"
                    elif 'day' in full_match or 'd' in full_match:
                        return f"{match_obj.group(1)}d"
                    elif 'week' in full_match or 'w' in full_match:
                        return f"{match_obj.group(1)}w"
                    elif 'year' in full_match or 'y' in full_match:
                        return f"{match_obj.group(1)}y"
                    
                    # For non-numeric patterns, return the text directly
                    if 'just now' in full_match or 'now' in full_match:
                        return 'just now'
                    if 'yesterday' in full_match:
                        return 'yesterday'
                    # ... add other non-numeric patterns here if needed
                    
                    # Fallback for patterns that don't need formatting
                    return full_match # This will return the raw matched string
                    
                # If no matches were found for the current pattern, continue to the next one
                # and log a success message for the match type.
                if re.search(pattern, page_text, re.IGNORECASE):
                    logger.info(f"✅ Pattern {i+1} matched: {pattern} -> (Found but not a specific numeric type)")
                    # This log is not ideal, you need to return something here if you want to stop.
                    # The 'return full_match' above will handle the first match.
                    break # Break out of the for loop after the first successful pattern match

        except Exception as e:
            logger.error(f"Text pattern search error: {e}")
        
        logger.info("❌ No timestamps found via text patterns")
        return None

    async def _find_by_javascript_advanced(self, page: Page) -> Optional[str]:
        """Advanced JavaScript timestamp search"""
        logger.info("🔍 Running advanced JavaScript timestamp search...")
        
        try:
            result = await page.evaluate('''() => {
                const results = [];
                
                // Strategy 1: Time elements with datetime
                const timeElements = document.querySelectorAll('time[datetime]');
                results.push(`Time elements found: ${timeElements.length}`);
                
                for (let el of timeElements) {
                    if (el.offsetParent !== null) {
                        const datetime = el.getAttribute('datetime');
                        const text = el.textContent.trim();
                        results.push(`Visible time element: datetime="${datetime}", text="${text}"`);
                        if (datetime) return { type: 'datetime', value: datetime };
                        if (text) return { type: 'text', value: text };
                    }
                }
                
                // Strategy 2: Elements with time-related classes
                const timeClasses = [
                    '.time-ago', '.posted-time', '.update-time', 
                    '.feed-shared-actor__sub-description',
                    '.org-recent-activity-card__meta',
                    '.activity-card__meta'
                ];
                
                for (let className of timeClasses) {
                    const els = document.querySelectorAll(className);
                    results.push(`${className}: ${els.length} elements`);
                    
                    for (let el of els) {
                        if (el.offsetParent !== null) {
                            const text = el.textContent.trim();
                            if (text && /\\b\\d+\\s*(second|minute|hour|day|week|month|year)s?\\s*ago\\b/i.test(text)) {
                                results.push(`Found via class ${className}: "${text}"`);
                                return { type: 'class', value: text };
                            }
                        }
                    }
                }
                
                // Strategy 3: Text node walking
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                
                const timeRegex = /\\b\\d+\\s*(second|minute|hour|day|week|month|year)s?\\s*ago\\b/i;
                let textNodesChecked = 0;
                let node;
                
                while (node = walker.nextNode() && textNodesChecked < 1000) {
                    textNodesChecked++;
                    if (timeRegex.test(node.textContent)) {
                        const match = node.textContent.match(timeRegex);
                        if (match) {
                            results.push(`Found via text walking: "${match[0]}"`);
                            return { type: 'textwalk', value: match[0] };
                        }
                    }
                }
                
                results.push(`Text nodes checked: ${textNodesChecked}`);
                
                return { type: 'debug', results: results };
            }''')
            
            if result:
                if result.get('type') in ['datetime', 'text', 'class', 'textwalk']:
                    logger.info(f"✅ JavaScript found timestamp: {result['value']}")
                    return result['value']
                elif result.get('type') == 'debug':
                    logger.info("📊 JavaScript debug info:")
                    for debug_line in result.get('results', []):
                        logger.info(f"  - {debug_line}")
            
        except Exception as e:
            logger.error(f"JavaScript search error: {e}")
        
        logger.info("❌ No timestamps found via JavaScript")
        return None

    async def _find_by_dom_walking(self, page: Page) -> Optional[str]:
        """DOM tree walking for timestamp discovery"""
        logger.info("🔍 Walking DOM tree for timestamps...")
        
        try:
            # Get all elements and check for timestamp-like content
            elements_info = await page.evaluate('''() => {
                const allElements = document.querySelectorAll('*');
                const timeElements = [];
                
                for (let el of allElements) {
                    if (el.offsetParent !== null) {  // Visible element
                        const text = el.textContent.trim();
                        const classes = el.className;
                        const id = el.id;
                        
                        // Check for time-related attributes or content
                        if (text && text.length < 50 && 
                            (/\\b\\d+\\s*(second|minute|hour|day|week|month|year)s?\\s*ago\\b/i.test(text) ||
                             /\\b(yesterday|today|just now)\\b/i.test(text))) {
                            
                            timeElements.push({
                                tag: el.tagName,
                                text: text,
                                classes: classes,
                                id: id
                            });
                        }
                    }
                }
                
                return {
                    total: allElements.length,
                    timeElements: timeElements.slice(0, 10)  // First 10 matches
                };
            }''')
            
            logger.info(f"📊 DOM Analysis: {elements_info['total']} total elements")
            logger.info(f"📊 Time elements found: {len(elements_info['timeElements'])}")
            
            for elem in elements_info['timeElements']:
                logger.info(f"  - {elem['tag']}: '{elem['text']}' (class: {elem['classes']})")
                if self._is_valid_timestamp_text(elem['text']):
                    logger.info(f"✅ Valid timestamp found: {elem['text']}")
                    return elem['text']
                    
        except Exception as e:
            logger.error(f"DOM walking error: {e}")
        
        logger.info("❌ No timestamps found via DOM walking")
        return None

    async def _find_in_content_areas_advanced(self, page: Page) -> Optional[str]:
        """Advanced content area analysis"""
        logger.info("🔍 Analyzing specific content areas...")
        
        content_areas = [
            ('main', 'Main content area'),
            ('.application-outlet', 'Application outlet'),
            ('.feed-container', 'Feed container'),
            ('.org-page-content', 'Organization page content'),
            ('[data-module-result-urn]', 'Module result'),
            ('.org-recent-activity', 'Recent activity'),
            ('.activity-card', 'Activity card'),
            ('.feed-shared-update-v2', 'Feed update')
        ]
        
        for selector, description in content_areas:
            try:
                logger.info(f"🔍 Checking {description}: {selector}")
                area = await page.query_selector(selector)
                
                if area:
                    # Count elements in this area
                    element_count = await area.evaluate('el => el.querySelectorAll("*").length')
                    logger.info(f"  📊 Area has {element_count} child elements")
                    
                    # Look for time elements
                    time_elements = await area.query_selector_all('time, .time-ago, [class*="time"], [class*="ago"]')
                    logger.info(f"  📊 Found {len(time_elements)} potential time elements")
                    
                    for i, el in enumerate(time_elements):
                        if await el.is_visible():
                            datetime_attr = await el.get_attribute('datetime')
                            text = await el.inner_text()
                            
                            logger.info(f"    Element {i+1}: datetime='{datetime_attr}', text='{text}'")
                            
                            if datetime_attr:
                                logger.info(f"✅ Found in {description}: {datetime_attr}")
                                return datetime_attr
                            elif text and self._is_valid_timestamp_text(text):
                                logger.info(f"✅ Found in {description}: {text}")
                                return text.strip()
                else:
                    logger.info(f"  ❌ Area not found: {selector}")
                    
            except Exception as e:
                logger.debug(f"Content area error {selector}: {e}")
                continue
        
        logger.info("❌ No timestamps found in content areas")
        return None

    def _is_valid_timestamp_text(self, text: str) -> bool:
        """
        Enhanced timestamp validation to be more robust for LinkedIn formats.
        It checks if the text contains a valid timestamp pattern, ignoring surrounding text like 'Edited'.
        """
        if not text:
            return False
            
        text = text.lower().strip()
        
        # Simple length check to filter out irrelevant long strings
        if len(text) > 100 or len(text) < 2:
            return False
        
        # Comprehensive timestamp patterns including abbreviations and full words
        # The 'edited' pattern is made optional to allow "2h" and "2h Edited" to pass
        patterns = [
            # Numeric patterns with optional 'ago' or 'edited' suffixes
            r'(\d+)\s*(?:second|minute|hour|day|week|month|year|sec|min|hr|h|d|w|mo|y)s?\b',
            # Non-numeric patterns
            r'just now',
            r'now',
            r'yesterday',
            r'today',
            r'last week',
            r'last month',
            # Handles timestamps with a surrounding word like 'edited'
            r'\b(?:edited)\b', 
        ]
        
        # Join patterns with OR operator
        combined_pattern = '|'.join(patterns)
        
        # The `re.search` function checks for a match anywhere in the string.
        # This is perfect for a string like '2h Edited'.
        if re.search(combined_pattern, text, re.IGNORECASE):
            # We found a valid pattern. Now, let's make sure it's not just a random word.
            # This is a bit of a sanity check to ensure the length is reasonable.
            return True
        
        return False

    def _standardize_timestamp(self, timestamp: str) -> str:
        """Enhanced timestamp standardization"""
        if not timestamp:
            return ""
        
        timestamp = timestamp.strip()
        
        # Handle ISO datetime format
        try:
            if 'T' in timestamp and ('Z' in timestamp or '+' in timestamp or '-' in timestamp):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                now = datetime.now()
                
                # Handle timezone-aware datetime
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                
                diff = now - dt
                
                if diff.days > 0:
                    return f"{diff.days}d ago"
                elif diff.seconds >= 3600:
                    hours = diff.seconds // 3600
                    return f"{hours}h ago"
                elif diff.seconds >= 60:
                    minutes = diff.seconds // 60
                    return f"{minutes}m ago"
                else:
                    return "just now"
        except Exception as e:
            logger.debug(f"Datetime parsing error: {e}")
        
        # Standardize relative timestamps
        timestamp = timestamp.lower()
        
        # Handle special cases
        special_cases = {
            'just now': 'just now',
            'now': 'just now',
            'today': 'today',
            'yesterday': '1d ago',
            'last week': '1w ago',
            'last month': '1mo ago'
        }
        
        for key, value in special_cases.items():
            if key in timestamp:
                return value
        
        # Standardize time units
        replacements = {
            'seconds': 's', 'second': 's', 'sec': 's',
            'minutes': 'm', 'minute': 'm', 'min': 'm',
            'hours': 'h', 'hour': 'h', 'hr': 'h',
            'days': 'd', 'day': 'd',
            'weeks': 'w', 'week': 'w',
            'months': 'mo', 'month': 'mo',
            'years': 'y', 'year': 'y'
        }
        
        for old, new in replacements.items():
            timestamp = re.sub(f'\\b{old}\\b', new, timestamp)
        
        # Clean up
        timestamp = re.sub(r'\s*ago\s*', '', timestamp).strip()
        timestamp = re.sub(r'\s+', ' ', timestamp)
        
        return timestamp

# Initialize the scraper
scraper = LinkedInScraperAPI(headless=True, slow_mo=100)

@app.get("/")
async def read_root():
    return {
        "message": "LinkedIn Company Post Timestamp API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/scrape")
async def scrape_company(
    url: str = Query(..., description="LinkedIn company profile URL (e.g., https://www.linkedin.com/company/icreatenextgen/)")
):
    try:
        timestamp = await scraper.scrape_company(url)
        return {
            "company_url": url,
            "last_post_timestamp": timestamp,
            "status": "success" if timestamp else "no_post_found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-scrape")
async def batch_scrape_companies(
    urls: List[str] = Query(..., description="List of LinkedIn company profile URLs")
):
    try:
        results = {}
        for url in urls:
            timestamp = await scraper.scrape_company(url)
            results[url] = timestamp
            # Add delay between requests to avoid rate limiting
            await asyncio.sleep(random.uniform(5, 10))
        
        return {
            "results": results,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)