import asyncio
import random
import logging
import time
import re
import json
from typing import Optional, Dict, List, Tuple, Any
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path

# Configure logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linkedin_scraper_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedLinkedInScraper:
    def __init__(self, headless: bool = False, slow_mo: int = 100):
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = 45000
        self.max_retries = 3
        self.max_posts = 3
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.debug_mode = False
        
        # More diverse user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ]
        
        # Enhanced post container selectors
        self.post_selectors = [
            'article.main-feed-activity-card',
            'div[data-id^="urn:li:activity"]',  # Primary LinkedIn post identifier
            '.feed-shared-update-v2',
            '.occludable-update',
            '.base-main-card',
            '.feed-shared-update-v2__description-wrapper',
            'div.update-components-text',
            '.share-update-card'
        ]
        
        # Text content selectors (multiple fallbacks)
        self.text_selectors = [
            '.update-components-text .feed-shared-text',
            '.update-components-text .break-words',
            '.feed-shared-text .break-words',
            '.feed-shared-text',
            '.update-components-text__text',
            'div[data-test-id="main-feed-activity-card__commentary"]',
            '.feed-shared-update-v2__commentary .break-words',
            '.feed-shared-update-v2__commentary',
            '.base-main-card__commentary .break-words',
            '.base-main-card__commentary',
            'span.break-words',
            '.update-components-text > div',
            '.feed-shared-text > span',
            '.attributed-text-segment-list__content'
        ]
        
        # Enhanced timestamp selectors
        self.timestamp_selectors = [
            'a.app-aware-link time',
            '.feed-shared-actor__sub-description time',
            '.update-components-actor__meta time',
            '.feed-shared-update-v2__description time',
            '.feed-shared-actor__meta time',
            'time[datetime]',
            '.feed-shared-actor__sub-description a',
            '.update-components-actor__meta .visually-hidden',
            'span.visually-hidden',
            '.feed-shared-actor__sub-description span',
            '.update-components-actor__sub-description'
        ]
        
        # Link/URL selectors
        self.link_selectors = [
            'a[data-id="main-feed-card__full-link"]',
            'a.main-feed-card__overlay-link',
            'a[href*="/posts/"][aria-label^="Update "]',
            'div[data-id="entire-feed-card-link"] > a'
        ]

    async def scrape_company_posts(self, company_url: str) -> List[Dict[str, Any]]:
        """Scrape latest posts from a company page"""
        logger.info(f"🔄 Starting scrape for: {company_url}")
        
        async with async_playwright() as p:
            browser = await self._launch_browser(p)
            context = await self._create_context(browser)
            page = await context.new_page()
            
            try:
                await self._navigate_with_retry(page, company_url)
                await self._handle_all_popups_comprehensive(page)
                
                # Navigate to posts section first
                # await self._navigate_to_posts_section(page)
                
                # Load content strategically
                await self._load_content_strategically(page)
                
                # Extract posts with retry mechanism
                posts = await self._extract_posts_with_retry(page)
                
                if not posts:
                    logger.warning(f"No posts found for {company_url}")
                    await self._save_comprehensive_debug(page, "failed_scrape", 0)
                    return []
                
                # Get top 3 most recent posts
                recent_posts = sorted(posts, key=lambda x: self._timestamp_to_seconds(x.get('timestamp', '')), reverse=False)[:self.max_posts]
                logger.info(f"✅ Successfully extracted {len(recent_posts)} posts")
                return recent_posts
                
            except Exception as e:
                logger.error(f"Error scraping {company_url}: {str(e)}")
                await self._save_comprehensive_debug(page, "error_scrape", 0)
                return []
            finally:
                await browser.close()

    async def _extract_posts_with_retry(self, page: Page) -> List[Dict[str, Any]]:
        """Extract posts with multiple retry strategies"""
        for attempt in range(3):
            try:
                logger.info(f"📝 Post extraction attempt {attempt + 1}")
                posts = await self._extract_posts(page)
                if posts:
                    return posts
                
                # If no posts found, try scrolling and waiting
                if attempt < 2:
                    logger.info("🔄 No posts found, trying scroll and reload strategy...")
                    await self._aggressive_content_loading(page)
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Post extraction attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(5)
                continue
        
        return []

    async def _extract_posts(self, page: Page) -> List[Dict[str, Any]]:
        """Enhanced post extraction with better selectors"""
        posts = []
        
        # Wait for page to load completely
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
        except:
            logger.warning("Page didn't reach networkidle state, continuing anyway")
        
        # CHANGE 1: Look for the parent containers that contain both the URL and the post content
        parent_containers = []
        try:
            # First, try to find the parent containers that have both the link and the article
            containers = await page.query_selector_all('div[data-id="entire-feed-card-link"]')
            if containers:
                logger.info(f"Found {len(containers)} parent containers with links")
                parent_containers = containers
        except Exception as e:
            logger.debug(f"Parent container selector failed: {e}")
        
        # Fallback to finding articles directly if parent containers not found
        if not parent_containers:
            try:
                articles = await page.query_selector_all('article.main-feed-activity-card')
                if articles:
                    logger.info(f"Found {len(articles)} article elements")
                    parent_containers = articles
            except Exception as e:
                logger.debug(f"Article selector failed: {e}")
        
        if not parent_containers:
            logger.error("No post containers found at all")
            return []
        
        logger.info(f"Processing {len(parent_containers)} potential post containers")
        
        # Process each container
        for i, container in enumerate(parent_containers[:self.max_posts * 3]):
            try:
                # Scroll container into view
                try:
                    await container.scroll_into_view_if_needed()
                    await asyncio.sleep(0.5)
                except:
                    logger.debug(f"Could not scroll container {i+1} into view")
                
                # Check if container is visible
                try:
                    is_visible = await container.is_visible()
                    if not is_visible:
                        logger.debug(f"Container {i+1} not visible, skipping")
                        continue
                except:
                    logger.debug(f"Could not check visibility of container {i+1}")
                
                # CHANGE 2: Extract URL from the container first
                post_url = await self._extract_post_url_from_container(container)
                
                # CHANGE 3: Find the article element within the container for content extraction
                article_element = container
                try:
                    # If this is a parent container, find the article inside it
                    inner_article = await container.query_selector('article.main-feed-activity-card')
                    if inner_article:
                        article_element = inner_article
                except:
                    pass
                
                # Extract post data from the article element
                post_data = await self._extract_single_post_with_url(article_element, post_url, i+1)
                
                if post_data and post_data.get('text') and len(post_data.get('text', '').strip()) > 10:
                    posts.append(post_data)
                    logger.info(f"✅ Extracted post {len(posts)}: {post_data.get('url', 'No URL')} - {post_data['timestamp']} - {post_data['text'][:50]}...")
                    
                    if len(posts) >= self.max_posts:
                        break
                else:
                    logger.debug(f"Skipping container {i+1} - invalid or empty data")
                    
            except Exception as e:
                logger.warning(f"Error processing container {i+1}: {e}")
                continue
        
        return posts

    async def _extract_post_url_from_container(self, container) -> Optional[str]:
        """Extract URL from the parent container - NEW METHOD"""
        try:
            # Method 1: Look for the overlay link in the container
            overlay_link = await container.query_selector('a.main-feed-card__overlay-link')
            if overlay_link:
                href = await overlay_link.get_attribute('href')
                if href and '/posts/' in href:
                    return href.split('?')[0]
            
            # Method 2: Look for the full link
            full_link = await container.query_selector('a[data-id="main-feed-card__full-link"]')
            if full_link:
                href = await full_link.get_attribute('href')
                if href and ('/posts/' in href or '/activity/' in href):
                    return href.split('?')[0]
            
            # Method 3: Look for any link with posts or activity
            links = await container.query_selector_all('a[href*="/posts/"], a[href*="/activity/"]')
            for link in links:
                href = await link.get_attribute('href')
                if href:
                    if href.startswith('/'):
                        return f"https://www.linkedin.com{href.split('?')[0]}"
                    else:
                        return href.split('?')[0]
            
            # Method 4: Check if the container itself has a data-id with activity info
            data_id = await container.get_attribute('data-id')
            if data_id == "entire-feed-card-link":
                # This confirms it's the right container, but we need to find the actual link
                all_links = await container.query_selector_all('a')
                for link in all_links:
                    href = await link.get_attribute('href')
                    if href and ('/posts/' in href or '/activity/' in href):
                        if href.startswith('/'):
                            return f"https://www.linkedin.com{href.split('?')[0]}"
                        else:
                            return href.split('?')[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"URL extraction from container error: {e}")
            return None

    async def _extract_single_post_with_url(self, post_element, post_url: str, post_num: int) -> Optional[Dict[str, Any]]:
        """Extract data from a single post element with pre-extracted URL - MODIFIED METHOD"""
        try:
            # Extract text content first
            text = await self._extract_text_enhanced(post_element)
            if not text or len(text.strip()) < 10:
                logger.debug(f"Post {post_num}: No meaningful text found")
                return None
                
            # Extract timestamp
            timestamp = await self._extract_timestamp_enhanced(post_element)
            if not timestamp:
                logger.debug(f"Post {post_num}: No valid timestamp found, using default")
                timestamp = "unknown"
                
            post_data = {
                'url': post_url,  # Use the pre-extracted URL
                'timestamp': timestamp,
                'text': text,
            }
            
            logger.debug(f"Post {post_num} extracted data: url={post_url}, timestamp={timestamp}, text_length={len(text)}")
            return post_data
            
        except Exception as e:
            logger.error(f"Error extracting single post {post_num}: {e}")
            return None

    async def _extract_single_post(self, post_element, post_num: int) -> Optional[Dict[str, Any]]:
            """Extract data from a single post element"""
            try:
                # Extract text content first
                text = await self._extract_text_enhanced(post_element)
                if not text or len(text.strip()) < 10:
                    logger.debug(f"Post {post_num}: No meaningful text found")
                    return None
                    
                # Extract timestamp
                timestamp = await self._extract_timestamp_enhanced(post_element)
                if not timestamp:
                    logger.debug(f"Post {post_num}: No valid timestamp found, using default")
                    timestamp = "unknown"
                    
                # Extract post URL
                url = await self._extract_post_url_enhanced(post_element)
                
                post_data = {
                    'url': url,
                    'timestamp': timestamp,
                    'text': text,
                }
                
                logger.debug(f"Post {post_num} extracted data: url={url}, timestamp={timestamp}, text_length={len(text)}")
                return post_data
                
            except Exception as e:
                logger.error(f"Error extracting single post {post_num}: {e}")
                return None

    async def _extract_text_enhanced(self, post_element) -> str:
        """Enhanced text extraction with multiple fallback strategies"""
        try:
            # Strategy 1: Try specific text selectors
            for selector in self.text_selectors:
                try:
                    text_element = await post_element.query_selector(selector)
                    if text_element:
                        text = await text_element.inner_text()
                        if text and len(text.strip()) > 10:  # Ensure meaningful text
                            logger.debug(f"Text found with selector: {selector}")
                            return text.strip()
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            # Strategy 2: Look for any text content in the post but filter out navigation/metadata
            try:
                # Get all text from the post
                full_text = await post_element.inner_text()
                if full_text:
                    # Split by lines and filter out short lines (likely metadata)
                    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                    content_lines = [line for line in lines if len(line) > 20 and not self._is_metadata_line(line)]
                    
                    if content_lines:
                        return '\n'.join(content_lines[:3])  # Take first 3 meaningful lines
            except Exception as e:
                logger.debug(f"Full text extraction failed: {e}")
            
            # Strategy 3: Look for specific text patterns in spans/divs
            try:
                # Look for spans or divs that contain substantial text
                text_elements = await post_element.query_selector_all('span, div, p')
                for element in text_elements:
                    try:
                        text = await element.inner_text()
                        if text and len(text.strip()) > 20 and not self._is_metadata_line(text):
                            return text.strip()
                    except:
                        continue
            except Exception as e:
                logger.debug(f"Element text extraction failed: {e}")
            
            # Strategy 4: Try to get text from specific LinkedIn content areas
            try:
                # Look for common LinkedIn post content classes
                content_selectors = [
                    '.feed-shared-text',
                    '.attributed-text-segment-list__content',
                    '.update-components-text',
                    '[data-test-id="main-feed-activity-card__commentary"]'
                ]
                
                for selector in content_selectors:
                    try:
                        element = await post_element.query_selector(selector)
                        if element:
                            text = await element.inner_text()
                            if text and len(text.strip()) > 5:
                                return text.strip()
                    except:
                        continue
            except Exception as e:
                logger.debug(f"LinkedIn content extraction failed: {e}")
            
            logger.debug("No meaningful text content found")
            return ""
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""

    def _is_metadata_line(self, line: str) -> bool:
        """Check if a line is likely metadata rather than post content"""
        line = line.lower().strip()
        
        # Skip very short lines
        if len(line) < 10:
            return True
        
        metadata_indicators = [
            'like', 'comment', 'share', 'follow', 'connect', 'view profile',
            'ago', 'hour', 'day', 'week', 'month', 'year', 'just now',
            'view', 'profile', 'company', 'job', 'hiring', 'see translation',
            'linkedin', 'see more', 'show more', 'show less', 'read more',
            'repost', 'celebrate', 'love', 'insightful', 'curious',
            'reactions', 'comments', 'reposts', 'send', 'save',
            'report', 'copy link', 'embed', 'send via',
            'followers', 'connections', 'mutual', 'premium'
        ]
        
        # Lines that are mostly metadata indicators
        words = line.split()
        if len(words) <= 4:
            metadata_word_count = sum(1 for word in words if any(indicator in word for indicator in metadata_indicators))
            if metadata_word_count >= len(words) * 0.7:  # 70% or more are metadata words
                return True
        
        # Skip lines that are just numbers (like counts)
        if line.replace(',', '').replace('.', '').isdigit():
            return True
        
        # Skip lines that look like timestamps
        if any(time_word in line for time_word in ['ago', 'hour', 'day', 'week', 'month', 'year']) and len(line) < 20:
            return True
        
        return False

    async def _extract_timestamp_enhanced(self, post_element) -> Optional[str]:
        """Enhanced timestamp extraction"""
        try:
            # Strategy 1: Look for time elements with datetime attribute
            time_elements = await post_element.query_selector_all('time[datetime]')
            for time_el in time_elements:
                datetime_attr = await time_el.get_attribute('datetime')
                if datetime_attr:
                    return self._standardize_timestamp(datetime_attr)
            
            # Strategy 2: Try specific timestamp selectors
            for selector in self.timestamp_selectors:
                try:
                    element = await post_element.query_selector(selector)
                    if element:
                        # Check for datetime attribute first
                        datetime_attr = await element.get_attribute('datetime')
                        if datetime_attr:
                            return self._standardize_timestamp(datetime_attr)
                        
                        # Check inner text
                        text = await element.inner_text()
                        if text and self._is_valid_timestamp_text(text):
                            return self._standardize_timestamp(text)
                except:
                    continue
            
            # Strategy 3: Look for timestamp patterns in all text
            all_text = await post_element.inner_text()
            if all_text:
                lines = all_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if self._is_valid_timestamp_text(line):
                        return self._standardize_timestamp(line)
            
            return None
            
        except Exception as e:
            logger.debug(f"Timestamp extraction error: {e}")
            return None

    async def _extract_post_url_enhanced(self, post_element) -> Optional[str]:
        """Enhanced URL extraction"""
        try:
            # Primary selector based on your HTML structure - most common pattern
            try:
                overlay_link = await post_element.query_selector('a.main-feed-card__overlay-link')
                if overlay_link:
                    href = await overlay_link.get_attribute('href')
                    if href and '/posts/' in href:
                        return href.split('?')[0]
            except:
                pass
            
            # Try other link selectors
            for selector in self.link_selectors:
                try:
                    link = await post_element.query_selector(selector)
                    if link:
                        href = await link.get_attribute('href')
                        if href and '/posts/' in href:
                            if href.startswith('/'):
                                return f"https://www.linkedin.com{href.split('?')[0]}"
                            else:
                                return href.split('?')[0]
                except:
                    continue
            
            # Look for any link containing posts
            try:
                links = await post_element.query_selector_all('a[href*="/posts/"]')
                for link in links:
                    href = await link.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            return f"https://www.linkedin.com{href.split('?')[0]}"
                        else:
                            return href.split('?')[0]
            except:
                pass
            
            # Fallback: look for any link with activity in the href
            try:
                links = await post_element.query_selector_all('a[href*="activity"]')
                for link in links:
                    href = await link.get_attribute('href')
                    if href and 'activity' in href:
                        if href.startswith('/'):
                            return f"https://www.linkedin.com{href.split('?')[0]}"
                        else:
                            return href.split('?')[0]
            except:
                pass
            
            # Final fallback: look for data-id attribute that might contain the post link
            try:
                link_container = await post_element.query_selector('div[data-id="entire-feed-card-link"]')
                if link_container:
                    link = await link_container.query_selector('a')
                    if link:
                        href = await link.get_attribute('href')
                        if href:
                            if href.startswith('/'):
                                return f"https://www.linkedin.com{href.split('?')[0]}"
                            else:
                                return href.split('?')[0]
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"URL extraction error: {e}")
            return None

    def _timestamp_to_seconds(self, timestamp: str) -> int:
        """Convert timestamp to seconds for sorting (recent = lower number)"""
        if not timestamp:
            return 999999
        
        timestamp = timestamp.lower().strip()
        
        # Parse time units
        if 'just now' in timestamp or timestamp == 'now':
            return 0
        elif 's' in timestamp or 'second' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) if re.findall(r'\d+', timestamp) else 60
        elif 'm' in timestamp or 'minute' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 60 if re.findall(r'\d+', timestamp) else 3600
        elif 'h' in timestamp or 'hour' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 3600 if re.findall(r'\d+', timestamp) else 86400
        elif 'd' in timestamp or 'day' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 86400 if re.findall(r'\d+', timestamp) else 604800
        elif 'w' in timestamp or 'week' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 604800 if re.findall(r'\d+', timestamp) else 2592000
        elif 'mo' in timestamp or 'month' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 2592000 if re.findall(r'\d+', timestamp) else 31536000
        elif 'y' in timestamp or 'year' in timestamp:
            return int(re.findall(r'\d+', timestamp)[0]) * 31536000 if re.findall(r'\d+', timestamp) else 31536000
        
        return 999999

    async def _aggressive_content_loading(self, page: Page):
        """More aggressive content loading strategy"""
        logger.info("🔄 Applying aggressive content loading...")
        
        try:
            # Multiple scroll and wait cycles
            for i in range(10):
                # Scroll down
                await page.evaluate(f"window.scrollTo(0, {(i + 1) * 500})")
                await asyncio.sleep(1)
                
                # Check if we have posts
                posts_found = await page.query_selector_all('div[data-id^="urn:li:activity"]')
                if len(posts_found) >= 3:
                    logger.info(f"Found {len(posts_found)} posts after scroll {i+1}")
                    break
            
            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(2)
            
            # Try to trigger lazy loading
            await page.evaluate("""
                const images = document.querySelectorAll('img[data-delayed-url]');
                images.forEach(img => {
                    if (img.dataset.delayedUrl) {
                        img.src = img.dataset.delayedUrl;
                    }
                });
            """)
            
            await asyncio.sleep(3)
            
        except Exception as e:
            logger.warning(f"Aggressive loading error: {e}")

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
                '--window-size=1920,1080',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                '--disable-extensions',
                '--disable-plugins'
            ]
        )

    async def _create_context(self, browser: Browser) -> BrowserContext:
        """Create context with realistic browser fingerprint"""
        user_agent = random.choice(self.user_agents)
        logger.info(f"🔧 Setting up context...")
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York'
        )
        
        # Block heavy resources
        await context.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,mp4,mp3,pdf}", lambda route: route.abort())
        
        return context

    async def _navigate_with_retry(self, page: Page, url: str):
        """Navigate with better error handling"""
        logger.info(f"🌐 Navigating to: {url}")
        
        await asyncio.sleep(random.uniform(2, 4))
        
        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
        await page.wait_for_timeout(3000)
        
        title = await page.title()
        logger.info(f"📄 Page title: {title}")

    async def _handle_all_popups_comprehensive(self, page: Page):
        """Comprehensive popup handling"""
        logger.info("🔍 Handling popups...")
        
        await page.wait_for_timeout(2000)
        
        popup_selectors = [
            'button[aria-label="Dismiss"]',
            'button[data-test-modal-close-btn]',
            '.artdeco-modal__dismiss',
            '.sign-in-modal__dismiss-btn',
            'button[aria-label="Close"]'
        ]
        
        for selector in popup_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if element and await element.is_visible():
                        await element.click()
                        await page.wait_for_timeout(1000)
            except:
                continue

    async def _load_content_strategically(self, page: Page):
        """Strategic content loading"""
        logger.info("📦 Loading content strategically...")
        
        try:
            # Wait for main content
            await page.wait_for_selector('main, .application-outlet, body', timeout=15000)
            
            # Wait a bit for dynamic content
            await page.wait_for_timeout(3000)
            
            # Try to wait for feed content specifically
            try:
                await page.wait_for_selector('div.feed-container-theme, .scaffold-layout__content, article', timeout=10000)
            except:
                logger.warning("Feed content selector not found, continuing")
            
            # Progressive scrolling with more attempts
            for i in range(8):  # Increased scroll attempts
                scroll_position = (i + 1) * 300
                await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                await page.wait_for_timeout(1500)
                
                # Check if posts are loading
                try:
                    posts = await page.query_selector_all('article, div[data-id]')
                    if len(posts) > 0:
                        logger.info(f"Found {len(posts)} potential posts after scroll {i+1}")
                        if len(posts) >= 3:  # If we have enough posts, break early
                            break
                except:
                    continue
            
            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(2000)
            
            # Try to trigger any lazy-loaded content
            try:
                await page.evaluate("""
                    // Trigger intersection observer for lazy loading
                    const observer = new IntersectionObserver((entries) => {
                        entries.forEach(entry => {
                            if (entry.isIntersecting) {
                                entry.target.dispatchEvent(new Event('load'));
                            }
                        });
                    });
                    
                    document.querySelectorAll('article, div[data-id]').forEach(el => {
                        observer.observe(el);
                    });
                """)
                await page.wait_for_timeout(2000)
            except:
                logger.debug("Could not trigger lazy loading")
            
        except Exception as e:
            logger.warning(f"Content loading issues: {e}")

    def _is_valid_timestamp_text(self, text: str) -> bool:
        """Enhanced timestamp validation"""
        if not text or len(text) > 50:
            return False

        text = text.lower().strip()

        patterns = [
            r'\d+\s*(?:second|minute|hour|day|week|month|year|s|m|hr|h|d|w|mo|y)s?\b',
            r'just now', r'now', r'yesterday', r'today'
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    def _standardize_timestamp(self, timestamp: str) -> str:
        """Enhanced timestamp standardization"""
        if not timestamp:
            return ""

        original_timestamp = timestamp.strip() # Keep original to check for 'edited'
        timestamp = original_timestamp.lower()
        
        is_edited = 'edited' in timestamp # Check for 'edited' early
        
        # Handle ISO datetime
        try:
            if 'T' in timestamp and ('Z' in timestamp or '+' in timestamp or '-' in timestamp):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                now = datetime.now()

                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)

                diff = now - dt

                if diff.days > 0:
                    base_time = f"{diff.days}d"
                elif diff.seconds >= 3600:
                    hours = diff.seconds // 3600
                    base_time = f"{hours}h"
                elif diff.seconds >= 60:
                    minutes = diff.seconds // 60
                    base_time = f"{minutes}m"
                else:
                    base_time = "just now"
                
                return f"{base_time} Edited" if is_edited else base_time

        except:
            pass # Continue to other parsing methods if ISO fails
        
        # Handle special cases (after ISO parsing)
        base_time = ""
        if 'just now' in timestamp or timestamp == 'now':
            base_time = 'just now'
        elif 'today' in timestamp:
            base_time = 'today'
        elif 'yesterday' in timestamp:
            base_time = '1d'
        
        if base_time: # If a special case was matched
            return f"{base_time} Edited" if is_edited else base_time


        # Standardize units
        replacements = {
            'seconds': 's', 'second': 's', 'sec': 's',
            'minutes': 'm', 'minute': 'm', 'min': 'm',
            'hours': 'h', 'hour': 'h', 'hr': 'h',
            'days': 'd', 'day': 'd',
            'weeks': 'w', 'week': 'w',
            'months': 'mo', 'month': 'mo'
        }

        for old, new in replacements.items():
            timestamp = re.sub(f'\\b{old}\\b', new, timestamp)

        # Clean up
        timestamp = re.sub(r'\s*ago\s*', '', timestamp).strip()
        timestamp = re.sub(r'\s+', '', timestamp)
        
        # Append "Edited" if it was found
        if is_edited and timestamp:
            # We need to remove 'edited' from the timestamp string if it was replaced or added directly
            # This handles cases like "1d edited" becoming "1dEdited", but then we add " Edited"
            # It's better to ensure 'edited' is only appended once and clearly.
            timestamp = timestamp.replace('edited', '').strip()
            return f"{timestamp} Edited".strip()
        elif is_edited and not timestamp: # Handles cases like just "Edited" being the input
             return "Edited"
        else:
            return timestamp

    async def _save_comprehensive_debug(self, page: Page, prefix: str, attempt: int):
        """Save debug information"""
        try:
            timestamp = int(time.time())
            screenshot_file = f"debug_{prefix}_{attempt}_{timestamp}.png"
            await page.screenshot(path=screenshot_file, full_page=True)
            logger.info(f"📸 Debug screenshot: {screenshot_file}")
        except:
            pass

    async def scrape_companies(self, company_urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Scrape multiple company pages"""
        results = {}
        
        for url in company_urls:
            try:
                posts = await self.scrape_company_posts(url)
                if posts:
                    results[url] = posts
                
                await asyncio.sleep(random.uniform(5, 10))
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                continue
                
        return results
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]], filename: str = None):
        """Save scraping results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linkedin_scraper_results_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        try:
            # Clean results for JSON serialization
            clean_results = {}
            for url, posts in results.items():
                clean_posts = []
                for post in posts:
                    clean_post = {
                        'url': post.get('url', ''),
                        'timestamp': post.get('timestamp', ''),
                        'text': post.get('text', ''),
                        'scraped_at': datetime.now().isoformat()
                    }
                    # Only include raw_html if debug mode is enabled
                    if self.debug_mode and post.get('raw_html'):
                        clean_post['raw_html'] = post['raw_html']
                    
                    clean_posts.append(clean_post)
                
                clean_results[url] = clean_posts
            
            # Add metadata
            output_data = {
                'metadata': {
                    'scraped_at': datetime.now().isoformat(),
                    'total_companies': len(clean_results),
                    'total_posts': sum(len(posts) for posts in clean_results.values()),
                    'scraper_version': '2.0',
                    'debug_mode': self.debug_mode
                },
                'results': clean_results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None

    def load_company_urls(self, file_path: str) -> List[str]:
        """Load company URLs from a text file"""
        try:
            urls = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Ensure URL is properly formatted
                        if not line.startswith('http'):
                            line = f"https://www.linkedin.com/company/{line}"
                        urls.append(line)
            
            logger.info(f"📋 Loaded {len(urls)} company URLs from {file_path}")
            return urls
            
        except Exception as e:
            logger.error(f"Error loading URLs from {file_path}: {e}")
            return []

    def generate_summary_report(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a summary report of the scraping results"""
        total_companies = len(results)
        total_posts = sum(len(posts) for posts in results.values())
        
        report = f"""
LinkedIn Scraper Summary Report
{'='*50}
Scraped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total companies processed: {total_companies}
Total posts extracted: {total_posts}

Company Details:
{'-'*30}
"""
        
        for url, posts in results.items():
            company_name = url.split('/')[-1] if url.split('/')[-1] else url.split('/')[-2]
            report += f"• {company_name}: {len(posts)} posts\n"
            
            for i, post in enumerate(posts, 1):
                timestamp = post.get('timestamp', 'Unknown')
                text_preview = post.get('text', '')[:100] + '...' if len(post.get('text', '')) > 100 else post.get('text', '')
                likes = post.get('likes', 0)
                comments = post.get('comments', 0)
                
                report += f"  {i}. {timestamp} | Likes: {likes} | Comments: {comments}\n"
                report += f"     Text: {text_preview}\n\n"
        
        return report

    async def validate_urls(self, urls: List[str]) -> List[Tuple[str, bool, str]]:
        """Validate company URLs before scraping"""
        logger.info(f"🔍 Validating {len(urls)} URLs...")
        
        results = []
        async with async_playwright() as p:
            browser = await self._launch_browser(p)
            context = await self._create_context(browser)
            page = await context.new_page()
            
            for url in urls:
                try:
                    response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    
                    if response.status == 200:
                        # Check if it's actually a LinkedIn company page
                        title = await page.title()
                        if 'linkedin' in title.lower() and ('company' in url or 'organization' in title.lower()):
                            results.append((url, True, "Valid"))
                        else:
                            results.append((url, False, "Not a valid company page"))
                    else:
                        results.append((url, False, f"HTTP {response.status}"))
                        
                except Exception as e:
                    results.append((url, False, str(e)))
                
                await asyncio.sleep(2)  # Rate limiting
            
            await browser.close()
        
        valid_count = sum(1 for _, valid, _ in results if valid)
        logger.info(f"✅ {valid_count}/{len(urls)} URLs are valid")
        
        return results

    async def scrape_with_progress(self, company_urls: List[str], save_intermediate: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Scrape companies with progress tracking and intermediate saves"""
        results = {}
        total_urls = len(company_urls)
        
        logger.info(f"🚀 Starting scrape of {total_urls} companies...")
        
        for i, url in enumerate(company_urls, 1):
            try:
                logger.info(f"📊 Progress: {i}/{total_urls} - Processing: {url}")
                
                posts = await self.scrape_company_posts(url)
                
                if posts:
                    results[url] = posts
                    logger.info(f"✅ Successfully scraped {len(posts)} posts from {url}")
                else:
                    logger.warning(f"⚠️ No posts found for {url}")
                    results[url] = []
                
                # Save intermediate results every 5 companies
                if save_intermediate and i % 5 == 0:
                    intermediate_filename = f"intermediate_results_{i}of{total_urls}.json"
                    self.save_results(results, intermediate_filename)
                    logger.info(f"💾 Intermediate results saved")
                
                # Rate limiting between requests
                if i < total_urls:  # Don't wait after the last URL
                    wait_time = random.uniform(8, 15)
                    logger.info(f"⏳ Waiting {wait_time:.1f}s before next company...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"❌ Failed to scrape {url}: {str(e)}")
                results[url] = []
                continue
        
        logger.info(f"🎉 Scraping completed! Processed {total_urls} companies")
        return results


async def main():
    """Main execution function"""
    logger.info("🔥 LinkedIn Scraper v2.0 Starting...")
    
    # Configuration
    HEADLESS = True 
    SLOW_MO = 100   
    
    # Initialize scraper
    scraper = AdvancedLinkedInScraper(headless=HEADLESS, slow_mo=SLOW_MO)
    
    # Company URLs to scrape - you can modify this list
    # company_urls = [
    #     "https://www.linkedin.com/company/icreatenextgen/",
    #     "https://www.linkedin.com/company/iitmandicatalyst/",
    #     # Add more URLs here or load from file
    # ]
    
    # Alternative: Load URLs from file
    company_urls = scraper.load_company_urls("company_urls.txt")
    
    if not company_urls:
        logger.error("❌ No company URLs provided!")
        return
    
    try:
        
        # Start scraping
        logger.info(f"🚀 Starting to scrape {len(company_urls)} valid URLs...")
        results = await scraper.scrape_with_progress(company_urls, save_intermediate=True)

        output_file = scraper.save_results(results)
        
        # Final statistics
        total_posts = sum(len(posts) for posts in results.values())
        successful_companies = sum(1 for posts in results.values() if posts)
        
        logger.info(f"""
🎊 SCRAPING COMPLETED SUCCESSFULLY!
{'='*50}
✅ Companies processed: {len(results)}
✅ Companies with posts: {successful_companies}
✅ Total posts extracted: {total_posts}
✅ Results saved to: {output_file}
✅ Average posts per company: {total_posts/len(results):.1f}
""")
        
    except KeyboardInterrupt:
        logger.info("🛑 Scraping interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
    finally:
        logger.info("🏁 LinkedIn Scraper finished")


def create_sample_urls_file():
    """Create a sample company URLs file"""
    sample_urls = [
        "# LinkedIn Company URLs - one per line",
        "# You can use full URLs or just company slugs",
        "https://www.linkedin.com/company/microsoft/",
        "https://www.linkedin.com/company/google/",
        "https://www.linkedin.com/company/apple/",
        "https://www.linkedin.com/company/amazon/",
        "https://www.linkedin.com/company/tesla-motors/",
        "netflix",  # This will be converted to full URL
        "spotify",
        "airbnb"
    ]
    
    with open("company_urls.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sample_urls))
    
    print("📝 Sample company_urls.txt file created!")


if __name__ == "__main__":
    # Uncomment the line below to create a sample URLs file
    # create_sample_urls_file()
    
    # Run the scraper
    asyncio.run(main())