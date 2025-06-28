import datetime
from datetime import datetime, timedelta
import os
import re
import random
import json
import asyncio
from typing import Dict, List, Any, Optional
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Page
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinkedInScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        self.timeout = 30000
        self.max_retries = 3
        self.headless = True
        
    async def _random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add random delay to avoid detection"""
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"Waiting {delay:.2f} seconds...")
        await asyncio.sleep(delay)
        
    async def scrape_company_posts(self, company_url: str, max_posts: int = 5) -> List[Dict[str, Any]]:
        """
        Main method to scrape company posts from LinkedIn
        """
        normalized_url = self._normalize_company_url(company_url)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries} to scrape LinkedIn posts for {normalized_url}")
                
                posts = await self._scrape_with_playwright(normalized_url, max_posts)
                if posts and len(posts) > 0:
                    logger.info(f"Successfully scraped {len(posts)} posts")
                    return posts
                else:
                    logger.warning("No posts found. Retrying...")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

            if attempt < self.max_retries - 1:
                await self._random_delay(3, 7)

        logger.error("All scraping attempts failed")
        return []

    async def _scrape_with_playwright(self, company_url: str, max_posts: int) -> List[Dict[str, Any]]:
        """Scrape company posts using Playwright with login capability"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                ]
            )
            
            context = await browser.new_context(
                user_agent=self.headers["User-Agent"],
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            page = await context.new_page()
            
            try:
                # First try to login if needed
                await self._linkedin_login(page)
                
                # Now try to scrape posts
                posts_url = f"{company_url}posts/"
                await page.goto(posts_url, wait_until="domcontentloaded")
                
                # Check if we're still on login page
                if await page.query_selector('input#username'):
                    raise Exception("Login failed - still seeing login page")
                
                # Set sort filter to "Recent" if available
                await self._set_sort_to_recent(page)
                
                # Wait for content and scroll just a bit to ensure all posts load
                await self._wait_for_content_load(page)
                
                # Get fresh posts with proper timestamp validation
                posts = await self._get_fresh_posts(page, max_posts)
                
                return posts[:max_posts]
                
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                await page.screenshot(path='error.png')
                return []
            finally:
                await browser.close()

    async def _get_fresh_posts(self, page: Page, max_posts: int) -> List[Dict[str, Any]]:
        """Get only fresh posts with proper timestamp validation"""
        fresh_posts = []
        attempts = 0
        max_attempts = 3
        
        while len(fresh_posts) < max_posts and attempts < max_attempts:
            attempts += 1
            logger.info(f"Attempt {attempts} to get fresh posts")
            
            # Extract all visible posts
            all_posts = await self._extract_posts_comprehensive(page, 50)  # Get more than needed
            
            # Filter for truly recent posts (within last month)
            current_time = datetime.now()
            for post in all_posts:
                timestamp = post.get('timestamp', '')
                if not timestamp:
                    continue
                    
                # Parse timestamp (handle formats like "8mo", "2w", "1d")
                post_time = self._parse_linkedin_timestamp(timestamp)
                if not post_time:
                    continue
                    
                # Only keep posts from last 30 days
                if (current_time - post_time).days <= 30:
                    fresh_posts.append(post)
                    if len(fresh_posts) >= max_posts:
                        break
            
            # If we didn't get enough fresh posts, scroll and try again
            if len(fresh_posts) < max_posts:
                await page.evaluate("window.scrollBy(0, 1000)")
                await self._random_delay(2, 3)
                await self._wait_for_content_load(page)
        
        # Sort by most recent first
        fresh_posts.sort(key=lambda x: self._parse_linkedin_timestamp(x.get('timestamp', '') or datetime.min), reverse=True)
        
        return fresh_posts[:max_posts]

    def _parse_linkedin_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse LinkedIn timestamp strings into datetime objects"""
        try:
            timestamp_str = timestamp_str.lower().strip()
            current_time = datetime.now()
            
            if 'just now' in timestamp_str or 'minute' in timestamp_str:
                return current_time
            elif 'hour' in timestamp_str:
                hours = int(re.search(r'(\d+)\s*hour', timestamp_str).group(1))
                return current_time - timedelta(hours=hours)
            elif 'day' in timestamp_str:
                days = int(re.search(r'(\d+)\s*day', timestamp_str).group(1))
                return current_time - timedelta(days=days)
            elif 'week' in timestamp_str:
                weeks = int(re.search(r'(\d+)\s*week', timestamp_str).group(1))
                return current_time - timedelta(weeks=weeks)
            elif 'month' in timestamp_str:
                months = int(re.search(r'(\d+)\s*month', timestamp_str).group(1))
                return current_time - timedelta(days=months*30)
            elif 'year' in timestamp_str:
                years = int(re.search(r'(\d+)\s*year', timestamp_str).group(1))
                return current_time - timedelta(days=years*365)
            else:
                # Try to parse ISO format or other date formats
                for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%b %d, %Y'):
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
        except Exception as e:
            logger.debug(f"Could not parse timestamp '{timestamp_str}': {e}")
            return None

    async def _set_sort_to_recent(self, page: Page):
        """Set the post sort filter to 'Recent'"""
        try:
            # Wait for the sort dropdown to be available
            await page.wait_for_selector('.sort-dropdown__dropdown', timeout=10000)
            
            # Check if already set to Recent
            current_sort = await page.evaluate('''() => {
                const sortButton = document.querySelector('#sort-dropdown-trigger');
                return sortButton ? sortButton.innerText.includes('Recent') : false;
            }''')
            
            if current_sort:
                logger.info("Posts already sorted by Recent")
                return
            
            # Click the sort dropdown
            await page.click('#sort-dropdown-trigger')
            await self._random_delay(1, 2)
            
            # Wait for dropdown options to appear
            await page.wait_for_selector('.artdeco-dropdown__content-inner', timeout=5000)
            
            # Find and click the Recent option
            recent_found = False
            dropdown_options = await page.query_selector_all('.artdeco-dropdown__item')
            
            for option in dropdown_options:
                option_text = await option.inner_text()
                if 'Recent' in option_text:
                    await option.click()
                    recent_found = True
                    logger.info("Changed sort to Recent")
                    await self._random_delay(2, 3)  # Wait for posts to reload
                    break
            
            if not recent_found:
                logger.warning("Could not find Recent sort option")
            
        except Exception as e:
            logger.error(f"Failed to set sort to Recent: {e}")
            # Continue even if we can't set the sort, we'll still try to get posts

    async def _perform_minimal_scrolling(self, page: Page):
        """Perform minimal scrolling just to ensure all posts load"""
        try:
            # Small scroll to trigger any lazy loading
            await page.evaluate("window.scrollBy(0, 500)")
            await self._random_delay(1, 2)
            
            # Check if we have posts loaded
            posts_count = await page.evaluate('''() => {
                return document.querySelectorAll('[data-test-id="main-feed-activity-card"], .feed-shared-update-v2').length;
            }''')
            
            logger.info(f"Found {posts_count} posts after minimal scrolling")
            
        except Exception as e:
            logger.error(f"Minimal scrolling failed: {e}")

    async def _linkedin_login(self, page):
        """Improved LinkedIn login method for your scraper"""
        try:
            # Step 1: Go to LinkedIn login page directly
            logger.info("Navigating to LinkedIn login page...")
            await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            await self._random_delay(2, 4)

            # Step 2: Check if we're already logged in
            if await self._check_if_logged_in(page):
                logger.info("Already logged in to LinkedIn")
                return True

            # Step 3: Handle potential captcha or security checks
            if await self._handle_security_challenge(page):
                logger.warning("Security challenge detected - manual intervention may be required")
                return False

            # Step 4: Get credentials
            username = os.getenv('LINKEDIN_USERNAME') or os.getenv('LINKEDIN_EMAIL')
            password = os.getenv('LINKEDIN_PASSWORD')
            
            if not username or not password:
                logger.error("LinkedIn credentials not found in environment variables")
                logger.info("Please set LINKEDIN_USERNAME (or LINKEDIN_EMAIL) and LINKEDIN_PASSWORD")
                raise Exception("LinkedIn credentials not provided")

            # Step 5: Fill username field
            username_filled = await self._fill_username(page, username)
            if not username_filled:
                raise Exception("Could not find or fill username field")

            await self._random_delay(0.5, 1.5)

            # Step 6: Fill password field
            password_filled = await self._fill_password(page, password)
            if not password_filled:
                raise Exception("Could not find or fill password field")

            await self._random_delay(0.5, 1.5)

            # Step 7: Submit the form
            login_submitted = await self._submit_login_form(page)
            if not login_submitted:
                raise Exception("Could not submit login form")

            # Step 8: Wait for login completion
            success = await self._wait_for_login_completion(page)
            if not success:
                raise Exception("Login failed - check credentials and account status")
                
            logger.info("Login successful!")
            return True

        except Exception as e:
            logger.error(f"Login process failed: {e}")
            await self._save_debug_screenshot(page, "login_error")
            raise

    async def _check_if_logged_in(self, page):
        """Check if already logged in using multiple indicators"""
        logged_in_selectors = [
            'nav.global-nav',
            '.global-nav__me',
            'button[data-test-id="nav-me-icon"]',
            '.feed-identity-module',
            'a[href*="/in/"]',
            '.global-nav__primary-link--me'
        ]
        
        for selector in logged_in_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=3000)
                if element:
                    logger.info(f"Already logged in (detected: {selector})")
                    return True
            except PlaywrightTimeoutError:
                continue
        
        return False

    async def _handle_security_challenge(self, page):
        """Detect and handle security challenges"""
        challenge_indicators = [
            'input[name="captcha"]',
            '.captcha-container',
            'iframe[src*="recaptcha"]',
            '.challenge-form',
            'h1:has-text("Security Verification")',
            'h1:has-text("Help us protect")',
            '.two-step-verification'
        ]
        
        for indicator in challenge_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    logger.warning(f"Security challenge detected: {indicator}")
                    return True
            except PlaywrightTimeoutError:
                continue
        
        return False

    async def _fill_username(self, page, username):
        """Fill username field with multiple selector attempts"""
        username_selectors = [
            'input#username',
            'input[name="session_key"]',
            'input[autocomplete="username"]',
            'input[type="email"]',
            'input[placeholder*="email"]',
            'input[placeholder*="Email"]'
        ]
        
        for selector in username_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if element:
                    await element.click()
                    await page.keyboard.press('Control+a')
                    await element.fill(username)
                    
                    filled_value = await element.input_value()
                    if filled_value == username:
                        logger.info(f"Username filled successfully using: {selector}")
                        return True
            except PlaywrightTimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Error filling username with {selector}: {e}")
                continue
        
        return False

    async def _fill_password(self, page, password):
        """Fill password field with multiple selector attempts"""
        password_selectors = [
            'input#password',
            'input[name="session_password"]',
            'input[type="password"]',
            'input[autocomplete="current-password"]'
        ]
        
        for selector in password_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if element:
                    await element.click()
                    await element.fill(password)
                    
                    filled_value = await element.input_value()
                    if len(filled_value) == len(password):
                        logger.info(f"Password filled successfully using: {selector}")
                        return True
            except PlaywrightTimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Error filling password with {selector}: {e}")
                continue
        
        return False

    async def _submit_login_form(self, page):
        """Submit login form with multiple button selectors"""
        submit_selectors = [
            'button[type="submit"]',
            'button[data-id="sign-in-form__submit-btn"]',
            'input[type="submit"]',
            '.login__form_action_container button',
            'button:has-text("Sign in")',
            'button:has-text("Sign In")'
        ]
        
        for selector in submit_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if element and await element.is_visible():
                    await element.click()
                    logger.info(f"Login form submitted using: {selector}")
                    return True
            except PlaywrightTimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Error submitting form with {selector}: {e}")
                continue
        
        # Fallback: try pressing Enter
        try:
            await page.keyboard.press('Enter')
            logger.info("Login form submitted using Enter key")
            return True
        except Exception as e:
            logger.debug(f"Enter key submission failed: {e}")
        
        return False

    async def _wait_for_login_completion(self, page):
        """Wait for login completion and handle various scenarios"""
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
            await self._random_delay(2, 4)
            
            # Check for successful login
            if await self._check_if_logged_in(page):
                return True
            
            # Check for errors
            await self._handle_login_errors(page)
            
            # Check if still on login page
            current_url = page.url
            if 'login' in current_url.lower():
                logger.error("Still on login page - login likely failed")
                return False
            
            # Final check
            await self._random_delay(2, 3)
            return await self._check_if_logged_in(page)
            
        except PlaywrightTimeoutError:
            logger.error("Login completion timeout")
            return False
        except Exception as e:
            logger.error(f"Error waiting for login completion: {e}")
            return False

    async def _handle_login_errors(self, page):
        """Handle various login error scenarios"""
        error_scenarios = [
            {
                'selectors': ['.alert--error', '.form__error', '.login-form__error'],
                'message': 'Login form error detected'
            },
            {
                'selectors': ['h1:has-text("Account locked")', '.account-locked'],
                'message': 'Account appears to be locked'
            },
            {
                'selectors': ['h1:has-text("Verify")', '.verification-required'],
                'message': 'Account verification required'
            }
        ]
        
        for scenario in error_scenarios:
            for selector in scenario['selectors']:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element and await element.is_visible():
                        error_text = await element.inner_text()
                        logger.error(f"{scenario['message']}: {error_text}")
                        return True
                except PlaywrightTimeoutError:
                    continue
        
        return False

    async def _save_debug_screenshot(self, page, prefix="debug"):
        """Save screenshot for debugging"""
        try:
            screenshot_path = f"{prefix}_{int(time.time())}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Debug screenshot saved: {screenshot_path}")
        except Exception as e:
            logger.debug(f"Could not save screenshot: {e}")

    async def _wait_for_content_load(self, page: Page):
        """Wait for page content to load using multiple strategies"""
        content_selectors = [
            '[data-test-id="main-feed-activity-card"]',
            '.feed-shared-update-v2',
            '.update-components-text',
            '[data-view-name="feed-update"]',
            '.occludable-update',
            'div[data-id]',
            'article',
            'main section',
            '.core-rail',
            '.scaffold-layout__main'
        ]
        
        for selector in content_selectors:
            try:
                logger.debug(f"Waiting for selector: {selector}")
                await page.wait_for_selector(selector, timeout=10000)
                logger.info(f"Content loaded with selector: {selector}")
                return
            except PlaywrightTimeoutError:
                continue
                
        # Fallback: just wait for body
        logger.info("Using fallback content wait")
        await page.wait_for_selector('body', timeout=10000)

    async def _perform_smart_scrolling(self, page: Page, max_posts: int):
        """Perform intelligent scrolling to load dynamic content"""
        try:
            # Calculate required scrolls based on number of posts needed
            scrolls_needed = min(max(3, max_posts), 10)  # Scroll 3-10 times max
            
            for i in range(scrolls_needed):
                # Get current scroll position
                scroll_height = await page.evaluate("document.body.scrollHeight")
                scroll_position = (i + 1) * 1000
                
                # Scroll to position
                await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                logger.debug(f"Scrolled to position {scroll_position}")
                
                # Wait for new content to load
                await self._random_delay(2, 3)
                
                # Check if new content loaded
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height > scroll_height:
                    logger.debug(f"New content loaded (height increased from {scroll_height} to {new_height})")
                
                # Check if we have enough posts loaded
                posts_count = await page.evaluate('''() => {
                    return document.querySelectorAll('[data-test-id="main-feed-activity-card"], .feed-shared-update-v2').length;
                }''')
                
                if posts_count >= max_posts * 2:  # Load extra posts to ensure we get enough unique ones
                    logger.debug(f"Found {posts_count} posts, stopping scroll")
                    break
                
            # Final scroll to top to ensure all elements are in view
            await page.evaluate("window.scrollTo(0, 0)")
            await self._random_delay(1, 2)
            
        except Exception as e:
            logger.error(f"Scrolling failed: {e}")

    async def _extract_posts_comprehensive(self, page: Page, max_posts: int) -> List[Dict[str, Any]]:
        """Comprehensive post extraction with deduplication"""
        posts = []
        seen_post_ids = set()
        
        # Modern LinkedIn selectors
        modern_selectors = [
            '[data-test-id="main-feed-activity-card"]',
            '.feed-shared-update-v2',
            '[data-view-name="feed-update"]',
            '.update-components-actor'
        ]
        
        for selector in modern_selectors:
            try:
                elements = await page.query_selector_all(selector)
                logger.info(f"Found {len(elements)} elements with selector: {selector}")
                
                for element in elements:
                    if len(posts) >= max_posts:
                        break
                        
                    post_data = await self._extract_comprehensive_post_data(element)
                    post_id = self._generate_post_id(post_data)
                    
                    if post_id and post_id not in seen_post_ids and self._is_valid_post(post_data):
                        seen_post_ids.add(post_id)
                        posts.append(post_data)
                        
                if len(posts) >= max_posts:
                    break
                    
            except Exception as e:
                logger.error(f"Error extracting posts with {selector}: {e}")
                continue
        
        return posts

    def _generate_post_id(self, post_data: Dict[str, Any]) -> Optional[str]:
        """Generate a unique ID for a post to avoid duplicates"""
        if not post_data:
            return None
            
        # Use URL if available
        if post_data.get('url'):
            return post_data['url'].split('?')[0]  # Remove query params
            
        # Fallback to content hash
        content = post_data.get('content', '')
        if content:
            return str(hash(content[:200]))  # Use first 200 chars for hash
            
        return None

    async def _extract_comprehensive_post_data(self, element) -> Dict[str, Any]:
        """Extract comprehensive data from a post element"""
        try:
            # Extract content using multiple methods
            content = await self._extract_text_content(element)
            
            # Extract metadata
            timestamp = await self._extract_timestamp_from_element(element)
            reactions = await self._extract_engagement_count(element, 'reactions')
            comments = await self._extract_engagement_count(element, 'comments')
            
            # Extract media
            images = await self._extract_images_from_element(element)
            links = await self._extract_links_from_element(element)
            
            # Extract post URL
            post_url = await self._extract_post_url_from_element(element)
            
            return {
                "content": content,
                "timestamp": timestamp,
                "reactions": reactions,
                "comments": comments,
                "images": images,
                "links": links,
                "url": post_url,
                "extraction_method": "comprehensive"
            }
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive post data: {e}")
            return {}

    async def _extract_text_content(self, element) -> str:
        """Extract text content using multiple strategies"""
        text_selectors = [
            '.feed-shared-text',
            '.update-components-text',
            '.feed-shared-update-v2__description',
            '[data-test-id="post-text"]',
            'span[dir="ltr"]',
            '.attributed-text-segment-list__content',
            '.break-words'
        ]
        
        # Try specific selectors first
        for selector in text_selectors:
            try:
                text_element = await element.query_selector(selector)
                if text_element:
                    text = await text_element.inner_text()
                    if text and text.strip():
                        return text.strip()
            except:
                continue
        
        # Fallback to element's own text
        try:
            text = await element.inner_text()
            return text.strip() if text else ""
        except:
            return ""

    async def _extract_timestamp_from_element(self, element) -> str:
        """Extract timestamp from element"""
        timestamp_selectors = [
            'time',
            '[data-test-id="post-timestamp"]',
            '.update-components-actor__meta',
            '.feed-shared-actor__meta'
        ]
        
        for selector in timestamp_selectors:
            try:
                time_element = await element.query_selector(selector)
                if time_element:
                    # Try datetime attribute first
                    datetime_attr = await time_element.get_attribute('datetime')
                    if datetime_attr:
                        return datetime_attr
                    
                    # Fallback to text content
                    text = await time_element.inner_text()
                    if text:
                        return text.strip()
            except:
                continue
        
        return ""

    async def _extract_engagement_count(self, element, metric_type: str) -> int:
        """Extract engagement counts (reactions, comments, etc.)"""
        if metric_type == 'reactions':
            selectors = [
                '.social-details-social-counts__reactions-count',
                '[data-test-id="reactions-count"]',
                '.feed-shared-social-action-bar__reaction-count'
            ]
        elif metric_type == 'comments':
            selectors = [
                '.social-details-social-counts__comments-count',
                '[data-test-id="comments-count"]',
                '.feed-shared-social-action-bar__comment-count'
            ]
        else:
            return 0
        
        for selector in selectors:
            try:
                count_element = await element.query_selector(selector)
                if count_element:
                    text = await count_element.inner_text()
                    if text:
                        # Extract numbers, handle K/M notation
                        numbers = re.findall(r'[\d,]+', text.replace(',', ''))
                        if numbers:
                            count = int(numbers[0])
                            if 'K' in text.upper():
                                count *= 1000
                            elif 'M' in text.upper():
                                count *= 1000000
                            return count
            except:
                continue
        
        return 0

    async def _extract_images_from_element(self, element) -> List[str]:
        """Extract image URLs from element"""
        images = []
        try:
            img_elements = await element.query_selector_all('img')
            for img in img_elements:
                src = await img.get_attribute('src')
                if src and 'http' in src and any(domain in src for domain in ['linkedin.com', 'licdn.com']):
                    images.append(src)
        except:
            pass
        return images

    async def _extract_links_from_element(self, element) -> List[Dict[str, str]]:
        """Extract links from element"""
        links = []
        try:
            link_elements = await element.query_selector_all('a[href]')
            for link in link_elements:
                href = await link.get_attribute('href')
                text = await link.inner_text()
                if (href and 'http' in href and 
                    not any(pattern in href for pattern in ['/in/', '/company/', 'linkedin.com/feed'])):
                    links.append({"url": href, "title": text.strip()})
        except:
            pass
        return links

    async def _extract_post_url_from_element(self, element) -> str:
        """Extract direct post URL"""
        url_selectors = [
            'a[href*="/feed/update/"]',
            'a[href*="/posts/"]',
            '[data-id]'
        ]
        
        for selector in url_selectors:
            try:
                url_element = await element.query_selector(selector)
                if url_element:
                    if 'href' in selector:
                        href = await url_element.get_attribute('href')
                        if href:
                            return href.split('?')[0]  # Remove tracking params
                    else:
                        data_id = await url_element.get_attribute('data-id')
                        if data_id:
                            return f"https://www.linkedin.com/feed/update/{data_id}"
            except:
                continue
        
        return ""

    def _normalize_company_url(self, url: str) -> str:
        """Normalize and validate company URL"""
        # Remove trailing slashes
        url = url.rstrip('/')
        
        # Ensure proper LinkedIn company URL format
        if 'linkedin.com/company/' not in url:
            # Extract company identifier
            if url.endswith('/company'):
                url = url[:-8]  # Remove /company
            
            # Get company name from URL
            if '/' in url:
                company_name = url.split('/')[-1]
            else:
                company_name = url
            
            # Build proper LinkedIn company URL
            url = f"https://www.linkedin.com/company/{company_name}"
        
        # Ensure trailing slash
        if not url.endswith('/'):
            url += '/'
            
        return url

    def _is_restricted_page(self, page_content: str) -> bool:
        """Check if page has access restrictions"""
        restriction_indicators = [
            'sign in to linkedin',
            'join linkedin to see',
            'login to continue',
            'authentication required',
            'access denied',
            'page not found',
            'this content isn\'t available',
            'verify your account',
            'complete your profile'
        ]
        
        content_lower = page_content.lower()
        return any(indicator in content_lower for indicator in restriction_indicators)

    def _looks_like_post_content(self, text: str) -> bool:
        """Check if text looks like a LinkedIn post"""
        if not text or len(text.strip()) < 20:
            return False
        
        # Exclude navigation and UI elements
        exclude_patterns = [
            'sign in', 'log in', 'join linkedin', 'home', 'my network',
            'jobs', 'messaging', 'notifications', 'skip to main content',
            'cookie policy', 'privacy policy', 'terms of service',
            'follow', 'connect', 'view profile', 'see all',
            'like', 'comment', 'share', 'send'
        ]
        
        text_lower = text.lower()
        
        # Check for excluded patterns
        if any(pattern in text_lower for pattern in exclude_patterns):
            return False
        
        # Look for post-like characteristics
        post_indicators = [
            '.',  # Sentences typically end with periods
            '!',  # Exclamation marks are common in posts
            '?',  # Questions are common
            '#',  # Hashtags
            '@',  # Mentions
            'we', 'our', 'excited', 'proud', 'happy', 'announcing'
        ]
        
        return any(indicator in text_lower for indicator in post_indicators)

    def _is_valid_post(self, post_data: Dict[str, Any]) -> bool:
        """Validate if extracted data represents a real post"""
        if not post_data or not isinstance(post_data, dict):
            return False
        
        content = post_data.get('content', '').strip()
        
        # Must have meaningful content
        if not content or len(content) < 20:
            return False
        
        # Check if content looks like a real post
        return self._looks_like_post_content(content)

    def _clean_and_validate_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate extracted posts with improved deduplication"""
        cleaned_posts = []
        seen_post_ids = set()
        
        for post in posts:
            if not self._is_valid_post(post):
                continue
            
            # Generate a unique ID for the post
            post_id = self._generate_post_id(post)
            if not post_id or post_id in seen_post_ids:
                continue
                
            seen_post_ids.add(post_id)
            
            # Clean up the post data
            cleaned_post = {
                "content": post.get('content', '').strip(),
                "timestamp": post.get('timestamp', ''),
                "reactions": max(0, post.get('reactions', 0)),
                "comments": max(0, post.get('comments', 0)),
                "images": post.get('images', [])[:5],  # Limit images
                "links": post.get('links', [])[:3],    # Limit links
                "url": post.get('url', ''),
                "extraction_method": post.get('extraction_method', 'unknown')
            }
            
            cleaned_posts.append(cleaned_post)
        
        # Sort posts by timestamp (newest first)
        cleaned_posts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return cleaned_posts

    def save_posts_to_file(self, posts: List[Dict[str, Any]], filename: str = "linkedin_posts.json"):
        """Save scraped posts to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(posts, f, indent=2, ensure_ascii=False)
            logger.info(f"Posts saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save posts: {e}")

async def main():
    """Example usage"""
    scraper = LinkedInScraper()
    
    # Example company URLs
    company_urls = [
        "https://www.linkedin.com/company/microsoft/",
        "https://www.linkedin.com/company/google/",
        "https://www.linkedin.com/company/amazon/"
    ]
    
    for url in company_urls:
        try:
            logger.info(f"\n{'='*50}\nScraping posts for: {url}\n{'='*50}")
            posts = await scraper.scrape_company_posts(url, max_posts=3)  # Get 3 most recent posts
            
            if posts:
                logger.info(f"\nScraped {len(posts)} posts:")
                for i, post in enumerate(posts, 1):
                    logger.info(f"\nPost {i}:")
                    logger.info(f"Content: {post['content'][:200]}...")
                    logger.info(f"Timestamp: {post['timestamp']}")
                    logger.info(f"Reactions: {post['reactions']}")
                    logger.info(f"Comments: {post['comments']}")
                    logger.info(f"URL: {post['url']}")
                    
                # Save to file
                filename = f"linkedin_posts_{urlparse(url).path.split('/')[-2]}.json"
                scraper.save_posts_to_file(posts, filename)
            else:
                logger.warning("No posts found")
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            
        await asyncio.sleep(random.uniform(5, 10))  # Be polite with delays

if __name__ == "__main__":
    asyncio.run(main())