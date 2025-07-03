import asyncio
import random
import logging
import time
import re
import os  
from typing import Optional, Dict, List, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import json
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from dateutil import parser as date_parser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linkedin_scraper_grant_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedGrantOpportunityAnalyzer:
    def __init__(self, hf_token: str = None):
        logger.info("🧠 Initializing Advanced Grant Opportunity Analyzer...")
        
        self.hf_token = hf_token
        
        try:
            model_name = "microsoft/DialoGPT-medium"
            if hf_token:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/bart-large-mnli",
                    token=hf_token
                )
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    token=hf_token,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            self.opportunity_labels = [
                "active funding opportunity",
                "startup grant program", 
                "accelerator program accepting applications",
                "investment competition open",
                "mentorship program available",
                "networking opportunity for startups",
                "completed past event",
                "general news update",
                "promotional content only"
            ]
            
            logger.info("✅ Advanced Grant Opportunity Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {e}")
            self.classifier = None
            self.tokenizer = None
    
    def extract_time_sensitivity(self, text: str) -> Dict:
        current_date = datetime.now()
        
        urgent_indicators = [
            r'deadline.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'apply.*?by.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'last.*?(\d+).*?days?',
            r'(\d+).*?days?.*?left',
            r'closing.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'until.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
        ]
        
        immediate_urgency = [
            'apply now', 'last day', 'deadline today', 'closing soon',
            'few days left', 'urgent', 'immediate', 'asap'
        ]
        
        ongoing_indicators = [
            'now accepting', 'applications open', 'currently accepting',
            'register now', 'join now', 'participate', 'enroll'
        ]
        
        past_indicators = [
            'congratulations', 'winners announced', 'completed', 'concluded',
            'was held', 'took place', 'participated in', 'attended',
            'recap', 'highlights from', 'thank you for'
        ]
        
        text_lower = text.lower()
        
        urgency_score = 0
        status = "unknown"
        reasons = []
        
        if any(indicator in text_lower for indicator in past_indicators):
            status = "past_event"
            urgency_score = 0
            reasons.append("Contains past event indicators")
        elif any(indicator in text_lower for indicator in immediate_urgency):
            status = "urgent_opportunity"
            urgency_score = 1.0
            reasons.append("Contains immediate urgency indicators")
        elif any(indicator in text_lower for indicator in ongoing_indicators):
            status = "active_opportunity"
            urgency_score = 0.8
            reasons.append("Contains active opportunity indicators")
        elif any(re.search(pattern, text_lower) for pattern in urgent_indicators):
            status = "time_sensitive_opportunity"
            urgency_score = 0.9
            reasons.append("Contains specific deadline information")
        
        for pattern in urgent_indicators:
            matches = re.findall(pattern, text_lower)
            if matches:
                reasons.append(f"Found deadline/time reference: {matches[0]}")
                break
        
        return {
            "status": status,
            "urgency_score": urgency_score,
            "reasons": reasons
        }
    
    def extract_opportunity_type(self, text: str) -> Dict:
        funding_patterns = {
            "direct_funding": [
                r'grant.*?(?:up to|upto|\₹|rs\.?|inr).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|crore|l|cr)',
                r'funding.*?(?:up to|upto|\₹|rs\.?|inr).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|crore|l|cr)',
                r'financial support.*?(?:up to|upto|\₹|rs\.?|inr).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|crore|l|cr)'
            ],
            "accelerator_program": [
                'accelerator', 'launchpad', 'cohort', 'bootcamp', 'incubation'
            ],
            "competition": [
                'competition', 'challenge', 'hackathon', 'pitch', 'demo day'
            ],
            "mentorship": [
                'mentorship', 'mentoring', 'guidance', 'expert support'
            ],
            "networking": [
                'networking', 'connect', 'ecosystem', 'community'
            ]
        }
        
        text_lower = text.lower()
        opportunity_types = []
        funding_amounts = []
        
        for opp_type, patterns in funding_patterns.items():
            if opp_type == "direct_funding":
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        opportunity_types.append("direct_funding")
                        funding_amounts.extend(matches)
            else:
                if any(pattern in text_lower for pattern in patterns):
                    opportunity_types.append(opp_type)
        
        return {
            "types": list(set(opportunity_types)),
            "funding_amounts": funding_amounts,
            "has_direct_funding": "direct_funding" in opportunity_types
        }
    
    def analyze_startup_relevance(self, text: str) -> float:
        startup_positive_terms = [
            'startup', 'entrepreneur', 'founder', 'early stage', 'seed stage',
            'prototype', 'mvp', 'innovation', 'disruptive', 'scalable',
            'tech startup', 'deep tech', 'fintech', 'edtech', 'healthtech'
        ]
        
        startup_stages = [
            'idea stage', 'prototype', 'proof of concept', 'poc', 'pre-revenue',
            'early traction', 'pilot', 'beta', 'validation'
        ]
        
        eligibility_terms = [
            'early stage startups', 'startups with', 'founded in',
            'registered startups', 'incorporated'
        ]
        
        text_lower = text.lower()
        
        relevance_score = 0
        
        startup_matches = sum(1 for term in startup_positive_terms if term in text_lower)
        stage_matches = sum(1 for term in startup_stages if term in text_lower)
        eligibility_matches = sum(1 for term in eligibility_terms if term in text_lower)
        
        relevance_score = min(1.0, (startup_matches * 0.15) + (stage_matches * 0.25) + (eligibility_matches * 0.3))
        
        return relevance_score
    
    def analyze_grant_opportunity(self, post_content: str, post_timestamp: str = None) -> Dict:
        if not self.classifier or not post_content:
            return {
                "opportunity_score": 0.0,
                "opportunity_type": "unknown",
                "actionable": False,
                "confidence": 0.0,
                "reason": "Unable to analyze"
            }
        
        try:
            result = self.classifier(post_content, self.opportunity_labels)
            
            time_analysis = self.extract_time_sensitivity(post_content)
            opportunity_analysis = self.extract_opportunity_type(post_content)
            startup_relevance = self.analyze_startup_relevance(post_content)
            
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            is_active_opportunity = top_label in [
                "active funding opportunity",
                "startup grant program", 
                "accelerator program accepting applications",
                "investment competition open",
                "mentorship program available"
            ]
            
            is_past_event = (
                top_label == "completed past event" or 
                time_analysis['status'] == 'past_event'
            )
            
            final_score = 0.0
            actionable = False
            opportunity_type = "none"
            
            if is_past_event:
                final_score = 0.0
                actionable = False
                opportunity_type = "past_event"
            elif is_active_opportunity:
                base_score = top_score
                time_boost = time_analysis['urgency_score'] * 0.3
                relevance_boost = startup_relevance * 0.4
                funding_boost = 0.2 if opportunity_analysis['has_direct_funding'] else 0.0
                
                final_score = min(1.0, base_score + time_boost + relevance_boost + funding_boost)
                actionable = final_score > 0.6
                opportunity_type = "active_opportunity"
            
            confidence = min(0.95, top_score + 0.1)
            
            analysis_reasons = []
            analysis_reasons.extend(time_analysis.get('reasons', []))
            
            if opportunity_analysis['types']:
                analysis_reasons.append(f"Opportunity types: {', '.join(opportunity_analysis['types'])}")
            
            if opportunity_analysis['funding_amounts']:
                analysis_reasons.append(f"Funding mentioned: {', '.join(opportunity_analysis['funding_amounts'])}")
            
            if startup_relevance > 0.3:
                analysis_reasons.append(f"High startup relevance score: {startup_relevance:.2f}")
            
            return {
                "opportunity_score": round(final_score, 3),
                "opportunity_type": opportunity_type,
                "actionable": actionable,
                "confidence": round(confidence, 3),
                "time_sensitivity": time_analysis,
                "opportunity_details": opportunity_analysis,
                "startup_relevance": round(startup_relevance, 3),
                "ml_classification": {
                    "top_label": top_label,
                    "score": round(top_score, 3),
                    "all_scores": {label: round(score, 3) for label, score in zip(result['labels'][:3], result['scores'][:3])}
                },
                "analysis_reasons": analysis_reasons[:5]
            }
            
        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}")
            return {
                "opportunity_score": 0.0,
                "opportunity_type": "error",
                "actionable": False,
                "confidence": 0.0,
                "reason": f"Analysis error: {str(e)}"
            }

class EnhancedLinkedInScraper:
    def __init__(self, headless: bool = False, slow_mo: int = 100, hf_token: str = None):
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = 45000
        self.max_retries = 3
        
        self.opportunity_analyzer = AdvancedGrantOpportunityAnalyzer(hf_token)
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        ]
        
        self.post_selectors = [
            '.main-feed-activity-card',
            '[data-test-id="main-feed-activity-card"]',
            '.org-recent-activity-card',
            '.feed-shared-update-v2',
            '.update-components-text',
            '.share-update-card'
        ]
        
        self.content_selectors = [
            '[data-test-id="main-feed-activity-card__commentary"]',
            '.attributed-text-segment-list__content',
            '.feed-shared-text',
            '.update-components-text__text-view',
            '.feed-shared-update-v2__description',
            '.org-recent-activity-card__text'
        ]
        
        self.timestamp_selectors = [
            'time',
            '.time-ago',
            '.posted-time',
            '[datetime]'
        ]

    async def get_company_posts_data(self, company_name: str, max_posts: int = 5) -> Dict:
        urls_to_try = [
            f"https://www.linkedin.com/company/{company_name}/",
            f"https://www.linkedin.com/company/{company_name}/posts/",
        ]
        
        for attempt in range(self.max_retries):
            logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries} for company: {company_name}")
            
            async with async_playwright() as p:
                browser = await self._launch_browser(p)
                context = await self._create_context(browser)
                page = await context.new_page()
                
                try:
                    for url in urls_to_try:
                        logger.info(f"🌐 Trying URL: {url}")
                        
                        try:
                            await self._navigate_with_retry(page, url)
                            await self._handle_popups(page)
                            await self._load_content(page)
                            
                            posts_data = await self._extract_posts_data(page, company_name, max_posts)
                            
                            if posts_data['posts']:
                                logger.info(f"✅ SUCCESS: Found {len(posts_data['posts'])} posts for {company_name}")
                                return posts_data
                            else:
                                logger.warning(f"⚠️ No posts found with URL: {url}")
                                
                        except Exception as e:
                            logger.error(f"❌ Error with URL {url}: {str(e)}")
                            continue
                    
                finally:
                    await browser.close()
            
            if attempt < self.max_retries - 1:
                delay = random.uniform(3, 8)
                logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
        
        logger.error(f"❌ FAILED: Could not get posts for {company_name}")
        return {'company': company_name, 'posts': [], 'error': 'No posts found'}

    async def _launch_browser(self, playwright) -> Browser:
        return await playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--window-size=1920,1080',
                '--disable-images',
            ]
        )

    async def _create_context(self, browser: Browser) -> BrowserContext:
        user_agent = random.choice(self.user_agents)
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
        )
        
        await context.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,mp4,mp3,pdf}", lambda route: route.abort())
        await context.route("**/analytics**", lambda route: route.abort())
        
        return context

    async def _navigate_with_retry(self, page: Page, url: str):
        logger.info(f"🌐 Navigating to: {url}")
        await asyncio.sleep(random.uniform(2, 4))
        
        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
        await page.wait_for_timeout(4000)

    async def _handle_popups(self, page: Page):
        logger.info("🔍 Handling popups...")
        await page.wait_for_timeout(3000)
        
        popup_selectors = [
            'button[aria-label="Dismiss"]',
            'button[data-test-modal-close-btn]',
            'button[aria-label="Close"]',
            '.artdeco-modal__dismiss',
            '.sign-in-modal__dismiss-btn',
        ]
        
        for selector in popup_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if element and await element.is_visible():
                        await element.click()
                        await page.wait_for_timeout(2000)
                        break
            except:
                continue

    async def _load_content(self, page: Page):
        logger.info("📦 Loading content...")
        
        try:
            await page.wait_for_selector('main, .application-outlet, body', timeout=15000)
            
            for i in range(3):
                scroll_position = (i + 1) * 500
                await page.evaluate(f"window.scrollTo(0, {scroll_position})")
                await page.wait_for_timeout(2000)
            
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(3000)
            
        except Exception as e:
            logger.warning(f"Content loading issues: {e}")

    async def _extract_posts_data(self, page: Page, company_name: str, max_posts: int) -> Dict:
        logger.info(f"🔍 Extracting posts data for {company_name}")
        
        posts_data = {
            'company': company_name,
            'scraped_at': datetime.now().isoformat(),
            'posts': []
        }
        
        post_elements = []
        for selector in self.post_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    post_elements.extend(elements[:max_posts])
                    logger.info(f"Found {len(elements)} posts with selector: {selector}")
                    break
            except Exception as e:
                logger.debug(f"Selector error {selector}: {e}")
                continue
        
        if not post_elements:
            logger.warning("No post elements found")
            return posts_data
        
        for i, post_element in enumerate(post_elements[:max_posts]):
            try:
                post_data = await self._extract_single_post_data(post_element, i + 1)
                if post_data:
                    if post_data.get('content'):
                        logger.info(f"🧠 Analyzing opportunity potential for post {i + 1}")
                        opportunity_analysis = self.opportunity_analyzer.analyze_grant_opportunity(
                            post_data['content'], 
                            post_data.get('timestamp')
                        )
                        post_data['opportunity_analysis'] = opportunity_analysis
                        
                        score = opportunity_analysis.get('opportunity_score', 0)
                        actionable = opportunity_analysis.get('actionable', False)
                        opp_type = opportunity_analysis.get('opportunity_type', 'unknown')
                        
                        logger.info(f"📊 Opportunity Score: {score:.3f} | Actionable: {actionable} | Type: {opp_type}")
                    
                    posts_data['posts'].append(post_data)
                    logger.info(f"✅ Extracted post {i + 1}: {post_data.get('timestamp', 'No timestamp')}")
                else:
                    logger.warning(f"⚠️ Failed to extract post {i + 1}")
                    
            except Exception as e:
                logger.error(f"Error extracting post {i + 1}: {e}")
                continue
        
        logger.info(f"📊 Total posts extracted: {len(posts_data['posts'])}")
        return posts_data

    async def _extract_single_post_data(self, post_element, post_index: int) -> Optional[Dict]:
        post_data = {
            'post_index': post_index,
            'timestamp': None,
            'content': None,
            'author': None,
            'engagement': {}
        }
        
        try:
            timestamp_elements = await post_element.query_selector_all('time')
            for time_el in timestamp_elements:
                if await time_el.is_visible():
                    datetime_attr = await time_el.get_attribute('datetime')
                    if datetime_attr:
                        post_data['timestamp'] = self._standardize_timestamp(datetime_attr)
                        break
                    
                    text = await time_el.inner_text()
                    if text and self._is_valid_timestamp_text(text):
                        post_data['timestamp'] = self._standardize_timestamp(text.strip())
                        break
            
            for content_selector in self.content_selectors:
                try:
                    content_el = await post_element.query_selector(content_selector)
                    if content_el and await content_el.is_visible():
                        content_text = await content_el.inner_text()
                        if content_text and len(content_text.strip()) > 10:
                            post_data['content'] = content_text.strip()
                            break
                except:
                    continue
            
            try:
                author_elements = await post_element.query_selector_all('a[href*="/company/"]')
                for author_el in author_elements:
                    author_text = await author_el.inner_text()
                    if author_text and len(author_text.strip()) > 0:
                        post_data['author'] = author_text.strip()
                        break
            except:
                pass
            
            try:
                engagement_elements = await post_element.query_selector_all('[aria-label*="reaction"], [aria-label*="comment"], [aria-label*="like"]')
                for eng_el in engagement_elements:
                    aria_label = await eng_el.get_attribute('aria-label')
                    if aria_label:
                        numbers = re.findall(r'\d+', aria_label)
                        if numbers:
                            if 'like' in aria_label.lower() or 'reaction' in aria_label.lower():
                                post_data['engagement']['reactions'] = int(numbers[0])
                            elif 'comment' in aria_label.lower():
                                post_data['engagement']['comments'] = int(numbers[0])
            except:
                pass
            
            if post_data['timestamp'] or post_data['content']:
                return post_data
            
        except Exception as e:
            logger.error(f"Error in single post extraction: {e}")
        
        return None

    def _is_valid_timestamp_text(self, text: str) -> bool:
        if not text or len(text) > 50:
            return False
            
        text = text.lower().strip()
        patterns = [
            r'\d+\s*(?:second|minute|hour|day|week|month|year|sec|min|hr|h|d|w|mo|y)s?\b',
            r'just now', r'now', r'yesterday', r'today'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _standardize_timestamp(self, timestamp: str) -> str:
        if not timestamp:
            return ""
        
        timestamp = timestamp.strip()
        
        try:
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                now = datetime.now()
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                
                diff = now - dt
                if diff.days > 0:
                    return f"{diff.days}d ago"
                elif diff.seconds >= 3600:
                    return f"{diff.seconds // 3600}h ago"
                elif diff.seconds >= 60:
                    return f"{diff.seconds // 60}m ago"
                else:
                    return "just now"
        except:
            pass
        
        timestamp = timestamp.lower()
        replacements = {
            'seconds': 's', 'second': 's', 'minutes': 'm', 'minute': 'm',
            'hours': 'h', 'hour': 'h', 'days': 'd', 'day': 'd',
            'weeks': 'w', 'week': 'w', 'months': 'mo', 'month': 'mo'
        }
        
        for old, new in replacements.items():
            timestamp = re.sub(f'\\b{old}\\b', new, timestamp)
        
        return re.sub(r'\s*ago\s*', '', timestamp).strip()

    async def scrape_companies_posts(self, companies: List[str], max_posts_per_company: int = 5) -> Dict[str, Dict]:
        results = {}
        
        logger.info(f"🚀 Starting posts scrape of {len(companies)} companies")
        
        for i, company in enumerate(companies, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🏢 PROCESSING {i}/{len(companies)}: {company}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            company_data = await self.get_company_posts_data(company, max_posts_per_company)
            end_time = time.time()
            
            results[company] = company_data
            
            logger.info(f"⏱️ {company} processed in {end_time - start_time:.1f}s")
            
            if i < len(companies):
                delay = random.uniform(5, 10)
                logger.info(f"⏳ Waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return results

    def save_results(self, results: Dict, filename_prefix: str = "linkedin_opportunity_analysis"):
        timestamp = int(time.time())
        filename = f"{filename_prefix}_{timestamp}.json"
        
        actionable_opportunities = 0
        high_score_opportunities = 0
        total_analyzed_posts = 0
        
        for data in results.values():
            for post in data.get('posts', []):
                if 'opportunity_analysis' in post:
                    total_analyzed_posts += 1
                    analysis = post['opportunity_analysis']
                    if analysis.get('actionable', False):
                        actionable_opportunities += 1
                    if analysis.get('opportunity_score', 0) > 0.7:
                        high_score_opportunities += 1
        
        summary = {
            'scraped_at': datetime.now().isoformat(),
            'total_companies': len(results),
            'companies_with_posts': sum(1 for data in results.values() if data.get('posts')),
            'total_posts': sum(len(data.get('posts', [])) for data in results.values()),
            'posts_analyzed': total_analyzed_posts,
            'actionable_opportunities': actionable_opportunities,
            'high_score_opportunities': high_score_opportunities,
            'actionable_rate': round(actionable_opportunities / max(total_analyzed_posts, 1) * 100, 1)
        }
        
        output_data = {
            'summary': summary,
            'data': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Results saved to: {filename}")
        return filename

async def main():
    companies = [
        "iitmandicatalyst", 
    ]
    
    HF_TOKEN = os.getenv("HF_API_KEY")
    
    scraper = EnhancedLinkedInScraper(
        headless=False,
        slow_mo=100,
        hf_token=HF_TOKEN
    )
    
    print("🚀 ADVANCED LINKEDIN OPPORTUNITY ANALYZER")
    print("="*50)
    
    try:
        results = await scraper.scrape_companies_posts(companies, max_posts_per_company=3)
        
        filename = scraper.save_results(results)
        
        total_posts = sum(len(data.get('posts', [])) for data in results.values())
        actionable = sum(1 for data in results.values() 
                        for post in data.get('posts', []) 
                        if post.get('opportunity_analysis', {}).get('actionable', False))
        
        high_score = sum(1 for data in results.values() 
                        for post in data.get('posts', []) 
                        if post.get('opportunity_analysis', {}).get('opportunity_score', 0) > 0.7)
        
        print(f"\n📊 SCRAPING SUMMARY:")
        print(f"   Companies processed: {len(companies)}")
        print(f"   Total posts found: {total_posts}")
        print(f"   Actionable opportunities: {actionable}")
        print(f"   High-score opportunities: {high_score}")
        print(f"   Results saved to: {filename}")
        
        # Display actionable opportunities
        print(f"\n🎯 ACTIONABLE OPPORTUNITIES:")
        print("="*50)
        
        for company, data in results.items():
            company_actionable = [post for post in data.get('posts', []) 
                                if post.get('opportunity_analysis', {}).get('actionable', False)]
            
            if company_actionable:
                print(f"\n🏢 {company.upper()}:")
                for i, post in enumerate(company_actionable, 1):
                    analysis = post['opportunity_analysis']
                    score = analysis.get('opportunity_score', 0)
                    opp_type = analysis.get('opportunity_type', 'unknown')
                    timestamp = post.get('timestamp', 'Unknown time')
                    
                    print(f"   {i}. Score: {score:.3f} | Type: {opp_type} | Posted: {timestamp}")
                    
                    # Show first 100 chars of content
                    content = post.get('content', '')
                    if content:
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"      Preview: {preview}")
                    
                    # Show key analysis reasons
                    reasons = analysis.get('analysis_reasons', [])
                    if reasons:
                        print(f"      Reasons: {reasons[0]}")
                    print()
        
        print("🎉 Analysis complete! Check the JSON file for detailed results.")
        
    except KeyboardInterrupt:
        logger.info("🛑 Scraping interrupted by user")
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())