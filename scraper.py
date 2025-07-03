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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


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

class AdvancedGrantDetector:
    """Advanced grant opportunity detection system for Indian startups"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.min_grant_amount = 50000  # Minimum amount in INR to consider
        
        # Initialize NLP models (will be loaded on first use)
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        
        # Indian government schemes and funding bodies
        self.indian_funding_bodies = {
            'government': [
                'BIRAC', 'DSIR', 'DST', 'DIPP', 'MSME', 'SIDBI', 'MUDRA',
                'Startup India', 'Ministry of MSME', 'NSTEDB', 'TIFAC',
                'C-CAMP', 'KIIT-TBI', 'CIIE', 'FITT', 'SINE', 'NSRCEL',
                'Department of Science and Technology', 'Department of Biotechnology',
                'Ministry of Electronics and Information Technology',
                'Atal Innovation Mission', 'NITI Aayog', 'Government of India'
            ],
            'corporate': [
                'Tata Trust', 'Reliance Foundation', 'Infosys Foundation',
                'Wipro Foundation', 'Mahindra Rise', 'Godrej', 'L&T',
                'ITC', 'Bajaj', 'Aditya Birla', 'JSW Foundation'
            ],
            'incubators': [
                'T-Hub', 'NASSCOM', 'Indian Angel Network', 'Blume Ventures',
                'Accel Partners', 'Sequoia Capital', 'Kalaari Capital',
                'Matrix Partners', 'Lightspeed Ventures', 'Nexus Venture Partners'
            ]
        }
        
        # Indian startup-specific keywords and contexts
        self.startup_contexts = {
            'sectors': [
                'fintech', 'healthtech', 'edtech', 'agritech', 'deeptech',
                'cleantech', 'biotech', 'foodtech', 'logistics', 'e-commerce',
                'SaaS', 'enterprise software', 'mobile apps', 'AI/ML',
                'IoT', 'blockchain', 'renewable energy', 'sustainable tech'
            ],
            'stages': [
                'pre-seed', 'seed', 'pre-series A', 'series A', 'early stage',
                'prototype', 'MVP', 'proof of concept', 'pilot', 'scale-up',
                'bootstrapped', 'revenue generating', 'growth stage'
            ],
            'amounts': [
                'lakh', 'crore', 'INR', '₹', 'rupees', 'lakhs', 'crores',
                'up to', 'upto', 'maximum', 'funding of', 'grant of',
                'support of', 'assistance of', 'investment of'
            ]
        }
        
        # Grant opportunity indicators with weights
        self.grant_indicators = {
            'strong_indicators': {
                'weight': 0.4,
                'patterns': [
                    r'\b(applications?\s+(?:are\s+)?(?:now\s+)?(?:open|invited|welcome))\b',
                    r'\b(apply\s+(?:now|today|by|before))\b',
                    r'\b(deadline\s+(?:is|for|extended|approaching))\b',
                    r'\b(funding\s+(?:available|opportunity|program|scheme))\b',
                    r'\b(grant\s+(?:available|opportunity|program|scheme|competition))\b',
                    r'\b(call\s+for\s+(?:applications|proposals))\b',
                    r'\b(invite\s+(?:applications|proposals))\b',
                    r'\b(accepting\s+(?:applications|proposals))\b'
                ]
            },
            'medium_indicators': {
                'weight': 0.3,
                'patterns': [
                    r'\b(startup\s+(?:funding|grants|support|program))\b',
                    r'\b(entrepreneur\s+(?:funding|grants|support|program))\b',
                    r'\b(innovation\s+(?:funding|grants|support|program))\b',
                    r'\b(seed\s+(?:funding|grant|support))\b',
                    r'\b(incubation\s+(?:program|support))\b',
                    r'\b(accelerator\s+(?:program|support))\b',
                    r'\b(mentorship\s+(?:program|support))\b'
                ]
            },
            'weak_indicators': {
                'weight': 0.2,
                'patterns': [
                    r'\b(support\s+(?:startups|entrepreneurs|innovation))\b',
                    r'\b(help\s+(?:startups|entrepreneurs|innovators))\b',
                    r'\b(opportunity\s+for\s+(?:startups|entrepreneurs))\b',
                    r'\b(initiative\s+for\s+(?:startups|entrepreneurs))\b'
                ]
            },
            'negative_indicators': {
                'weight': -0.3,
                'patterns': [
                    r'\b(congratulations?\s+(?:to|on))\b',
                    r'\b(winners?\s+(?:of|announced))\b',
                    r'\b(selected\s+(?:for|as))\b',
                    r'\b(awarded\s+(?:to|the))\b',
                    r'\b(closed\s+(?:applications|program))\b',
                    r'\b(past\s+(?:event|program|opportunity))\b'
                ]
            }
        }
        
        # Date extraction patterns
        self.date_patterns = [
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})\b',
            r'\b((?:by|before|until|deadline)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?)\b'
        ]
        
        # Amount extraction patterns for Indian currency
        self.amount_patterns = [
            r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|crore|thousand)?',
            r'(?:INR|Rs\.?|Rupees)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|crore|thousand)?',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|crore)\s*(?:INR|Rs\.?|Rupees|₹)?',
            r'(?:up\s+to|upto|maximum|funding\s+of|grant\s+of)\s*(?:₹|INR|Rs\.?|Rupees)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|crore|thousand)?'
        ]

    def _load_nlp_models(self):
        """Load NLP models lazily to avoid initialization overhead"""
        if self.sentence_transformer is None:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_transformer = None
        
        if self.sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                logger.info("✅ Sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentiment analyzer: {e}")
                self.sentiment_analyzer = None

    def analyze_grant_opportunity(self, post_text: str, post_url: str = "", 
                                post_timestamp: str = "") -> Dict[str, Any]:
        """
        Comprehensive analysis of post content for grant opportunities
        
        Args:
            post_text: The text content of the post
            post_url: URL of the post (optional)
            post_timestamp: Timestamp of the post (optional)
            
        Returns:
            Dict containing detailed analysis results
        """
        if not post_text or len(post_text.strip()) < 50:
            return self._create_analysis_result(False, 0.0, "Insufficient content")
        
        # Clean and normalize text
        cleaned_text = self._clean_text(post_text)
        
        # Multi-layered analysis
        analysis = {
            'text_length': len(cleaned_text),
            'pattern_analysis': self._analyze_patterns(cleaned_text),
            'entity_analysis': self._analyze_entities(cleaned_text),
            'semantic_analysis': self._analyze_semantics(cleaned_text),
            'temporal_analysis': self._analyze_temporal_context(cleaned_text),
            'financial_analysis': self._analyze_financial_context(cleaned_text),
            'context_analysis': self._analyze_startup_context(cleaned_text),
            'sentiment_analysis': self._analyze_sentiment(cleaned_text)
        }
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(analysis)
        
        # Determine if it's a grant opportunity
        is_grant = confidence_score >= self.confidence_threshold
        
        # Extract key information if it's a grant
        grant_info = {}
        if is_grant:
            grant_info = self._extract_grant_details(cleaned_text, analysis)
        
        return self._create_analysis_result(is_grant, confidence_score, 
                                          "Analysis complete", analysis, grant_info)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove LinkedIn-specific noise
        text = re.sub(r'#\w+', '', text)  # Remove hashtags for cleaner analysis
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        
        return text.strip()

    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for grant indicators"""
        pattern_scores = {}
        total_score = 0.0
        matched_patterns = []
        
        for category, config in self.grant_indicators.items():
            weight = config['weight']
            patterns = config['patterns']
            category_matches = 0
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    category_matches += len(matches)
                    matched_patterns.extend(matches)
            
            category_score = min(category_matches * weight, abs(weight))
            pattern_scores[category] = category_score
            total_score += category_score
        
        return {
            'total_score': max(0, total_score),  # Ensure non-negative
            'category_scores': pattern_scores,
            'matched_patterns': matched_patterns,
            'pattern_count': len(matched_patterns)
        }

    def _analyze_entities(self, text: str) -> Dict[str, Any]:
        """Analyze entities relevant to Indian startup funding"""
        entities = {
            'funding_bodies': [],
            'government_schemes': [],
            'corporate_funders': [],
            'incubators': []
        }
        
        text_lower = text.lower()
        
        # Check for funding bodies
        for category, bodies in self.indian_funding_bodies.items():
            for body in bodies:
                if body.lower() in text_lower:
                    entities['funding_bodies'].append(body)
                    if category == 'government':
                        entities['government_schemes'].append(body)
                    elif category == 'corporate':
                        entities['corporate_funders'].append(body)
                    elif category == 'incubators':
                        entities['incubators'].append(body)
        
        # Calculate entity relevance score
        entity_score = (
            len(entities['funding_bodies']) * 0.3 +
            len(entities['government_schemes']) * 0.4 +
            len(entities['corporate_funders']) * 0.2 +
            len(entities['incubators']) * 0.1
        )
        
        return {
            'entities': entities,
            'entity_score': min(entity_score, 1.0),
            'has_credible_source': len(entities['funding_bodies']) > 0
        }

    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Semantic analysis using sentence transformers"""
        # Reference grant opportunity descriptions
        grant_references = [
            "startup funding opportunity application deadline",
            "government grant for entrepreneurs innovation",
            "seed funding program for startups",
            "incubator program accepting applications",
            "financial support for small business"
        ]
        
        semantic_score = 0.0
        similarities = []
        
        try:
            if self.sentence_transformer is None:
                self._load_nlp_models()
            
            if self.sentence_transformer is not None:
                # Calculate semantic similarity
                text_embedding = self.sentence_transformer.encode([text])
                ref_embeddings = self.sentence_transformer.encode(grant_references)
                
                similarities = cosine_similarity(text_embedding, ref_embeddings)[0]
                semantic_score = np.mean(similarities)
            
        except Exception as e:
            logger.debug(f"Semantic analysis failed: {e}")
        
        return {
            'semantic_score': float(semantic_score),
            'similarities': [float(s) for s in similarities],
            'is_semantically_relevant': str(semantic_score > 0.5)
        }

    def _analyze_temporal_context(self, text: str) -> Dict[str, Any]:
        """Analyze temporal context for deadlines and time-sensitive information"""
        dates_found = []
        deadline_indicators = []
        
        # Extract dates
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates_found.extend(matches)
        
        # Find deadline indicators
        deadline_patterns = [
            r'\b(deadline|due|expires?|closes?|ends?|last\s+date)\b',
            r'\b(apply\s+(?:by|before|until))\b',
            r'\b(submissions?\s+(?:by|before|until))\b'
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            deadline_indicators.extend(matches)
        
        # Calculate temporal relevance
        temporal_score = 0.0
        if dates_found:
            temporal_score += 0.3
        if deadline_indicators:
            temporal_score += 0.4
        if len(dates_found) > 1:  # Multiple dates might indicate application period
            temporal_score += 0.2
        
        return {
            'dates_found': dates_found,
            'deadline_indicators': deadline_indicators,
            'temporal_score': min(temporal_score, 1.0),
            'has_deadlines': len(deadline_indicators) > 0
        }

    def _analyze_financial_context(self, text: str) -> Dict[str, Any]:
        """Analyze financial context and funding amounts"""
        amounts_found = []
        financial_terms = []
        
        # Extract amounts
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts_found.extend(matches)
        
        # Find financial terms
        financial_patterns = [
            r'\b(funding|grant|investment|capital|finance|money)\b',
            r'\b(seed|series|round|equity|debt)\b',
            r'\b(₹|INR|Rs\.?|Rupees|lakh|crore)\b'
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_terms.extend(matches)
        
        # Calculate financial relevance
        financial_score = 0.0
        if amounts_found:
            financial_score += 0.5
        if financial_terms:
            financial_score += 0.3
        if len(set(financial_terms)) > 3:  # Diverse financial vocabulary
            financial_score += 0.2
        
        return {
            'amounts_found': amounts_found,
            'financial_terms': list(set(financial_terms)),
            'financial_score': min(financial_score, 1.0),
            'has_funding_amounts': len(amounts_found) > 0
        }

    def _analyze_startup_context(self, text: str) -> Dict[str, Any]:
        """Analyze startup and entrepreneurship context"""
        context_matches = {
            'sectors': [],
            'stages': [],
            'amounts': []
        }
        
        text_lower = text.lower()
        
        for category, terms in self.startup_contexts.items():
            for term in terms:
                if term.lower() in text_lower:
                    context_matches[category].append(term)
        
        # Calculate context relevance
        context_score = (
            len(context_matches['sectors']) * 0.3 +
            len(context_matches['stages']) * 0.4 +
            len(context_matches['amounts']) * 0.3
        ) / 3
        
        return {
            'context_matches': context_matches,
            'context_score': min(context_score, 1.0),
            'is_startup_relevant': any(context_matches.values())
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment to distinguish opportunities from announcements"""
        sentiment_score = 0.0
        sentiment_label = "NEUTRAL"
        
        try:
            if self.sentiment_analyzer is None:
                self._load_nlp_models()
            
            if self.sentiment_analyzer is not None:
                result = self.sentiment_analyzer(text[:512])  # Limit text length
                sentiment_label = result[0]['label']
                sentiment_score = result[0]['score']
        
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
        
        # Adjust score based on sentiment (positive sentiment more likely for opportunities)
        adjusted_score = sentiment_score if sentiment_label == "POSITIVE" else sentiment_score * 0.5
        
        return {
            'sentiment_label': sentiment_label,
            'sentiment_score': float(sentiment_score),
            'adjusted_score': float(adjusted_score),
            'is_positive': sentiment_label == "POSITIVE"
        }

    def _calculate_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for grant opportunity"""
        weights = {
            'pattern_analysis': 0.25,
            'entity_analysis': 0.20,
            'semantic_analysis': 0.15,
            'temporal_analysis': 0.15,
            'financial_analysis': 0.10,
            'context_analysis': 0.10,
            'sentiment_analysis': 0.05
        }
        
        total_score = 0.0
        
        for component, weight in weights.items():
            if component in analysis:
                component_score = 0.0
                
                if component == 'pattern_analysis':
                    component_score = analysis[component]['total_score']
                elif component == 'entity_analysis':
                    component_score = analysis[component]['entity_score']
                elif component == 'semantic_analysis':
                    component_score = analysis[component]['semantic_score']
                elif component == 'temporal_analysis':
                    component_score = analysis[component]['temporal_score']
                elif component == 'financial_analysis':
                    component_score = analysis[component]['financial_score']
                elif component == 'context_analysis':
                    component_score = analysis[component]['context_score']
                elif component == 'sentiment_analysis':
                    component_score = analysis[component]['adjusted_score']
                
                total_score += component_score * weight
        
        return min(max(total_score, 0.0), 1.0)  # Ensure score is between 0 and 1

    def _extract_grant_details(self, text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific grant details from high-confidence posts"""
        details = {
            'funding_body': None,
            'program_name': None,
            'deadlines': [],
            'funding_amounts': [],
            'eligibility_criteria': [],
            'application_process': [],
            'contact_information': []
        }
        
        # Extract funding body
        entities = analysis.get('entity_analysis', {}).get('entities', {})
        if entities.get('funding_bodies'):
            details['funding_body'] = entities['funding_bodies'][0]
        
        # Extract deadlines
        temporal = analysis.get('temporal_analysis', {})
        if temporal.get('dates_found'):
            details['deadlines'] = temporal['dates_found']
        
        # Extract funding amounts
        financial = analysis.get('financial_analysis', {})
        if financial.get('amounts_found'):
            details['funding_amounts'] = financial['amounts_found']
        
        # Extract program name (simple heuristic)
        program_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Program|Scheme|Initiative|Grant|Fund)\b',
            r'\b(?:Program|Scheme|Initiative|Grant|Fund)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        for pattern in program_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                details['program_name'] = matches[0]
                break
        
        return details

    def _create_analysis_result(self, is_grant: bool, confidence: float, 
                              message: str, analysis: Dict = None, 
                              grant_info: Dict = None) -> Dict[str, Any]:
        """Create standardized analysis result"""
        result = {
            'is_grant_opportunity': is_grant,
            'confidence_score': confidence,
            'analysis_message': message,
            'analyzed_at': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        if analysis:
            result['detailed_analysis'] = analysis
        
        if grant_info:
            result['grant_details'] = grant_info
        
        return result

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
        self.enable_grant_analysis = True  # Add this flag to control grant analysis
        
        # Initialize the advanced grant detector
        self.grant_detector = AdvancedGrantDetector()
        
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

                # Add advanced grant opportunity analysis
                if self.enable_grant_analysis:
                    for post in recent_posts:
                        post_text = post.get('text', '')
                        post_url = post.get('url', '')
                        post_timestamp = post.get('timestamp', '')
                        
                        # Perform comprehensive analysis
                        grant_analysis = self.grant_detector.analyze_grant_opportunity(
                            post_text, post_url, post_timestamp
                        )
                        
                        post['grant_analysis'] = grant_analysis
                        
                        # Log significant findings
                        if grant_analysis.get('is_grant_opportunity'):
                            confidence = grant_analysis.get('confidence_score', 0)
                            logger.info(f"🎯 Grant opportunity detected! Confidence: {confidence:.2f}")
                            
                            # Log grant details if available
                            grant_details = grant_analysis.get('grant_details', {})
                            if grant_details.get('funding_body'):
                                logger.info(f"📊 Funding Body: {grant_details['funding_body']}")
                            if grant_details.get('deadlines'):
                                logger.info(f"⏰ Deadlines: {grant_details['deadlines']}")
                
                logger.info(f"✅ Successfully extracted {len(recent_posts)} posts")
                return recent_posts
                
            except Exception as e:
                logger.error(f"Error scraping {company_url}: {str(e)}")
                await self._save_comprehensive_debug(page, "error_scrape", 0)
                return []
            finally:
                await browser.close()

    def _analyze_grant_opportunity(self, text: str) -> Dict[str, Any]:
        """Analyze post text for grant opportunities"""
        if not text:
            return {"is_grant": False, "analysis": {}}
        
        analysis = {
            "keywords": self._find_grant_keywords(text),
            "patterns": self._find_grant_patterns(text),
            "is_grant": False
        }
        
        # Simple logic to determine if this is likely a grant opportunity
        if (len(analysis["keywords"]) >= 3 or 
            any(p["pattern_name"] == "application_deadline" for p in analysis["patterns"])):
            analysis["is_grant"] = True
        
        return analysis
    
    def _find_grant_keywords(self, text: str) -> List[str]:
        """Find grant-related keywords in text"""
        text_lower = text.lower()
        return [kw for kw in self.grant_keywords if kw.lower() in text_lower]
    
    def _find_grant_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Apply regex patterns to identify grant characteristics"""
        matches = []
        
        for pattern_info in self.grant_patterns:
            try:
                pattern = re.compile(pattern_info['pattern'], flags=pattern_info.get('flags', 0))
                found = pattern.search(text)
                if found:
                    matches.append({
                        'pattern_name': pattern_info['name'],
                        'match': found.group(),
                        'span': (found.start(), found.end())
                    })
            except Exception as e:
                logger.debug(f"Pattern matching error for {pattern_info['name']}: {e}")
                continue
        
        return matches

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
            grant_stats = {
                "total_posts": 0,
                "grant_opportunities": 0,
                "high_confidence_grants": 0,
                "companies_with_grants": 0,
                "funding_bodies_identified": set(),
                "average_confidence": 0.0,
                "grant_details": []
            }

            total_confidence = 0.0
            confidence_count = 0

            for url, posts in results.items():
                clean_posts = []
                company_has_grants = False
                
                for post in posts:
                    grant_stats["total_posts"] += 1
                    
                    clean_post = {
                        'url': post.get('url', ''),
                        'timestamp': post.get('timestamp', ''),
                        'text': post.get('text', ''),
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    # Include comprehensive grant analysis
                    if 'grant_analysis' in post:
                        grant_analysis = post['grant_analysis']
                        clean_post['grant_analysis'] = grant_analysis
                        
                        # Update statistics
                        if grant_analysis.get('is_grant_opportunity', False):
                            grant_stats["grant_opportunities"] += 1
                            company_has_grants = True
                            
                            confidence = grant_analysis.get('confidence_score', 0.0)
                            total_confidence += confidence
                            confidence_count += 1
                            
                            if confidence >= 0.8:
                                grant_stats["high_confidence_grants"] += 1
                            
                            # Extract funding body info
                            grant_details = grant_analysis.get('grant_details', {})
                            if grant_details.get('funding_body'):
                                grant_stats["funding_bodies_identified"].add(grant_details['funding_body'])
                            
                            # Store detailed grant information
                            grant_info = {
                                'company_url': url,
                                'post_url': clean_post['url'],
                                'confidence_score': confidence,
                                'funding_body': grant_details.get('funding_body'),
                                'program_name': grant_details.get('program_name'),
                                'deadlines': grant_details.get('deadlines', []),
                                'funding_amounts': grant_details.get('funding_amounts', []),
                                'post_excerpt': clean_post['text'][:200] + "..." if len(clean_post['text']) > 200 else clean_post['text']
                            }
                            grant_stats["grant_details"].append(grant_info)

                    # Only include raw_html if debug mode is enabled
                    if self.debug_mode and post.get('raw_html'):
                        clean_post['raw_html'] = post['raw_html']
                    
                    clean_posts.append(clean_post)
                
                clean_results[url] = clean_posts
                if company_has_grants:
                    grant_stats["companies_with_grants"] += 1
            
            # Calculate average confidence
            if confidence_count > 0:
                grant_stats["average_confidence"] = total_confidence / confidence_count
            
            # Convert set to list for JSON serialization
            grant_stats["funding_bodies_identified"] = list(grant_stats["funding_bodies_identified"])
            
            # Add comprehensive metadata
            output_data = {
                'metadata': {
                    'scraped_at': datetime.now().isoformat(),
                    'total_companies': len(clean_results),
                    'total_posts': grant_stats["total_posts"],
                    'scraper_version': '3.0',
                    'grant_analysis_enabled': self.enable_grant_analysis,
                    'grant_detector_version': '2.0',
                    'confidence_threshold': self.grant_detector.confidence_threshold,
                    'debug_mode': self.debug_mode
                },
                'grant_statistics': grant_stats,
                'results': clean_results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Log comprehensive results
            logger.info(f"💾 Results saved to: {filepath}")
            logger.info(f"📊 Grant Analysis Summary:")
            logger.info(f"   • Total Posts Analyzed: {grant_stats['total_posts']}")
            logger.info(f"   • Grant Opportunities Found: {grant_stats['grant_opportunities']}")
            logger.info(f"   • High Confidence Grants: {grant_stats['high_confidence_grants']}")
            logger.info(f"   • Companies with Grants: {grant_stats['companies_with_grants']}")
            logger.info(f"   • Average Confidence: {grant_stats['average_confidence']:.2f}")
            logger.info(f"   • Funding Bodies Identified: {len(grant_stats['funding_bodies_identified'])}")
            
            if grant_stats["funding_bodies_identified"]:
                logger.info(f"   • Key Funding Bodies: {', '.join(grant_stats['funding_bodies_identified'][:5])}")
            
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

    def get_grant_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a summary of grant opportunities found"""
        summary = {
            'total_grants': 0,
            'high_confidence_grants': 0,
            'funding_bodies': set(),
            'grant_types': {},
            'upcoming_deadlines': [],
            'funding_amounts': [],
            'companies_with_opportunities': []
        }
        
        for company_url, posts in results.items():
            company_grants = []
            
            for post in posts:
                grant_analysis = post.get('grant_analysis', {})
                
                if grant_analysis.get('is_grant_opportunity', False):
                    summary['total_grants'] += 1
                    
                    confidence = grant_analysis.get('confidence_score', 0.0)
                    if confidence >= 0.8:
                        summary['high_confidence_grants'] += 1
                    
                    # Extract grant details
                    grant_details = grant_analysis.get('grant_details', {})
                    
                    if grant_details.get('funding_body'):
                        summary['funding_bodies'].add(grant_details['funding_body'])
                    
                    if grant_details.get('deadlines'):
                        summary['upcoming_deadlines'].extend(grant_details['deadlines'])
                    
                    if grant_details.get('funding_amounts'):
                        summary['funding_amounts'].extend(grant_details['funding_amounts'])
                    
                    company_grants.append({
                        'confidence': confidence,
                        'funding_body': grant_details.get('funding_body'),
                        'program_name': grant_details.get('program_name')
                    })
            
            if company_grants:
                summary['companies_with_opportunities'].append({
                    'company_url': company_url,
                    'grants': company_grants
                })
        
        # Convert sets to lists for JSON serialization
        summary['funding_bodies'] = list(summary['funding_bodies'])
        
        return summary

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