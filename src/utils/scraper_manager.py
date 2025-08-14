"""
Enhanced scraper manager module for handling advanced scraping scenarios
"""
import random
import time
from typing import Dict, Any, List, Callable, Optional
import requests
from bs4 import BeautifulSoup
from loguru import logger
import asyncio

class ScraperManager:
    """Helper class for managing web scraping tasks with advanced features"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the scraper manager
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will be exponentially increased)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a new session with randomized headers"""
        session = requests.Session()
        
        # Randomize user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
        
        # Create browser-like headers
        session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-user': '?1',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        return session
    
    async def fetch_with_retry(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Fetch a URL with retry logic
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Add a small random delay between attempts
                if attempt > 0:
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                    await asyncio.sleep(delay)
                
                # Refresh session if not first attempt
                if attempt > 0:
                    self.session = self._create_session()
                
                # Make the request
                logger.debug(f"Fetching {url} (attempt {attempt+1}/{self.max_retries})")
                response = self.session.get(url, timeout=timeout)
                
                # Check for common anti-bot challenges
                if self._is_blocked(response):
                    logger.warning(f"Detected anti-bot measure on {url}. Retrying with new session.")
                    continue
                    
                return response
                
            except requests.RequestException as e:
                logger.warning(f"Request failed: {e} (attempt {attempt+1}/{self.max_retries})")
                
        logger.error(f"All {self.max_retries} attempts to fetch {url} failed")
        return None
    
    def _is_blocked(self, response: requests.Response) -> bool:
        """Check if response indicates we're being blocked
        
        Args:
            response: Response to check
            
        Returns:
            True if blocked, False otherwise
        """
        if response.status_code in (403, 429, 503):
            return True
            
        # Check for common captcha indicators
        captcha_indicators = ['captcha', 'robot', 'automated', 'blocked', 'suspicious']
        
        for indicator in captcha_indicators:
            if indicator in response.text.lower():
                return True
                
        return False
    
    async def paginated_fetch(self, 
                             base_url: str, 
                             extract_func: Callable[[BeautifulSoup], List[Any]],
                             max_pages: int = 2,
                             page_param: str = 'page',
                             starting_page: int = 1) -> List[Any]:
        """Fetch and extract data from multiple pages
        
        Args:
            base_url: Base URL to fetch from
            extract_func: Function to extract items from each page
            max_pages: Maximum number of pages to fetch
            page_param: URL parameter name for pagination
            starting_page: Page number to start from
            
        Returns:
            List of extracted items from all pages
        """
        all_items = []
        
        for page_num in range(starting_page, starting_page + max_pages):
            # Construct page URL
            if '?' in base_url:
                page_url = f"{base_url}&{page_param}={page_num}"
            else:
                page_url = f"{base_url}?{page_param}={page_num}"
                
            logger.info(f"Fetching page {page_num}: {page_url}")
            
            # Fetch the page
            response = await self.fetch_with_retry(page_url)
            
            if not response or response.status_code != 200:
                logger.warning(f"Failed to fetch page {page_num}. Stopping pagination.")
                break
                
            # Parse and extract items
            soup = BeautifulSoup(response.text, 'html.parser')
            page_items = extract_func(soup)
            
            logger.info(f"Found {len(page_items)} items on page {page_num}")
            all_items.extend(page_items)
            
            # If we got fewer items than expected, we might be on the last page
            if len(page_items) == 0:
                logger.info(f"No more items found on page {page_num}. Stopping pagination.")
                break
                
            # Add a small delay between pages to be respectful
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
        return all_items
