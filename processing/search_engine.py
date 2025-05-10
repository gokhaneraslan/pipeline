import re
import logging
import urllib.parse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup, SoupStrainer
from googlesearch import search as google_search_lib

from core.utils import fetch_with_retry, HEADERS as DEFAULT_HTTP_HEADERS


logger = logging.getLogger("DataPipeline")

GOOGLE_SEARCH_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"


def search_with_duckduckgo(query: str, num_results: int = 10, lang: str = "en") -> List[Dict[str, str]]:
    """
    Performs a search on DuckDuckGo's HTML interface and returns results.
    NOTE: This relies on HTML scraping and can break if DDG changes its layout.
    """
    
    if not query:
        logger.warning("Empty query provided to DuckDuckGo search.")
        return []

    encoded_query = urllib.parse.quote_plus(query)
    
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    logger.info(f"Searching DuckDuckGo for: '{query}' (requesting ~{num_results} results, lang via header: {lang})")

    ddg_headers = DEFAULT_HTTP_HEADERS.copy()
    ddg_headers['Accept-Language'] = f'{lang.lower()}-{lang.upper()}, {lang.lower()};q=0.9,en-US;q=0.8,en;q=0.7' if lang else DEFAULT_HTTP_HEADERS.get('Accept-Language', 'en-US,en;q=0.9')
    ddg_headers['Referer'] = 'https://duckduckgo.com/'

    response = fetch_with_retry(url, headers=ddg_headers)

    if not response or response.status_code != 200:
        logger.error(
            f"DuckDuckGo search failed for '{query}'. Status: {response.status_code if response else 'No Response'}"
        )
        return []

    try:

        #parse_only = SoupStrainer('div', class_='result')
        #soup = BeautifulSoup(response.text, 'html.parser', parse_only=parse_only)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        search_results_data: List[Dict[str, str]] = []
        
        html_results = soup.find_all(['div', 'article'], class_=re.compile(r'\bresult\b'))

        if not html_results:
            logger.warning(f"No result elements found in DuckDuckGo HTML for query: '{query}'. Page structure might have changed.")
            return []

        logger.debug(f"Found {len(html_results)} potential result elements from DuckDuckGo HTML for query '{query}'.")

        for result_idx, result_tag in enumerate(html_results):
            if len(search_results_data) >= num_results:
                break

            title_tag = result_tag.find(['h2','a'], class_=re.compile(r'result__title|result__a|title|links_main'))
            title = title_tag.get_text(strip=True) if title_tag else "No title found"
            
            link_tag = None
            if title_tag and title_tag.name == 'a' and title_tag.has_attr('href'):
                link_tag = title_tag
            else:
                link_tag = result_tag.find('a', class_=re.compile(r'result__url|result__a|links_main'), href=True)

            raw_link = link_tag['href'] if link_tag and link_tag.has_attr('href') else None

            if not raw_link:
                logger.debug(f"Skipping DDG result #{result_idx+1} due to missing link. Title: '{title}'")
                continue

            cleaned_link = raw_link
            if raw_link.startswith("/l/"):
                match = re.search(r'uddg=([^&]+)', raw_link)
                if match:
                    try:
                        cleaned_link = urllib.parse.unquote(match.group(1))
                    except Exception as e:
                        logger.warning(f"Error unquoting DDG link part '{match.group(1)}': {e}. Using raw link segment.")
                        cleaned_link = match.group(1) 
                else: 
                    logger.debug(f"DDG link '{raw_link}' does not contain 'uddg=', trying to use as is or find alternative.")

            
            if not cleaned_link.startswith("http"):
                if cleaned_link.startswith("//"):
                    cleaned_link = "https:" + cleaned_link
                    
                elif cleaned_link.startswith("/"):
                    cleaned_link = urllib.parse.urljoin("https://duckduckgo.com", cleaned_link)
                    
                else:
                    logger.warning(f"Skipping DDG result with unusual relative link: {title} - {cleaned_link}")
                    continue


            snippet_tag = result_tag.find(['div', 'span', 'a'], class_=re.compile(r'result__snippet|snippet|result__body'))
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet found"
            
            if cleaned_link and cleaned_link.startswith("http"):
                search_results_data.append({
                    'title': title,
                    'link': cleaned_link,
                    'snippet': snippet,
                    'source': 'DuckDuckGo'
                })
                
            elif cleaned_link:
                logger.debug(f"Skipping DDG result with non-HTTP link after processing: {title} - {cleaned_link}")

        logger.info(f"DuckDuckGo search for '{query}' yielded {len(search_results_data)} valid results (requested {num_results}).")
        
        return search_results_data[:num_results]

    except Exception as e:
        logger.error(f"Error parsing DuckDuckGo search results for '{query}': {e}", exc_info=True)
        return []


def search_with_google(query: str, num_results: int = 10, lang: str = "en") -> List[Dict[str, str]]:
    """
    Performs a search on Google using the googlesearch-python library.
    NOTE: This library scrapes Google and is prone to rate limiting or CAPTCHAs.
    """
    
    if not query:
        logger.warning("Empty query provided to Google search.")
        return []

    logger.info(f"Searching Google for: '{query}' (requesting {num_results} results, lang: {lang})")
    search_results_data: List[Dict[str, str]] = []

    try:

        google_result_urls_generator = google_search_lib(
            query,
            lang=lang,
            num_results=num_results,
        )
        
        count = 0
        
        for url_str in google_result_urls_generator:
            if count >= num_results:
                break
            
            if isinstance(url_str, str) and url_str.startswith("http"):
                 search_results_data.append({
                    'title': f'Title not available via googlesearch-python ({query[:30]}...)',
                    'link': url_str,
                    'snippet': 'Snippet not available via googlesearch-python.',
                    'source': 'Google (googlesearch-python)'
                })
                 
                 count += 1
                 
            else:
                logger.debug(f"Skipping unexpected Google result format/value: {type(url_str)} - {str(url_str)[:100]}")

        if count == 0 and num_results > 0:
            logger.warning(f"Google search (googlesearch-python) for '{query}' yielded 0 results. This might be due to rate limiting/CAPTCHA.")
        
        else:
            logger.info(f"Google search (googlesearch-python) for '{query}' yielded {len(search_results_data)} URLs.")
        
        return search_results_data


    except urllib.error.HTTPError as he:
        
        if he.code == 429:
             logger.error(f"Google search for '{query}' failed due to HTTP 429 (Too Many Requests). Google is rate limiting. Try increasing 'pause' or using a proxy/VPN.", exc_info=True)
       
        else:
             logger.error(f"Google search for '{query}' failed due to HTTPError {he.code}: {he}", exc_info=True)
    
    except Exception as e:
        logger.error(
            f"An error occurred during Google search (googlesearch-python) for '{query}': {e}. "
            "This might be due to Google blocking requests, network issues, or library limitations.",
            exc_info=True
        )
        return []


def perform_search(
    query: str,
    search_engine_type: str,
    num_results: int,
    search_language: Optional[str] = "en") -> List[Dict[str, str]]:
    """
    Facade function to perform search using the specified engine.
    Returns a list of result dictionaries, each with 'title', 'link', 'snippet', 'source'.
    """
    
    engine_type_lower = search_engine_type.lower()
    lang = search_language if search_language else "en"

    logger.info(f"Performing search for query '{query}' using engine: {engine_type_lower}, num_results: {num_results}, lang: {lang}")

    if engine_type_lower == "google":
        return search_with_google(query, num_results, lang=lang)
    
    elif engine_type_lower == "duckduckgo":
        return search_with_duckduckgo(query, num_results, lang=lang)
    
    else:
        logger.error(f"Unsupported search engine type: {search_engine_type}. Supported: 'google', 'duckduckgo'.")
        return []
