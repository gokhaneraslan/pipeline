import time
import uuid
import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, List, Any, Union

import trafilatura
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


from core.utils import fetch_with_retry, make_safe_filename

#import nest_asyncio
#nest_asyncio.apply()


logger = logging.getLogger("DataPipeline")


async def _async_web_crawler_crawl4ai(
    url: str, 
    crawler_run_config: CrawlerRunConfig,
    max_text_length: Optional[int] = None) -> Union[str, List[Dict[str, Any]], None]:
    """
    Asynchronously crawls a URL using crawl4ai and returns markdown content.
    """
    
    logger.info(f"Starting crawl4ai for URL: {url} with config: {crawler_run_config}")
    
    try:

        async with AsyncWebCrawler() as crawler:
            result_obj = await crawler.arun(
                url=url,
                config=crawler_run_config
            )

        if not result_obj:
            logger.warning(f"crawl4ai returned no result object for URL: {url}")
            return None

        processed_data: Union[str, List[Dict[str, Any]], None] = None

        if isinstance(result_obj, list):
            processed_results_list = []
            for page_info in result_obj:
                if not page_info or not hasattr(page_info, 'markdown'):
                    logger.warning(f"Invalid page_info object in crawl4ai list result for {url}: {page_info}")
                    continue
                
                markdown_content = page_info.markdown
                if max_text_length and markdown_content:
                    markdown_content = markdown_content[:max_text_length]
                    
                processed_results_list.append({
                    "url": getattr(page_info, 'url', url),
                    "title": getattr(page_info, 'title', "No title from crawl4ai"),
                    "markdown": markdown_content or "",
                    "raw_html": getattr(page_info, 'raw_html', None)
                })
                
            if processed_results_list:
                logger.info(f"crawl4ai successfully processed {len(processed_results_list)} pages starting from {url}.")
                processed_data = processed_results_list
                
            else:
                logger.warning(f"crawl4ai deep crawl from {url} yielded no processable pages.")
                processed_data = None

        elif hasattr(result_obj, 'markdown'):
            markdown_content = result_obj.markdown
            if max_text_length and markdown_content:
                markdown_content = markdown_content[:max_text_length]
            
            logger.info(f"crawl4ai successfully processed URL: {url}. Markdown length: {len(markdown_content or '')}")
            
            processed_data = markdown_content or "" 
            
        else:
            logger.warning(f"crawl4ai returned an unexpected result object type for {url}: {type(result_obj)}")
            processed_data = None
            
        return processed_data

    except Exception as e:
        logger.error(f"Error during crawl4ai processing for {url}: {e}", exc_info=True)
        return None


def scrape_with_crawl4ai(
    url: str,
    max_text_length: Optional[int] = None,
    deep_crawl: bool = False,
    max_depth_deep_crawl: int = 0,
    include_external_deep_crawl: bool = False,
    scraping_strategy_name: str = "lxml" # "lxml" or "trafilatura"
    ) -> Union[str, List[Dict[str, Any]], None]:
    """
    Scrapes a URL using crawl4ai. Can perform deep crawling.
    Returns markdown string for single page, or list of dicts for deep crawl, or None on error.
    """
    
    if not url or not url.strip().startswith("http"):
        logger.warning(f"Invalid or empty URL provided to scrape_with_crawl4ai: '{url}'")
        return None

    deep_crawl_config = None
    if deep_crawl:
        deep_crawl_config = BFSDeepCrawlStrategy(
            max_depth=max_depth_deep_crawl,
            include_external=include_external_deep_crawl,
        )

    selected_scraping_strategy: Any
    if scraping_strategy_name.lower() == "lxml":
        selected_scraping_strategy = LXMLWebScrapingStrategy()
        
    else:
        logger.warning(f"Unsupported scraping_strategy_name '{scraping_strategy_name}' for crawl4ai. Defaulting to LXML.")
        selected_scraping_strategy = LXMLWebScrapingStrategy()


    crawler_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_config,
        cache_mode=CacheMode.BYPASS, 
        scraping_strategy=selected_scraping_strategy,
    )
    
    logger.debug(f"crawl4ai configuration for {url}: {crawler_config}")

    try:

        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.warning(
                "scrape_with_crawl4ai: Event loop was already running. "
                "Ensure nest_asyncio.apply() is called at the start of your main script "
                "or manage the event loop carefully if calling from an async context."
            )

            logger.info("Attempting to run crawl4ai with a new event loop due to existing running loop (this is risky).")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(
                    _async_web_crawler_crawl4ai(url, crawler_config, max_text_length)
                )
            finally:
                asyncio.set_event_loop(loop)
                new_loop.close()
        else:
            result = asyncio.run(_async_web_crawler_crawl4ai(url, crawler_config, max_text_length))
        
        return result
    
    
    except RuntimeError as e:
        
        if "cannot be called when another loop is running" in str(e).lower() or "event loop is closed" in str(e).lower():
            logger.error(
                f"Asyncio loop error in scrape_with_crawl4ai for {url}: {e}. "
                "This often happens with nested asyncio.run() calls. "
                "It's highly recommended to use nest_asyncio.apply() at the beginning of your script, "
                "or refactor the calling code to be async and use 'await'."
            )
            return None
        
        else:
            logger.error(f"RuntimeError in scrape_with_crawl4ai for {url}: {e}", exc_info=True)
            return None
        
    except Exception as e:
        logger.error(f"Unexpected error in scrape_with_crawl4ai for {url}: {e}", exc_info=True)
        return None



def scrape_with_beautifulsoup(
    url: str, 
    max_retries: int,
    backoff_factor: float,
    initial_wait_seconds: float = 1.0 ) -> Optional[str]:
    """
    Scrapes a URL using httpx and BeautifulSoup, then extracts text using soup.get_text().
    Returns cleaned text content or None on failure.
    """
    
    logger.info(f"Attempting to scrape URL with BeautifulSoup: {url}")
    response = fetch_with_retry(
        url,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_wait_seconds=initial_wait_seconds
    )
    
    if not response or not response.text:
        logger.warning(f"Failed to fetch content or content is empty for BeautifulSoup from {url}")
        return None

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        
        if not text_content:
            logger.warning(f"BeautifulSoup extracted no text from {url}. HTML might be JavaScript-heavy or empty.")
            return ""

        logger.info(f"Successfully scraped and extracted text from {url} using BeautifulSoup. Length: {len(text_content)}")
        return text_content
    
    except Exception as e:
        logger.error(f"Error processing HTML with BeautifulSoup for {url}: {e}", exc_info=True)
        return None


def scrape_with_trafilatura(
    url: str, 
    max_retries: int,
    backoff_factor: float,
    initial_wait_seconds: float = 1.0, 
    include_formatting: bool = False,
    include_links: bool = False,
    include_comments: bool = False,
    target_language: Optional[str] = None ) -> Optional[str]:
    """
    Scrapes main content from a URL using Trafilatura.
    Returns extracted text or None on failure.
    """
    
    logger.info(f"Attempting to scrape URL with Trafilatura: {url}")

    response = fetch_with_retry(
        url,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_wait_seconds=initial_wait_seconds
    )
    
    if not response or not response.text:
        logger.warning(f"Failed to fetch content or content is empty for Trafilatura from {url}")
        return None

    try:

        extracted_text = trafilatura.extract(
            response.text,
            include_formatting=include_formatting,
            include_links=include_links,
            include_comments=include_comments,
            output_format='txt',
            target_language=target_language,
        )
        
        if extracted_text:
            logger.info(f"Successfully extracted text from {url} using Trafilatura. Length: {len(extracted_text)}")
            return extracted_text.strip()
        else:
            logger.warning(f"Trafilatura extracted no significant content from {url}. The page might be non-article or JavaScript-heavy.")
            return ""
        
    except Exception as e:
        logger.error(f"Error extracting content with Trafilatura for {url}: {e}", exc_info=True)
        return None


def download_and_save_pdf(
    url: str,
    output_dir: Path,
    max_retries: int,
    backoff_factor: float,
    initial_wait_seconds: float = 1.0 ) -> Optional[Path]:
    """
    Downloads a PDF from the given URL and saves it to the specified directory.
    Returns the path to the saved PDF file if successful, else None.
    """
    
    logger.info(f"Processing PDF URL for download: {url}")
    
    response = fetch_with_retry(
        url,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_wait_seconds=initial_wait_seconds
    )

    if response and response.content:
        try:
            
            parsed_url = urlparse(url)
            base_name_from_url = Path(parsed_url.path).name
            
            doc_name_base: str
            if base_name_from_url and base_name_from_url.lower().endswith(".pdf"):
                doc_name_base = base_name_from_url[:-4]
                
            else:
                logger.warning(
                    f"Could not determine a PDF filename from URL path '{parsed_url.path}' or it doesn't end with .pdf. "
                    f"Using UUID for base name."
                )
                doc_name_base = f"pdf_doc_{uuid.uuid4().hex[:10]}"

            safe_filename_str = make_safe_filename(doc_name_base, "pdf", allow_dots_in_name=False)
            
        
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            output_file_path = output_dir_path / safe_filename_str

            logger.info(f"Saving PDF from {url} to: {output_file_path}")
            
            with open(output_file_path, "wb") as f:
                f.write(response.content)
                
            logger.info(f"Successfully saved PDF: {output_file_path} (Size: {len(response.content)} bytes)")
            
            return output_file_path
        
        except IOError as e:
            logger.error(f"IOError saving PDF file from {url} (intended for {output_dir}): {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Unexpected error saving PDF from {url}: {e}", exc_info=True)
            
    else:
        logger.warning(f"Failed to download PDF content from: {url} (No response or empty content).")
    
    return None


def scrape_and_save_webpage_text(
    url: str,
    output_dir: Path,
    max_retries: int,
    backoff_factor: float,
    initial_wait_seconds: float = 1.0,
    extraction_strategy: str = "trafilatura" # 'trafilatura' or 'beautifulsoup'
    ) -> Optional[Path]:
    """
    Extracts text content from a webpage URL using the specified strategy and saves it.
    Returns the path to the saved text file if successful, else None.
    """
    
    logger.info(f"Processing webpage URL for text extraction: {url} using strategy: {extraction_strategy}")
    
    extracted_text: Optional[str] = None
    strategy_lower = extraction_strategy.lower()

    if strategy_lower == "trafilatura":
        extracted_text = scrape_with_trafilatura(
            url, max_retries, backoff_factor, initial_wait_seconds
        )
        
    elif strategy_lower == "beautifulsoup":
        extracted_text = scrape_with_beautifulsoup(
            url, max_retries, backoff_factor, initial_wait_seconds
        )
        
    else:
        logger.error(f"Unsupported text extraction strategy: '{extraction_strategy}' for URL: {url}. Supported: 'trafilatura', 'beautifulsoup'.")
        return None

    if extracted_text is not None:
        if not extracted_text:
            logger.warning(f"No text content extracted by {extraction_strategy} from: {url}, but no error. Saving empty file.")

        try:
            
            parsed_url = urlparse(url)

            name_base_from_url = (parsed_url.netloc + parsed_url.path).replace('/', '_').replace('.', '_').strip('_')
            
            if not name_base_from_url:
                name_base_from_url = f"web_content_{uuid.uuid4().hex[:10]}"

            safe_filename_str = make_safe_filename(name_base_from_url, "txt", allow_dots_in_name=True) # Metin dosyalarında nokta olabilir
            
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            output_file_path = output_dir_path / safe_filename_str

            logger.info(f"Saving extracted text (len: {len(extracted_text)}) from {url} to: {output_file_path}")
            
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            
            logger.info(f"Successfully saved text: {output_file_path}")
            
            return output_file_path
        
        except IOError as e:
            logger.error(f"IOError saving text file for {url} (intended for {output_dir}): {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Unexpected error saving text for {url}: {e}", exc_info=True)
            
    else:
        logger.warning(f"Text extraction by {extraction_strategy} failed for URL: {url} (returned None).")
        
    return None


def process_urls_for_content(
    urls_to_process: List[str],
    download_output_dir: Path,
    fetch_config: Dict[str, Any],
    delay_between_requests: float, 
    default_text_extraction_strategy: str = "trafilatura" ) -> Dict[str, Optional[Path]]:
    """
    Processes a list of URLs. Downloads PDFs or scrapes and saves text from webpages.
    Returns a dictionary mapping original URLs to their saved file paths (or None if failed).
    """
    
    if not urls_to_process:
        logger.info("No URLs provided for content processing.")
        return {}

    output_dir = Path(download_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Content will be saved to: {output_dir.resolve()}")

    processed_results: Dict[str, Optional[Path]] = {}
    total_urls = len(urls_to_process)
    logger.info(f"Starting processing of {total_urls} URLs...")

    max_r = fetch_config.get("max_retries", 3)
    backoff_f = fetch_config.get("backoff_factor", 2.0)
    initial_w = fetch_config.get("initial_wait_seconds", 1.0)

    for i, url in enumerate(urls_to_process):
        url = url.strip()
        if not url or not url.startswith("http"):
            logger.warning(f"Skipping invalid or empty URL at index {i}: '{url}'")
            processed_results[url or f"empty_or_invalid_url_{i}"] = None
            continue

        logger.info(f"\n--- Processing URL {i+1}/{total_urls}: {url} ---")
        saved_path: Optional[Path] = None
        try:
            parsed_url = urlparse(url)
            path_lower = parsed_url.path.lower()
            
            is_pdf = any(path_lower.endswith(ext) for ext in [".pdf", ".PDF"])

            if is_pdf:
                saved_path = download_and_save_pdf(
                    url,
                    output_dir,
                    max_r,
                    backoff_f,
                    initial_w
                )
            else:
                saved_path = scrape_and_save_webpage_text(
                    url,
                    output_dir,
                    max_r,
                    backoff_f,
                    initial_w,
                    extraction_strategy=default_text_extraction_strategy
                )
                
            processed_results[url] = saved_path 
            
        except Exception as e: 
            logger.error(f"Unhandled exception during top-level processing of URL {url}: {e}", exc_info=True)
            processed_results[url] = None

        if i < total_urls - 1 and delay_between_requests > 0:
            logger.debug(f"Pausing for {delay_between_requests:.2f} seconds before next URL...")
            time.sleep(delay_between_requests)

    successful_saves = sum(1 for path in processed_results.values() if path is not None)
    
    failed_saves = total_urls - successful_saves
    
    logger.info(f"--- Finished processing {total_urls} URLs ---")
    logger.info(f"Successfully saved content for {successful_saves} URLs.")
    
    if failed_saves > 0:
        logger.warning(f"Failed to save content for {failed_saves} URLs.")
    
    return processed_results