import logging
from pathlib import Path
from typing import List, Any, Dict, Optional

from processing.search_engine import perform_search
from processing.llm_interactor import llm_filter_search_results
from processing.web_scraper import process_urls_for_content
from core.prompts import get_search_results_filtering_prompt

logger = logging.getLogger("DataPipeline")

def search_filter_and_download(
    selected_queries: List[str],
    search_engine_config: Dict[str, Any],
    llm_client: Any,
    llm_provider_name: str,
    llm_filter_model_name: str,
    llm_temperature: float,
    llm_max_retries: int,
    llm_retry_delay: int,
    download_output_dir: Path,
    fetch_config: Dict[str, Any],
    scraper_delay_between_urls: float,
    scraper_default_text_extraction_strategy: str) -> Dict[str, Optional[Path]]:
    """
    Executes Stage 2: For each selected query, performs a web search,
    filters the results using an LLM, and then downloads/scrapes content from the filtered URLs.
    """
    
    logger.info(f"--- Starting Stage 2: Search, Filter, and Download ---")
    
    if not selected_queries:
        logger.warning("No queries selected from Stage 1. Skipping Stage 2.")
        return {}

    all_urls_to_process_set = set()

    for i, query_text in enumerate(selected_queries):
        
        logger.info(f"\nProcessing query {i+1}/{len(selected_queries)}: '{query_text}'")
        logger.info(f"Step 2.1: Performing web search for query: '{query_text}' with engine '{search_engine_config.get('type', 'N/A')}'...")
        
        raw_search_results: List[Dict[str,str]] = perform_search(
            query=query_text,
            search_engine_type=search_engine_config["type"],
            num_results=search_engine_config["num_results"],
            search_language=search_engine_config.get("language", "en")
        )

        if not raw_search_results:
            logger.warning(f"No search results returned for query: '{query_text}'. Skipping LLM filtering for this query.")
            continue
        
        logger.info(f"Found {len(raw_search_results)} raw search results for query: '{query_text}'.")
        logger.info(f"Step 2.2: Filtering search results with LLM model '{llm_filter_model_name}' for query: '{query_text}'...")
        
        filtered_urls_from_llm: List[str] = llm_filter_search_results(
            llm_client=llm_client,
            provider_name=llm_provider_name,
            model_name=llm_filter_model_name,
            original_query=query_text,
            search_results_list=raw_search_results,
            temperature=llm_temperature,
            max_retries=llm_max_retries,
            retry_delay=llm_retry_delay,
            results_filtering_prompt_func=get_search_results_filtering_prompt
        )

        current_query_added_urls_count = 0
        if filtered_urls_from_llm:
            
            for url in filtered_urls_from_llm:
                
                if url not in all_urls_to_process_set and url.startswith("http"): # Tekrar kontrol
                    all_urls_to_process_set.add(url)
                    current_query_added_urls_count += 1
                    
            logger.info(f"LLM filtering added {current_query_added_urls_count} new, unique URLs for query '{query_text}'.")
        
        else:
            logger.warning(
                f"LLM filtering yielded no usable URLs for query '{query_text}'. "
                "Considering fallback to raw search URLs for this query."
            )
            
            fallback_added_count = 0
            for res in raw_search_results:
                url = res.get("link")
                
                if url and url.startswith("http") and url not in all_urls_to_process_set:
                    all_urls_to_process_set.add(url)
                    fallback_added_count += 1
            
            if fallback_added_count > 0:
                 logger.info(f"Using {fallback_added_count} new, unique URLs from raw search results as fallback for query '{query_text}'.")
            
            else:
                logger.warning(f"No URLs (neither LLM-filtered nor raw fallback) to process for query: '{query_text}'.")
        

    final_unique_urls_list = list(all_urls_to_process_set)

    if not final_unique_urls_list:
        logger.warning("No URLs were collected from any query after searching and filtering. Stage 2 ends with no content to download.")
        return {}
    
    logger.info(f"Total of {len(final_unique_urls_list)} unique URLs collected across all queries to download/scrape.")
    
    logger.info(f"Step 2.3: Processing {len(final_unique_urls_list)} collected URLs for content...")
    
    processed_url_to_filepath_map = process_urls_for_content(
        urls_to_process=final_unique_urls_list,
        download_output_dir=download_output_dir,
        fetch_config=fetch_config,
        delay_between_requests=scraper_delay_between_urls,
        default_text_extraction_strategy=scraper_default_text_extraction_strategy
    )
    
    logger.info("--- Finished Stage 2: Search, Filter, and Download ---")
    
    return processed_url_to_filepath_map