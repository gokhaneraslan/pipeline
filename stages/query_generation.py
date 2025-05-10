import logging
from typing import List, Any

from core.prompts import get_query_generation_prompt, get_query_evaluation_prompt
from processing.llm_interactor import llm_generate_search_queries, llm_evaluate_and_select_queries

logger = logging.getLogger("DataPipeline")


def query_generation_and_selection(
    llm_client: Any,
    llm_provider_name: str,
    generation_model_name: str,
    evaluation_model_name: str,
    topic: str,
    num_queries_to_select: int,
    llm_temperature: float,
    llm_max_retries: int,
    llm_retry_delay: int ) -> List[str]:
    """
    Executes Stage 1: Generates search queries for the given topic using an LLM,
    then uses another (or the same) LLM call to evaluate and select a subset of these queries.
    """
    
    logger.info(f"--- Starting Stage 1: Query Generation and Selection for Topic: '{topic}' ---")
    logger.info(f"Step 1.1: Generating initial search queries using model '{generation_model_name}'...")
    
    generated_queries_list: List[str] = llm_generate_search_queries(
        llm_client=llm_client,
        provider_name=llm_provider_name,
        model_name=generation_model_name,
        topic=topic,
        temperature=llm_temperature,
        max_retries=llm_max_retries,
        retry_delay=llm_retry_delay,
        query_generation_prompt_func=get_query_generation_prompt
    )

    if not generated_queries_list:
        logger.error("Query generation (Step 1.1) failed or yielded no queries. Stage 1 cannot proceed with selection.")
        return []
    
    logger.info(f"Successfully generated {len(generated_queries_list)} initial queries. First 5: {generated_queries_list[:5]}")

    if num_queries_to_select <= 0:
        logger.info(f"Number of queries to select ({num_queries_to_select}) is not positive. Returning all {len(generated_queries_list)} generated queries.")
        return generated_queries_list
    
    if num_queries_to_select >= len(generated_queries_list):
        logger.info(f"Number of queries to select ({num_queries_to_select}) is >= number of generated queries ({len(generated_queries_list)}). "
                    f"No further LLM evaluation needed. Returning all generated queries.")
        return generated_queries_list

    logger.info(f"Step 1.2: Evaluating and selecting {num_queries_to_select} queries from {len(generated_queries_list)} generated queries using model '{evaluation_model_name}'...")
    
    
    selected_queries_list: List[str] = llm_evaluate_and_select_queries(
        llm_client=llm_client,
        provider_name=llm_provider_name,
        model_name=evaluation_model_name,
        generated_queries=generated_queries_list,
        num_to_select=num_queries_to_select,
        temperature=llm_temperature,
        max_retries=llm_max_retries,
        retry_delay=llm_retry_delay,
        query_evaluation_prompt_func=get_query_evaluation_prompt
    )

    if not selected_queries_list:
        logger.warning(
            "Query evaluation/selection (Step 1.2) failed or yielded no queries. "
            f"Falling back to using the first {num_queries_to_select} initially generated queries."
        )
        return generated_queries_list[:num_queries_to_select]

    logger.info(f"Successfully selected {len(selected_queries_list)} queries after evaluation: {selected_queries_list}")
    logger.info("--- Finished Stage 1: Query Generation and Selection ---")
    
    return selected_queries_list