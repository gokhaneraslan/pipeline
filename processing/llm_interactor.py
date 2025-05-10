import re
import time
import json
import logging
from typing import Optional, Any, List, Dict, Callable

from core.utils import (
    parse_json_from_llm_evaluate, 
    extract_urls_from_json_or_text,
    parse_json_from_llm_response,
    parse_questions_from_response
)


logger = logging.getLogger("DataPipeline")


def _make_llm_api_call(
    client: Any,
    provider_name: str,
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    temperature: float,
    max_retries: int,
    retry_delay_base_seconds: int,
    response_format_type = None) -> Optional[str]:
    """
    Helper function to make an API call to the specified LLM provider with exponential backoff retries.
    Returns the raw string content from the LLM response or None on failure.
    """
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    messages.append({"role": "user", "content": user_prompt})


    groq_mistral_response_format_arg = None
    ollama_format_param = None
    if response_format_type == "json_object":
        
        if provider_name in ["groq", "mistral"]:
            groq_mistral_response_format_arg = {"type": "json_object"}
            
        elif provider_name == "ollama":
            ollama_format_param = "json"


    for attempt in range(max_retries):
        try:
            
            logger.debug(
                f"LLM API call attempt {attempt + 1}/{max_retries} to {provider_name} model {model_name}. "
                f"User prompt length: {len(user_prompt)}, Temp: {temperature}, Format: {response_format_type}"
            )
            
            completion_content: Optional[str] = None
            
            if provider_name == "groq":
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    #response_format=groq_mistral_response_format_arg
                )
                
                completion_content = completion.choices[0].message.content
            
        
            elif provider_name == "mistral":
                completion = client.chat.complete(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    #response_format=groq_mistral_response_format_arg
                )
                completion_content = completion.choices[0].message.content
            
            elif provider_name == "ollama":
                
                ollama_options = {"temperature": temperature}
                
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    options=ollama_options,
                    #format=ollama_format_param
                )
                
                content_data = response['message']['content']
                
                if content_data is None:
                    logger.error(f"Ollama response missing 'message.content' for model {model_name}.")
                    
                elif ollama_format_param == "json" and isinstance(content_data, dict):
                    completion_content = json.dumps(content_data)
                    
                elif isinstance(content_data, str):
                    completion_content = content_data
                    
                else:
                    logger.error(f"Ollama returned unexpected content type in 'message.content': {type(content_data)}")
                    
            else:
                logger.error(f"Unsupported LLM provider '{provider_name}' in _make_llm_api_call.")
                return None

            if completion_content is not None:
                logger.debug(f"LLM call successful for {provider_name} model {model_name}. Response length: {len(completion_content)}")
                return completion_content
            
            else:
                logger.warning(f"LLM call for {provider_name} model {model_name} resulted in None content. Retrying if attempts remain.")

        except Exception as e:
            logger.warning(f"LLM API call error on attempt {attempt + 1}/{max_retries} for {provider_name} model {model_name}: {e}")

            if attempt == max_retries - 1:
                logger.error(
                    f"Max retries reached. LLM API call failed for {provider_name} model {model_name} after {max_retries} attempts.",
                    exc_info=True
                )
                return None


            wait_seconds = retry_delay_base_seconds * (2 ** attempt)
            max_wait = 120
            actual_wait_seconds = min(wait_seconds, max_wait)
            
            error_str = str(e).lower()
            status_code = getattr(e, 'status_code', None)
            
            if "rate limit" in error_str or status_code == 429:
                logger.warning(f"Rate limit likely hit for {provider_name}. Retrying in {actual_wait_seconds:.2f}s.")
            
            logger.info(f"Retrying LLM call in {actual_wait_seconds:.2f} seconds...")
            time.sleep(actual_wait_seconds)
                
    return None


def llm_generate_search_queries(
    llm_client: Any, provider_name: str, model_name: str, topic: str,
    temperature: float, max_retries: int, retry_delay: int,
    query_generation_prompt_func: Callable[[str], str] ) -> List[str]:
    """
    Uses LLM to generate search queries for a given topic.
    Returns a list of query strings.
    Prompts.get_query_generation_prompt expects JSON: {"queries": ["q1", "q2"]}
    """
    
    logger.info(f"Generating search queries for topic: '{topic}' using {provider_name} model {model_name}")
    
    user_prompt = query_generation_prompt_func(topic=topic)
    
    raw_response = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert search query generator, skilled in creating diverse and effective queries.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type="json_object"
    )

    if not raw_response:
        logger.error(f"LLM failed to generate search queries (no response) for topic: {topic}")
        return []

    try:
        data = json.loads(raw_response)
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            
            queries = [str(q) for q in data["queries"] if isinstance(q, str) and q.strip()]
            
            if not queries:
                logger.warning(f"LLM generated an empty list of queries or invalid format for topic: {topic}. Raw: {raw_response[:200]}")
                return []
            
            logger.info(f"Successfully generated {len(queries)} search queries for topic: {topic}.")
            
            return queries
        
        else:
            logger.error(f"LLM response for query generation was not in expected JSON format {{'queries': [...]}}: {raw_response[:200]}")
            return []
    
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM query generation response: {raw_response[:200]}", exc_info=True)
        return []
    
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM query generation response: {e}", exc_info=True)
        return []


def llm_evaluate_and_select_queries(
    llm_client: Any, provider_name: str, model_name: str,
    generated_queries: List[str],
    num_to_select: int,
    temperature: float, max_retries: int, retry_delay: int,
    query_evaluation_prompt_func: Callable[[str, int], str] ) -> List[str]:
    """
    Uses LLM to evaluate a list of generated queries and select a diverse subset.
    Returns a list of selected query strings.
    utils.parse_json_from_llm_evaluate handles parsing of {"selected_queries": [{"query": ..., "justification": ...}]}
    """
    
    logger.info(f"Evaluating and selecting {num_to_select} queries using {provider_name} model {model_name}")
    
    if not generated_queries:
        logger.warning("No generated queries provided for evaluation.")
        return []
    
    queries_as_json_for_prompt = json.dumps(generated_queries, indent=2)
        
    user_prompt = query_evaluation_prompt_func(
        query_list_json_str=queries_as_json_for_prompt,
        num_queries_to_select=num_to_select
    )

    raw_response = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert search query evaluator and selector, skilled in identifying diverse and effective queries.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type="json_object"
    )

    if not raw_response:
        logger.error("LLM failed to evaluate and select queries (no response).")
        return []


    selected_query_strings = parse_json_from_llm_evaluate(raw_response)

    if selected_query_strings:
        logger.info(f"LLM selected {len(selected_query_strings)} queries. Requested: {num_to_select}")
        return selected_query_strings
    
    else:
        logger.error(f"LLM response for query evaluation did not yield any selectable queries or was in invalid format: {raw_response[:200]}...")
        return []


def llm_filter_search_results(
    llm_client: Any, provider_name: str, model_name: str,
    original_query: str,
    search_results_list: List[Dict[str, str]],
    temperature: float, max_retries: int, retry_delay: int,
    results_filtering_prompt_func: Callable[[str, str], str]) -> List[str]:
    """
    Uses LLM to filter search results based on relevance and suitability.
    Returns a list of filtered URL strings.
    utils.extract_urls_from_json_or_text handles parsing of LLM response (JSON list)
    and can use search_results_list as fallback.
    """
    
    logger.info(f"Filtering search results for query: '{original_query}' using {provider_name} model {model_name}")
    
    if not search_results_list:
        logger.warning(f"No search results (Python list) provided for filtering (query: '{original_query}').")
        return []


    search_results_as_json_for_prompt = json.dumps(search_results_list, indent=2)

    user_prompt = results_filtering_prompt_func(
        original_query=original_query,
        search_results_json_str=search_results_as_json_for_prompt
    )

    raw_response_from_llm = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert search result filter for LLM dataset curation, skilled at identifying relevant and high-quality sources.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type="json_object"
    )

    if not raw_response_from_llm:
        logger.error(f"LLM failed to filter search results (no response) for query: {original_query}")
        
        urls = [res.get('url') or res.get('link') for res in search_results_list if res.get('url') or res.get('link')]
        
        logger.info(f"Returning all {len(urls)} raw URLs due to LLM failure for query: {original_query}")
        
        return urls


    filtered_url_strings = extract_urls_from_json_or_text(
        llm_filtered_response_content=raw_response_from_llm,
        fallback_search_results_list=search_results_list
    )

    if filtered_url_strings:
        logger.info(f"LLM (and/or fallback) filtering resulted in {len(filtered_url_strings)} URLs for query: {original_query}")
        return filtered_url_strings
    else:
        logger.warning(f"LLM response for search result filtering did not yield any URLs or was invalid: {raw_response_from_llm[:200]}...")
        return []


def llm_get_score_for_chunk(
    llm_client: Any, provider_name: str, model_name: str,
    topic: str,
    chunk_text: str,
    temperature: float, max_retries: int, retry_delay: int,
    scoring_prompt_func: Callable[[str, str], str] ) -> Optional[int]:
    """
    Uses LLM to get a relevance/focus score (0-100) for a text chunk.
    """
    
    if not chunk_text.strip():
        logger.debug("Skipping scoring for empty chunk.")
        return None

    logger.debug(f"Requesting score for chunk (len: {len(chunk_text)}) related to topic: '{topic}' using {provider_name} model {model_name}")
    
    user_prompt = scoring_prompt_func(topic=topic, chunk_text=chunk_text)

    raw_response = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert content scorer, assigning numerical scores based on provided criteria.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type=None
    )

    if not raw_response:
        logger.warning(f"LLM failed to provide a score for chunk (topic: {topic}). No response.")
        return None

    try:

        score = int(raw_response.strip())
        if 0 <= score <= 100:
            logger.debug(f"Extracted score (direct int conversion): {score} for chunk (topic: {topic})")
            return score
        else:
            logger.warning(f"LLM returned a number '{score}' outside the 0-100 range. Raw: '{raw_response}'. Treating as invalid.")

    except ValueError:
        logger.debug(f"Direct integer conversion failed for score response '{raw_response}'. Trying regex.")
        
        
    match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', raw_response)
    if match:
        
        try:
            score = int(match.group(1))
            logger.debug(f"Extracted score (regex): {score} from response '{raw_response}' for chunk (topic: {topic})")
            return score
        
        except ValueError:
            logger.warning(f"Regex match '{match.group(1)}' could not be converted to int, though regex should ensure it. Response: '{raw_response}'")
            return None
        
    else:
        logger.warning(f"Could not find any valid integer score (0-100) in LLM response: '{raw_response}'")
        return None


def llm_extract_clean_content_from_chunk(
    llm_client: Any, provider_name: str, model_name: str,
    topic: str,
    chunk_text: str,
    temperature: float, max_retries: int, retry_delay: int,
    content_extraction_prompt_func: Callable[[str, str], str] ) -> Optional[str]:
    """
    Uses LLM to extract clean, topic-focused content from a text chunk.
    Returns the extracted text string, or None/empty string on failure/no content.
    utils.parse_json_from_llm_response handles {"output": "..."}
    """
    
    if not chunk_text.strip():
        logger.debug("Skipping content extraction for empty chunk.")
        return ""

    logger.debug(f"Requesting clean content extraction for chunk (len: {len(chunk_text)}, topic: '{topic}') using {provider_name} model {model_name}")
    user_prompt = content_extraction_prompt_func(topic=topic, chunk_text=chunk_text)

    raw_response = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert content extractor and consolidator, focused on extracting specific information.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type="json_object"
    )

    if not raw_response:
        logger.warning(f"LLM failed to provide extracted content (no response) for chunk (topic: {topic}).")
        return None

    extracted_text = parse_json_from_llm_response(raw_response)
    
    if extracted_text is not None:
        
        if not extracted_text:
            logger.debug(f"LLM indicated no relevant content to extract from chunk (topic: {topic}).")
        return extracted_text
    
    else:
        logger.error(f"Failed to parse 'output' from LLM content extraction response or key missing: {raw_response[:200]}...")
        return None


def llm_generate_questions_from_chunk(
    llm_client: Any, provider_name: str, model_name: str,
    num_questions: int,
    chunk_text: str,
    temperature: float, max_retries: int, retry_delay: int,
    questions_generate_prompt_func: Callable[[str, int], str] ) -> List[str]:
    """
    Generates a list of questions from a text chunk using an LLM.
    utils.parse_questions_from_response handles parsing of numbered list.
    """
    
    if not chunk_text.strip():
        logger.debug("Skipping question generation for empty chunk.")
        return []

    logging.info(f"Generating {num_questions} questions from chunk (len: {len(chunk_text)}) using {provider_name} model {model_name} (temp={temperature})...")
    
    user_prompt = questions_generate_prompt_func(chunk_text=chunk_text, num_questions_to_generate=num_questions)
    
    raw_response = _make_llm_api_call(
        client=llm_client, provider_name=provider_name, model_name=model_name,
        system_prompt="You are an expert question generator, creating insightful questions from provided text.",
        user_prompt=user_prompt, temperature=temperature,
        max_retries=max_retries, retry_delay_base_seconds=retry_delay,
        response_format_type=None
    )

    if not raw_response:
        logger.warning(f"LLM failed to provide response for question generation from chunk.")
        return []

    extracted_questions = parse_questions_from_response(raw_response)
    
    if not extracted_questions:
        logging.warning(f"Could not parse any questions from the model's response: {raw_response[:200]}")
        return []

    if len(extracted_questions) != num_questions:
        logging.warning(
            f"LLM generated {len(extracted_questions)} questions, but {num_questions} were requested. "
            f"Using the generated {len(extracted_questions)}."
        )
    
    #extracted_questions = extracted_questions[:num_questions]
    
    return extracted_questions
    
    
def llm_answers_for_questions_from_chunk(
    llm_client: Any, provider_name: str, model_name: str,
    generated_questions: List[str],
    chunk_text: str,
    temperature: float, max_retries: int, retry_delay: int,
    answered_questions_prompt_func: Callable[[str, str], str]) -> List[str]:
    """
    Generates answers for a list of questions based strictly on a text chunk.
    """
    
    if not chunk_text.strip():
        logger.debug("Skipping answer generation as chunk_text is empty.")
        return ["Chunk text was empty, cannot answer." for _ in generated_questions] if generated_questions else []
    
    if not generated_questions:
        logger.debug("No questions provided for answer generation.")
        return []

    logging.info(f"Answering {len(generated_questions)} questions from chunk (len: {len(chunk_text)}) using {provider_name} model {model_name} (temp={temperature})...")
    
    answers: List[str] = []
    for i, question_text in enumerate(generated_questions):
        
        if not question_text.strip():
            logger.warning(f"Skipping empty question at index {i}.")
            answers.append("Provided question was empty.")
            continue

        user_prompt = answered_questions_prompt_func(chunk_text=chunk_text, question=question_text)
        
        raw_response = _make_llm_api_call(
            client=llm_client, provider_name=provider_name, model_name=model_name,
            system_prompt=(
                "You are an AI assistant specialized in synthesizing information and generating "
                "clear, concise responses based *strictly* on provided context. "
                "If the answer is not in the context, you MUST state that."
            ),
            user_prompt=user_prompt, temperature=temperature,
            max_retries=max_retries, retry_delay_base_seconds=retry_delay,
            response_format_type=None
        )

        if raw_response is None:
            logging.error(f"LLM call failed while trying to answer question {i+1}: '{question_text[:50]}...'")
            answers.append("Error: LLM failed to generate an answer for this question.")
            
        else:
            answers.append(raw_response.strip())
        
    if len(answers) != len(generated_questions):
        logger.error(f"Mismatch in number of answers ({len(answers)}) and questions ({len(generated_questions)}). This should not happen.")
        
    return answers


def llm_refined_json_response(
    llm_client: Any, provider_name: str, model_name: str,
    topic: str,
    dataset_json_string: str,
    temperature: float, max_retries: int, retry_delay: int,
    check_dataset_prompt_func: Callable[[str, str], str]) -> Optional[str]:
    
    if not dataset_json_string.strip():
        logger.debug("Dataset JSON string is empty. Skipping refinement LLM call.")
        return ""

    user_prompt = check_dataset_prompt_func(topic=topic, dataset_json_string=dataset_json_string)
    
    logger.debug(f"Requesting dataset refinement for topic: '{topic}' using {provider_name} model {model_name}. Dataset length: {len(dataset_json_string)}")

    raw_response = _make_llm_api_call(
        client=llm_client,
        provider_name=provider_name,
        model_name=model_name,
        system_prompt="You are an AI expert specializing in refining and enhancing question-answer datasets.",
        user_prompt=user_prompt,
        temperature=temperature,
        max_retries=max_retries,
        retry_delay_base_seconds=retry_delay,
        response_format_type="json_object"
    )

    if raw_response is None:
        logger.warning(f"LLM failed to provide refined dataset (None response) for topic: {topic}.")
        return None

    match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response, re.IGNORECASE)
    
    if match:
        logger.info("Found JSON response wrapped in markdown by LLM, extracting.")
        cleaned_response = match.group(1).strip()
    
    try:
        json.loads(cleaned_response)
        logger.debug(f"LLM returned a valid JSON string for dataset refinement. Length: {len(cleaned_response)}")
        return cleaned_response
    
    except json.JSONDecodeError as e:
        logger.error(f"LLM response for dataset refinement is not valid JSON after cleaning: {e}. Raw response snippet: {raw_response[:200]}...")
        return raw_response