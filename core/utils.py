import re
import uuid
import json
import html
import httpx
import time
import logging
import os
import pandas as pd
from typing import Optional, Dict, List, Union


logger = logging.getLogger("DataPipeline")


DEFAULT_MAX_RETRIES_HTTP = 3
DEFAULT_BACKOFF_FACTOR_HTTP = 2


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def make_safe_filename(name: str, extension: str, allow_dots_in_name: bool = False) -> str:
  """
    Removes/replaces invalid characters for a filename and ensures uniqueness.
    Args:
        name: The base name for the file.
        extension: The file extension (without a leading dot).
        allow_dots_in_name: If True, allows '.' characters in the filename (before extension).
  """
  
  name = re.sub(r'\s+', '_', name)
  
  if allow_dots_in_name:
    name = re.sub(r'[^\w\-\.]+', '', name)
    
  else:
    name = re.sub(r'[^\w\-]+', '', name)

  name = name.strip('._-')
  name = name[:100]

  if not name or name == '_':
    name = uuid.uuid4().hex[:12]
    
  return f"{name}.{extension}"


def fetch_with_retry(
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES_HTTP,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR_HTTP,
    initial_wait_seconds: float = 1.0,
    headers: Optional[Dict[str, str]] = None ) -> Optional[httpx.Response]:
    """
    Fetches a URL with retries for transient errors using exponential backoff.
    """
    
    current_headers = headers if headers else HEADERS
      
    for attempt in range(max_retries):
        wait_time = initial_wait_seconds * (backoff_factor ** attempt)
        
        try:
            
            logger.debug(f"Attempt {attempt + 1}/{max_retries} to fetch URL: {url}")
            
            response = httpx.get(url, headers=current_headers, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched: {url} (Status: {response.status_code})")
            
            return response
        
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error for {url} on attempt {attempt + 1}: {e.response.status_code} - {e.request.url}")
            
            if e.response.status_code in [400, 401, 403, 404, 410]:
                logger.error(f"Non-retryable HTTP error {e.response.status_code} for {url}. Stopping retries.")
                return None

            if e.response.status_code == 429:
                retry_after_header = e.response.headers.get('Retry-After')
                if retry_after_header:
                    try:

                        wait_override = int(retry_after_header)
                        logger.info(f"Server requested Retry-After: {wait_override} seconds for {url}.")
                        wait_time = min(wait_override, 300)
                        
                    except ValueError:
                        logger.warning(f"Could not parse Retry-After header value '{retry_after_header}' as int for {url}. Using default backoff.")

                      
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts (last error: {e.response.status_code}).")
                return None

            wait_time_to_sleep = min(wait_time, 120)
            logger.info(f"Retrying {url} in {wait_time_to_sleep:.2f} seconds...")
            time.sleep(wait_time_to_sleep)

        except httpx.RequestError as e:
            logger.warning(f"Request error for {url} on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts due to RequestError: {e}")
                return None
            
            wait_time_to_sleep = min(wait_time, 120)
            logger.info(f"Retrying {url} in {wait_time_to_sleep:.2f} seconds...")
            time.sleep(wait_time_to_sleep)
            
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching {url} on attempt {attempt + 1}: {e}", exc_info=True)
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts due to unexpected error.")
                return None
            
            wait_time_to_sleep = min(wait_time, 120)
            logger.info(f"Retrying {url} in {wait_time_to_sleep:.2f} seconds due to unexpected error...")
            time.sleep(wait_time_to_sleep)
            
    return None


def clean_html_content(html_content: Optional[str]) -> str:
    """
    Cleans HTML content by removing scripts, styles, comments, and tags,
    then normalizing whitespace.
    NOTE: For robust HTML to text extraction, consider using libraries like
    BeautifulSoup's get_text() or Trafilatura, which are already in web_scraper.py.
    This function is for more basic HTML string cleaning if needed.
    """
    
    if html_content is None:
        return ""

    text = str(html_content)
    # Remove script and style elements
    cleaned_text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    cleaned_text = re.sub(r'<!--.*?-->', '', cleaned_text, flags=re.DOTALL)
    # Remove all remaining HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    # Unescape HTML entities like &, <, etc.
    cleaned_text = html.unescape(cleaned_text)
    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def parse_json_from_llm_evaluate(response_text: str) -> List[str]:
    """
    Parses the LLM response for query evaluation, expecting a JSON object
    with a "selected_queries" key containing a list of objects,
    each with a "query" key.
    Example: {"selected_queries": [{"query": "query text", "justification": "..."}]}
    Returns a list of selected query strings.
    """
    
    if not response_text:
        logger.warning("Received empty response_text for LLM query evaluation.")
        return []

    try:
        
        data = json.loads(response_text)
        if not isinstance(data, dict):
            logger.error(f"LLM evaluation response is not a JSON object: {response_text[:200]}")
            return []

        selected_queries_list_of_objects = data.get("selected_queries")
        if not isinstance(selected_queries_list_of_objects, list):
            logger.error(f"'selected_queries' key not found or not a list in LLM response: {response_text[:200]}")
            return []

        extracted_query_strings: List[str] = []
        for item in selected_queries_list_of_objects:
            if isinstance(item, dict):
                query = item.get("query")
                
                if isinstance(query, str):
                    extracted_query_strings.append(query)
                    
                else:
                    logger.warning(f"Query item in 'selected_queries' is missing 'query' string: {item}")
                    
            else:
                logger.warning(f"Item in 'selected_queries' list is not a dictionary: {item}")
        
        if not extracted_query_strings:
            logger.warning(f"No valid queries extracted from 'selected_queries' list: {response_text[:200]}")
        
        return extracted_query_strings

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM query evaluation response: {response_text[:200]}", exc_info=True)
        
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            logger.info("Attempting to parse JSON found via regex fallback in LLM evaluation response.")
            
            return parse_json_from_llm_evaluate(json_match.group(0))

        return []
    
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM query evaluation response: {e}", exc_info=True)
        return []


def extract_urls_from_json_or_text(
    llm_filtered_response_content: str,
    fallback_search_results_list: Optional[List[Dict[str,str]]] = None) -> List[str]:
    """
    Extracts URLs primarily from an LLM's JSON response (expected to be a list of dicts,
    each containing a 'url' or 'link').
    If that fails or yields too few results, it can use a fallback list of search result dicts.
    Finally, it can attempt regex extraction from the initial LLM response if it wasn't valid JSON.

    Args:
        llm_filtered_response_content: The string content from LLM, expected to be a JSON list.
        fallback_search_results_list: An optional list of raw search result dictionaries.

    Returns:
        A list of unique URL strings.
    """
    
    extracted_urls = set()


    if llm_filtered_response_content:
        
        try:
            data = json.loads(llm_filtered_response_content)
            
            if isinstance(data, list):
                
                for item in data:
                    if isinstance(item, dict):
                        url = item.get('url') or item.get('link')
                        if isinstance(url, str) and url.startswith("http"):
                            extracted_urls.add(url)
                            
                logger.info(f"Extracted {len(extracted_urls)} unique URLs from LLM JSON response.")
            
            else:
                logger.warning(f"LLM response for URL extraction was not a JSON list as expected: {llm_filtered_response_content[:200]}")
        
        except json.JSONDecodeError:
            logger.warning(f"LLM response content is not valid JSON. Attempting regex. Content: {llm_filtered_response_content[:200]}")


    if not extracted_urls and llm_filtered_response_content:
        logger.info("Attempting regex URL extraction from LLM response string as fallback.")
        
        pattern = r'https?://[^\s"\'<>]+'
        try:
            found_by_regex = re.findall(pattern, llm_filtered_response_content)
            
            for u in found_by_regex:
                extracted_urls.add(u)
                
            if found_by_regex:
                logger.info(f"Extracted {len(found_by_regex)} URLs via regex from LLM response string.")
                
        except Exception as e:
            logger.warning(f"Regex URL extraction from LLM response string failed: {e}")


    if len(extracted_urls) < 2 and fallback_search_results_list:
        logger.info(f"Extracted URLs count ({len(extracted_urls)}) is low, using fallback search results list.")
        
        for item in fallback_search_results_list:
            if isinstance(item, dict):
                url = item.get('url') or item.get('link')
                
                if isinstance(url, str) and url.startswith("http"):
                    extracted_urls.add(url)
                    
        logger.info(f"Total unique URLs after considering fallback list: {len(extracted_urls)}")
    
    final_url_list = list(extracted_urls)
    if not final_url_list:
        logger.warning("No URLs extracted from any method.")
        
    return final_url_list


def parse_json_from_llm_response(response_content: str) -> Optional[str]:
    """
    Parses LLM response expecting a JSON with an "output" key containing a string.
    Returns the extracted string from "output" or None if not found/invalid.
    """
    
    if not response_content:
        return None

    try:
        json_response = json.loads(response_content)
        if isinstance(json_response, dict) and "output" in json_response:
            extracted_text = json_response["output"]
            if isinstance(extracted_text, str):
                extracted_text = extracted_text.strip()
                # extracted_text = extracted_text.replace('  ', ' ') # Handled by re.sub later if needed
                # extracted_text = extracted_text.replace('\n\n', '\n')
                # extracted_text = extracted_text.replace('\n  ', '\n')
                
                if not extracted_text:
                    logger.debug("LLM 'output' key was an empty string.")
                    return ""
                
                else:
                    logger.debug(f"Successfully extracted text from 'output' (length: {len(extracted_text)}).")
                    return extracted_text
                
            else:
                logger.warning(f"LLM response 'output' key exists but its value is not a string: {type(extracted_text)}")
                return None
            
        else:
            logger.warning(f"LLM response was valid JSON but missing 'output' key or it's not a dict: {response_content[:200]}")
            return None

    except json.JSONDecodeError:
        logger.warning(f"LLM response was not valid JSON. Attempting regex fallback: {response_content[:200]}")

        json_match = re.search(r'\{[\s\S]*\}', response_content)
        if json_match:
            try:
                logger.debug("Found JSON-like structure with regex, attempting to parse it.")
                
                return parse_json_from_llm_response(json_match.group(0))
            
            except Exception as e:
                logger.warning(f"Could not decode JSON part extracted via regex due to: {e}. Original: {response_content[:200]}")
                return None
            
        else:
            logger.warning(f"No JSON-like structure found via regex in LLM response: {response_content[:200]}")
            return None
        
    except Exception as e:
        logger.error(f"An unexpected error occurred in parse_json_from_llm_response: {e}", exc_info=True)
        return None


def parse_questions_from_response(response_text: str) -> List[str]:
    """
    Parses a numbered list of questions from the LLM response string.
    Tries to handle potential variations in formatting.
    """
    
    if not response_text or not response_text.strip():
        logging.warning("Empty response received when expecting questions.")
        return []
        
    questions = []
    question_pattern = re.compile(r"^\s*\d+\.\s*(.*?)\s*$", re.MULTILINE)
    matches = question_pattern.findall(response_text)

    for match_content in matches:
        question = match_content.strip()
        if question:
            
            question = re.sub(r"^[-\*\s]+", "", question)
            question = question.strip(' *"\'')
            
            if question:
                questions.append(question)


    if not questions and response_text:
        logging.warning("Primary question pattern matching failed or found too few, trying fallback method.")
        
        lines = response_text.splitlines()
        
        potential_questions = []
        for line in lines:
            line = line.strip()

            if line and len(line) > 5 and (line[0].isdigit() or line[0] in ['-', '*', '+']):
                cleaned_line = re.sub(r"^\s*[\d\.\-\*\+]+\s*", '', line).strip(' *"\'')

                if cleaned_line and ('?' in cleaned_line or len(cleaned_line.split()) > 3):
                    potential_questions.append(cleaned_line)

            elif line and '?' in line and len(line.split()) > 3:
                potential_questions.append(line.strip(' *"\''))

        if potential_questions:
            logging.info(f"Used fallback method to parse questions. Found {len(potential_questions)} potential questions.")
            questions.extend(potential_questions)


    if questions:
        unique_questions = []
        seen = set()
        
        for q in questions:
            if q.lower() not in seen:
                unique_questions.append(q)
                seen.add(q.lower())
                
        questions = unique_questions
            
    return questions



def create_default_template(generated_questions: List[str], answers_questions: List[str]) -> List[Dict[str, str]]:
    """Create a basic template with input/output pairs."""
    
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match. Cannot create default template.")
        return []
    
    basic_temp = []
    for i in range(len(generated_questions)):
        basic_temp.append({"input": generated_questions[i], "output": answers_questions[i]})
    
    logging.debug(f"Created basic template with {len(basic_temp)} question-answer pairs.")
    
    return basic_temp


def create_gemma_template(generated_questions: List[str], answers_questions: List[str]) -> List[Dict[str, str]]:
    """Create a template formatted for Gemma models."""
    
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match. Cannot create Gemma template.")
        return []
    
    gemma_temp = []
    for i in range(len(generated_questions)):
        gemma_temp.append({"role": "user", "content": generated_questions[i]})
        gemma_temp.append({"role": "assistant", "content": answers_questions[i]})
    
    logging.debug(f"Created Gemma template with {len(gemma_temp)//2} conversation turns.")
    
    return gemma_temp

def create_llama_template(generated_questions: List[str], answers_questions: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Create a template formatted for LLaMA models (ShareGPT format)."""
    
    if len(generated_questions) != len(answers_questions):
        logging.error(f"Question count ({len(generated_questions)}) and answer count ({len(answers_questions)}) do not match. Cannot create Llama template.")
        return {"conversations": []}
    
    conversations = []
    for i in range(len(generated_questions)):
        conversations.append({"from": "human", "value": generated_questions[i]})
        conversations.append({"from": "gpt", "value": answers_questions[i]})
    
    llama_temp = {
        "id": uuid.uuid4().hex,
        "conversations": conversations
    }
    
    logging.debug(f"Created Llama template with {len(conversations)//2} conversation turns.")
    
    return llama_temp


def llm_create_template(
    template_name: str,
    generated_questions: List[str], 
    answers_questions: List[str],
    supported_templates: List[str]) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]], str]:
    
    """Create the appropriate template based on the specified format."""
    
    template_lower = template_name.lower()
    if template_lower not in [t.lower() for t in supported_templates]:
        error_msg = f"Unsupported template: '{template_name}'. Supported templates: {', '.join(supported_templates)}"
        logging.error(error_msg)
        
        return error_msg
    
    if template_lower == "default":
        return create_default_template(generated_questions, answers_questions)
    
    elif template_lower == "gemma":
        return create_gemma_template(generated_questions, answers_questions)
    
    elif template_lower == "llama":
        return create_llama_template(generated_questions, answers_questions)
    

    unknown_template_msg = f"Unknown error: Template '{template_name}' was in supported_templates but not handled."
    
    logging.error(unknown_template_msg)
    
    return unknown_template_msg


def llm_dataset_save_dataset(
    dataset_data: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]],
    template_name: str,
    dataset_output_base_path: str ) -> Optional[str]:
    
    """Save the dataset in the appropriate format based on the template."""

    if not dataset_data:
        logging.error("No dataset data provided to save.")
        return None
        

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_name_for_file = os.path.splitext(os.path.basename(dataset_output_base_path))[0]
    output_dir = os.path.dirname(dataset_output_base_path)
    
    if not output_dir:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    json_filename = os.path.join(output_dir, f"{base_name_for_file}_{timestamp}.json")
    csv_filename = os.path.join(output_dir, f"{base_name_for_file}_{timestamp}.csv")

    try:
        with open(json_filename, 'w', encoding="utf-8") as nf:
            json.dump(dataset_data, nf, ensure_ascii=False, indent=2)
            
        logging.info(f"Dataset saved to {json_filename}")
        
    except Exception as e:
        logging.error(f"Error saving JSON dataset to {json_filename}: {e}", exc_info=True)
        return None
    

    if template_name.lower() == "default" and isinstance(dataset_data, list):
        if all(isinstance(item, dict) for item in dataset_data):
            try:
                
                df = pd.DataFrame(dataset_data)
                df.to_csv(csv_filename, index=False, encoding="utf-8")
                
                logging.info(f"Dataset (default template) also saved to {csv_filename}")
                
            except Exception as e:
                logging.warning(f"Could not save CSV for default template to {csv_filename}: {e}", exc_info=True)
        
        else:
            logging.warning("Default template data is not a list of dictionaries, cannot save as CSV.")
            
    return json_filename