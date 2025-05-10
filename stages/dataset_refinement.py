import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List

from processing.llm_interactor import llm_refined_json_response
from core.prompts import get_check_dataset_prompt


logger = logging.getLogger("DataPipeline")


def refine_dataset_with_llm(
    input_dataset_path: Path,
    output_refined_dataset_dir: Path,
    refined_dataset_filename_base: str,
    refined_filename_suffix: str,
    topic: str,
    llm_client: Any,
    llm_provider_name: str,
    llm_refinement_model_name: str,
    llm_temperature: float,
    llm_max_retries: int,
    llm_retry_delay: int ) -> Optional[Path]:
    """
    Executes Stage 7: Reads a Q&A dataset (JSON or JSONL), refines it using an LLM
    based on a specialized prompt, and saves the improved dataset.
    """
    
    logger.info(f"--- Starting Stage 7: Q&A Dataset Refinement ---")
    logger.info(f"Input dataset file: {input_dataset_path.resolve()}")
    logger.info(f"Output directory for refined dataset: {output_refined_dataset_dir.resolve()}")
    logger.info(f"Topic for refinement context: '{topic}'")

    output_refined_dataset_dir.mkdir(parents=True, exist_ok=True)

    if not input_dataset_path.exists():
        logger.error(f"Input dataset file not found: {input_dataset_path}. Stage 7 cannot proceed.")
        return None

    try:
        
        dataset_json_string: Optional[str] = None
        original_file_suffix = input_dataset_path.suffix.lower()

        logger.info(f"Reading dataset from: {input_dataset_path} (format: {original_file_suffix})")
        
        with open(input_dataset_path, 'r', encoding='utf-8') as f:
            
            if original_file_suffix == ".jsonl":
                
                dataset_list = []
                for line_num, line in enumerate(f, 1):
                    line_stripped = line.strip()
                    
                    if not line_stripped:
                        continue
                    
                    try:
                        dataset_list.append(json.loads(line_stripped))
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Skipping invalid JSON line {line_num} in {input_dataset_path}: '{line_stripped[:100]}...' - Error: {e}")
                        continue
                    
                if not dataset_list:
                    logger.error(f"No valid JSON objects found in JSONL file: {input_dataset_path}")
                    return None
                
                dataset_json_string = json.dumps(dataset_list)
                
            elif original_file_suffix == ".json":
                
                raw_content = f.read()
                try:

                    parsed_json = json.loads(raw_content)
                    if not isinstance(parsed_json, list):
                        logger.error(f"Input JSON file {input_dataset_path} does not contain a JSON array at the root.")
                        return None
                    
                    for item in parsed_json:
                        
                        if not isinstance(item, dict) or "input" not in item or "output" not in item:
                            logger.error(f"Invalid item structure in {input_dataset_path}. Expected objects with 'input' and 'output' keys. Found: {str(item)[:100]}")
                            return None
                        
                    dataset_json_string = raw_content
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Input dataset {input_dataset_path} is not valid JSON: {e}")
                    return None
                
            else:
                logger.error(f"Unsupported file extension for dataset: {original_file_suffix}. Expected .json or .jsonl.")
                return None

        if not dataset_json_string or not dataset_json_string.strip():
            logger.error(f"Dataset content from {input_dataset_path} is empty or could not be processed into a string.")
            return None

        try:
            num_qa_pairs = len(json.loads(dataset_json_string))
            logger.info(f"Prepared dataset string with {num_qa_pairs} Q&A pairs for refinement.")
        
        except json.JSONDecodeError:
            logger.error("Failed to parse the prepared dataset_json_string to count pairs. This is unexpected.")

        logger.info(f"Sending dataset to LLM ({llm_provider_name} - {llm_refinement_model_name}) for refinement...")

        refined_json_response_str: Optional[str] = None
        
        refined_json_response_str = llm_refined_json_response(
                llm_client=llm_client,
                provider_name=llm_provider_name,
                model_name=llm_refinement_model_name,
                topic=topic,
                dataset_json_string=dataset_json_string,
                temperature=llm_temperature,
                max_retries=llm_max_retries,
                retry_delay=llm_retry_delay,
                check_dataset_prompt_func=get_check_dataset_prompt
        )

      
        if refined_json_response_str is None:
            logger.error("LLM (via llm_refined_json_response) did not return a response string after all retries for dataset refinement.")
            return None
        
        if not refined_json_response_str.strip():
            logger.warning("LLM (via llm_refined_json_response) returned an empty string. Assuming no refinement or error in LLM output.")
            return None

        logger.info("LLM responded. Validating refined dataset JSON structure...")
        
        try:
            parsed_output_from_llm = json.loads(refined_json_response_str)
            refined_dataset_list: Optional[List[Dict[str, Any]]] = None

            if isinstance(parsed_output_from_llm, list):
                refined_dataset_list = parsed_output_from_llm
                
            elif isinstance(parsed_output_from_llm, dict):
                logger.warning("LLM returned a JSON object instead of a direct array. Attempting to find array within.")
                
                found_list_in_dict = False
                for key, value in parsed_output_from_llm.items():
                    
                    if isinstance(value, list):
                        
                        logger.info(f"Found list under key '{key}'. Using this as the dataset.")
                        
                        refined_dataset_list = value
                        found_list_in_dict = True
                        break
                    
                if not found_list_in_dict:
                    logger.error("LLM returned a JSON object, but no list was found within it.")
                    logger.debug(f"LLM Raw Output (first 500 chars): {refined_json_response_str[:500]}")
                    return None
                
            else:
                logger.error(f"Refined dataset from LLM is not a JSON array or a JSON object containing an array. Type: {type(parsed_output_from_llm)}")
                logger.debug(f"LLM Raw Output (first 500 chars): {refined_json_response_str[:500]}")
                return None

            if refined_dataset_list is None:
                logger.error("No valid list data could be extracted from LLM response after checks.")
                return None

            for item_idx, item in enumerate(refined_dataset_list):
                if not isinstance(item, dict) or "input" not in item or "output" not in item:
                    logger.error(f"Invalid item structure in refined dataset (item {item_idx+1}): {str(item)[:100]}. Missing 'input' or 'output' key.")
                    logger.debug(f"LLM Raw Output (first 500 chars): {refined_json_response_str[:500]}")
                    return None
            
            logger.info(f"Refined dataset JSON structure is valid. Contains {len(refined_dataset_list)} Q&A pairs.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse refined dataset string from LLM as JSON: {e}")
            logger.debug(f"LLM Raw Output (first 500 chars): {refined_json_response_str[:500]}")
            return None
        
        output_file_name = f"{refined_dataset_filename_base}{refined_filename_suffix}{original_file_suffix}"
        output_file_path = output_refined_dataset_dir / output_file_name

        logger.info(f"Saving refined dataset to: {output_file_path}")
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            
            if original_file_suffix == ".jsonl":
                
                for item in refined_dataset_list:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                    
            elif original_file_suffix == ".json":
                json.dump(refined_dataset_list, f, indent=4, ensure_ascii=False)

        logger.info(f"Stage 7: Dataset refinement completed successfully. Output at {output_file_path}")
        
        return output_file_path

    except FileNotFoundError:
        logger.error(f"Input file {input_dataset_path} was not found (should have been caught earlier).")
        return None

    except Exception as e:
        logger.critical(f"An unexpected error occurred during Stage 7 (Dataset Refinement): {e}", exc_info=True)
        return None