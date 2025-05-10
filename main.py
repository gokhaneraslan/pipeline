import sys
from typing import Optional, List, Dict
from pathlib import Path
import config
from logger_setup import setup_logger
from core.llm_client import get_llm_client

from stages.query_generation import query_generation_and_selection
from stages.search_and_download import search_filter_and_download
from stages.initial_filtering import initial_file_filtering
from stages.detailed_focus_filtering import run_detailed_focus_filtering
from stages.content_extraction import final_content_extraction
from stages.question_generation_and_answers import generate_questions_and_answers
from stages.dataset_refinement import refine_dataset_with_llm 

try:
    config.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"FATAL: Could not create base output directory {config.BASE_OUTPUT_DIR}. Error: {e}")
    sys.exit(1)


log_file_path = config.BASE_OUTPUT_DIR / config.LOG_FILE_NAME if config.LOG_TO_FILE else None
logger = setup_logger(log_level=config.LOG_LEVEL, log_file=log_file_path)


def run_pipeline():
    """
    Orchestrates the execution of the entire data processing pipeline.
    """
    logger.info("==================================================")
    logger.info("=== STARTING DATA PROCESSING PIPELINE ===")
    logger.info(f"Pipeline Output Directory: {config.BASE_OUTPUT_DIR.resolve()}")
    logger.info("==================================================")

    topic = config.TOPIC_FOR_PROCESSING
    logger.info(f"Processing for Topic: '{topic}'")

    primary_llm_provider = config.PRIMARY_LLM_PROVIDER
    llm_client = get_llm_client(primary_llm_provider, vars(config))
    
    if not llm_client:
      
        logger.critical(
            f"Failed to initialize LLM client for provider: {primary_llm_provider}. "
            "Ensure API keys and configurations (e.g., OLLAMA_BASE_URL) are correct. Pipeline cannot continue."
        )
        return

    try:
        initial_download_dir = config.get_output_dir_path(config.INITIAL_DOWNLOAD_DIR_NAME)
        stage3_output_dir = config.get_output_dir_path(config.STAGE3_RELEVANT_FILES_DIR_NAME)
        stage4_output_dir = config.get_output_dir_path(config.STAGE4_FOCUSED_FILES_DIR_NAME)
        final_content_output_dir = config.get_output_dir_path(config.FINAL_OUTPUT_DIR_NAME)
        dataset_output_dir = config.get_output_dir_path(config.DATASET_OUTPUT_DIR_NAME)
        refined_dataset_dir = config.get_output_dir_path(config.REFINED_DATASET_DIR_NAME)
        
    except OSError as e:
        logger.critical(f"Failed to create one or more output directories: {e}. Aborting pipeline.", exc_info=True)
        return

    # ==================================================
    # === STAGE 1: Query Generation and Selection ===
    # ==================================================
    
    selected_queries: List[str] = []
    try:
      
        logger.info("\n--- EXECUTING STAGE 1: Query Generation and Selection ---")
        
        selected_queries = query_generation_and_selection(
            llm_client=llm_client,
            llm_provider_name=primary_llm_provider,
            generation_model_name=config.get_llm_model_for_task("query_generation"),
            evaluation_model_name=config.get_llm_model_for_task("query_evaluation"),
            topic=topic,
            num_queries_to_select=config.NUM_QUERIES_TO_SELECT_AFTER_GENERATION,
            llm_temperature=config.LLM_TEMPERATURE,
            llm_max_retries=config.LLM_MAX_RETRIES,
            llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS
        )
        
        if not selected_queries:
            logger.error("Stage 1 did not produce any selected queries. Subsequent stages might not yield results or will be skipped.")
        
        else:
            logger.info(f"Stage 1 completed. Selected {len(selected_queries)} queries.")
    
    except Exception as e:
        logger.critical(f"Critical error in Stage 1 (Query Generation): {e}. Aborting pipeline.", exc_info=True)
        return

    # ==================================================
    # === STAGE 2: Search, Filter, and Download ===
    # ==================================================
    
    downloaded_files_map: Dict[str, Optional[Path]] = {}
    if selected_queries:
        try:
          
            logger.info("\n--- EXECUTING STAGE 2: Search, Filter, and Download ---")
            
            search_engine_cfg = {
                "type": config.SEARCH_ENGINE_TYPE,
                "num_results": config.SEARCH_MAX_RESULTS_PER_QUERY,
                "language": config.SEARCH_LANGUAGE
            }
            
            downloaded_files_map = search_filter_and_download(
                selected_queries=selected_queries,
                search_engine_config=search_engine_cfg,
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_filter_model_name=config.get_llm_model_for_task("results_filtering"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS,
                download_output_dir=initial_download_dir,
                fetch_config=config.WEB_FETCHER_CONFIG,
                scraper_delay_between_urls=config.WEB_SCRAPER_DELAY_BETWEEN_URLS,
                scraper_default_text_extraction_strategy=config.DEFAULT_TEXT_EXTRACTION_STRATEGY
            )
            
            num_successful_downloads = sum(1 for path in downloaded_files_map.values() if path is not None)
            
            if num_successful_downloads == 0:
                logger.warning("Stage 2 did not download or scrape any files successfully.")
            
            else:
                logger.info(f"Stage 2 completed. Successfully processed {num_successful_downloads} URLs resulting in downloaded files.")
        
        except Exception as e:
            logger.critical(f"Critical error in Stage 2 (Search & Download): {e}. Aborting pipeline.", exc_info=True)
            return
    
    else:
        logger.warning("Skipping Stage 2 as no queries were selected in Stage 1.")

    # ==================================================
    # === STAGE 3: Initial File Filtering (Relevance) ===
    # ==================================================
    
    stage3_relevant_files: List[Path] = []
    if initial_download_dir.exists() and any(initial_download_dir.iterdir()):
        try:
          
            logger.info("\n--- EXECUTING STAGE 3: Initial File Filtering (Relevance) ---")
            
            stage3_relevant_files = initial_file_filtering(
                input_files_dir=initial_download_dir,
                output_relevant_files_dir=stage3_output_dir,
                topic=topic,
                relevance_threshold=config.STAGE3_RELEVANCE_THRESHOLD,
                splitter_config=config.STAGE3_SPLITTER_CONFIG,
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_scoring_model_name=config.get_llm_model_for_task("relevance_scoring"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS
            )
            
            if not stage3_relevant_files:
                logger.warning("Stage 3 did not identify any relevant files.")
                
            else:
                logger.info(f"Stage 3 completed. Identified {len(stage3_relevant_files)} relevant files.")
                
        except Exception as e:
            logger.critical(f"Critical error in Stage 3 (Initial Filtering): {e}. Aborting pipeline.", exc_info=True)
            return
          
    else:
        logger.warning("Skipping Stage 3 as the download directory from Stage 2 is empty, does not exist, or Stage 2 was skipped.")

    # ==================================================
    # === STAGE 4: Detailed File Filtering (Focus) ===
    # ==================================================
    
    stage4_focused_files: List[Path] = []
    if stage3_relevant_files:
        try:
          
            logger.info("\n--- EXECUTING STAGE 4: Detailed File Filtering (Focus) ---")
            
            stage4_focused_files = run_detailed_focus_filtering(
                input_files_from_stage3=stage3_relevant_files,
                output_focused_files_dir=stage4_output_dir,
                topic=topic,
                focus_threshold=config.STAGE4_FOCUS_THRESHOLD,
                embedding_provider=config.EMBEDDING_MODEL_PROVIDER,
                ollama_embed_model=config.OLLAMA_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                ollama_url=config.OLLAMA_BASE_URL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                st_embed_model=config.SENTENCE_TRANSFORMERS_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "sentence_transformers" else None,
                semantic_splitter_buffer=config.BASE_SEMANTIC_SPLITTER_CONFIG["buffer_size"],
                semantic_splitter_breakpoint_perc=config.BASE_SEMANTIC_SPLITTER_CONFIG["breakpoint_percentile_threshold"],
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_scoring_model_name=config.get_llm_model_for_task("focus_scoring"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS
            )
            
            if not stage4_focused_files:
                logger.warning("Stage 4 did not identify any focused files.")
                
            else:
                logger.info(f"Stage 4 completed. Identified {len(stage4_focused_files)} focused files.")
                
        except Exception as e:
            logger.critical(f"Critical error in Stage 4 (Detailed Filtering): {e}. Aborting pipeline.", exc_info=True)
            return
          
    else:
        logger.warning("Skipping Stage 4 as no relevant files were produced by Stage 3.")

    # ==================================================
    # === STAGE 5: Final Content Extraction ===
    # ==================================================
    
    final_compiled_text_path: Optional[Path] = None
    if stage4_focused_files:
        try:
          
            logger.info("\n--- EXECUTING STAGE 5: Final Content Extraction ---")
            
            final_compiled_text_path = final_content_extraction(
                input_files_from_stage4=stage4_focused_files,
                final_output_dir=final_content_output_dir,
                final_output_filename=config.FINAL_EXTRACTED_TEXT_FILENAME,
                topic=topic,
                text_wrap_width=config.STAGE5_TEXT_WRAP_WIDTH,
                embedding_provider=config.EMBEDDING_MODEL_PROVIDER,
                ollama_embed_model=config.OLLAMA_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                ollama_url=config.OLLAMA_BASE_URL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                st_embed_model=config.SENTENCE_TRANSFORMERS_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "sentence_transformers" else None,
                semantic_splitter_buffer=config.BASE_SEMANTIC_SPLITTER_CONFIG["buffer_size"],
                semantic_splitter_breakpoint_perc=config.BASE_SEMANTIC_SPLITTER_CONFIG["breakpoint_percentile_threshold"],
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_extraction_model_name=config.get_llm_model_for_task("content_extraction"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS
            )
            
            if final_compiled_text_path:
                logger.info(f"Stage 5 completed. Final compiled text created at: {final_compiled_text_path.resolve()}")
                
            else:
                logger.warning("Stage 5 did not produce a final compiled text file (possibly no content extracted or error during save).")
        
        except Exception as e:
            logger.critical(f"Critical error in Stage 5 (Content Extraction): {e}. Pipeline may finish with incomplete data.", exc_info=True)
            
    else:
        logger.warning("Skipping Stage 5 as no focused files were produced by Stage 4.")

    # ==================================================
    # === STAGE 6: Q&A Dataset Creation ===
    # ==================================================
    
    initial_dataset_path_str: Optional[str] = None
    if final_compiled_text_path and final_compiled_text_path.exists():
        try:
          
            logger.info("\n--- EXECUTING STAGE 6: Q&A Dataset Creation ---")
            initial_dataset_path_str = generate_questions_and_answers(
                input_final_content_file=final_compiled_text_path,
                dataset_output_dir=dataset_output_dir,
                dataset_output_filename_base=config.DATASET_OUTPUT_FILENAME_BASE,
                num_questions_per_chunk=config.STAGE6_NUM_QUESTIONS_PER_CHUNK,
                llm_dataset_template=config.DEFAULT_DATASET_TEMPLATE,
                llm_dataset_supported_templates=config.SUPPORTED_DATASET_TEMPLATES,
                embedding_provider=config.EMBEDDING_MODEL_PROVIDER,
                ollama_embed_model=config.OLLAMA_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                ollama_url=config.OLLAMA_BASE_URL if config.EMBEDDING_MODEL_PROVIDER == "ollama" else None,
                st_embed_model=config.SENTENCE_TRANSFORMERS_EMBEDDING_MODEL if config.EMBEDDING_MODEL_PROVIDER == "sentence_transformers" else None,
                semantic_splitter_buffer=config.BASE_SEMANTIC_SPLITTER_CONFIG["buffer_size"],
                semantic_splitter_breakpoint_perc=config.BASE_SEMANTIC_SPLITTER_CONFIG["breakpoint_percentile_threshold"],
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_qa_model_name=config.get_llm_model_for_task("qa_generation"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS,
            )
            
            if initial_dataset_path_str:
                logger.info(f"Stage 6 completed. Dataset created at: {Path(initial_dataset_path_str).resolve()}") 
                
            else:
                logger.error("Stage 6 (Q&A Dataset Creation) did not produce a dataset file or failed.")
        
        except Exception as e:
            logger.critical(f"Critical error in Stage 6 (Q&A Dataset Creation): {e}. Dataset generation failed.", exc_info=True)
    
    elif stage4_focused_files:
        logger.warning("Skipping Stage 6 as Stage 5 (Final Content Extraction) did not produce a usable output file.")
    
    else:
        logger.warning("Skipping Stage 6 as no focused files were available from previous stages.")


    # ==================================================
    # === STAGE 7: Q&A Dataset Refinement ===
    # ==================================================
    
    refined_dataset_path: Optional[Path] = None
    if initial_dataset_path_str and Path(initial_dataset_path_str).exists():
        
        try:
            logger.info("\n--- EXECUTING STAGE 7: Q&A Dataset Refinement ---")
            
            input_dataset_for_refinement = Path(initial_dataset_path_str)

            refined_dataset_path = refine_dataset_with_llm(
                input_dataset_path=input_dataset_for_refinement,
                output_refined_dataset_dir=refined_dataset_dir,
                refined_dataset_filename_base=config.DATASET_OUTPUT_FILENAME_BASE,
                refined_filename_suffix=config.REFINED_DATASET_FILENAME_SUFFIX,
                topic=topic,
                llm_client=llm_client,
                llm_provider_name=primary_llm_provider,
                llm_refinement_model_name=config.get_llm_model_for_task("dataset_refinement"),
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_retries=config.LLM_MAX_RETRIES,
                llm_retry_delay=config.LLM_RETRY_DELAY_SECONDS,
            )

            if refined_dataset_path:
                logger.info(f"Stage 7 completed. Refined dataset saved at: {refined_dataset_path.resolve()}")
            
            else:
                logger.error("Stage 7 (Dataset Refinement) did not produce a refined dataset file or failed.")
        
        except Exception as e:
            logger.critical(f"Critical error in Stage 7 (Dataset Refinement): {e}. Refinement failed.", exc_info=True)
    
    elif final_compiled_text_path and final_compiled_text_path.exists(): 
        logger.warning("Skipping Stage 7 as Stage 6 (Q&A Dataset Creation) did not produce an initial dataset.")
    
    else:
        logger.warning("Skipping Stage 7 as no dataset was available from previous stages for refinement.")

    logger.info("==================================================")
    

    if refined_dataset_path and refined_dataset_path.exists():
        logger.info(f"=== DATA PROCESSING PIPELINE FINISHED SUCCESSFULLY ===")
        logger.info(f"Final refined Q&A dataset available at: {refined_dataset_path.resolve()}")
    
    elif initial_dataset_path_str and Path(initial_dataset_path_str).exists():
        logger.info(f"=== DATA PROCESSING PIPELINE FINISHED (Refinement stage may have issues or was skipped) ===")
        logger.info(f"Initial Q&A dataset (before refinement) available at: {Path(initial_dataset_path_str).resolve()}")
    
    elif final_compiled_text_path and final_compiled_text_path.exists():
        logger.info(f"=== DATA PROCESSING PIPELINE FINISHED (Q&A/Refinement stages may have issues) ===")
        logger.info(f"Compiled text (input for Q&A) is at: {final_compiled_text_path.resolve()}")
    
    else:
        logger.info(f"=== DATA PROCESSING PIPELINE FINISHED (with potential issues or no data processed) ===")
        
    logger.info(f"All pipeline outputs are under: {config.BASE_OUTPUT_DIR.resolve()}")
    logger.info("==================================================")



if __name__ == "__main__":
    run_pipeline()
