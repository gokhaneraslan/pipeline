import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# --- General Configuration ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper() # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "True").lower() == "true"
LOG_FILE_NAME: str = os.getenv("LOG_FILE_NAME", "data_pipeline.log")
BASE_OUTPUT_DIR: Path = Path(os.getenv("BASE_OUTPUT_DIR", "./pipeline_output"))

# --- LLM Provider Configuration ---
# Options: "groq", "mistral", "ollama"
PRIMARY_LLM_PROVIDER: str = os.getenv("PRIMARY_LLM_PROVIDER", "groq")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CLIENT_TIMEOUT: int = int(os.getenv("OLLAMA_CLIENT_TIMEOUT", "120"))


# --- LLM Model Names ---
# Task types: 
# "query_generation", "query_evaluation", "results_filtering",
# "relevance_scoring", "focus_scoring", "content_extraction",
# "qa_generation", "dataset_refinement"

LLM_MODELS: Dict[str, Dict[str, str]] = {
    "groq": {
        "default": "llama-3.3-70b-versatile",
        "query_generation": "llama-3.3-70b-versatile",
        "query_evaluation": "llama-3.3-70b-versatile", 
        "results_filtering": "llama-3.3-70b-versatile", 
        "relevance_scoring": "llama-3.3-70b-versatile",
        "focus_scoring": "llama-3.3-70b-versatile",   
        "content_extraction": "llama-3.3-70b-versatile", 
        "qa_generation": "llama-3.3-70b-versatile",
        "dataset_refinement": "llama-3.3-70b-versatile",
    },
    "mistral": {
        "default": "mistral-small-latest",
        "query_generation": "mistral-small-latest",
        "query_evaluation": "mistral-small-latest", 
        "results_filtering": "mistral-small-latest", 
        "relevance_scoring": "mistral-small-latest",
        "focus_scoring": "mistral-small-latest",   
        "content_extraction": "mistral-small-latest", 
        "qa_generation": "mistral-small-latest",
        "dataset_refinement": "mistral-small-latest",
    },
    "ollama": {
        "default": "llama3:latest",
        #"default": "mistral:latest",
        "query_generation": "llama3:latest",
        "query_evaluation": "llama3:latest", 
        "results_filtering": "llama3:latest", 
        "relevance_scoring": "llama3:latest",
        "focus_scoring": "llama3:latest",   
        "content_extraction": "llama3:latest", 
        "qa_generation": "llama3:latest",
        "dataset_refinement": "llama3:latest",
    }
}

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAY_SECONDS: int = int(os.getenv("LLM_RETRY_DELAY_SECONDS", "15"))

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_PROVIDER: str = os.getenv("EMBEDDING_MODEL_PROVIDER", "sentence_transformers") # "ollama" or "sentence_transformers"

OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
SENTENCE_TRANSFORMERS_EMBEDDING_MODEL: str = os.getenv("SENTENCE_TRANSFORMERS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Search Engine Configuration ---
SEARCH_ENGINE_TYPE: str = os.getenv("SEARCH_ENGINE_TYPE", "google") # "google" or "duckduckgo"
SEARCH_MAX_RESULTS_PER_QUERY: int = int(os.getenv("SEARCH_MAX_RESULTS_PER_QUERY", "5"))
SEARCH_LANGUAGE: str = os.getenv("SEARCH_LANGUAGE", "en")

# --- Web Scraping and Download Configuration ---
DEFAULT_TEXT_EXTRACTION_STRATEGY: str = os.getenv("DEFAULT_TEXT_EXTRACTION_STRATEGY", "trafilatura") # "trafilatura" or "beautifulsoup"
WEB_FETCHER_MAX_RETRIES: int = int(os.getenv("WEB_FETCHER_MAX_RETRIES", "3"))
WEB_FETCHER_BACKOFF_FACTOR: float = float(os.getenv("WEB_FETCHER_BACKOFF_FACTOR", "1.5"))
WEB_FETCHER_INITIAL_WAIT: float = float(os.getenv("WEB_FETCHER_INITIAL_WAIT", "1.0"))
WEB_SCRAPER_DELAY_BETWEEN_URLS: float = float(os.getenv("WEB_SCRAPER_DELAY_BETWEEN_URLS", "1.5"))

# --- Pipeline Stage Specific Directories ---
INITIAL_DOWNLOAD_DIR_NAME: str = "01_initial_downloads"
STAGE3_RELEVANT_FILES_DIR_NAME: str = "02_relevant_files"
STAGE4_FOCUSED_FILES_DIR_NAME: str = "03_focused_files"
FINAL_OUTPUT_DIR_NAME: str = "04_extracted_content"
FINAL_EXTRACTED_TEXT_FILENAME: str = "compiled_dataset_for_qa.txt"
DATASET_OUTPUT_DIR_NAME: str = "05_final_datasets"
DATASET_OUTPUT_FILENAME_BASE: str = "qna_dataset"

# --- Processing Configuration ---
# Stage 1: Query Generation
TOPIC_FOR_PROCESSING: str = os.getenv("TOPIC_FOR_PROCESSING", "")
NUM_QUERIES_TO_SELECT_AFTER_GENERATION: int = int(os.getenv("NUM_QUERIES_TO_SELECT_AFTER_GENERATION", "5"))

# Stage 3: Initial Filtering (Relevance) using SentenceSplitter
STAGE3_CHUNK_SIZE: int = int(os.getenv("STAGE3_CHUNK_SIZE", "2048"))
STAGE3_CHUNK_OVERLAP: int = int(os.getenv("STAGE3_CHUNK_OVERLAP", "200"))
STAGE3_RELEVANCE_THRESHOLD: int = int(os.getenv("STAGE3_RELEVANCE_THRESHOLD", "60")) # 0-100

# Stage 4 & 5 & 6: Semantic Splitting Config
SEMANTIC_SPLITTER_BUFFER_SIZE: int = int(os.getenv("SEMANTIC_SPLITTER_BUFFER_SIZE", "2"))
SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE: int = int(os.getenv("SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE", "96"))

# Stage 4: Detailed Focus Filtering
STAGE4_FOCUS_THRESHOLD: int = int(os.getenv("STAGE4_FOCUS_THRESHOLD", "60")) # 0-100

# Stage 5: Final Content Extraction
STAGE5_TEXT_WRAP_WIDTH: int = int(os.getenv("STAGE5_TEXT_WRAP_WIDTH", "120"))

# Stage 6: Q&A Dataset Creation
STAGE6_NUM_QUESTIONS_PER_CHUNK: int = int(os.getenv("STAGE6_NUM_QUESTIONS_PER_CHUNK", "5"))
DEFAULT_DATASET_TEMPLATE: str = os.getenv("DEFAULT_DATASET_TEMPLATE", "default")
SUPPORTED_DATASET_TEMPLATES: List[str] = ["default", "gemma", "llama"]

# Stage 7: Dataset Refinement
REFINED_DATASET_DIR_NAME = "refined_dataset"
REFINED_DATASET_FILENAME_SUFFIX = "_refined"

def get_llm_model_for_task(task_key: str) -> str:
    """
    Gets the appropriate LLM model name for a given task_key based on the
    PRIMARY_LLM_PROVIDER. Falls back to "default" if task_key specific model is not found.
    """
    provider_models = LLM_MODELS.get(PRIMARY_LLM_PROVIDER)
    if not provider_models:
        raise ValueError(f"LLM provider '{PRIMARY_LLM_PROVIDER}' not configured in LLM_MODELS.")
    
    model_name = provider_models.get(task_key)
    if not model_name:
        model_name = provider_models.get("default")
        if not model_name:
            raise ValueError(f"No model specified for task '{task_key}' or a default model for provider '{PRIMARY_LLM_PROVIDER}'.")
    return model_name


def get_output_dir_path(stage_dir_name: str) -> Path:
    """Creates and returns the absolute Path for a given stage output directory name."""
    dir_path = BASE_OUTPUT_DIR / stage_dir_name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Could not create directory {dir_path}: {e}") from e
    return dir_path


WEB_FETCHER_CONFIG: Dict[str, Any] = {
    "max_retries": WEB_FETCHER_MAX_RETRIES,
    "backoff_factor": WEB_FETCHER_BACKOFF_FACTOR,
    "initial_wait_seconds": WEB_FETCHER_INITIAL_WAIT
}


STAGE3_SPLITTER_CONFIG: Dict[str, Any] = {
    "chunk_size": STAGE3_CHUNK_SIZE,
    "chunk_overlap": STAGE3_CHUNK_OVERLAP

}

BASE_SEMANTIC_SPLITTER_CONFIG: Dict[str, Any] = {
    "buffer_size": SEMANTIC_SPLITTER_BUFFER_SIZE,
    "breakpoint_percentile_threshold": SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE
}
