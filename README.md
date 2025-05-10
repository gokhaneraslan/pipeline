# LLM-Powered Q&A Dataset Generation Pipeline

This project implements a multi-stage automated pipeline designed to generate high-quality Question & Answer (Q&A) datasets for a given technical topic. It leverages Large Language Models (LLMs) for various tasks including query generation, content filtering, relevance scoring, content extraction, and Q&A pair creation. The pipeline is configurable, supporting different LLM providers (Groq, Mistral, Ollama), search engines, and processing parameters.

## Overview

The core objective of this pipeline is to automate the laborious process of finding relevant information on a specific topic, extracting the core explanatory content, and then transforming that content into a structured Q&A dataset suitable for training smaller LLMs or for knowledge base construction.

## Features

*   **Multi-LLM Provider Support:** Easily switch between Groq, Mistral, and local Ollama instances.
*   **Configurable LLM Models:** Specify different LLM models for various tasks (query generation, scoring, extraction, etc.).
*   **Multi-Search Engine Support:** Utilizes Google or DuckDuckGo for initial information retrieval.
*   **Advanced Web Scraping:** Employs `crawl4ai`, `Trafilatura`, and `BeautifulSoup` for robust content extraction from web pages and PDFs.
*   **Multi-Stage Filtering:**
    *   LLM-based filtering of initial search results.
    *   Relevance scoring of downloaded documents using LLM and sentence splitting.
    *   Detailed focus scoring of relevant documents using LLM and semantic splitting.
*   **Semantic Chunking:** Uses `llama-index` (SentenceSplitter and SemanticSplitter) for intelligent document segmentation.
*   **LLM-Powered Content Extraction:** Isolates core explanatory content from noisy text chunks.
*   **Automated Q&A Generation:** Creates question-answer pairs from the extracted content using an LLM.
*   **Dataset Refinement:** An optional LLM-powered step to review and improve the generated Q&A pairs.
*   **Flexible Configuration:** Most parameters are configurable via a `.env` file and `config.py`.
*   **Comprehensive Logging:** Detailed logging for monitoring and debugging.
*   **Structured Output:** Organizes outputs from each stage into dedicated directories.

## Pipeline Stages

The pipeline is divided into several distinct stages, each performing a specific part of the data processing workflow:

### Stage 1: Query Generation and Selection

*   **Purpose:** To generate a diverse set of search queries relevant to the input `TOPIC_FOR_PROCESSING` and select the most promising ones for web searching.
*   **Process:**
    1.  An LLM (configured by `PRIMARY_LLM_PROVIDER` and `LLM_MODELS["query_generation"]`) generates an initial list of search queries based on the topic. The prompt for this is `prompts.get_query_generation_prompt`.
    2.  Another LLM call (model: `LLM_MODELS["query_evaluation"]`) evaluates these generated queries and selects a diverse subset (`NUM_QUERIES_TO_SELECT_AFTER_GENERATION`). The prompt for this is `prompts.get_query_evaluation_prompt`.
*   **Input:** `TOPIC_FOR_PROCESSING` (from `config.py`).
*   **Output:** A list of selected query strings.
*   **Modules:** `stages/query_generation.py`, `core/prompts.py`, `processing/llm_interactor.py`.

### Stage 2: Search, Filter, and Download

*   **Purpose:** To use the selected queries to find relevant web pages/documents, filter these results using an LLM, and download/scrape their content.
*   **Process:**
    1.  For each selected query:
        *   Perform a web search using the configured `SEARCH_ENGINE_TYPE` (Google or DuckDuckGo) to get `SEARCH_MAX_RESULTS_PER_QUERY` results.
        *   An LLM (model: `LLM_MODELS["results_filtering"]`) filters these search results to identify the most relevant URLs. The prompt is `prompts.get_search_results_filtering_prompt`.
    2.  All unique, filtered URLs are collected.
    3.  The content from these URLs is downloaded (for PDFs) or scraped (for HTML pages using `DEFAULT_TEXT_EXTRACTION_STRATEGY` like `trafilatura` or `beautifulsoup`).
*   **Input:** List of selected queries from Stage 1.
*   **Output:** Downloaded PDF files and scraped text files stored in the `pipeline_output/01_initial_downloads/` directory. A dictionary mapping processed URLs to their local file paths.
*   **Modules:** `stages/search_and_download.py`, `processing/search_engine.py`, `processing/web_scraper.py`, `processing/llm_interactor.py`, `core/utils.py`.

### Stage 3: Initial File Filtering (Relevance Scoring)

*   **Purpose:** To perform an initial pass on all downloaded/scraped files, scoring them for overall relevance to the topic.
*   **Process:**
    1.  For each file in `01_initial_downloads/`:
        *   The content is read (PDFs are parsed using `unstructured`).
        *   The text is split into chunks using `llama-index.SentenceSplitter` (configured by `STAGE3_CHUNK_SIZE`, `STAGE3_CHUNK_OVERLAP`).
        *   An LLM (model: `LLM_MODELS["relevance_scoring"]`) assigns a relevance score (0-100) to each chunk based on the `prompts.get_chunk_relevance_scoring_prompt`.
        *   The average relevance score for the file is calculated.
    2.  Files with an average score above `STAGE3_RELEVANCE_THRESHOLD` are copied to `pipeline_output/02_relevant_files/`.
*   **Input:** Files from `pipeline_output/01_initial_downloads/`.
*   **Output:** A list of paths to relevant files copied to `pipeline_output/02_relevant_files/`.
*   **Modules:** `stages/initial_filtering.py`, `processing/document_parser.py`, `processing/llm_interactor.py`.

### Stage 4: Detailed File Filtering (Focus Scoring)

*   **Purpose:** To further refine the set of relevant files by assessing how *focused* their content is on explaining the topic, rather than just being generally related.
*   **Process:**
    1.  For each file in `02_relevant_files/`:
        *   The content is split into chunks using `llama-index.SemanticSplitterNodeParser`. This uses an embedding model (`EMBEDDING_MODEL_PROVIDER` like `ollama` or `sentence_transformers`) to create more semantically coherent chunks.
        *   An LLM (model: `LLM_MODELS["focus_scoring"]`) assigns a focus score (0-100) to each semantic chunk based on `prompts.get_chunk_focus_scoring_prompt`.
        *   The average focus score for the file is calculated.
    2.  Files with an average score above `STAGE4_FOCUS_THRESHOLD` are copied to `pipeline_output/03_focused_files/`.
*   **Input:** Files from `pipeline_output/02_relevant_files/`.
*   **Output:** A list of paths to focused files copied to `pipeline_output/03_focused_files/`.
*   **Modules:** `stages/detailed_focus_filtering.py`, `processing/document_parser.py`, `processing/llm_interactor.py`.

### Stage 5: Final Content Extraction

*   **Purpose:** To extract only the core, explanatory text related to the topic from the focused files, discarding irrelevant sections.
*   **Process:**
    1.  For each file in `03_focused_files/`:
        *   The content is split into semantic chunks (as in Stage 4).
        *   An LLM (model: `LLM_MODELS["content_extraction"]`) processes each chunk, guided by `prompts.get_clean_content_extraction_prompt`, to extract only the relevant explanatory paragraphs.
    2.  All extracted text segments from all files are compiled into a single text file, with separators indicating original file sources. Text is wrapped for readability (`STAGE5_TEXT_WRAP_WIDTH`).
*   **Input:** Files from `pipeline_output/03_focused_files/`.
*   **Output:** A single compiled text file (`compiled_dataset_for_qa.txt`) in `pipeline_output/04_extracted_content/`.
*   **Modules:** `stages/content_extraction.py`, `processing/document_parser.py`, `processing/llm_interactor.py`.

### Stage 6: Q&A Dataset Creation

*   **Purpose:** To generate question-answer pairs from the final compiled text.
*   **Process:**
    1.  The compiled text file from Stage 5 is read.
    2.  The content is split into semantic chunks.
    3.  For each chunk:
        *   An LLM (model: `LLM_MODELS["qa_generation"]`) generates a specified number of questions (`STAGE6_NUM_QUESTIONS_PER_CHUNK`) based *only* on the content of that chunk. The prompt is `prompts.get_questions_generate_prompt`.
        *   The same LLM then answers each generated question based *strictly* on the same chunk's content. The prompt is `prompts.get_answered_questions_prompt`.
    4.  All generated Q&A pairs are collected and formatted into a dataset (e.g., default JSON list of input/output pairs, or Gemma/Llama specific formats based on `DEFAULT_DATASET_TEMPLATE`).
*   **Input:** The compiled text file from `pipeline_output/04_extracted_content/`.
*   **Output:** A Q&A dataset file (e.g., `qna_dataset_YYYYMMDD_HHMMSS.json`) in `pipeline_output/05_final_datasets/`.
*   **Modules:** `stages/question_generation_and_answers.py`, `processing/document_parser.py`, `processing/llm_interactor.py`, `core/utils.py` (for templating).

### Stage 7: Q&A Dataset Refinement (Optional)

*   **Purpose:** To use an LLM to review and potentially improve the generated Q&A dataset for accuracy, consistency, clarity, and completeness.
*   **Process:**
    1.  The Q&A dataset file from Stage 6 is read.
    2.  The entire dataset (as a JSON string) is provided to an LLM (model: `LLM_MODELS["dataset_refinement"]`) along with `prompts.get_check_dataset_prompt`. This prompt instructs the LLM to:
        *   Verify factual accuracy.
        *   Resolve inconsistencies.
        *   Attempt to answer questions previously marked as "information not available" using general knowledge of the topic (if appropriate).
        *   Enhance clarity and completeness of answers.
        *   Improve natural language flow.
    3.  The LLM returns a revised JSON dataset.
*   **Input:** The Q&A dataset file from `pipeline_output/05_final_datasets/`.
*   **Output:** A refined Q&A dataset file (e.g., `qna_dataset_refined_YYYYMMDD_HHMMSS.json`) in `pipeline_output/refined_dataset/`.
*   **Modules:** `stages/dataset_refinement.py`, `processing/llm_interactor.py`.

## Prerequisites

*   Python 3.9+
*   Access to LLM APIs:
    *   Groq API Key (if using Groq)
    *   Mistral API Key (if using Mistral)
    *   A running Ollama instance with desired models pulled (if using Ollama, e.g., `ollama pull llama3`, `ollama pull nomic-embed-text`)
*   `git` for cloning the repository.
*   (Recommended) Tesseract OCR and Poppler-utils for robust PDF processing with `unstructured`:
    *   **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr poppler-utils`
    *   **MacOS:** `brew install tesseract poppler`
    *   **Windows:** Requires manual installation. Add Tesseract to PATH.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers (for `crawl4ai` if its default browser-based strategies are used, though this project primarily uses LXML/Trafilatura):**
    ```bash
    playwright install
    ```
    (Note: The current `web_scraper.py` uses `LXMLWebScrapingStrategy` for `crawl4ai` which may not require this, but it's good practice for full `crawl4ai` functionality.)

## Configuration

The pipeline is configured primarily through environment variables, which can be set in a `.env` file in the project root directory. A `config.py` file loads these variables and provides defaults.

1.  **Create a `.env` file** in the root of the project:
    ```
    cp .env.example .env # If an example file is provided, otherwise create manually
    ```

2.  **Edit `.env` and `config.py`** with your settings:

    **Key Environment Variables for `.env`:**
    ```env
    # --- General ---
    LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_FILE=True
    LOG_FILE_NAME=data_pipeline.log
    BASE_OUTPUT_DIR=./pipeline_output

    # --- LLM Provider ---
    PRIMARY_LLM_PROVIDER=groq # Options: "groq", "mistral", "ollama"

    # API Keys (only fill for the provider you use)
    GROQ_API_KEY=your_groq_api_key
    MISTRAL_API_KEY=your_mistral_api_key

    # Ollama Settings (if PRIMARY_LLM_PROVIDER="ollama")
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_CLIENT_TIMEOUT=120

    # --- Embedding Model (for Semantic Splitting) ---
    EMBEDDING_MODEL_PROVIDER=sentence_transformers # "ollama" or "sentence_transformers"
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text # If EMBEDDING_MODEL_PROVIDER="ollama"
    SENTENCE_TRANSFORMERS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 # If EMBEDDING_MODEL_PROVIDER="sentence_transformers"

    # --- Search Engine ---
    SEARCH_ENGINE_TYPE=google # "google" or "duckduckgo"
    SEARCH_MAX_RESULTS_PER_QUERY=5
    SEARCH_LANGUAGE=en

    # --- Topic ---
    TOPIC_FOR_PROCESSING="Your chosen technical topic here" # e.g., "Quantum Entanglement Explained"

    # --- Other configurations can be adjusted directly in config.py if needed ---
    # LLM_MODELS (per provider, per task)
    # LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_RETRY_DELAY_SECONDS
    # STAGE3_CHUNK_SIZE, STAGE3_CHUNK_OVERLAP, STAGE3_RELEVANCE_THRESHOLD
    # SEMANTIC_SPLITTER_BUFFER_SIZE, SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE
    # STAGE4_FOCUS_THRESHOLD
    # STAGE5_TEXT_WRAP_WIDTH
    # STAGE6_NUM_QUESTIONS_PER_CHUNK
    # DEFAULT_DATASET_TEMPLATE
    ```

    **`config.py` adjustments:**
    *   You can customize `LLM_MODELS` in `config.py` to specify different models for each LLM provider and for each task (e.g., `query_generation`, `relevance_scoring`).
    *   Other numerical thresholds and parameters for pipeline stages are also defined in `config.py`.

## Usage

After installation and configuration:

1.  Ensure your LLM provider is accessible (e.g., Ollama server is running if selected).
2.  Run the main pipeline script:
    ```bash
    python main.py
    ```
3.  Monitor the console output and the log file (`pipeline_output/data_pipeline.log` by default) for progress and any errors.
4.  Outputs from each stage will be saved in subdirectories within `pipeline_output/` (or your configured `BASE_OUTPUT_DIR`). The final refined dataset will typically be in `pipeline_output/refined_dataset/`.

## Directory Structure
Use code with caution.
Markdown
.
├── core/ # Core utilities and prompt definitions
│ ├── llm_client.py # LLM client initialization
│ ├── prompts.py # LLM prompt templates
│ └── utils.py # General utility functions
├── processing/ # Modules for specific data processing tasks
│ ├── document_parser.py # PDF/text parsing, document splitting
│ ├── llm_interactor.py # Logic for LLM interactions for various tasks
│ ├── search_engine.py # Web search functionalities
│ └── web_scraper.py # Web scraping and PDF downloading
├── stages/ # Modules defining each pipeline stage
│ ├── query_generation.py
│ ├── search_and_download.py
│ ├── initial_filtering.py
│ ├── detailed_focus_filtering.py
│ ├── content_extraction.py
│ ├── question_generation_and_answers.py
│ └── dataset_refinement.py
├── pipeline_output/ # Default directory for all generated outputs and logs
│ ├── 01_initial_downloads/ # Raw downloaded/scraped files
│ ├── 02_relevant_files/ # Files passing initial relevance filter
│ ├── 03_focused_files/ # Files passing detailed focus filter
│ ├── 04_extracted_content/ # Compiled extracted text
│ ├── 05_final_datasets/ # Generated Q&A datasets (before refinement)
│ ├── refined_dataset/ # Refined Q&A datasets
│ └── data_pipeline.log # Log file
├── config.py # Main configuration loading settings and defaults
├── logger_setup.py # Logging configuration
├── main.py # Main script to run the pipeline
├── requirements.txt # Python dependencies
├── .env.example # Example environment file (optional)
└── README.md # This file

## Key Technologies & Libraries

*   **LLM Interaction:**
    *   `groq`: For Groq API access.
    *   `mistralai`: For Mistral API access.
    *   `ollama`: For local LLM interaction via Ollama.
*   **Web Interaction & Scraping:**
    *   `googlesearch-python`: For Google search.
    *   `httpx`: For robust HTTP requests.
    *   `BeautifulSoup4`: For HTML parsing (DDG search, basic scraping).
    *   `trafilatura`: For extracting main content from web pages.
    *   `crawl4ai`: Advanced asynchronous web crawling framework.
*   **Document Processing & Chunking:**
    *   `unstructured`: For partitioning PDFs and extracting text from various elements.
    *   `PyMuPDF` (fitz): Often a dependency for PDF processing.
    *   `llama-index-core`: Core components for document representation and node parsing.
    *   `llama-index-embeddings-ollama`: Ollama embeddings for LlamaIndex.
    *   `llama-index-embeddings-huggingface`: HuggingFace Sentence Transformers embeddings.
    *   `sentence-transformers`: For local text embeddings.
*   **Utilities:**
    *   `python-dotenv`: For loading environment variables.
    *   `pandas`: For saving datasets to CSV (for default template).
    *   `nest_asyncio`: To manage asyncio event loops, especially when `crawl4ai` is used.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one is created).
