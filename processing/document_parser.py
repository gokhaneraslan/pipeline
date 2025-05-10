import logging
from pathlib import Path
from typing import Optional, List, Any, Union

import ollama
from unstructured.documents.elements import Text, Title, ListItem, NarrativeText, Table
from unstructured.partition.pdf import partition_pdf

from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


logger = logging.getLogger("DataPipeline")


def read_text_file_content(file_path: Path) -> Optional[str]:
    """Reads content from a text file."""
    logger.debug(f"Attempting to read text file: {file_path}")
    
    if not file_path.exists():
        logger.error(f"Text file not found: {file_path}")
        return None
    
    try:
        content = file_path.read_text(encoding='utf-8')
        if not content.strip():
            logger.warning(f"Text file is empty or contains only whitespace: {file_path}")
            return ""
        
        logger.info(f"Successfully read text file: {file_path} (Length: {len(content)})")
        return content
    
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}", exc_info=True)
        return None


def extract_text_from_pdf(
    file_path: Path, 
    strategy: str = "fast",
    languages: Optional[List[str]] = None ) -> Optional[str]:
    """
    Uses unstructured to partition a PDF and extract text from relevant elements.
    Excludes table text by default unless specifically handled with [TABLE:] prefix.
    """
    
    logger.debug(f"Attempting to extract text from PDF: {file_path} with strategy: {strategy}")
    if not file_path.exists():
        logger.error(f"PDF file not found: {file_path}")
        return None
    
    if languages is None:
        languages = ['eng']

    try:
        
        elements = partition_pdf(
            filename=str(file_path),
            strategy=strategy,
            infer_table_structure=True,
            languages=languages
            #include_page_breaks=False,
        )
        
        content_list = []
        relevant_element_types = (NarrativeText, Title, ListItem, Text) 
        
        for element_idx, element in enumerate(elements):
            element_text = getattr(element, 'text', '').strip()

            if not element_text:
                continue

            if isinstance(element, Table):
                content_list.append(f"[TABLE:]\n{element_text}")
                logger.debug(f"Extracted table (element {element_idx}) from PDF {file_path.name}")
                
            elif isinstance(element, relevant_element_types):
                content_list.append(element_text)
                
            else:
                logger.debug(f"Skipping element of type {type(element).__name__} (text: '{element_text[:50]}...') from PDF {file_path.name}")

        if not content_list:
            logger.warning(f"No relevant text elements extracted by unstructured from PDF: {file_path.name} with strategy {strategy}")
            return ""
            
        full_text = "\n\n".join(content_list)
        logger.info(f"Successfully extracted text from PDF: {file_path.name} (Length: {len(full_text)}) using strategy {strategy}")
        
        return full_text
    
    except Exception as e:
        logger.error(f"Error partitioning PDF file {file_path.name} with unstructured (strategy: {strategy}): {e}", exc_info=True)
        return None


def get_document_content(file_path: Path) -> Optional[str]:
    """
    Reads content from PDF or TXT file.
    """
    
    file_ext = file_path.suffix.lower()
    logger.debug(f"Getting document content for: {file_path} (type: {file_ext})")
    
    if file_ext == ".pdf":
        return extract_text_from_pdf(file_path, strategy="fast") # Default strategy "fast"
    
    elif file_ext == ".txt":
        return read_text_file_content(file_path)
    
    else:
        logger.warning(f"Unsupported file type for content extraction: {file_path}. Only .pdf and .txt are supported.")
        return None


def get_sentence_splitter(
    chunk_size: int, 
    chunk_overlap: int,
    paragraph_separator: str = "\n\n\n",
    secondary_chunking_regex: str = "[^,.;。？！]+[,.;。？！]?" # LlamaIndex default for sentence splitting
    ) -> SentenceSplitter:
    
    """Initializes and returns a LlamaIndex SentenceSplitter."""
    
    logger.info(f"Initializing SentenceSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    try:
        
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator=paragraph_separator,
            #secondary_chunking_regex=secondary_chunking_regex # For finer sentence boundary detection
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize SentenceSplitter: {e}", exc_info=True)
        raise


def get_semantic_splitter(
    embedding_model_provider: str,
    ollama_embedding_model_name: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    sentence_transformers_model_name: Optional[str] = None,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    ollama_client_timeout: int = 120 ) -> SemanticSplitterNodeParser:
    """Initializes and returns a LlamaIndex SemanticSplitterNodeParser."""
    
    logger.info(f"Initializing SemanticSplitter with provider: {embedding_model_provider.lower()}")
    
    embed_model: Any = None
    try:
        
        provider = embedding_model_provider.lower()
        if provider == "ollama":
            if not ollama_embedding_model_name or not ollama_base_url:
                
                msg = "Ollama embedding model name and base URL are required for Ollama provider."
                logger.error(msg)
                raise ValueError(msg)
            
            logger.info(f"Attempting to use Ollama embedding model: {ollama_embedding_model_name} from {ollama_base_url}")

            try:
                
                temp_client = ollama.Client(host=ollama_base_url, timeout=ollama_client_timeout)
                temp_client.list()
                logger.info(f"Ollama server at {ollama_base_url} is reachable.")
                
            except Exception as e:
                 logger.error(f"Ollama server at {ollama_base_url} not reachable or 'ollama list' failed: {e}. Ensure Ollama is running.")
                 raise

            embed_model = OllamaEmbedding(
                model_name=ollama_embedding_model_name,
                base_url=ollama_base_url,
                ollama_options={"timeout": ollama_client_timeout}
            )
            
            logger.info(f"OllamaEmbedding initialized with model: {ollama_embedding_model_name}")
            
        elif provider == "sentence_transformers":
            if not sentence_transformers_model_name:
                msg = "Sentence Transformers model name is required for sentence_transformers provider."
                logger.error(msg)
                raise ValueError(msg)
            
            logger.info(f"Using Sentence-Transformers embedding model: {sentence_transformers_model_name}")
            
            embed_model = HuggingFaceEmbedding(model_name=sentence_transformers_model_name)
            
            logger.info(f"HuggingFaceEmbedding initialized with model: {sentence_transformers_model_name}")
            
        else:
            msg = f"Unsupported embedding_model_provider: {embedding_model_provider}. Choose 'ollama' or 'sentence_transformers'."
            logger.error(msg)
            raise ValueError(msg)

        return SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize SemanticSplitter or its embedding model: {e}", exc_info=True)
        raise

def split_document_into_nodes(
    document_content: str,
    file_name_for_metadata: str,
    splitter: Union[SentenceSplitter, SemanticSplitterNodeParser]) -> List[TextNode]:
    """
    Splits document content into TextNode objects using the provided splitter.
    """
    
    if not document_content or not document_content.strip():
        logger.warning(f"Cannot split empty or null document content for: {file_name_for_metadata}")
        return []
    
    splitter_name = type(splitter).__name__
    logger.debug(f"Splitting document: {file_name_for_metadata} (len: {len(document_content)}) using {splitter_name}")
    
    try:

        llama_doc = LlamaDocument(
            text=document_content, 
            metadata={
                "filename": file_name_for_metadata, 
                "source_type": "parsed_file",
                "splitter_used": splitter_name
            }
        )
        
        nodes = splitter.get_nodes_from_documents([llama_doc], show_progress=False)
        
        if not nodes:
            logger.warning(f"Splitting produced no nodes for {file_name_for_metadata} using {splitter_name}.")
            return []
        
        logger.info(f"Generated {len(nodes)} nodes from {file_name_for_metadata} using {splitter_name}.")

        if nodes and nodes[0].get_content():
            snippet = nodes[0].get_content()[:100].replace("\n", " ")
            
            logger.debug(f"First node content snippet for {file_name_for_metadata}: '{snippet}...'")
             
        return nodes
    
    except Exception as e:
        logger.error(f"Error splitting document {file_name_for_metadata} into nodes using {splitter_name}: {e}", exc_info=True)
        return []
