import os
import ollama
import logging
from mistralai import Mistral
from groq import Groq, GroqError
from typing import Optional, Dict, Any


logger = logging.getLogger("DataPipeline")


def initialize_groq_client(api_key: Optional[str] = None):
    """Initializes and returns a Groq client."""
    
    key = api_key or os.getenv("GROQ_API_KEY")
    
    if not key:
        logger.error("Groq API key not found. Set GROQ_API_KEY environment variable or provide it as an argument.")
        return None
      
    try:
        client = Groq(api_key=key)
        client.models.list() 
        logger.info("Groq client initialized and connection tested successfully.")
        
        return client
    
    except GroqError as e:
        logger.error(f"Failed to initialize Groq client or test connection: {e}")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during Groq client initialization: {e}", exc_info=True)
        
    return None


def initialize_mistral_client(api_key: Optional[str] = None):
    """Initializes and returns a Mistral client."""
    
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        logger.error("Mistral API key not found. Set MISTRAL_API_KEY environment variable or provide it.")
        return None
      
    try:
        
        client = Mistral(api_key=key)
        client.models.list()
        logger.info("Mistral client initialized and connection tested successfully.")
        
        return client
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during Mistral client initialization: {e}")
        
    return None


def initialize_ollama_client(base_url: Optional[str] = None, client_timeout: Optional[int] = 30) -> Optional[ollama.Client]:
    """
    Initializes and returns an Ollama client object.
    Tests connection by listing local models.
    Args:
        base_url: The base URL for the Ollama server.
        client_timeout: Timeout in seconds for the client connection.
    """
    
    effective_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
  
    try:
        
        client = ollama.Client(host=effective_base_url, timeout=client_timeout)
        client.list()
        logger.info(f"Ollama client connection to {effective_base_url} (timeout: {client_timeout}s) tested successfully.")
        return client
    
    except ollama.ResponseError as e:
        logger.error(f"Ollama API responded with an error during client initialization: {e.status_code} - {e.error}")
        logger.error(f"Ensure Ollama is running at {effective_base_url} and accessible.")
    
    except ollama.RequestError as e:
        logger.error(f"Failed to connect to Ollama at {effective_base_url}: {e}")
        logger.error("Ensure Ollama server is running and the OLLAMA_BASE_URL (or argument) is correct.")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ollama client initialization: {e}", exc_info=True)
    
    return None


def get_llm_client(provider_name: str, config_vars: Dict[str, Any]) -> Optional[Any]:
    """
    Factory function to get an LLM client based on the provider name.
    Args:
        provider_name: Name of the LLM provider (e.g., "groq", "mistral", "ollama").
        config_vars: A dictionary containing configuration variables (e.g., API keys, URLs).
                     Typically obtained from `vars(config_module)`.
    """
    
    logger.info(f"Attempting to initialize LLM client for provider: {provider_name}")
    
    provider_lower = provider_name.lower()

    if provider_lower == "groq":
        return initialize_groq_client(api_key=config_vars.get("GROQ_API_KEY"))
    
    elif provider_lower == "mistral":
        return initialize_mistral_client(api_key=config_vars.get("MISTRAL_API_KEY"))
    
    elif provider_lower == "ollama":
        
        return initialize_ollama_client(
            base_url=config_vars.get("OLLAMA_BASE_URL"),
            client_timeout=config_vars.get("OLLAMA_CLIENT_TIMEOUT", 120)
        )
        
    else:
        logger.error(f"Unsupported LLM provider: {provider_name}")
        return None
