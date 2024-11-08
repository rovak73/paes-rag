"""
Configuration module for environment variables and settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_MODEL = "gemini-1.5-pro-002"

# Vector store settings
CHROMA_PERSIST_DIRECTORY = "data/chroma"

# Model settings
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
MODEL_NAME = "gpt-3.5-turbo"  # Used when USE_LOCAL_LLM is False
EMBEDDING_MODEL = "text-embedding-ada-002"

# Llama model settings
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-2-7b-chat.gguf")
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))  # Context window
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "4"))  # CPU threads to use
LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.0"))

# Chunk settings for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
