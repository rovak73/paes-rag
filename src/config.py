"""
Configuration module for environment variables and settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store settings
CHROMA_PERSIST_DIRECTORY = "data/chroma"

# Model settings
MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Chunk settings for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
