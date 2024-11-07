import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Chunk configuration for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'claude-2.1')

# Chroma Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = 'data/chroma'
