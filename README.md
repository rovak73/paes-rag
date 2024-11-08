# PAES RAG Question Answering System

## Project Description
Automated Question Answering System for PAES (University Admission Test) using Retrieval Augmented Generation (RAG). The system processes exam PDFs and provides detailed answers based on the source material.

PAES stands for the Prueba de Acceso a la Educación Superior, which is Chile’s standardized test for university admissions. Introduced in 2022, PAES replaced the previous test known as the Prueba de Selección Universitaria (PSU).

### The PAES assesses competencies across several core areas:

- Reading Comprehension
- Mathematics (with PAES Math 1 for all applicants, and an additional PAES Math 2 for more demanding math-focused programs)
- Sciences (covering biology, chemistry, and physics)
- History and Social Sciences

PAES emphasizes practical reasoning and problem-solving skills rather than rote memorization, aiming to provide a more holistic and fair assessment of students’ readiness for higher education in Chile.

## Key Features
- PDF text extraction from PAES exams
- Vector storage using ChromaDB for efficient retrieval
- Multiple LLM support:
  - Google's Gemini Pro (default)
  - OpenAI GPT models
  - Local Llama models
- Command-line interface for queries
- Source citation in responses
- Context-aware answers

## Requirements
- Python 3.12+
- Google Gemini API Key (default) or OpenAI API Key
- Dependencies listed in `requirements.txt`

## Initial Setup

### 1. Clone Repository
```bash
git clone https://github.com/rovak73/paes-rag.git
cd paes-rag
```

### 2. Set Up Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file with or use .env-example provided:
```
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
USE_GEMINI=true  # Set to false to use OpenAI
USE_LOCAL_LLM=false  # Set to true to use Llama
```

## Usage

### Run the Application
```bash
python -m src.main
```

The system will:
1. Load and process PDF documents
2. Create vector embeddings
3. Start an interactive prompt
4. Answer questions with relevant source citations

### Run Tests
```bash
pytest tests/
```

## Project Structure
```
paes-rag/
│
├── src/
│   ├── main.py           # Core application logic
│   ├── pdf_processor.py  # PDF processing utilities
│   └── config.py         # Environment and model settings
├── models/               # Inference models (Llama3, etc..)
├── tests/                # Test suite
├── data/                 # PDF documents
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
└── README.md
```

## Model Configuration
The system supports three LLM options:

1. Google Gemini (Default)
   - High-quality responses
   - Cost-effective
   - Set `USE_GEMINI=true`

2. OpenAI GPT
   - Alternative option
   - Set `USE_GEMINI=false`

3. Local Llama
   - Offline operation
   - Set `USE_LOCAL_LLM=true`
   - Configure in `config.py`

## Contributing
- Report issues via GitHub
- Pull requests welcome
- Follow existing code style

## License
MIT License

## Contact
rovak73@gmail.com
