"""
Test suite for main RAG application module.
"""
import os
import pytest
from src.main import PAESQuestionAnswerer
import src.config as config

class TestPAESQuestionAnswerer:
    @pytest.fixture
    def qa_system(self):
        """Create a PAESQuestionAnswerer instance for testing."""
        return PAESQuestionAnswerer()

    def test_load_documents(self, qa_system):
        """Test loading documents."""
        documents = qa_system.load_documents()
        
        # Check documents are loaded
        assert len(documents) > 0, "No documents loaded"
        
        # Check document structure
        for doc in documents:
            assert "source" in doc, "Document missing source"
            assert "text" in doc, "Document missing text"
            assert doc["text"], "Document text is empty"

    def test_create_vectorstore(self, qa_system):
        """Test creating vector store."""
        # Load documents
        documents = qa_system.load_documents()
        
        # Create vector store
        qa_system.create_vectorstore(documents)
        
        # Check vector store is created
        assert qa_system.vectorstore is not None, "Vector store not created"

    def test_setup_qa_chain(self, qa_system):
        """Test setting up QA chain."""
        # Load documents and create vector store
        documents = qa_system.load_documents()
        qa_system.create_vectorstore(documents)
        
        # Setup QA chain
        qa_system.setup_qa_chain()
        
        # Check QA chain is created
        assert qa_system.qa_chain is not None, "QA chain not created"

    def test_answer_question(self, qa_system):
        """Test answering a question."""
        # Load documents, create vector store, and setup QA chain
        documents = qa_system.load_documents()
        qa_system.create_vectorstore(documents)
        qa_system.setup_qa_chain()
        
        # Test question
        test_question = "¿Cuál es el propósito de la prueba PAES?"
        
        # Answer question
        result = qa_system.answer_question(test_question)
        
        # Check result structure
        assert "question" in result, "Result missing question"
        assert "answer" in result, "Result missing answer"
        assert "sources" in result, "Result missing sources"
        
        # Check answer is not empty
        assert result["answer"], "Answer is empty"
        assert len(result["sources"]) > 0, "No source documents found"

    def test_config_settings(self):
        """Test configuration settings."""
        # Check OpenAI API key is set
        assert config.OPENAI_API_KEY is not None, "OpenAI API key not set"
        
        # Check model settings
        assert config.MODEL_NAME, "Model name not specified"
        assert config.EMBEDDING_MODEL, "Embedding model not specified"
        
        # Check chunk settings
        assert config.CHUNK_SIZE > 0, "Invalid chunk size"
        assert config.CHUNK_OVERLAP >= 0, "Invalid chunk overlap"
