"""
Test suite for PDF Processor module.
"""
import os
import pytest
from src.pdf_processor import PDFProcessor

class TestPDFProcessor:
    @pytest.fixture
    def pdf_processor(self):
        """Create a PDFProcessor instance for testing."""
        return PDFProcessor(pdf_dir="data")

    def test_extract_text_from_pdf(self, pdf_processor):
        """Test text extraction from a PDF file."""
        pdf_path = "data/2024-23-11-28-paes-regular-oficial-competencia-lectora-p2024.pdf"
        
        # Ensure the PDF exists
        assert os.path.exists(pdf_path), f"Test PDF not found at {pdf_path}"
        
        # Extract text
        text = pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Check basic text extraction
        assert text is not None, "No text extracted from PDF"
        assert len(text) > 0, "Extracted text is empty"
        
        # Check for some expected content
        assert "PAES" in text, "Expected PAES content not found"
        assert "Competencia Lectora" in text, "Expected section title not found"

    def test_process_pdfs(self, pdf_processor):
        """Test processing multiple PDFs."""
        documents = pdf_processor.process_pdfs()
        
        # Check documents are processed
        assert len(documents) > 0, "No documents processed"
        
        # Check document structure
        for doc in documents:
            assert "source" in doc, "Document missing source"
            assert "text" in doc, "Document missing text"
            assert doc["text"], "Document text is empty"

    def test_save_processed_text(self, pdf_processor, tmp_path):
        """Test saving processed text to files."""
        # Override output directory to a temporary path
        output_dir = tmp_path / "processed"
        pdf_processor.save_processed_text(str(output_dir))
        
        # Check files are created
        processed_files = list(output_dir.glob("*.txt"))
        assert len(processed_files) > 0, "No processed text files created"
        
        # Check file content
        for file in processed_files:
            content = file.read_text(encoding="utf-8")
            assert content, f"Processed text file {file.name} is empty"
