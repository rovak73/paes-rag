"""
PDF Processor module for extracting text from PAES exam PDFs.
"""
from typing import List, Dict
from pypdf import PdfReader
import os

class PDFProcessor:
    def __init__(self, pdf_dir: str = "data"):
        """Initialize PDF processor with directory containing PDFs."""
        self.pdf_dir = pdf_dir

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def process_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDFs in the pdf_dir and extract their text.
        
        Returns:
            List of dictionaries containing PDF metadata and extracted text
        """
        documents = []
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, filename)
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    documents.append({
                        "source": filename,
                        "text": text
                    })
        return documents

    def save_processed_text(self, output_dir: str = "data/processed") -> None:
        """
        Save extracted text from PDFs to text files.
        
        Args:
            output_dir: Directory to save processed text files
        """
        os.makedirs(output_dir, exist_ok=True)
        documents = self.process_pdfs()
        
        for doc in documents:
            output_path = os.path.join(output_dir, f"{doc['source']}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc["text"])
