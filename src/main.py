
"""
Main module for the PAES RAG application.
"""
from typing import List, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from .pdf_processor import PDFProcessor
import src.config as config

class PAESQuestionAnswerer:
    def __init__(self):
        """Initialize the PAES question answerer."""
        self.pdf_processor = PDFProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Choose embeddings based on model type
        if config.USE_GEMINI:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GEMINI_API_KEY,
                task_type="retrieval_document",  # Ensures 384 dimensions
                dimensions=384
            )
        elif config.USE_LOCAL_LLM:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY
            )
            
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load and process PDF documents.
        
        Returns:
            List of processed documents
        """
        return self.pdf_processor.process_pdfs()

    def create_vectorstore(self, documents: List[Dict[str, str]]) -> None:
        """
        Create vector store from processed documents.
        
        Args:
            documents: List of processed documents
        """
        # Extract text from documents
        texts = [doc["text"] for doc in documents]
        
        # Split texts into chunks
        chunks = self.text_splitter.create_documents(texts)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        self.vectorstore.persist()

    def setup_qa_chain(self) -> None:
        """Set up the question-answering chain."""
        if not self.vectorstore:
            raise ValueError("Vector store must be created first")

        if config.USE_GEMINI:
            # Set up Gemini model
            llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0
            )
        elif config.USE_LOCAL_LLM:
            # Set up Llama model
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            llm = LlamaCpp(
                model_path=config.LLAMA_MODEL_PATH,
                n_ctx=config.LLAMA_N_CTX,
                n_threads=config.LLAMA_N_THREADS,
                temperature=config.LLAMA_TEMPERATURE,
                callback_manager=callback_manager,
                verbose=True
            )
        else:
            # Set up OpenAI model
            llm = ChatOpenAI(
                model_name=config.MODEL_NAME,
                temperature=0,
                openai_api_key=config.OPENAI_API_KEY
            )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def answer_question(self, question: str) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain must be set up first")

        # Create prompt that encourages detailed explanations
        prompt = f"""
        Eres un asistente experto en análisis de textos. Tu tarea es responder preguntas sobre el contenido de los documentos proporcionados.

        Instrucciones específicas:
        1. Si encuentras la información en los documentos:
           - Proporciona la respuesta basada en el contenido exacto
           - Cita el texto relevante
           - Explica el contexto si es necesario
        2. Si la información aparece en un texto literario o narrativo:
           - Describe lo que sucede en la escena
           - Cita las partes relevantes del texto
           - Mantén la interpretación fiel al texto
        
        Pregunta: {question}
        
        Responde de manera clara y directa, citando el texto cuando sea posible.
        """

        result = self.qa_chain.invoke({"query": prompt})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

def main():
    """Main function to run the PAES question answerer."""
    # Initialize question answerer
    qa = PAESQuestionAnswerer()
    
    # Load and process documents
    print("Loading and processing documents...")
    documents = qa.load_documents()
    
    # Create vector store
    print("Creating vector store...")
    qa.create_vectorstore(documents)
    
    # Set up QA chain
    print("Setting up QA chain...")
    qa.setup_qa_chain()
    
    # Interactive question answering loop
    print("\nPAES Question Answerer ready!")
    model_type = "Gemini" if config.USE_GEMINI else ("Llama" if config.USE_LOCAL_LLM else "OpenAI")
    print(f"Using {model_type} model")
    print("Type 'exit' to quit")
    
    while True:
        question = input("\nIngrese su pregunta: ")
        if question.lower() == 'exit':
            break
            
        try:
            result = qa.answer_question(question)
            print("\n=== Respuesta ===")
            print(result["answer"])
            print("\n=== Fragmentos Relevantes ===")
            seen_sources = set()
            for i, source in enumerate(result["sources"], 1):
                # Clean up the source text
                clean_source = source.replace('\n', ' ').strip()
                # Only show unique sources
                if clean_source not in seen_sources:
                    seen_sources.add(clean_source)
                    print(f"\nFragmento {len(seen_sources)}:")
                    print(f"{clean_source}")
            print("\n" + "="*50)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
