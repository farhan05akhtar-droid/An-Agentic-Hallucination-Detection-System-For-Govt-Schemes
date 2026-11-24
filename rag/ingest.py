import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths relative to the root directory
PDF_DIR = "data/raw_pdfs"
DB_DIR = "data/vector_db"

def load_documents(directory_path):
    """Loads all PDFs from the specified directory."""
    print(f"Loading PDFs from {directory_path}...")
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    return documents

def chunk_text(documents):
    """Splits documents into smaller chunks for the RAG pipeline."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Number of characters per chunk
        chunk_overlap=200,     # Overlap to maintain context between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory):
    """Embeds the chunks and stores them in a local Chroma database."""
    print("Initializing embedding model...")
    # Using a free, local embedding model from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Building Chroma vector store at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector database built and saved successfully!")
    return vectorstore

if __name__ == "__main__":
    # Ensure the PDF directory exists
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # Check if there are actually files to process
    if not os.listdir(PDF_DIR):
        print(f"⚠️ No PDFs found in {PDF_DIR}. Please add some government scheme PDFs and run again.")
    else:
        docs = load_documents(PDF_DIR)
        doc_chunks = chunk_text(docs)
        create_vector_store(doc_chunks, DB_DIR)