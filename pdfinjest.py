import os
from typing import List, Dict, Any
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


class PDFIngestor:
    """
    A class to ingest PDF files, create embeddings using sentence transformers,
    and store them in FAISS vector database.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the PDF Ingestor with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       Default is 'all-MiniLM-L6-v2' (fast and efficient)
        """
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store metadata (text chunks and source info)
        self.metadata = []
        
    def extract_text_from_pdf(self, pdf_path: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Extract text from PDF and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Number of characters per chunk
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"Processing PDF: {pdf_path}")
                print(f"Total pages: {num_pages}")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Split text into chunks
                    for i in range(0, len(text), chunk_size):
                        chunk_text = text[i:i + chunk_size].strip()
                        if chunk_text:  # Only add non-empty chunks
                            chunks.append({
                                'text': chunk_text,
                                'source': pdf_path,
                                'page': page_num + 1,
                                'chunk_index': i // chunk_size
                            })
                
                print(f"Extracted {len(chunks)} chunks from {num_pages} pages")
                
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def add_to_index(self, chunks: List[Dict[str, Any]]):
        """
        Add text chunks to the FAISS index.
        
        Args:
            chunks: List of chunk dictionaries containing text and metadata
        """
        if not chunks:
            print("No chunks to add")
            return
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(chunks)
        
        print(f"Added {len(chunks)} chunks to the index")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500):
        """
        Ingest a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Number of characters per chunk
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return
        
        # Extract text chunks
        chunks = self.extract_text_from_pdf(pdf_path, chunk_size)
        
        # Add to index
        self.add_to_index(chunks)
    
    def ingest_pdf_directory(self, directory_path: str, chunk_size: int = 500):
        """
        Ingest all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            chunk_size: Number of characters per chunk
        """
        if not os.path.exists(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return
        
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            self.ingest_pdf(pdf_path, chunk_size)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar text chunks using a query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing matching chunks and their distances
        """
        if self.index.ntotal == 0:
            print("Index is empty. Please ingest some PDFs first.")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distances[0][i])
                result['similarity_score'] = 1 / (1 + float(distances[0][i]))
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str = 'faiss_index.bin', metadata_path: str = 'metadata.pkl'):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str = 'faiss_index.bin', metadata_path: str = 'metadata.pkl'):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("Index or metadata file not found")
            return
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.metadata)} metadata entries")


# Example usage
if __name__ == "__main__":
    import json
    from datetime import datetime
    
    # Initialize the PDF Ingestor
    ingestor = PDFIngestor(model_name='all-MiniLM-L6-v2')
    
    # Ingest the CineBook PDF
    pdf_path = '/Users/akshayhegde/Documents/communication/Project Requirements Document_ CineBook - Next-Gen Movie Ticket Booking Platform-1-10.pdf'
    
    # Check if we need to re-ingest
    config_path = 'pdf_config.json'
    should_ingest = True
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Check if PDF is the same
            if config.get('pdf_path') == pdf_path:
                should_ingest = False
                print("\n✓ Same PDF detected. Using existing database.")
                print(f"  PDF: {os.path.basename(pdf_path)}")
                print("  To force refresh, delete 'pdf_config.json' or change PDF path.\n")
    
    if should_ingest:
        print("\n🔄 Processing PDF and creating embeddings...")
        ingestor.ingest_pdf(pdf_path, chunk_size=500)
        
        # Save the index
        ingestor.save_index('faiss_index.bin', 'metadata.pkl')
        
        # Save configuration
        config = {
            'pdf_path': pdf_path,
            'last_updated': datetime.now().isoformat()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_path}\n")
    else:
        print("✅ Using existing PDF database.")
