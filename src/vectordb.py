import os
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import uuid
from PyPDF2 import PdfReader



class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
           "CHROMA_COLLECTION_NAME", "rag_documents")
      

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Load embedding model - ignored as preferring to use chroma's default embedding model
        

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )
          
        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Your implementation here
        text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
        chunks = text_splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        print(f"Processing {len(documents)} documents...")
        for doc in documents:
            r_path = f"data/{doc['name']}"
            if ".txt" in doc['name']:
                chunks = self.chunk_text(doc['content'])
            
            stats = os.stat(r_path)
            # Some platforms (Windows, older filesystems) may not provide st_birthtime.
            # Use st_ctime as a fallback for created/creation timestamp when necessary.
            created_ts = getattr(stats, "st_birthtime", stats.st_ctime)
            metadata = {
                        "file_name": os.path.basename(r_path),
                        "file_path": r_path,
                        "created_at": datetime.fromtimestamp(created_ts).isoformat(),
                        "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        "size_bytes": stats.st_size
                                    }
            
                metadatas = [metadata] * len(chunks)
                self.collection.add(
                    ids=[str(uuid.uuid4()) for _ in chunks],
                    documents=chunks,
                    metadatas=metadatas,
                )
                print(f"Successfully ingested {doc['name']}")
          
            else:
                chunks = []
                reader = PdfReader(f"./data/{doc['name']}")
                print(f"Pdf file found - {doc['name']}")
                print("Reading pages")
                for page in doc['content']:
                    # --- Extract metadata ---
                    raw_metadata = reader.metadata or {}
                    metadata = {key.lstrip('/'): value for key, value in raw_metadata.items()}
                    metadata['page_number'] = page['page_number']
                    metadata['source'] = doc['name']

                    text = page['text']
                    chunks = self.chunk_text(text)

                    metadatas = [metadata] * len(chunks)

                    self.collection.add(
                        ids=[str(uuid.uuid4()) for _ in chunks],
                        documents=chunks,
                        metadatas=metadatas,
                    )

                print(f"Successfully Ingested {len(reader.pages)} pages from {doc['name']}.")
        print("Documents added to vector database")
                            
            

          

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        return {
            "documents": [results["documents"]],
            "metadatas": [results["metadatas"]],
            "distances": [results["distances"]],
            "ids": [results["ids"]],
        }
