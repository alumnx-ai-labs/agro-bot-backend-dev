# services/chatbot_service.py
import os
import time
import logging
from typing import Dict, Any, List
import asyncio

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ChatbotService:
    """
    Service class for handling chatbot operations with expert-filtered retrieval.
    """
    
    def __init__(self):
        """Initialize the chatbot service with required components."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "expert-chatbot")
        
        if not self.gemini_api_key or not self.pinecone_api_key:
            raise ValueError("Missing required API keys: GEMINI_API_KEY and PINECONE_API_KEY")
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize Gemini
            genai.configure(api_key=self.gemini_api_key)
            self.llm = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Google embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                logger.info(f"Index {self.index_name} created and ready.")
            
            self.index = self.pc.Index(self.index_name)
            
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            
            logger.info("All components initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def get_response(self, query: str, expert: str, top_k: int = 5) -> str:
        """
        Get chatbot response for a query filtered by expert.
        
        Args:
            query: User's question
            expert: Selected expert to filter by
            top_k: Number of similar documents to retrieve
            
        Returns:
            Generated response string
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search with expert filter
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"expert": expert}
            )
            
            # Extract context from results
            context_chunks = []
            if search_results.matches:
                for match in search_results.matches:
                    if match.score > 0.7:  # Similarity threshold
                        context_chunks.append(match.metadata.get('text', ''))
            
            # Build context
            if context_chunks:
                context = "\n\n---\n\n".join(context_chunks)
            else:
                context = "No relevant information found in the knowledge base for this expert."
            
            # Create prompt
            prompt = f"""
You are an AI assistant helping farmers by providing expert advice. You are currently representing {expert}.

Based on the following context information, please answer the user's question. If the information is not available in the context, politely say you don't have that specific information.

Context:
{context}

Question: {query}

Please provide a helpful, accurate response based on the available information:
"""
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.generate_content, prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def upload_pdf(self, file_path: str, pdf_name: str, expert: str) -> bool:
        """
        Upload and process PDF document for an expert.
        
        Args:
            file_path: Path to the PDF file
            pdf_name: Name of the PDF file
            expert: Expert this document belongs to
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Processing PDF: {pdf_name} for expert: {expert}")
            
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            
            if not documents:
                logger.warning(f"No content extracted from PDF: {pdf_name}")
                return False
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                logger.warning(f"No chunks created from PDF: {pdf_name}")
                return False
            
            logger.info(f"Split PDF into {len(chunks)} chunks")
            
            # Process chunks in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare vectors for batch
                vectors = []
                for j, chunk in enumerate(batch):
                    # Generate embedding
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    
                    # Create vector
                    vector = {
                        "id": f"{pdf_name}_{i+j}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk.page_content,
                            "pdf_name": pdf_name,
                            "expert": expert,
                            "page": chunk.metadata.get('page', 0)
                        }
                    }
                    vectors.append(vector)
                
                # Upsert batch
                await asyncio.get_event_loop().run_in_executor(
                    None, self.index.upsert, vectors
                )
                
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            logger.info(f"Successfully uploaded {len(chunks)} chunks for {expert}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading PDF {pdf_name}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of all service components.
        
        Returns:
            Health status dictionary
        """
        status = {
            "gemini_api": False,
            "pinecone_connection": False,
            "embeddings": False,
            "overall_health": False
        }
        
        try:
            # Test Gemini
            test_response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.generate_content, "Say 'OK'"
            )
            status["gemini_api"] = "OK" in test_response.text
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
        
        try:
            # Test Pinecone
            index_stats = self.index.describe_index_stats()
            status["pinecone_connection"] = True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
        
        try:
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            status["embeddings"] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embeddings health check failed: {e}")
        
        status["overall_health"] = all([
            status["gemini_api"],
            status["pinecone_connection"],
            status["embeddings"]
        ])
        
        return status