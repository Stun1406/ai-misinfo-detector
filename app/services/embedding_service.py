"""
Embedding generation service using sentence transformers
"""
import torch
from typing import List, Union, Dict
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.core.config import settings
from app.services.text_processor_simple import text_processor


class EmbeddingService:
    """Service for generating semantic embeddings"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty")
        
        # Preprocess text for embedding
        processed_text = text_processor.preprocess_for_embedding(text)
        
        try:
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_tensor=False)
            
            # Convert to list for JSON serialization
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model:
            raise Exception("Embedding model not loaded")
        
        if not texts:
            return []
        
        # Preprocess all texts
        processed_texts = [text_processor.preprocess_for_embedding(text) for text in texts]
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(processed_texts) if text.strip()]
        
        if not valid_texts:
            return [[] for _ in texts]
        
        try:
            # Generate embeddings in batch
            valid_indices, valid_processed = zip(*valid_texts)
            embeddings = self.model.encode(list(valid_processed), convert_to_tensor=False, batch_size=settings.batch_size)
            
            # Convert to list format
            if hasattr(embeddings[0], 'tolist'):
                embeddings = [emb.tolist() for emb in embeddings]
            
            # Map back to original order
            result = [[] for _ in texts]
            for i, embedding in zip(valid_indices, embeddings):
                result[i] = embedding
            
            logger.info(f"Generated {len(valid_texts)} embeddings in batch")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_most_similar_texts(self, query_text: str, candidate_texts: List[str], top_k: int = 5) -> List[Dict]:
        """Find most similar texts to a query using embeddings"""
        if not candidate_texts:
            return []
        
        try:
            # Generate embeddings
            query_embedding = self.generate_embedding(query_text)
            candidate_embeddings = self.generate_embeddings_batch(candidate_texts)
            
            # Calculate similarities
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                if candidate_emb:  # Skip empty embeddings
                    similarity = self.calculate_similarity(query_embedding, candidate_emb)
                    similarities.append({
                        "index": i,
                        "text": candidate_texts[i],
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar texts: {e}")
            return []


# Global embedding service instance
embedding_service = EmbeddingService()
