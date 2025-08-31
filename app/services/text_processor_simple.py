"""
Simplified text preprocessing service without NLTK dependency
"""
import re
import string
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from collections import Counter
from loguru import logger


class TextProcessor:
    """Service for text preprocessing and cleaning"""
    
    def __init__(self):
        # Basic English stopwords
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once'
        }
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        if not text:
            return ""
        
        # Parse HTML and extract text
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML if present
        text = self.clean_html(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra punctuation but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using simple regex"""
        if not text:
            return []
        
        # Simple sentence splitting on periods, exclamations, questions
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Simple tokenization"""
        if not text:
            return []
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', text.lower())
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text for keyword search"""
        if not text:
            return []
        
        # Clean and tokenize
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text, remove_stopwords=True)
        
        # Simple frequency-based keyword extraction
        token_freq = Counter(tokens)
        
        # Get most common tokens as keywords
        keywords = [token for token, count in token_freq.most_common(max_keywords)]
        
        return keywords
    
    def preprocess_for_embedding(self, text: str) -> str:
        """Preprocess text specifically for embedding generation"""
        if not text:
            return ""
        
        # Clean text but keep sentence structure for better embeddings
        text = self.clean_html(text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Trim to max length for transformer models
        if len(text) > 1000:  # Conservative limit for processing
            sentences = self.extract_sentences(text)
            # Keep first few sentences if text is too long
            text = ' '.join(sentences[:3])
        
        return text.strip()
    
    def extract_claim_features(self, text: str) -> Dict[str, Any]:
        """Extract various features from a claim for analysis"""
        features = {
            "original_text": text,
            "clean_text": self.clean_text(text),
            "sentences": self.extract_sentences(text),
            "keywords": self.extract_keywords(text),
            "word_count": len(self.tokenize(text, remove_stopwords=False)),
            "sentence_count": len(self.extract_sentences(text)),
            "has_urls": bool(re.search(r'http[s]?://', text)),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_caps": bool(re.search(r'[A-Z]{3,}', text)),
            "preprocessed_for_embedding": self.preprocess_for_embedding(text)
        }
        
        return features


# Global text processor instance
text_processor = TextProcessor()
