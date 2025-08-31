"""
ML classification service for misinformation detection
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from app.core.config import settings
from app.services.text_processor_simple import text_processor


class ClassificationService:
    """Service for classifying claims as true/false/misleading/unverified"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
        
        # Label mapping
        self.label_map = {
            0: "True",
            1: "False", 
            2: "Misleading",
            3: "Unverified"
        }
    
    def _load_model(self):
        """Load the classification model"""
        try:
            logger.info(f"Loading classification model: {settings.classification_model}")
            
            # For MVP, we'll use a sentiment analysis model as a proxy
            # In production, this would be a fine-tuned model for misinformation detection
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Classification model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            self.classifier = None
    
    def classify_claim(self, claim_text: str, evidence_snippets: List[str] = None) -> Dict[str, any]:
        """
        Classify a claim with supporting evidence
        Returns classification, confidence score, and reasoning
        """
        try:
            # Preprocess the claim
            processed_claim = text_processor.preprocess_for_embedding(claim_text)
            
            if not processed_claim.strip():
                return {
                    "classification": "Unverified",
                    "reliability_score": 0.0,
                    "confidence": 0.0,
                    "reasoning": "Empty or invalid claim text"
                }
            
            # For MVP: Use rule-based heuristics combined with sentiment analysis
            classification_result = self._analyze_claim_heuristics(processed_claim, evidence_snippets)
            
            # Add sentiment analysis as additional signal if model is available
            if self.classifier:
                try:
                    sentiment_result = self.classifier(processed_claim)[0]
                    sentiment_score = sentiment_result['score']
                    
                    # Combine heuristic and sentiment analysis
                    final_result = self._combine_analysis_results(classification_result, sentiment_result)
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")
                    final_result = classification_result
            else:
                final_result = classification_result
            
            logger.info(f"Classified claim as: {final_result['classification']} (score: {final_result['reliability_score']:.2f})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Failed to classify claim: {e}")
            return {
                "classification": "Unverified",
                "reliability_score": 0.0,
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }
    
    def _analyze_claim_heuristics(self, claim: str, evidence_snippets: List[str] = None) -> Dict:
        """
        Analyze claim using rule-based heuristics for MVP
        In production, this would be replaced with fine-tuned models
        """
        score = 50.0  # Start with neutral score
        reasoning_parts = []
        
        # Text quality indicators
        features = text_processor.extract_claim_features(claim)
        
        # Check for suspicious patterns
        if features['has_caps']:
            score -= 10
            reasoning_parts.append("Contains excessive capitalization")
        
        if len(features['sentences']) == 1 and features['word_count'] < 5:
            score -= 15
            reasoning_parts.append("Very short claim with limited context")
        
        # Check for positive scientific language (increase score)
        scientific_words = ['study', 'research', 'clinical trial', 'scientific', 'evidence', 'data', 'according to', 'published', 'peer-reviewed', 'journal']
        if any(word in claim.lower() for word in scientific_words):
            score += 15
            reasoning_parts.append("Contains scientific language")
        
        # Check for authoritative sources (increase score)
        authority_words = ['cdc', 'who', 'fda', 'nasa', 'government', 'official', 'university', 'medical center', 'hospital', 'institute']
        if any(word in claim.lower() for word in authority_words):
            score += 10
            reasoning_parts.append("References authoritative sources")
        
        # Check for uncertainty language (slight decrease)
        uncertainty_words = ['maybe', 'possibly', 'might', 'could', 'allegedly', 'reportedly', 'supposedly', 'rumor', 'unconfirmed']
        if any(word in claim.lower() for word in uncertainty_words):
            score -= 3
            reasoning_parts.append("Contains uncertainty language")
        
        # Check for absolute statements (moderate decrease)
        absolute_words = ['always', 'never', 'all', 'every', 'none', 'completely', 'totally', 'absolutely', 'definitely', 'guaranteed']
        if any(word in claim.lower() for word in absolute_words):
            score -= 5
            reasoning_parts.append("Contains absolute statements")
        
        # Check for sensational language (major decrease)
        sensational_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 'secret', 'exposed', 'conspiracy', 'cover-up', 'scandal', 'breaking']
        if any(word in claim.lower() for word in sensational_words):
            score -= 15
            reasoning_parts.append("Contains sensational language")
        
        # Check for emotional manipulation (decrease score)
        emotional_words = ['fear', 'panic', 'terrifying', 'horrifying', 'outrageous', 'disgusting', 'shameful']
        if any(word in claim.lower() for word in emotional_words):
            score -= 8
            reasoning_parts.append("Contains emotional manipulation")
        
        # Check for specific factual claims (increase score)
        factual_indicators = ['percent', '%', 'million', 'billion', 'study shows', 'research indicates', 'data shows']
        if any(indicator in claim.lower() for indicator in factual_indicators):
            score += 8
            reasoning_parts.append("Contains specific factual claims")
        
        # Evidence analysis
        if evidence_snippets:
            evidence_score = self._analyze_evidence_quality(evidence_snippets)
            score += evidence_score
            reasoning_parts.append(f"Evidence quality analysis: +{evidence_score:.1f}")
        else:
            score -= 20
            reasoning_parts.append("No supporting evidence found")
        
        # Determine classification based on score
        if score >= 70:
            classification = "True"
        elif score >= 40:
            classification = "Unverified"
        elif score >= 20:
            classification = "Misleading"
        else:
            classification = "False"
        
        return {
            "classification": classification,
            "score": max(0, min(100, score)),
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Standard analysis applied"
        }
    
    def _analyze_evidence_quality(self, evidence_snippets: List[str]) -> float:
        """Analyze the quality of supporting evidence"""
        if not evidence_snippets:
            return -20.0
        
        quality_score = 0.0
        
        for snippet in evidence_snippets:
            # Check evidence length (longer is generally better)
            if len(snippet) > 100:
                quality_score += 5
            
            # Check for source indicators
            source_indicators = ['study', 'research', 'according to', 'expert', 'official', 'government', 'university', 'journal', 'published']
            if any(indicator in snippet.lower() for indicator in source_indicators):
                quality_score += 8
            
            # Check for date indicators (recent is better)
            if any(year in snippet for year in ['2023', '2024', '2025']):
                quality_score += 3
            
            # Check for specific data or statistics
            if any(char in snippet for char in ['%', '$', 'million', 'billion', 'thousand']):
                quality_score += 4
        
        return min(quality_score, 25.0)  # Cap the evidence bonus
    
    def _combine_analysis_results(self, heuristic_result: Dict, sentiment_result: Dict) -> Dict:
        """Combine heuristic analysis with sentiment analysis"""
        base_score = heuristic_result['score']
        classification = heuristic_result['classification']
        
        # Sentiment analysis adjustment
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']
        
        # Negative sentiment might indicate misinformation
        if sentiment_label == 'LABEL_0':  # Negative
            adjustment = -5 * sentiment_score
        elif sentiment_label == 'LABEL_2':  # Positive  
            adjustment = 2 * sentiment_score
        else:  # Neutral
            adjustment = 0
        
        final_score = max(0, min(100, base_score + adjustment))
        
        # Recalculate classification if score changed significantly
        if final_score >= 70 and classification != "True":
            classification = "True"
        elif final_score < 20 and classification != "False":
            classification = "False"
        elif 20 <= final_score < 40 and classification not in ["Misleading", "False"]:
            classification = "Misleading"
        elif 40 <= final_score < 70 and classification not in ["Unverified", "True"]:
            classification = "Unverified"
        
        reasoning = heuristic_result['reasoning']
        if adjustment != 0:
            reasoning += f"; Sentiment analysis adjustment: {adjustment:+.1f}"
        
        return {
            "classification": classification,
            "reliability_score": final_score,
            "confidence": min(sentiment_score * 100, 95),  # Convert to percentage
            "reasoning": reasoning
        }
    
    def analyze_claim_with_context(self, claim: str, similar_claims: List[Dict]) -> Dict:
        """
        Analyze claim considering similar previously analyzed claims
        """
        base_analysis = self.classify_claim(claim)
        
        if not similar_claims:
            return base_analysis
        
        # Analyze consistency with similar claims
        similar_classifications = [claim.get('classification') for claim in similar_claims]
        similar_scores = [claim.get('reliability_score', 50) for claim in similar_claims]
        
        if similar_classifications:
            # If most similar claims have same classification, increase confidence
            most_common_class = max(set(similar_classifications), key=similar_classifications.count)
            if most_common_class == base_analysis['classification']:
                base_analysis['confidence'] = min(base_analysis['confidence'] + 10, 95)
                base_analysis['reasoning'] += f"; Consistent with {len(similar_claims)} similar claims"
        
        return base_analysis


# Global classification service instance
classification_service = ClassificationService()
