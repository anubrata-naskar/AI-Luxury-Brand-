"""
Sentiment analysis utilities for fashion reviews and descriptions
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Dict, List, Optional
import re
from loguru import logger

class SentimentAnalyzer:
    """Sentiment analysis for fashion content"""
    
    def __init__(self, use_transformer: bool = True):
        """
        Initialize sentiment analyzer
        
        Args:
            use_transformer: Whether to use transformer-based models
        """
        self.use_transformer = use_transformer
        
        # Initialize NLTK VADER
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize VADER: {e}")
            self.vader_analyzer = None
        
        # Initialize transformer model for more accurate sentiment
        if use_transformer:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"Could not initialize transformer model: {e}")
                self.sentiment_pipeline = None
                self.use_transformer = False
        else:
            self.sentiment_pipeline = None
        
        # Fashion-specific sentiment keywords
        self.fashion_positive_words = self._load_fashion_positive_words()
        self.fashion_negative_words = self._load_fashion_negative_words()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return self._empty_sentiment()
        
        results = {}
        
        # VADER sentiment
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['vader'] = vader_scores
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
        
        # Transformer sentiment
        if self.use_transformer and self.sentiment_pipeline:
            try:
                transformer_result = self.sentiment_pipeline(text[:512])  # Limit length
                results['transformer'] = transformer_result[0]
            except Exception as e:
                logger.warning(f"Transformer analysis failed: {e}")
        
        # Fashion-specific sentiment
        fashion_sentiment = self._analyze_fashion_sentiment(text)
        results['fashion_specific'] = fashion_sentiment
        
        # Combine results
        combined_sentiment = self._combine_sentiments(results)
        
        return combined_sentiment
    
    def analyze_reviews(self, reviews: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment of multiple reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Aggregated sentiment scores
        """
        if not reviews:
            return self._empty_sentiment()
        
        all_sentiments = []
        
        for review in reviews:
            sentiment = self.analyze_text(review)
            all_sentiments.append(sentiment)
        
        # Aggregate results
        return self._aggregate_sentiments(all_sentiments)
    
    def extract_sentiment_aspects(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Extract sentiment for different aspects of fashion products
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores for different aspects
        """
        aspects = {
            'quality': ['quality', 'material', 'fabric', 'construction', 'durability'],
            'style': ['style', 'design', 'look', 'appearance', 'fashion'],
            'fit': ['fit', 'size', 'sizing', 'comfortable', 'tight', 'loose'],
            'value': ['price', 'value', 'worth', 'expensive', 'cheap', 'affordable'],
            'comfort': ['comfort', 'comfortable', 'soft', 'breathable', 'cozy']
        }
        
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            # Extract sentences containing aspect keywords
            aspect_sentences = self._extract_aspect_sentences(text, keywords)
            
            if aspect_sentences:
                # Analyze sentiment of aspect-specific sentences
                aspect_text = ' '.join(aspect_sentences)
                sentiment = self.analyze_text(aspect_text)
                aspect_sentiments[aspect] = sentiment
            else:
                aspect_sentiments[aspect] = self._empty_sentiment()
        
        return aspect_sentiments
    
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text (joy, anger, fear, etc.)
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotion scores
        """
        # Simple emotion detection based on keywords
        emotion_keywords = {
            'joy': ['love', 'amazing', 'beautiful', 'gorgeous', 'perfect', 'excellent'],
            'anger': ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disappointed'],
            'surprise': ['surprising', 'unexpected', 'wow', 'incredible', 'unbelievable'],
            'trust': ['recommend', 'reliable', 'trustworthy', 'quality', 'satisfied'],
            'anticipation': ['excited', 'looking forward', 'cant wait', 'eager'],
            'fear': ['worried', 'concerned', 'afraid', 'nervous', 'hesitant'],
            'sadness': ['sad', 'disappointed', 'regret', 'unfortunate'],
            'disgust': ['disgusting', 'gross', 'repulsive', 'offensive']
        }
        
        emotion_scores = {}
        text_lower = text.lower()
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length
            emotion_scores[emotion] = score / max(len(text.split()), 1)
        
        return emotion_scores
    
    def _analyze_fashion_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using fashion-specific keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.fashion_positive_words if word in text_lower)
        negative_count = sum(1 for word in self.fashion_negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        return {
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': max(neutral_ratio, 0.0)
        }
    
    def _combine_sentiments(self, results: Dict) -> Dict[str, float]:
        """Combine sentiment results from different analyzers"""
        # Initialize combined result
        combined = {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'confidence': 0.0
        }
        
        weights = {'vader': 0.3, 'textblob': 0.2, 'transformer': 0.3, 'fashion_specific': 0.2}
        total_weight = 0.0
        
        # VADER contribution
        if 'vader' in results:
            weight = weights['vader']
            vader = results['vader']
            combined['compound'] += vader.get('compound', 0.0) * weight
            combined['positive'] += vader.get('pos', 0.0) * weight
            combined['negative'] += vader.get('neg', 0.0) * weight
            combined['neutral'] += vader.get('neu', 0.0) * weight
            total_weight += weight
        
        # TextBlob contribution
        if 'textblob' in results:
            weight = weights['textblob']
            textblob = results['textblob']
            polarity = textblob.get('polarity', 0.0)
            
            combined['compound'] += polarity * weight
            if polarity > 0:
                combined['positive'] += polarity * weight
            elif polarity < 0:
                combined['negative'] += abs(polarity) * weight
            else:
                combined['neutral'] += weight
            
            total_weight += weight
        
        # Transformer contribution
        if 'transformer' in results:
            weight = weights['transformer']
            transformer = results['transformer']
            label = transformer.get('label', '').lower()
            score = transformer.get('score', 0.0)
            
            if 'positive' in label:
                combined['compound'] += score * weight
                combined['positive'] += score * weight
            elif 'negative' in label:
                combined['compound'] -= score * weight
                combined['negative'] += score * weight
            else:
                combined['neutral'] += score * weight
            
            total_weight += weight
        
        # Fashion-specific contribution
        if 'fashion_specific' in results:
            weight = weights['fashion_specific']
            fashion = results['fashion_specific']
            
            pos_score = fashion.get('positive', 0.0)
            neg_score = fashion.get('negative', 0.0)
            
            combined['compound'] += (pos_score - neg_score) * weight
            combined['positive'] += pos_score * weight
            combined['negative'] += neg_score * weight
            combined['neutral'] += fashion.get('neutral', 0.0) * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in combined:
                if key != 'confidence':
                    combined[key] /= total_weight
        
        # Calculate confidence based on agreement between methods
        combined['confidence'] = self._calculate_confidence(results)
        
        return combined
    
    def _aggregate_sentiments(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Aggregate multiple sentiment analyses"""
        if not sentiments:
            return self._empty_sentiment()
        
        aggregated = {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'confidence': 0.0
        }
        
        for sentiment in sentiments:
            for key in aggregated:
                aggregated[key] += sentiment.get(key, 0.0)
        
        # Average the scores
        count = len(sentiments)
        for key in aggregated:
            aggregated[key] /= count
        
        return aggregated
    
    def _extract_aspect_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing specific aspect keywords"""
        sentences = re.split(r'[.!?]+', text)
        aspect_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                aspect_sentences.append(sentence)
        
        return aspect_sentences
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence based on agreement between methods"""
        if len(results) < 2:
            return 0.5
        
        # Simple confidence calculation based on variance
        compounds = []
        
        if 'vader' in results:
            compounds.append(results['vader'].get('compound', 0.0))
        
        if 'textblob' in results:
            compounds.append(results['textblob'].get('polarity', 0.0))
        
        if 'transformer' in results:
            transformer = results['transformer']
            label = transformer.get('label', '').lower()
            score = transformer.get('score', 0.0)
            
            if 'positive' in label:
                compounds.append(score)
            elif 'negative' in label:
                compounds.append(-score)
            else:
                compounds.append(0.0)
        
        if len(compounds) < 2:
            return 0.5
        
        # Calculate variance (lower variance = higher confidence)
        mean_compound = sum(compounds) / len(compounds)
        variance = sum((x - mean_compound) ** 2 for x in compounds) / len(compounds)
        
        # Convert variance to confidence (inverse relationship)
        confidence = max(0.0, min(1.0, 1.0 - variance))
        
        return confidence
    
    def _empty_sentiment(self) -> Dict[str, float]:
        """Return empty sentiment result"""
        return {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'confidence': 0.0
        }
    
    def _load_fashion_positive_words(self) -> List[str]:
        """Load fashion-specific positive words"""
        return [
            'beautiful', 'gorgeous', 'stunning', 'elegant', 'chic', 'stylish',
            'fashionable', 'trendy', 'sophisticated', 'luxurious', 'premium',
            'high-quality', 'comfortable', 'flattering', 'perfect', 'amazing',
            'excellent', 'outstanding', 'impressive', 'classy', 'timeless',
            'versatile', 'well-made', 'durable', 'soft', 'smooth', 'sleek'
        ]
    
    def _load_fashion_negative_words(self) -> List[str]:
        """Load fashion-specific negative words"""
        return [
            'ugly', 'hideous', 'unflattering', 'cheap', 'poor-quality',
            'uncomfortable', 'tight', 'loose', 'ill-fitting', 'tacky',
            'outdated', 'boring', 'plain', 'disappointing', 'overpriced',
            'flimsy', 'rough', 'scratchy', 'thin', 'poorly-made',
            'defective', 'damaged', 'worn', 'faded', 'stretched'
        ]
