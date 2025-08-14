"""
Text processing utilities for fashion analysis
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob
from typing import List, Dict, Set
import string

class TextProcessor:
    """Text processing utilities for fashion content"""
    
    def __init__(self):
        """Initialize text processor with required models"""
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            print("Warning: spaCy model not found. Some features may be limited.")
        
        # Fashion-specific keywords
        self.fashion_keywords = self._load_fashion_keywords()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    print(f"Warning: Could not download {data}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^\w\s\-\'\.]', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and short words
        keywords = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2 
            and token.isalpha()
        ]
        
        # Lemmatize
        keywords = [self.lemmatizer.lemmatize(word) for word in keywords]
        
        # Filter for fashion-relevant terms
        fashion_keywords = [
            word for word in keywords 
            if word in self.fashion_keywords or self._is_fashion_relevant(word)
        ]
        
        # If we have fashion keywords, prioritize them
        if fashion_keywords:
            keywords = fashion_keywords
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)
        
        return unique_keywords[:max_keywords]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'brands': [],
            'materials': [],
            'colors': [],
            'styles': [],
            'sizes': []
        }
        
        if not text:
            return entities
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    if self._is_brand(ent.text):
                        entities['brands'].append(ent.text)
        
        # Extract fashion-specific entities
        entities['materials'].extend(self._extract_materials(text))
        entities['colors'].extend(self._extract_colors(text))
        entities['styles'].extend(self._extract_styles(text))
        entities['sizes'].extend(self._extract_sizes(text))
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_features(self, text: str) -> List[str]:
        """Extract product features from description"""
        features = []
        
        if not text:
            return features
        
        # Feature patterns
        feature_patterns = [
            r'\b(\w+(?:\s+\w+)*)\s+(?:fabric|material|construction|design|style)\b',
            r'\b(?:made\s+(?:of|from|with))\s+(\w+(?:\s+\w+)*)\b',
            r'\b(\w+(?:\s+\w+)*)\s+(?:finish|treatment|coating)\b',
            r'\b(?:features?)\s+(\w+(?:\s+\w+)*)\b',
            r'\b(\w+(?:\s+\w+)*)\s+(?:closure|fastening)\b'
        ]
        
        text_lower = text.lower()
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, text_lower)
            features.extend(matches)
        
        # Clean and filter features
        cleaned_features = []
        for feature in features:
            feature = feature.strip()
            if len(feature) > 2 and feature not in self.stop_words:
                cleaned_features.append(feature)
        
        return list(set(cleaned_features))
    
    def extract_sentiment_phrases(self, text: str) -> Dict[str, List[str]]:
        """Extract sentiment-bearing phrases"""
        if not text:
            return {'positive': [], 'negative': [], 'neutral': []}
        
        blob = TextBlob(text)
        sentences = blob.sentences
        
        sentiment_phrases = {'positive': [], 'negative': [], 'neutral': []}
        
        for sentence in sentences:
            sentiment = sentence.sentiment.polarity
            sentence_text = str(sentence)
            
            if sentiment > 0.1:
                sentiment_phrases['positive'].append(sentence_text)
            elif sentiment < -0.1:
                sentiment_phrases['negative'].append(sentence_text)
            else:
                sentiment_phrases['neutral'].append(sentence_text)
        
        return sentiment_phrases
    
    def _load_fashion_keywords(self) -> Set[str]:
        """Load fashion-specific keywords"""
        return {
            # Materials
            'cotton', 'silk', 'wool', 'cashmere', 'linen', 'polyester', 'leather',
            'suede', 'denim', 'velvet', 'satin', 'chiffon', 'lace', 'mesh',
            
            # Styles
            'classic', 'vintage', 'modern', 'contemporary', 'bohemian', 'minimalist',
            'elegant', 'casual', 'formal', 'sporty', 'edgy', 'romantic',
            
            # Colors
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
            'brown', 'gray', 'navy', 'beige', 'cream', 'burgundy', 'emerald',
            
            # Features
            'button', 'zipper', 'pocket', 'collar', 'sleeve', 'hem', 'waist',
            'neckline', 'seam', 'lining', 'padding', 'stretch', 'breathable',
            
            # Categories
            'dress', 'shirt', 'pants', 'skirt', 'jacket', 'coat', 'sweater',
            'blouse', 'jeans', 'shorts', 'shoes', 'boots', 'sandals', 'bag',
            
            # Brands (luxury)
            'gucci', 'prada', 'chanel', 'hermes', 'dior', 'versace', 'armani',
            'valentino', 'givenchy', 'balenciaga', 'bottega', 'fendi'
        }
    
    def _is_fashion_relevant(self, word: str) -> bool:
        """Check if a word is fashion-relevant"""
        fashion_suffixes = ['wear', 'style', 'fashion', 'design', 'cut', 'fit']
        
        for suffix in fashion_suffixes:
            if word.endswith(suffix):
                return True
        
        return False
    
    def _is_brand(self, text: str) -> bool:
        """Check if text is likely a brand name"""
        # Simple heuristic - check if it's capitalized and in our brand list
        luxury_brands = {
            'gucci', 'prada', 'chanel', 'hermes', 'dior', 'versace',
            'armani', 'valentino', 'givenchy', 'balenciaga', 'fendi'
        }
        
        return text.lower() in luxury_brands or (text[0].isupper() and len(text) > 2)
    
    def _extract_materials(self, text: str) -> List[str]:
        """Extract material mentions from text"""
        materials = [
            'cotton', 'silk', 'wool', 'cashmere', 'linen', 'polyester',
            'leather', 'suede', 'denim', 'velvet', 'satin', 'chiffon',
            'lace', 'mesh', 'nylon', 'spandex', 'bamboo', 'modal'
        ]
        
        found_materials = []
        text_lower = text.lower()
        
        for material in materials:
            if material in text_lower:
                found_materials.append(material)
        
        return found_materials
    
    def _extract_colors(self, text: str) -> List[str]:
        """Extract color mentions from text"""
        colors = [
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink',
            'purple', 'brown', 'gray', 'navy', 'beige', 'cream', 'tan',
            'burgundy', 'emerald', 'sapphire', 'gold', 'silver', 'bronze'
        ]
        
        found_colors = []
        text_lower = text.lower()
        
        for color in colors:
            if color in text_lower:
                found_colors.append(color)
        
        return found_colors
    
    def _extract_styles(self, text: str) -> List[str]:
        """Extract style mentions from text"""
        styles = [
            'classic', 'vintage', 'modern', 'contemporary', 'bohemian',
            'minimalist', 'elegant', 'casual', 'formal', 'sporty', 'edgy',
            'romantic', 'chic', 'trendy', 'timeless', 'sophisticated'
        ]
        
        found_styles = []
        text_lower = text.lower()
        
        for style in styles:
            if style in text_lower:
                found_styles.append(style)
        
        return found_styles
    
    def _extract_sizes(self, text: str) -> List[str]:
        """Extract size mentions from text"""
        size_pattern = r'\b(?:size\s+)?([XS|S|M|L|XL|XXL|\d+|small|medium|large])\b'
        matches = re.findall(size_pattern, text, re.IGNORECASE)
        
        return [match.upper() for match in matches if match]
