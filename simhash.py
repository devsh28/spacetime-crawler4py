import numpy as np
from bs4 import BeautifulSoup
import mmh3  # MurmurHash3
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math

class EnhancedSimHash:
    def __init__(self, hash_bits: int = 64, ngram_range: Tuple[int, int] = (1, 3),
                 threshold: float = 0.8):
        """
        Args:
            hash_bits: Number of bits in the hash
            ngram_range: Range of n-grams to use (min, max) -> unigram, bigram, trigram, etc.
            threshold: Similarity threshold for duplicate detection
        """
        self.hash_bits = hash_bits
        self.ngram_range = ngram_range
        self.threshold = threshold
        self.idf_weights = defaultdict(float)
        self.document_count = 0

    def _extract_features(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Weights for HTML content.
        """
        features = {
            'title': [],
            'headers': [],
            'meta': [],
            'content': []
        }
        
        # Title
        if soup.title:
            features['title'] = self._tokenize_text(soup.title.text)
            
        # Headers
        for header in soup.find_all(['h1', 'h2', 'h3']):
            features['headers'].extend(self._tokenize_text(header.text))
            
        # Meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            features['meta'] = self._tokenize_text(meta_desc['content'])
            
        # Main content
        content_text = soup.get_text(separator=' ', strip=True)
        features['content'] = self._tokenize_text(content_text)
        
        return features

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Text to n-grams.
        """
        # Normalize text
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        tokens = []
        # Generate n-grams
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                tokens.append(ngram)
                
        return tokens

    def _compute_feature_weights(self, features: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute weights based on position and importance.
        """
        weights = {}
        
        # Weight multiplier values for feature types
        weight_multipliers = {
            'title': 4.0,
            'headers': 3.0,
            'meta': 2.0,
            'content': 1.0
        }
        
        for feature_type, tokens in features.items():
            multiplier = weight_multipliers[feature_type]
            for pos, token in enumerate(tokens):
                # Position-based decay
                position_weight = 1.0 / (1 + 0.1 * pos)
                
                # Combine with IDF weight, if available
                idf_weight = self.idf_weights.get(token, 1.0)
                
                final_weight = multiplier * position_weight * idf_weight
                
                # Accumulate weights for tokens with multiple appearences 
                weights[token] = weights.get(token, 0) + final_weight
                
        return weights

    def _compute_feature_hash(self, feature: str) -> int:
        """
        Generate hash with MurmurHash3.
        """
        return mmh3.hash64(feature.encode(), seed=42)[0] & ((1 << self.hash_bits) - 1)

    def compute_document_hash(self, html_content: str) -> np.ndarray:
        """
        Compute SimHash for HTML document.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        features = self._extract_features(soup)
        feature_weights = self._compute_feature_weights(features)
        
        # Initialize vector for hash computation
        vector = np.zeros(self.hash_bits)
        
        # Compute weighted hash
        for feature, weight in feature_weights.items():
            feature_hash = self._compute_feature_hash(feature)
            
            # Update vector based on hash bits
            for i in range(self.hash_bits):
                bit = (feature_hash >> i) & 1
                vector[i] += (2 * bit - 1) * weight
        
        # Convert to binary hash
        return vector > 0

    def compute_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> Dict:
        """
        Similarity between two SimHashes.
        """
        hamming_dist = np.sum(hash1 != hash2)
        normalized_dist = hamming_dist / self.hash_bits
        similarity = 1.0 - normalized_dist
        
        return {
            'is_duplicate': similarity >= self.threshold,
            'similarity_score': similarity,
            'hamming_distance': int(hamming_dist)
        }

    def update_idf_weights(self, document_collection: List[str]):
        """
        Inverse document frequency weights based on a collection of documents.
        """
        token_document_count = defaultdict(int)
        self.document_count = len(document_collection)
        
        for doc in document_collection:
            soup = BeautifulSoup(doc, 'html.parser')
            features = self._extract_features(soup)
            
            # Count unique tokens in document
            unique_tokens = set()
            for feature_list in features.values():
                unique_tokens.update(feature_list)
                
            for token in unique_tokens:
                token_document_count[token] += 1
        
        # IDF weights
        for token, doc_count in token_document_count.items():
            self.idf_weights[token] = math.log(self.document_count / (1 + doc_count))
