"""
Modern Markov Chain Text Generator
A sophisticated implementation with higher-order chains, smoothing, and advanced features.
"""

import random
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarkovConfig:
    """Configuration for Markov chain generation."""
    order: int = 2  # Higher order for better context
    min_length: int = 10
    max_length: int = 100
    smoothing: str = 'laplace'  # 'laplace', 'good_turing', 'none'
    start_tokens: List[str] = None
    end_tokens: List[str] = None
    
    def __post_init__(self):
        if self.start_tokens is None:
            self.start_tokens = ['<START>']
        if self.end_tokens is None:
            self.end_tokens = ['<END>']


class AdvancedMarkovChain:
    """Advanced Markov chain implementation with higher-order models and smoothing."""
    
    def __init__(self, config: MarkovConfig = None):
        self.config = config or MarkovConfig()
        self.chain: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.start_states: Counter = Counter()
        self.vocabulary: set = set()
        self.total_transitions = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Simple tokenization (can be enhanced with NLTK/spaCy)
        tokens = text.lower().split()
        
        # Add special tokens
        tokens = self.config.start_tokens * self.config.order + tokens + self.config.end_tokens
        
        return tokens
    
    def train(self, texts: Union[str, List[str]]) -> None:
        """Train the Markov chain on provided texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        for text in texts:
            tokens = self.preprocess_text(text)
            self.vocabulary.update(tokens)
            
            # Build n-gram transitions
            for i in range(len(tokens) - self.config.order):
                state = tuple(tokens[i:i + self.config.order])
                next_token = tokens[i + self.config.order]
                
                self.chain[state][next_token] += 1
                self.total_transitions += 1
                
                # Track start states
                if i < self.config.order:
                    self.start_states[state] += 1
    
    def apply_smoothing(self, state: Tuple[str, ...], next_token: str) -> float:
        """Apply smoothing to probability estimation."""
        if state not in self.chain:
            return 0.0
            
        if self.config.smoothing == 'none':
            total_count = sum(self.chain[state].values())
            if total_count == 0:
                return 0.0
            return self.chain[state][next_token] / total_count
        
        elif self.config.smoothing == 'laplace':
            # Laplace smoothing (add-one smoothing)
            vocab_size = len(self.vocabulary)
            count = self.chain[state][next_token]
            total_count = sum(self.chain[state].values())
            return (count + 1) / (total_count + vocab_size)
        
        elif self.config.smoothing == 'good_turing':
            # Simplified Good-Turing smoothing
            count = self.chain[state][next_token]
            total_count = sum(self.chain[state].values())
            if count == 0:
                return 1 / (self.total_transitions + len(self.vocabulary))
            return count / total_count
        
        return 0.0
    
    def get_next_token(self, state: Tuple[str, ...]) -> Optional[str]:
        """Get next token based on current state with weighted random selection."""
        if state not in self.chain:
            return None
            
        # Get weighted probabilities
        tokens = list(self.chain[state].keys())
        weights = [self.apply_smoothing(state, token) for token in tokens]
        
        if not weights or sum(weights) == 0:
            return None
            
        # Weighted random selection
        return random.choices(tokens, weights=weights, k=1)[0]
    
    def generate_text(self, 
                     start_phrase: Optional[str] = None, 
                     max_length: Optional[int] = None) -> str:
        """Generate text using the trained Markov chain."""
        max_length = max_length or self.config.max_length
        
        # Check if model is empty
        if not self.start_states:
            return ""
        
        # Initialize state
        if start_phrase:
            start_tokens = self.preprocess_text(start_phrase)[:self.config.order]
            # Pad with start tokens if needed
            while len(start_tokens) < self.config.order:
                start_tokens = self.config.start_tokens + start_tokens
            state = tuple(start_tokens[-self.config.order:])
        else:
            # Random start state
            if not self.start_states:
                return ""
            state = random.choices(
                list(self.start_states.keys()),
                weights=list(self.start_states.values()),
                k=1
            )[0]
        
        generated = list(state)
        
        # Generate tokens
        for _ in range(max_length - len(generated)):
            next_token = self.get_next_token(state)
            
            if next_token is None or next_token in self.config.end_tokens:
                break
                
            generated.append(next_token)
            state = tuple(generated[-self.config.order:])
        
        # Clean up output
        result = ' '.join(generated)
        
        # Remove special tokens
        for token in self.config.start_tokens + self.config.end_tokens:
            result = result.replace(token, '')
        
        return result.strip()
    
    def get_stats(self) -> Dict:
        """Get statistics about the trained model."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'total_states': len(self.chain),
            'total_transitions': self.total_transitions,
            'average_transitions_per_state': self.total_transitions / len(self.chain) if self.chain else 0,
            'order': self.config.order
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        model_data = {
            'chain': dict(self.chain),
            'start_states': dict(self.start_states),
            'vocabulary': list(self.vocabulary),
            'total_transitions': self.total_transitions,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.chain = defaultdict(Counter, model_data['chain'])
        self.start_states = Counter(model_data['start_states'])
        self.vocabulary = set(model_data['vocabulary'])
        self.total_transitions = model_data['total_transitions']
        self.config = model_data['config']


class TextCorpusManager:
    """Manages text corpora for training Markov chains."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.corpora = {}
    
    def add_corpus(self, name: str, texts: Union[str, List[str]]) -> None:
        """Add a text corpus."""
        if isinstance(texts, str):
            texts = [texts]
        self.corpora[name] = texts
        
        # Save to file
        corpus_file = self.data_dir / f"{name}.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
    
    def load_corpus(self, name: str) -> List[str]:
        """Load a corpus from file."""
        corpus_file = self.data_dir / f"{name}.json"
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def get_corpus_names(self) -> List[str]:
        """Get list of available corpus names."""
        return list(self.corpora.keys())
    
    def create_sample_corpora(self) -> None:
        """Create sample corpora for demonstration."""
        # AI/ML corpus
        ai_corpus = [
            "Artificial intelligence is revolutionizing industries across the globe.",
            "Machine learning algorithms can identify patterns in vast datasets.",
            "Deep learning neural networks mimic the human brain's structure.",
            "Natural language processing enables computers to understand human speech.",
            "Computer vision allows machines to interpret visual information.",
            "AI ethics is becoming increasingly important as technology advances.",
            "Robotics combines AI with mechanical engineering for autonomous systems.",
            "Data science extracts insights from complex information.",
            "Algorithmic bias can perpetuate unfair discrimination in AI systems.",
            "Explainable AI helps users understand how decisions are made."
        ]
        
        # Literature corpus
        literature_corpus = [
            "The sun was setting behind the mountains as she walked home.",
            "He opened the old wooden door with a creaking sound.",
            "Memories flooded back as she looked at the photograph.",
            "The storm raged outside while they sat by the fireplace.",
            "She had always dreamed of traveling to distant lands.",
            "The library was filled with ancient books and dusty shelves.",
            "Time seemed to stand still in that magical moment.",
            "The garden bloomed with colorful flowers in spring.",
            "He found solace in the quiet of the early morning.",
            "The journey ahead would test their courage and determination."
        ]
        
        # Science corpus
        science_corpus = [
            "The scientific method involves observation, hypothesis, and experimentation.",
            "Quantum mechanics describes the behavior of matter at atomic scales.",
            "Evolution explains the diversity of life on Earth.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The periodic table organizes elements by their atomic properties.",
            "Genetics studies how traits are inherited from parents.",
            "Climate change affects weather patterns worldwide.",
            "Space exploration expands our understanding of the universe.",
            "Chemistry studies the composition and properties of matter.",
            "Physics seeks to understand the fundamental laws of nature."
        ]
        
        self.add_corpus("ai_ml", ai_corpus)
        self.add_corpus("literature", literature_corpus)
        self.add_corpus("science", science_corpus)


def main():
    """Demonstrate the advanced Markov chain implementation."""
    print("🧠 Advanced Markov Chain Text Generator")
    print("=" * 50)
    
    # Create corpus manager and sample data
    corpus_manager = TextCorpusManager()
    corpus_manager.create_sample_corpora()
    
    # Configure Markov chain
    config = MarkovConfig(
        order=2,
        min_length=10,
        max_length=50,
        smoothing='laplace'
    )
    
    # Train on AI/ML corpus
    ai_corpus = corpus_manager.load_corpus("ai_ml")
    markov_chain = AdvancedMarkovChain(config)
    markov_chain.train(ai_corpus)
    
    print(f"\n📊 Model Statistics:")
    stats = markov_chain.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Generated Texts:")
    print("-" * 30)
    
    # Generate multiple texts
    for i in range(5):
        generated = markov_chain.generate_text(max_length=30)
        print(f"{i+1}. {generated}")
    
    print(f"\n🎯 Generated with Custom Start:")
    print("-" * 30)
    
    # Generate with custom start phrase
    custom_generated = markov_chain.generate_text(start_phrase="machine learning", max_length=25)
    print(f"Starting with 'machine learning': {custom_generated}")
    
    # Save model
    markov_chain.save_model("ai_markov_model.pkl")
    print(f"\n💾 Model saved to 'ai_markov_model.pkl'")


if __name__ == "__main__":
    main()
