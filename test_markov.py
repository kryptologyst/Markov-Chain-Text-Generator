"""
Comprehensive test suite for Markov Chain Text Generator
Tests all components including core logic, API endpoints, and edge cases.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from markov_chain import AdvancedMarkovChain, MarkovConfig, TextCorpusManager
from app import app


class TestMarkovConfig:
    """Test MarkovConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MarkovConfig()
        assert config.order == 2
        assert config.min_length == 10
        assert config.max_length == 100
        assert config.smoothing == 'laplace'
        assert config.start_tokens == ['<START>']
        assert config.end_tokens == ['<END>']
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MarkovConfig(
            order=3,
            min_length=5,
            max_length=50,
            smoothing='good_turing',
            start_tokens=['<BEGIN>'],
            end_tokens=['<FINISH>']
        )
        assert config.order == 3
        assert config.min_length == 5
        assert config.max_length == 50
        assert config.smoothing == 'good_turing'
        assert config.start_tokens == ['<BEGIN>']
        assert config.end_tokens == ['<FINISH>']


class TestAdvancedMarkovChain:
    """Test AdvancedMarkovChain class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MarkovConfig(order=2, smoothing='laplace')
        self.chain = AdvancedMarkovChain(self.config)
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A quick brown fox jumps over a lazy dog.",
            "The lazy dog jumps over the quick brown fox."
        ]
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Hello World! How are you?"
        tokens = self.chain.preprocess_text(text)
        
        assert tokens[0] == '<START>'
        assert tokens[1] == '<START>'
        assert 'hello' in tokens
        assert 'world!' in tokens  # Note: punctuation is preserved
        assert tokens[-1] == '<END>'
    
    def test_train_single_text(self):
        """Test training with a single text."""
        text = "The quick brown fox"
        self.chain.train(text)
        
        assert len(self.chain.vocabulary) > 0
        assert len(self.chain.chain) > 0
        assert self.chain.total_transitions > 0
    
    def test_train_multiple_texts(self):
        """Test training with multiple texts."""
        self.chain.train(self.sample_texts)
        
        assert len(self.chain.vocabulary) > 0
        assert len(self.chain.chain) > 0
        assert self.chain.total_transitions > 0
        
        # Check that transitions were created
        assert ('<START>', '<START>') in self.chain.chain
        assert ('the', 'quick') in self.chain.chain
    
    def test_apply_smoothing_laplace(self):
        """Test Laplace smoothing."""
        self.chain.train("a b c")
        state = ('<START>', 'a')  # Correct state for order=2
        
        # Test existing token
        prob = self.chain.apply_smoothing(state, 'b')
        assert prob > 0
        
        # Test non-existing token
        prob = self.chain.apply_smoothing(state, 'z')
        assert prob > 0  # Should be smoothed
    
    def test_apply_smoothing_none(self):
        """Test no smoothing."""
        config = MarkovConfig(smoothing='none')
        chain = AdvancedMarkovChain(config)
        chain.train("a b c")
        state = ('<START>', 'a')  # Correct state for order=2
        
        prob = chain.apply_smoothing(state, 'b')
        assert prob >= 0
    
    def test_get_next_token(self):
        """Test getting next token."""
        self.chain.train("a b c d")
        state = ('<START>', 'a')  # Correct state for order=2
        
        next_token = self.chain.get_next_token(state)
        assert next_token is not None
        assert next_token in self.chain.vocabulary
    
    def test_get_next_token_empty_state(self):
        """Test getting next token for empty state."""
        next_token = self.chain.get_next_token(('nonexistent',))
        assert next_token is None
    
    def test_generate_text_no_start(self):
        """Test text generation without start phrase."""
        self.chain.train(self.sample_texts)
        
        generated = self.chain.generate_text(max_length=10)
        assert isinstance(generated, str)
        assert len(generated.split()) <= 10
    
    def test_generate_text_with_start(self):
        """Test text generation with start phrase."""
        self.chain.train(self.sample_texts)
        
        generated = self.chain.generate_text(start_phrase="the quick", max_length=10)
        assert isinstance(generated, str)
        assert generated.lower().startswith("the quick")
    
    def test_generate_text_empty_model(self):
        """Test text generation with empty model."""
        generated = self.chain.generate_text(max_length=10)
        assert generated == ""
    
    def test_get_stats(self):
        """Test getting model statistics."""
        self.chain.train(self.sample_texts)
        stats = self.chain.get_stats()
        
        assert 'vocabulary_size' in stats
        assert 'total_states' in stats
        assert 'total_transitions' in stats
        assert 'order' in stats
        assert stats['order'] == 2
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        self.chain.train(self.sample_texts)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.chain.save_model(tmp.name)
            
            # Create new chain and load model
            new_chain = AdvancedMarkovChain()
            new_chain.load_model(tmp.name)
            
            assert new_chain.vocabulary == self.chain.vocabulary
            assert new_chain.total_transitions == self.chain.total_transitions
            assert new_chain.config.order == self.chain.config.order
            
            # Clean up
            Path(tmp.name).unlink()


class TestTextCorpusManager:
    """Test TextCorpusManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TextCorpusManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_corpus_string(self):
        """Test adding corpus with string input."""
        texts = "This is a test corpus."
        self.manager.add_corpus("test", texts)
        
        assert "test" in self.manager.corpora
        assert len(self.manager.corpora["test"]) == 1
    
    def test_add_corpus_list(self):
        """Test adding corpus with list input."""
        texts = ["Text 1", "Text 2", "Text 3"]
        self.manager.add_corpus("test", texts)
        
        assert "test" in self.manager.corpora
        assert len(self.manager.corpora["test"]) == 3
    
    def test_load_corpus(self):
        """Test loading corpus from file."""
        texts = ["Text 1", "Text 2"]
        self.manager.add_corpus("test", texts)
        
        loaded = self.manager.load_corpus("test")
        assert loaded == texts
    
    def test_load_nonexistent_corpus(self):
        """Test loading non-existent corpus."""
        loaded = self.manager.load_corpus("nonexistent")
        assert loaded == []
    
    def test_get_corpus_names(self):
        """Test getting corpus names."""
        self.manager.add_corpus("corpus1", "Text 1")
        self.manager.add_corpus("corpus2", "Text 2")
        
        names = self.manager.get_corpus_names()
        assert "corpus1" in names
        assert "corpus2" in names
    
    def test_create_sample_corpora(self):
        """Test creating sample corpora."""
        self.manager.create_sample_corpora()
        
        names = self.manager.get_corpus_names()
        assert "ai_ml" in names
        assert "literature" in names
        assert "science" in names
        
        # Check that files were created
        for name in names:
            corpus_file = Path(self.temp_dir) / f"{name}.json"
            assert corpus_file.exists()


class TestFastAPIApp:
    """Test FastAPI application endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the corpus manager to use temp directory
        with patch('app.corpus_manager') as mock_manager:
            mock_manager.data_dir = Path(self.temp_dir)
            mock_manager.get_corpus_names.return_value = ["test_corpus"]
            mock_manager.load_corpus.return_value = ["Test text 1", "Test text 2"]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_get_corpora_endpoint(self):
        """Test getting corpora information."""
        with patch('app.corpus_manager') as mock_manager:
            mock_manager.get_corpus_names.return_value = ["test_corpus"]
            mock_manager.load_corpus.return_value = ["Text 1", "Text 2"]
            
            response = self.client.get("/api/corpora")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "test_corpus"
            assert data[0]["text_count"] == 2
    
    def test_generate_text_endpoint(self):
        """Test text generation endpoint."""
        with patch('app.markov_models') as mock_models:
            # Mock the model
            mock_model = MagicMock()
            mock_model.generate_text.return_value = "Generated text here"
            mock_model.get_stats.return_value = {"vocabulary_size": 100}
            mock_models.__getitem__.return_value = mock_model
            
            with patch('app.corpus_manager') as mock_manager:
                mock_manager.get_corpus_names.return_value = ["test_corpus"]
                
                request_data = {
                    "corpus_name": "test_corpus",
                    "max_length": 20,
                    "order": 2,
                    "smoothing": "laplace"
                }
                
                response = self.client.post("/api/generate", json=request_data)
                assert response.status_code == 200
                
                data = response.json()
                assert "generated_text" in data
                assert data["corpus_name"] == "test_corpus"
    
    def test_generate_text_nonexistent_corpus(self):
        """Test generating text with non-existent corpus."""
        with patch('app.corpus_manager') as mock_manager:
            mock_manager.get_corpus_names.return_value = ["existing_corpus"]
            
            request_data = {
                "corpus_name": "nonexistent_corpus",
                "max_length": 20
            }
            
            response = self.client.post("/api/generate", json=request_data)
            assert response.status_code == 404
    
    def test_train_corpus_endpoint(self):
        """Test training new corpus endpoint."""
        with patch('app.corpus_manager') as mock_manager:
            mock_manager.add_corpus.return_value = None
            
            request_data = {
                "corpus_name": "new_corpus",
                "texts": ["Text 1", "Text 2"]
            }
            
            response = self.client.post("/api/train", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
    
    def test_get_model_stats_endpoint(self):
        """Test getting model statistics endpoint."""
        with patch('app.markov_models') as mock_models:
            mock_model = MagicMock()
            mock_model.get_stats.return_value = {"vocabulary_size": 100}
            
            # Mock the models dictionary iteration
            mock_models.items.return_value = [("test_corpus_2_laplace", mock_model)]
            
            with patch('app.corpus_manager') as mock_manager:
                mock_manager.get_corpus_names.return_value = ["test_corpus"]
                
                response = self.client.get("/api/model/test_corpus/stats")
                assert response.status_code == 200
                
                data = response.json()
                assert "vocabulary_size" in data
    
    def test_delete_corpus_endpoint(self):
        """Test deleting corpus endpoint."""
        with patch('app.corpus_manager') as mock_manager:
            mock_manager.get_corpus_names.return_value = ["test_corpus"]
            mock_manager.data_dir = Path(self.temp_dir)
            
            # Create a dummy file
            dummy_file = Path(self.temp_dir) / "test_corpus.json"
            dummy_file.write_text('["test"]')
            
            response = self.client.delete("/api/corpus/test_corpus")
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TextCorpusManager(self.temp_dir)
        self.config = MarkovConfig(order=2, smoothing='laplace')
        self.chain = AdvancedMarkovChain(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from corpus creation to text generation."""
        # Create corpus
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A quick brown fox jumps over a lazy dog.",
            "The lazy dog jumps over the quick brown fox."
        ]
        self.manager.add_corpus("test", texts)
        
        # Train model
        loaded_texts = self.manager.load_corpus("test")
        self.chain.train(loaded_texts)
        
        # Generate text
        generated = self.chain.generate_text(max_length=10)
        assert isinstance(generated, str)
        assert len(generated.split()) <= 10
        
        # Check stats
        stats = self.chain.get_stats()
        assert stats['vocabulary_size'] > 0
        assert stats['total_transitions'] > 0
    
    def test_different_orders(self):
        """Test Markov chains with different orders."""
        texts = ["a b c d e f g h i j"]
        
        for order in [1, 2, 3]:
            config = MarkovConfig(order=order)
            chain = AdvancedMarkovChain(config)
            chain.train(texts)
            
            generated = chain.generate_text(max_length=5)
            assert isinstance(generated, str)
            
            stats = chain.get_stats()
            assert stats['order'] == order
    
    def test_different_smoothing_methods(self):
        """Test different smoothing methods."""
        texts = ["a b c d e f g h i j"]
        
        for smoothing in ['laplace', 'good_turing', 'none']:
            config = MarkovConfig(smoothing=smoothing)
            chain = AdvancedMarkovChain(config)
            chain.train(texts)
            
            generated = chain.generate_text(max_length=5)
            assert isinstance(generated, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
