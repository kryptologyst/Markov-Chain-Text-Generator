#!/usr/bin/env python3
"""
Demo script for Markov Chain Text Generator
Demonstrates the core functionality and web interface.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from markov_chain import AdvancedMarkovChain, MarkovConfig, TextCorpusManager


def demo_core_functionality():
    """Demonstrate core Markov chain functionality."""
    print("🧠 Markov Chain Text Generator - Core Demo")
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
    for i in range(3):
        generated = markov_chain.generate_text(max_length=30)
        print(f"{i+1}. {generated}")
    
    print(f"\n🎯 Generated with Custom Start:")
    print("-" * 30)
    
    # Generate with custom start phrase
    custom_generated = markov_chain.generate_text(start_phrase="machine learning", max_length=25)
    print(f"Starting with 'machine learning': {custom_generated}")
    
    return markov_chain


def demo_different_orders():
    """Demonstrate different Markov chain orders."""
    print(f"\n🔢 Different Markov Chain Orders Demo")
    print("=" * 50)
    
    corpus_manager = TextCorpusManager()
    corpus_manager.create_sample_corpora()
    literature_corpus = corpus_manager.load_corpus("literature")
    
    for order in [1, 2, 3]:
        print(f"\n📝 Order {order} Markov Chain:")
        print("-" * 20)
        
        config = MarkovConfig(order=order, smoothing='laplace')
        chain = AdvancedMarkovChain(config)
        chain.train(literature_corpus)
        
        # Generate text
        generated = chain.generate_text(max_length=20)
        print(f"Generated: {generated}")
        
        # Show stats
        stats = chain.get_stats()
        print(f"States: {stats['total_states']}, Transitions: {stats['total_transitions']}")


def demo_different_smoothing():
    """Demonstrate different smoothing methods."""
    print(f"\n🎲 Different Smoothing Methods Demo")
    print("=" * 50)
    
    corpus_manager = TextCorpusManager()
    corpus_manager.create_sample_corpora()
    science_corpus = corpus_manager.load_corpus("science")
    
    for smoothing in ['laplace', 'good_turing', 'none']:
        print(f"\n🔧 {smoothing.title()} Smoothing:")
        print("-" * 20)
        
        config = MarkovConfig(order=2, smoothing=smoothing)
        chain = AdvancedMarkovChain(config)
        chain.train(science_corpus)
        
        # Generate text
        generated = chain.generate_text(max_length=20)
        print(f"Generated: {generated}")


def demo_model_persistence():
    """Demonstrate model save/load functionality."""
    print(f"\n💾 Model Persistence Demo")
    print("=" * 50)
    
    corpus_manager = TextCorpusManager()
    corpus_manager.create_sample_corpora()
    ai_corpus = corpus_manager.load_corpus("ai_ml")
    
    # Train and save model
    config = MarkovConfig(order=2, smoothing='laplace')
    original_chain = AdvancedMarkovChain(config)
    original_chain.train(ai_corpus)
    
    model_file = "demo_model.pkl"
    original_chain.save_model(model_file)
    print(f"✅ Model saved to {model_file}")
    
    # Load model
    loaded_chain = AdvancedMarkovChain()
    loaded_chain.load_model(model_file)
    print(f"✅ Model loaded from {model_file}")
    
    # Compare outputs
    original_text = original_chain.generate_text(max_length=20)
    loaded_text = loaded_chain.generate_text(max_length=20)
    
    print(f"\nOriginal model output: {original_text}")
    print(f"Loaded model output: {loaded_text}")
    
    # Clean up
    Path(model_file).unlink()
    print(f"🗑️  Cleaned up {model_file}")


def main():
    """Run all demos."""
    try:
        # Core functionality demo
        demo_core_functionality()
        
        # Different orders demo
        demo_different_orders()
        
        # Different smoothing demo
        demo_different_smoothing()
        
        # Model persistence demo
        demo_model_persistence()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"\n🌐 To run the web interface:")
        print(f"   python app.py")
        print(f"   Then open: http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
