"""
FastAPI Web Application for Markov Chain Text Generation
Modern web interface with real-time text generation capabilities.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from pathlib import Path

from markov_chain import AdvancedMarkovChain, MarkovConfig, TextCorpusManager


# Pydantic models for API
class GenerationRequest(BaseModel):
    corpus_name: str = Field(..., description="Name of the corpus to use")
    start_phrase: Optional[str] = Field(None, description="Starting phrase for generation")
    max_length: int = Field(50, ge=5, le=200, description="Maximum length of generated text")
    order: int = Field(2, ge=1, le=5, description="Order of the Markov chain")
    smoothing: str = Field("laplace", description="Smoothing method")

class GenerationResponse(BaseModel):
    generated_text: str
    corpus_name: str
    parameters: Dict[str, Any]
    model_stats: Dict[str, Any]

class CorpusInfo(BaseModel):
    name: str
    text_count: int
    total_words: int

class TrainingRequest(BaseModel):
    corpus_name: str
    texts: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="Markov Chain Text Generator",
    description="Advanced Markov chain text generation with web interface",
    version="1.0.0"
)

# Global variables
corpus_manager = TextCorpusManager()
markov_models: Dict[str, AdvancedMarkovChain] = {}

# Create templates directory
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Create static directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    # Create sample corpora
    corpus_manager.create_sample_corpora()
    
    # Pre-train models for available corpora
    for corpus_name in corpus_manager.get_corpus_names():
        await train_model_for_corpus(corpus_name)


async def train_model_for_corpus(corpus_name: str):
    """Train a Markov model for a specific corpus."""
    texts = corpus_manager.load_corpus(corpus_name)
    if texts:
        config = MarkovConfig(order=2, smoothing='laplace')
        model = AdvancedMarkovChain(config)
        model.train(texts)
        markov_models[corpus_name] = model


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/corpora", response_model=List[CorpusInfo])
async def get_corpora():
    """Get information about available corpora."""
    corpora_info = []
    for name in corpus_manager.get_corpus_names():
        texts = corpus_manager.load_corpus(name)
        total_words = sum(len(text.split()) for text in texts)
        corpora_info.append(CorpusInfo(
            name=name,
            text_count=len(texts),
            total_words=total_words
        ))
    return corpora_info


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the specified parameters."""
    # Check if corpus exists
    if request.corpus_name not in corpus_manager.get_corpus_names():
        raise HTTPException(status_code=404, detail=f"Corpus '{request.corpus_name}' not found")
    
    # Get or create model
    model_key = f"{request.corpus_name}_{request.order}_{request.smoothing}"
    
    if model_key not in markov_models:
        # Train new model with specified parameters
        config = MarkovConfig(
            order=request.order,
            smoothing=request.smoothing
        )
        model = AdvancedMarkovChain(config)
        texts = corpus_manager.load_corpus(request.corpus_name)
        model.train(texts)
        markov_models[model_key] = model
    
    model = markov_models[model_key]
    
    # Generate text
    generated_text = model.generate_text(
        start_phrase=request.start_phrase,
        max_length=request.max_length
    )
    
    return GenerationResponse(
        generated_text=generated_text,
        corpus_name=request.corpus_name,
        parameters={
            "start_phrase": request.start_phrase,
            "max_length": request.max_length,
            "order": request.order,
            "smoothing": request.smoothing
        },
        model_stats=model.get_stats()
    )


@app.post("/api/train")
async def train_corpus(request: TrainingRequest):
    """Train a new corpus."""
    try:
        corpus_manager.add_corpus(request.corpus_name, request.texts)
        await train_model_for_corpus(request.corpus_name)
        return {"message": f"Corpus '{request.corpus_name}' trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/{corpus_name}/stats")
async def get_model_stats(corpus_name: str):
    """Get statistics for a trained model."""
    if corpus_name not in corpus_manager.get_corpus_names():
        raise HTTPException(status_code=404, detail=f"Corpus '{corpus_name}' not found")
    
    # Find any model for this corpus
    model = None
    for key, m in markov_models.items():
        if key.startswith(corpus_name):
            model = m
            break
    
    if not model:
        raise HTTPException(status_code=404, detail=f"No trained model found for corpus '{corpus_name}'")
    
    return model.get_stats()


@app.delete("/api/corpus/{corpus_name}")
async def delete_corpus(corpus_name: str):
    """Delete a corpus and its trained models."""
    if corpus_name not in corpus_manager.get_corpus_names():
        raise HTTPException(status_code=404, detail=f"Corpus '{corpus_name}' not found")
    
    # Remove from corpus manager
    corpus_file = corpus_manager.data_dir / f"{corpus_name}.json"
    if corpus_file.exists():
        corpus_file.unlink()
    
    # Remove trained models
    keys_to_remove = [key for key in markov_models.keys() if key.startswith(corpus_name)]
    for key in keys_to_remove:
        del markov_models[key]
    
    return {"message": f"Corpus '{corpus_name}' deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
