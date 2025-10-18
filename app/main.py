from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our model
from .model.model import Model

app = FastAPI(title="GPT Language Model API", version="1.0.0")

# Initialize model
model = Model()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str

# API operations
@app.get("/")
async def root():
    return {"message": "GPT Language Model API is running!"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        # Generate text using our model
        generated_text = model.predict(request.prompt, request.max_tokens)
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": True,
        "vocab_size": model.get_vocab_size(),
        "device": model.get_device()
    }