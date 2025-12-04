#!/usr/bin/env python3
"""Inference server for Syntra model using FastAPI."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI(title="Syntra LLM Inference API")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    generated_text: str

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained("models/syntra_model")
        tokenizer = GPT2Tokenizer.from_pretrained("models/tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        num_return_sequences=1,
        temperature=request.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return GenerateResponse(generated_text=generated_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)