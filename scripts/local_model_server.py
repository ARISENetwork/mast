#!/usr/bin/env python3
"""
Local API server for running MedGemma-27b-it model with MAST benchmarks.

This server wraps the Hugging Face MedGemma model to provide an API endpoint
that matches the MAST benchmark requirements. Supports CUDA, Metal (MPS), and CPU.

Usage:
    python scripts/local_model_server.py

Then configure scripts/config.json to point to http://localhost:8080
"""

import os
import json
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import uvicorn

# Configuration
MODEL_ID = "google/medgemma-27b-it"
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "test_token")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))  # Increased for longer rationales

# Device detection: prefer MPS (Metal) on macOS, then CUDA, then CPU
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"

DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

app = FastAPI(title="MedGemma MAST API Server")

# Global model and processor (loaded on startup)
model = None
processor = None


def load_model():
    """Load the MedGemma model and processor."""
    global model, processor
    
    print(f"Loading model {MODEL_ID} on {DEVICE}...")
    print("This may take several minutes on first run...")
    
    # Device-specific considerations
    if DEVICE == "mps":
        print("Using Metal Performance Shaders (MPS) for Apple Silicon GPU acceleration")
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device_to_use = "cpu"
        else:
            device_to_use = "mps"
            use_bfloat16 = True
    elif DEVICE == "cuda":
        device_to_use = "cuda"
        use_bfloat16 = True
    else:
        device_to_use = "cpu"
        use_bfloat16 = False  # CPU typically uses float32
    
    try:
        # Determine dtype based on device
        if use_bfloat16 and device_to_use != "cpu":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            if device_to_use != "cpu":
                print(f"Using {dtype} precision (MPS/CPU compatibility)")
        
        # Load model with device-specific handling
        if device_to_use == "mps":
            # For MPS, load to CPU first then move to MPS
            # This avoids potential issues with device_map on MPS
            print("Loading model to CPU first, then moving to MPS...")
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=None,  # Don't use auto device_map with MPS
            )
            model = model.to(device_to_use)
        elif device_to_use == "cuda":
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=None,
            )
            model = model.to(device_to_use)
        
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        print(f"Model loaded successfully on {device_to_use}")
        print(f"Using dtype: {dtype}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        # If bfloat16 fails on MPS, try float32
        if device_to_use == "mps" and use_bfloat16:
            print("Retrying with float32 precision...")
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float32,
                    device_map=None,
                )
                model = model.to("mps")
                processor = AutoProcessor.from_pretrained(MODEL_ID)
                print(f"Model loaded successfully on mps with float32")
                return True
            except Exception as e2:
                print(f"Error with float32: {e2}")
        raise


def extract_json_from_text(text: str) -> Optional[list]:
    """
    Extract JSON (array or object) from model output text.
    Handles markdown code blocks (```json ... ```), raw JSON, and embedded patterns.
    """
    def try_parse(s: str):
        s = s.strip()
        # JSON does not allow leading + on numbers; model may output {"Rating": +2}
        s = re.sub(r":\s*\+(\d+)", r": \1", s)
        try:
            return json.loads(s)
        except Exception:
            return None

    raw = text.strip()
    # Parse entire text as JSON
    parsed = try_parse(raw)
    if parsed is not None:
        return parsed if isinstance(parsed, (list, dict)) else None

    # Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_re = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", re.DOTALL)
    for block in code_block_re.findall(text):
        parsed = try_parse(block)
        if parsed is not None and isinstance(parsed, (list, dict)):
            return parsed

    # Try embedded JSON array or object (greedy match from first { or [ to last } or ])
    for start, end in (("{", "}"), ("[", "]")):
        i = text.find(start)
        if i == -1:
            continue
        j = text.rfind(end)
        if j != -1 and j > i:
            parsed = try_parse(text[i : j + 1])
            if parsed is not None and isinstance(parsed, (list, dict)):
                return parsed
    return None


def format_prompt_for_medgemma(prompt_text: str) -> list:
    """
    Convert MAST prompt format to MedGemma chat format.
    
    MAST sends: prompt.md content + "\n" + test_input.txt content
    We need to convert this to MedGemma's message format.
    """
    # Split prompt and case (they're joined by "\n")
    # The prompt ends with "Details of the case and options follow:"
    parts = prompt_text.split("\nDetails of the case and options follow:", 1)
    
    if len(parts) == 2:
        system_prompt = parts[0].strip()
        case_text = parts[1].strip()
    else:
        # Fallback: treat entire text as user message
        system_prompt = "You are an expert medical AI system providing clinical decision support."
        case_text = prompt_text.strip()
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": case_text}]
        }
    ]
    
    return messages


def generate_response(prompt_text: str) -> dict:
    """
    Generate response from MedGemma model.
    
    Returns:
        dict with 'success', 'response', and optional 'error'
    """
    try:
        # Format prompt for MedGemma
        messages = format_prompt_for_medgemma(prompt_text)
        
        # Process messages
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to the correct device and dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Embedding layers require integer indices (Long); do not cast input_ids to model dtype
        out = {}
        for k, v in inputs.items():
            v = v.to(device=device)
            if k == "input_ids":
                out[k] = v.long()
            elif k == "pixel_values":
                out[k] = v.to(dtype)
            else:
                out[k] = v
        inputs = out
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        # MPS may have issues with some operations, so we use inference_mode
        with torch.inference_mode():
            try:
                generation = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                )
                generation = generation[0][input_len:]
            except RuntimeError as e:
                # MPS-specific error handling
                if "MPS" in str(e) or "Metal" in str(e):
                    print(f"MPS error during generation: {e}")
                    print("This may indicate MPS compatibility issues with this model size.")
                    raise
                else:
                    raise
        
        # Decode
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        # Extract JSON from response
        json_response = extract_json_from_text(decoded)
        
        # SCT benchmark expects a single object {Rating, Rationale}; model often returns [{...}]
        if isinstance(json_response, list) and len(json_response) == 1:
            one = json_response[0]
            if isinstance(one, dict) and "Rating" in one and "Rationale" in one:
                json_response = one
        if json_response is None:
            # If we can't extract JSON, return error with the raw response for debugging
            return {
                "success": False,
                "error": "Could not extract valid JSON from model response",
                "raw_response": decoded[:500]  # First 500 chars for debugging
            }
        
        return {
            "success": True,
            "response": json_response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    device_info = DEVICE
    if model is not None:
        actual_device = str(next(model.parameters()).device)
        device_info = f"{DEVICE} (actual: {actual_device})"
    
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": device_info,
        "model_loaded": model is not None,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    }


@app.post("/", response_class=JSONResponse)
async def generate(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Main endpoint for MAST benchmark.
    
    Accepts:
        - POST request with plain text body (Content-Type: text/plain)
        - Authorization: Bearer {token} header
    
    Returns:
        JSON array matching MAST schema
    """
    # Check authentication
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.replace("Bearer ", "").strip()
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    
    # Read request body as plain text
    try:
        body = await request.body()
        prompt_text = body.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading request body: {e}")
    
    if not prompt_text.strip():
        raise HTTPException(status_code=400, detail="Empty request body")
    
    # Generate response
    result = generate_response(prompt_text)
    
    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        raw_response = result.get("raw_response", "")
        
        # Log error for debugging
        print(f"Error generating response: {error_msg}")
        if raw_response:
            print(f"Raw response (first 500 chars): {raw_response}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Model generation failed: {error_msg}"
        )
    
    # Return JSON response
    return JSONResponse(content=result["response"])


if __name__ == "__main__":
    print(f"Starting MedGemma API server on {HOST}:{PORT}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    if DEVICE == "mps":
        if torch.backends.mps.is_available():
            print("Metal Performance Shaders (MPS) is available")
        else:
            print("MPS not available, will use CPU")
    print(f"Bearer token: {BEARER_TOKEN[:10]}..." if len(BEARER_TOKEN) > 10 else f"Bearer token: {BEARER_TOKEN}")
    print(f"\nTo test, configure scripts/config.json with:")
    print(f'  "url": "http://localhost:{PORT}"')
    print(f'  "token": "{BEARER_TOKEN}"')
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host=HOST, port=PORT)
