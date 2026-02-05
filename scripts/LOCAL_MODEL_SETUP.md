# Setting Up Local Model Testing with MedGemma-27b-it

This guide explains how to run the MedGemma-27b-it model locally and test it with the MAST benchmarks.

## Prerequisites

1. **Python 3.13+** installed
2. **GPU acceleration** (recommended):
   - **CUDA**: NVIDIA GPU with sufficient VRAM (27B model requires ~50GB+ VRAM)
   - **Metal**: Apple Silicon Mac (M1/M2/M3/M4) with sufficient unified memory (32GB+ recommended for 27B model)
   - For CPU-only testing, expect very slow inference times
3. **Hugging Face account** with access to MedGemma model
   - Accept the model's terms: https://huggingface.co/google/medgemma-27b-it
   - You may need to log in: `huggingface-cli login`

## Installation

### 1. Install Base Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Install Local Server Dependencies

```bash
# Using uv
uv sync --extra local-server

# Or using pip
pip install fastapi uvicorn[standard] transformers>=4.50.0 torch accelerate pillow
```

### 3. Authenticate with Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. Get your token from: https://huggingface.co/settings/tokens

## Running the Local Server

### Start the API Server

```bash
python scripts/local_model_server.py
```

The server will:
- Automatically detect the best available device (MPS > CUDA > CPU)
- Download the model on first run (this may take 30+ minutes and ~50GB disk space)
- Load the model into memory (requires significant VRAM/Unified Memory)
- Start serving on `http://localhost:8080`

**Device Support:**
- **Metal (MPS)**: Automatically used on Apple Silicon (M1/M2/M3/M4) Macs
- **CUDA**: Automatically used on NVIDIA GPUs
- **CPU**: Fallback if no GPU is available (very slow for 27B model)

**Note for Metal/MPS users:**
- The server automatically detects and uses MPS on Apple Silicon
- May use float32 instead of bfloat16 for better MPS compatibility
- The 27B model is very large; ensure you have sufficient unified memory (32GB+ recommended)

### Configuration Options

You can customize the server using environment variables:

```bash
# Change port
PORT=8080 python scripts/local_model_server.py

# Change bearer token
BEARER_TOKEN=your_secret_token python scripts/local_model_server.py

# Set max tokens
MAX_NEW_TOKENS=2048 python scripts/local_model_server.py

# Force CPU (not recommended for 27B model)
DEVICE=cpu python scripts/local_model_server.py
```

## Configure MAST to Use Local Server

### Update config.json

Edit `scripts/config.json`:

```json
{
  "endpoint": {
    "url": "http://localhost:8080",
    "token": "test_token",
    "timeout": 300
  }
}
```

**Note:** Make sure the `token` matches the `BEARER_TOKEN` environment variable (default is `test_token`).

## Running Benchmarks

Once the server is running, in a separate terminal:

```bash
python scripts/validate_all.py
```

This will:
1. Make API calls to your local server
2. Validate responses against the benchmark schemas
3. Save results to `results/` directory

## Troubleshooting

### Out of Memory Errors

The 27B model requires significant VRAM/Unified Memory. Options:
- Use a smaller model variant (4B) if available
- Enable CPU offloading (modify `device_map` in the script)
- Use model quantization (requires code changes)
- **For Metal/MPS**: Ensure you have sufficient unified memory (32GB+ recommended for 27B model)

### Model Download Issues

- Ensure you're logged in: `huggingface-cli login`
- Check you've accepted the model's terms on Hugging Face
- Verify you have sufficient disk space (~50GB)

### Slow Inference

- Ensure you're using GPU acceleration:
  - **CUDA**: Check `DEVICE` is `cuda`
  - **Metal**: Check `DEVICE` is `mps` (automatically detected on Apple Silicon)
- The 27B model is large; expect 10-60 seconds per request depending on hardware
- Metal/MPS may be slower than CUDA for very large models
- Consider using a smaller model for faster testing

### JSON Parsing Errors

The server attempts to extract JSON from model responses. If you see parsing errors:
- Check `results/noharm/test_XXX_response.json` for the raw model output
- The model may need better prompting or fine-tuning for structured output

## Alternative: Using Smaller Models

If the 27B model is too large, you can modify `scripts/local_model_server.py` to use:
- `google/medgemma-4b-it` (much smaller, faster, but less capable)

Change the `MODEL_ID` constant in the script.

## Production Considerations

This local server is for **testing only**. For production:
- Add proper error handling and logging
- Implement request queuing for concurrent requests
- Add response caching if appropriate
- Use proper authentication and security
- Consider using vLLM or TGI for better performance

## Resources

- [MedGemma Model Card](https://huggingface.co/google/medgemma-27b-it)
- [MedGemma Technical Report](https://arxiv.org/abs/2507.05201)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
