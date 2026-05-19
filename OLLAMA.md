# GEPA POC — Local Testing with Ollama

**Run the GEPA POC locally without AWS Bedrock** using [Ollama](https://ollama.ai) — free, open-source LLM inference.

---

## Quick Start (5 minutes)

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Download a Model

```bash
# Start Ollama daemon
ollama serve

# In another terminal, pull a model (one-time setup)
ollama pull mistral      # 4.1GB, fastest, great for testing
# OR
ollama pull neural-chat  # 4.1GB, good for instruction following
# OR
ollama pull llama2       # 3.8GB, good general purpose
```

Choose based on your system:
- **Fast testing** → `mistral` (fastest, smallest)
- **Best quality** → `neural-chat` or `llama2` (slower, better reasoning)

### 3. Configure GEPA POC

```bash
cd gepa-poc

# Edit .env
cp .env.example .env

# Set:
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### 4. Verify Setup

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Expected output: {"models": [{"name": "mistral:latest", ...}]}

# Test GEPA config loads
python -c "
from src.config import load_config
config = load_config()
print(f'✅ LLM Provider: {config.llm_provider}')
print(f'✅ Ollama Model: {config.ollama.model}')
print(f'✅ Ollama URL: {config.ollama.base_url}')
"
```

### 5. Run Tests

```bash
# Unit tests (no LLM calls)
pytest tests/test_integration.py -v

# Smoke test (minimal POC with Ollama)
python main.py --phase smoke-test

# Full POC (200 applicants, 25 iterations)
python main.py --phase all
```

---

## Model Selection Guide

### For Quick Testing (< 5 minutes per phase)
```bash
ollama pull mistral
# Then in .env:
OLLAMA_MODEL=mistral
```
- ✅ Fastest (runs on CPU)
- ✅ Smallest (4.1GB)
- ⚠️ Sometimes rambles (may need stricter JSON parsing)
- Smoke test: ~2-3 minutes
- Full POC: ~30-40 minutes

### For Better Quality (< 10 minutes per phase)
```bash
ollama pull neural-chat
# Then in .env:
OLLAMA_MODEL=neural-chat
```
- ✅ Good instruction following
- ✅ Better JSON parsing
- ⚠️ Slightly slower than Mistral
- Smoke test: ~3-5 minutes
- Full POC: ~45-60 minutes

### For Best Quality (< 15 minutes per phase)
```bash
ollama pull llama2
# Then in .env:
OLLAMA_MODEL=llama2
```
- ✅ Excellent reasoning
- ✅ Better prompt understanding
- ⚠️ Slowest of the three
- Smoke test: ~5-8 minutes
- Full POC: ~60-90 minutes

---

## Performance Comparison

| Model | Speed | Quality | VRAM | Disk | Smoke Test | Full POC |
|-------|-------|---------|------|------|-----------|----------|
| mistral | ⚡⚡⚡ | ⭐⭐⭐ | 7GB | 4.1GB | 2-3m | 30-40m |
| neural-chat | ⚡⚡ | ⭐⭐⭐⭐ | 7GB | 4.1GB | 3-5m | 45-60m |
| llama2 | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | 3.8GB | 5-8m | 60-90m |

**Tip**: Start with `mistral` for quick iteration, switch to `neural-chat` for final runs.

---

## Configuration Examples

### Use Mistral (default)
```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### Use Local Ollama on Different Port
```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11435
OLLAMA_MODEL=llama2
```

### Use Remote Ollama
```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://192.168.1.100:11434
OLLAMA_MODEL=mistral
```

### Switch Back to AWS Bedrock
```bash
# .env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-west-2
```

---

## Troubleshooting

### Error: Read timed out (read timeout=30)
```
HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=30)
```

**Cause:** Local models (especially larger ones like `gemma3`, `llama2`) need more than 30 seconds to generate resume batches. The smoke test generates up to 10 resumes per LLM call.

**Fix:**
```bash
# Add to .env (600 seconds = 10 minutes)
OLLAMA_TIMEOUT=600

# Use a faster model (recommended for testing)
ollama pull mistral
# Then in .env:
OLLAMA_MODEL=mistral
```

**Tips:**
- `mistral` is fastest for smoke tests (~2-3 min); `gemma3` can take 5-10+ min per batch on CPU
- Re-run smoke test after updating `.env`
- Close other heavy apps to free RAM for Ollama

### Error: Cannot connect to Ollama
```
❌ Cannot connect to Ollama at http://localhost:11434
   Start Ollama with: ollama serve
```

**Fix:**
```bash
# Start Ollama daemon in a new terminal
ollama serve

# OR on macOS (if installed via Homebrew)
# It may already be running as a service
# Check: brew services list
```

### Error: Model not found
```
⚠️  No models installed. Run: ollama pull mistral
```

**Fix:**
```bash
# Download the model
ollama pull mistral

# List available models
ollama list
```

### Model running out of memory
```
Error: too many tokens
```

**Fix:**
- Reduce `max_tokens` in `config.yaml`
- Use a smaller model (mistral instead of llama2)
- Reduce dataset size: `GEPA_DATASET_TOTAL=100`
- Close other applications

### LLM responses are rambling / invalid JSON
```
Failed to parse resume batch 0: invalid json
```

**Fix:**
- Use a better model (neural-chat or llama2)
- Increase `max_retries` in config.yaml to 5
- Add stricter prompt formatting

---

## Comparing Ollama vs Bedrock

| Aspect | Ollama | Bedrock |
|--------|--------|---------|
| **Cost** | $0 (free) | ~$41 per full run |
| **Speed** | Depends on hardware | Consistent (API) |
| **Model Choice** | Many (Mistral, Llama2, etc.) | Claude only |
| **Hardware** | Requires local GPU/CPU | Cloud-based |
| **Privacy** | Fully local | AWS-hosted |
| **Internet** | Not required (except setup) | Required |
| **Scalability** | Limited by local hardware | Scales to many users |

**Recommendation**:
- **Testing/Development** → Use Ollama (free, fast iteration)
- **Production/Real Data** → Use Bedrock (better quality, consistency)

---

## Ollama Setup Checklist

- [ ] Install Ollama from https://ollama.ai
- [ ] Run `ollama serve` in background
- [ ] Download a model: `ollama pull mistral`
- [ ] Verify: `curl http://localhost:11434/api/tags`
- [ ] Update `.env`: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=mistral`
- [ ] Run unit tests: `pytest tests/test_integration.py -v`
- [ ] Run smoke test: `python main.py --phase smoke-test`
- [ ] Run full POC: `python main.py --phase all`

---

## Advanced: Running Ollama in Docker

For reproducible testing across machines:

```bash
# Start Ollama in Docker
docker run -d --name ollama -p 11434:11434 ollama/ollama

# Download model (inside container)
docker exec ollama ollama pull mistral

# Verify
curl http://localhost:11434/api/tags
```

---

## Advanced: GPU Acceleration

For faster Ollama inference:

**NVIDIA GPU**:
```bash
# NVIDIA CUDA support automatically detected
# Make sure CUDA is installed
ollama serve
```

**Apple Silicon (M1/M2)**:
```bash
# Metal acceleration automatically enabled
ollama serve
```

**CPU-only**:
```bash
# Uses all available CPU cores
ollama serve
```

---

## FAQ

**Q: Which model should I use?**
A: Start with `mistral` (fast), move to `neural-chat` (better) or `llama2` (best) for final runs.

**Q: Can I use Ollama and Bedrock in the same .env?**
A: No, choose one via `LLM_PROVIDER`. But you can have two `.env` files and switch between them.

**Q: Is Ollama free?**
A: Yes, Ollama and all its models are free and open-source.

**Q: Can I use Ollama for production?**
A: Yes, but with caveats. Bedrock is more reliable for production due to API stability, cost tracking, and better model quality.

**Q: Will my local Ollama setup match AWS Bedrock results?**
A: No, Ollama uses different models (Mistral, Llama2) vs Bedrock's Claude. Results will differ, but POC success criteria (≥15% improvement) should still be achievable.

**Q: How much disk space do I need?**
A: Models are 3.8GB-4.1GB each. SSD recommended for faster loading.

**Q: How much RAM do I need?**
A: Minimum 8GB. 16GB+ recommended for faster inference.

---

## Performance Tips

1. **Use SSD** — Models load faster from SSD than HDD
2. **Close other apps** — Free up RAM for Ollama
3. **Use mistral for iteration** — Fastest model, best for development
4. **Use better model for final run** — llama2 or neural-chat for production
5. **Run on machine with GPU** — Much faster than CPU-only

---

## Example Workflow

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Setup and test with fast model
cd gepa-poc
source venv/bin/activate
set -a; source .env; set +a

# Quick test
python main.py --phase smoke-test

# Switch to better model for final run
sed -i 's/OLLAMA_MODEL=mistral/OLLAMA_MODEL=llama2/' .env

# Full run
python main.py --phase all

# View results
python main.py --phase dashboard
```

---

**Next**: Run `python main.py --phase smoke-test` with Ollama! 🚀
