# GEPA POC — Quick Start Card

**Proof of Concept**: Validate GEPA can discover evaluation prompts ≥15% better than human-authored.

---

## 1️⃣ Setup (5 minutes)

```bash
cd gepa-poc

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure LLM provider
cp .env.example .env
```

**Option A: AWS Bedrock** (recommended for production)
```bash
# Edit .env with AWS credentials
# LLM_PROVIDER=bedrock
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AWS_DEFAULT_REGION=us-west-2
```

**Option B: Local Ollama** (free, no AWS required)
```bash
# Edit .env:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=mistral

# Start Ollama in another terminal:
# ollama serve
# ollama pull mistral
```

```bash
# Load environment
set -a; source .env; set +a
```

---

## 2️⃣ Validate Setup (1 minute)

```bash
# Check system works
python -c "
from src.config import load_config
from src.database import get_db, init_schema
from src.llm.client import LLMClient

config = load_config()
conn = get_db(':memory:')
init_schema(conn)
client = LLMClient(config.llm.model_rollout, config.llm.model_reflection,
                   config.llm.model_generation, 3, 30, 100, 80, 10000, conn)
print('✅ All systems ready!')
"
```

---

## 3️⃣ Run Tests (< 1 minute)

### Unit Tests (no AWS calls)
```bash
pytest tests/test_integration.py -v
# All 5 test suites should PASS
```

### Smoke Test (minimal POC, <$1)
```bash
python main.py --phase smoke-test
# Runs 10 applicants, 3 iterations
# Proves end-to-end connectivity
# Time: ~2 minutes
```

### Full POC (complete validation, ~$41)
```bash
python main.py --phase all
# Runs 200 applicants, 25 iterations
# Proves ≥15% improvement achieved
# Time: ~50-60 minutes
```

---

## 4️⃣ Run Individual Phases

If needed, run phases separately:

```bash
# Phase 1: Generate 200 resumes with oracle labels
python main.py --phase prep

# Phase 2: Baseline evaluation (human-authored prompts)
python main.py --phase baseline

# Phase 3: GEPA optimization (25 iterations)
python main.py --phase optimize

# Phase 4: Evaluate on holdout, print comparison
python main.py --phase evaluate

# Phase 5: Launch Streamlit dashboard
python main.py --phase dashboard
```

---

## 5️⃣ Resume Interrupted Run

```bash
# Check status
sqlite3 data/gepa.db "SELECT status, current_iteration FROM optimization_state;"

# Resume from checkpoint
python main.py --phase optimize --resume
```

---

## 6️⃣ View Results

### Terminal
```bash
# After --phase evaluate:
# Shows: Baseline accuracy, Optimized accuracy, % improvement
# ✅ SUCCESS if improvement ≥15%
```

### Dashboard
```bash
python main.py --phase dashboard
# Opens http://localhost:8501
# 5 pages: Overview, Baseline, Progress, Prompts, Examples
```

### Database
```bash
# View applicants
sqlite3 data/gepa.db "SELECT COUNT(*), SUM(hired) FROM applicants;"

# View optimization progress
sqlite3 data/gepa.db \
  "SELECT iteration, accuracy, precision_h FROM iteration_logs WHERE split='train';"

# View costs
sqlite3 data/gepa.db \
  "SELECT phase, COUNT(*), ROUND(SUM(cost_usd),2) FROM llm_call_log GROUP BY phase;"
```

---

## 📊 Expected Results

| Phase | Time | Cost | What It Proves |
|-------|------|------|---|
| Unit Tests | <1s | $0 | All components work correctly |
| Smoke Test | 2m | <$1 | End-to-end connectivity |
| Full POC | 60m | ~$41 | ≥15% improvement achieved |

---

## ✅ Success Criteria

After `python main.py --phase all`:

```
✅ Holdout accuracy improvement ≥15% vs baseline
✅ Pareto frontier has ≥3 candidates
✅ Final prompts are human-readable
✅ Dashboard loads with all 5 pages
✅ Cost tracking shows usage < $100
```

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| AWS credentials not found | `set -a; source .env; set +a` |
| Model not enabled | AWS Console → Bedrock → Enable model |
| Budget exceeded | Increase `budget_hard_cap_usd` in config.yaml or reduce iterations |
| Parse errors in LLM | Automatic fallback handles; if frequent, reduce `max_tokens` |

---

## 📚 Documentation

- **README.md** → Full setup & testing guide
- **TESTING.md** → Detailed test case descriptions
- **config.yaml** → Adjust job description, baseline prompts, GEPA params
- **Plan** → See `/Users/mridhul/.claude/plans/lively-napping-biscuit.md`

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI orchestrator |
| `config.yaml` | Job description, baseline prompts, parameters |
| `data/gepa.db` | SQLite database (all results) |
| `src/gepa/optimizer.py` | GEPA loop |
| `src/dashboard/app.py` | Streamlit visualization |
| `tests/test_integration.py` | 5 validation tests |

---

**Start here**: `python main.py --phase smoke-test`

**Then run**: `python main.py --phase all`

**Finally check**: `python main.py --phase dashboard`

Good luck! 🚀
