# GEPA POC — Complete Documentation Index

This directory contains a fully-functional Proof of Concept for GEPA (Genetic-Pareto optimization) Talent Lens — an AI system that discovers better hiring evaluation prompts than human-authored baselines.

---

## 📖 Documentation Overview

### For Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** ← **START HERE**
  - 5-minute setup
  - Run tests in 3 commands
  - Quick reference card

### For LLM Provider Options
- **[OLLAMA.md](OLLAMA.md)** — Run locally without AWS
  - Install Ollama (free, open-source)
  - Choose model (Mistral, Llama2, Neural-Chat)
  - Configure in .env
  - Performance tips and troubleshooting

- **README.md** — AWS Bedrock setup (recommended for production)
  - Complete setup guide
  - AWS credential configuration

### For Testing & Validation
- **[TESTING.md](TESTING.md)**
  - 5 test cases explained
  - How to run each test
  - Expected outputs
  - Troubleshooting guide

- **[README.md](README.md)**
  - Complete setup guide
  - Testing strategy (unit/integration/e2e)
  - Understanding outputs
  - Detailed troubleshooting

### For Understanding the Design
- **[Plan](/Users/mridhul/.claude/plans/lively-napping-biscuit.md)**
  - Architecture overview
  - Phase-by-phase breakdown
  - Module interfaces
  - Key design decisions

---

## 🧪 The 5 Test Cases (Proof of Correctness)

| Test | What It Proves | Time | Cost |
|------|---|---|---|
| **Test 1: Database Schema** | SQLite works, data is atomic, operations are idempotent | <1s | $0 |
| **Test 2: Config Loading** | YAML loads correctly, all required fields present | <1s | $0 |
| **Test 3: Synthetic Data** | Resume generation works, oracle labeling is deterministic | <1s | $0 |
| **Test 4: Metrics Calculation** | Accuracy, precision, recall computed correctly | <1s | $0 |
| **Test 5: Pareto Frontier** | Optimization constraints are correct, frontier management works | <1s | $0 |

**Run all tests:**
```bash
pytest tests/test_integration.py -v
```

---

## 🎯 Test Execution Levels

### Level 1: Unit Tests (No LLM calls)
```bash
pytest tests/test_integration.py -v
```
**Validates**: All components work in isolation
**Time**: <5 seconds
**Cost**: $0

### Level 2: Smoke Test (Minimal LLM calls)
```bash
python main.py --phase smoke-test
```
**Validates**: End-to-end pipeline works
**Data**: 10 applicants, 3 iterations
**Time**: ~2 minutes
**Cost**: <$1

### Level 3: Full POC (Real optimization)
```bash
python main.py --phase all
```
**Validates**: Achieves ≥15% improvement on holdout
**Data**: 200 applicants, 25 iterations
**Time**: ~50-60 minutes
**Cost**: ~$41

---

## 📁 Project Structure

```
gepa-poc/
├── main.py                    # CLI orchestrator (--phase all|prep|baseline|...)
├── config.yaml                # Job description, baseline prompts, GEPA params
├── requirements.txt           # Dependencies
├── README.md                  # Complete guide (setup, testing, troubleshooting)
├── TESTING.md                 # Test cases explained (5 detailed tests)
├── QUICKSTART.md              # Quick reference (copy-paste commands)
├── DOCUMENTATION.md           # This file (index of all docs)
│
├── src/
│   ├── config.py              # Config loading & validation
│   ├── database.py            # SQLite schema + 50+ helpers
│   ├── data_pipeline/
│   │   ├── synthetic_generator.py    # Claude batch resume generator
│   │   └── dataset_manager.py        # Oracle, train/test split
│   ├── llm/
│   │   ├── client.py                 # Bedrock wrapper, retry, budget
│   │   └── cost_tracker.py           # Price table, cost math
│   ├── evaluator/
│   │   ├── prompt_executor.py        # Run 5-prompt set
│   │   └── metrics.py                # Accuracy, Spearman ρ, F1
│   ├── gepa/
│   │   ├── pareto.py                 # 2-objective frontier
│   │   ├── rollout.py                # Evaluate candidates
│   │   ├── reflector.py              # Meta-LLM mutation
│   │   ├── seed_generator.py         # Initial candidates
│   │   └── optimizer.py              # Main GEPA loop
│   └── dashboard/
│       └── app.py                    # Streamlit visualization
│
├── data/
│   ├── synthetic/              # Generated resume JSONs
│   └── gepa.db                 # SQLite database (all results)
│
└── tests/
    ├── test_integration.py     # 5 validation test cases
    └── __init__.py
```

---

## 🚀 Getting Started (5 minutes)

### 1. Setup
```bash
cd gepa-poc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with AWS credentials
```

### 2. Verify Setup
```bash
set -a; source .env; set +a
python -c "from src.config import load_config; from src.database import get_db, init_schema; print('✅ Ready')"
```

### 3. Run Tests
```bash
# Unit tests (proof of correctness)
pytest tests/test_integration.py -v

# Smoke test (proof of connectivity)
python main.py --phase smoke-test

# Full POC (proof of improvement)
python main.py --phase all
```

---

## 📊 What Each Test Proves

### Test 1: Database Schema & Idempotency
**Proves:**
- ✅ SQLite schema is valid and creates successfully
- ✅ CRUD operations work correctly
- ✅ Idempotent operations (can repeat without error)
- ✅ Checkpoint/resume functionality

**Why it matters:**
- Without this, checkpoints would corrupt data
- Without this, crashed runs couldn't resume

### Test 2: Config Loading & Validation
**Proves:**
- ✅ config.yaml parses correctly
- ✅ All required fields present
- ✅ 5 baseline prompts loaded correctly
- ✅ Typed validation works

**Why it matters:**
- Without this, missing config would cause silent failures
- Without this, baseline would be incomplete

### Test 3: Synthetic Data & Oracle
**Proves:**
- ✅ Claude generates valid resume JSON
- ✅ Oracle labeling is deterministic (reproducible)
- ✅ Oracle has learnable signal (mentoring boost)
- ✅ Train/test split is stratified

**Why it matters:**
- Without determinism, can't compare results across runs
- Without signal, GEPA has nothing to optimize
- Without stratification, test set would be biased

### Test 4: Metrics Calculation
**Proves:**
- ✅ Accuracy = correct predictions / total
- ✅ Precision = TP / (TP+FP) for hired class
- ✅ Recall = TP / (TP+FN) for hired class
- ✅ F1 = harmonic mean of precision/recall
- ✅ Spearman ρ = rank correlation

**Why it matters:**
- Without correct metrics, improvement calculation is meaningless
- Without precision/recall, can't detect degenerate solutions

### Test 5: Pareto Frontier Logic
**Proves:**
- ✅ Dominance logic is correct
- ✅ Frontier doesn't exceed max size
- ✅ Dominated candidates are removed
- ✅ Pareto constraint prevents degenerate solutions

**Why it matters:**
- Without Pareto logic, optimizer might find useless solutions
- Without frontier management, memory would grow unbounded

---

## 🧭 Testing Roadmap

```
Setup (5 min)
    ↓
Unit Tests (< 1s) ← Proves all components correct
    ↓
Smoke Test (2 min) ← Proves end-to-end works
    ↓
Full POC (60 min) ← Proves ≥15% improvement
    ↓
Dashboard ← Visualize results
    ↓
Success! 🎉
```

---

## 📈 Expected Results

### After Smoke Test
```
============================================================
SMOKE TEST
============================================================

✅ Phase 1 (prep): 10 applicants, stratified split
✅ Phase 2 (baseline): baseline accuracy ~0.60
✅ Phase 3 (optimize): 3 iterations, Pareto pool grows
✅ Phase 4 (evaluate): holdout evaluation complete

✅ Smoke test passed!
```

### After Full POC
```
============================================================
HOLDOUT SET COMPARISON
============================================================

Baseline (human-authored):
  Accuracy: 0.620
  Precision (hired): 0.571

Optimized (GEPA):
  Accuracy: 0.740
  Precision (hired): 0.690

Improvement:
  Accuracy: +19.4%
  ✅ SUCCESS: 19.4% improvement (target: ≥15%)
```

---

## 🔍 Detailed Documentation by Topic

### Setup & Installation
- **QUICKSTART.md** - 5-minute setup
- **README.md** - Complete setup guide with troubleshooting

### Testing & Validation
- **TESTING.md** - 5 test cases detailed, run instructions
- **README.md** - Test plan section, validation tests

### Understanding the System
- **README.md** - Architecture section
- **Plan** - Complete technical design
- **config.yaml** - Configuration reference

### Running the POC
- **QUICKSTART.md** - Copy-paste commands
- **README.md** - Phase-by-phase walkthrough

### Troubleshooting
- **README.md** - Comprehensive troubleshooting
- **TESTING.md** - Test-specific debugging

---

## ✅ Success Checklist

Before declaring POC complete, verify:

- [ ] All 5 unit tests pass: `pytest tests/test_integration.py -v`
- [ ] Smoke test completes: `python main.py --phase smoke-test`
- [ ] Full POC achieves ≥15%: `python main.py --phase all`
- [ ] Database has 200 applicants: `sqlite3 data/gepa.db "SELECT COUNT(*) FROM applicants;"`
- [ ] Cost tracked correctly: `sqlite3 data/gepa.db "SELECT ROUND(SUM(cost_usd),2) FROM llm_call_log;"`
- [ ] Dashboard loads: `python main.py --phase dashboard`
- [ ] Pareto frontier has ≥3 candidates
- [ ] Final prompts differ from baseline

---

## 📚 Key References

| Question | Answer | Document |
|----------|--------|-----------|
| How do I set up? | See Quick Start | QUICKSTART.md |
| How do I run tests? | See test commands | TESTING.md |
| What do tests prove? | See detailed explanations | TESTING.md |
| How does the system work? | See architecture | Plan |
| What's the full guide? | See complete guide | README.md |
| What if something breaks? | See troubleshooting | README.md |

---

## 🎯 Summary

**GEPA POC** is a complete, testable system to prove that AI-driven optimization can discover better hiring evaluation prompts.

**Three levels of proof:**
1. **Unit Tests** - All components work correctly
2. **Smoke Test** - System works end-to-end
3. **Full POC** - Achieves ≥15% improvement goal

**Get started:**
```bash
# Setup
source venv/bin/activate
set -a; source .env; set +a

# Test
pytest tests/test_integration.py -v
python main.py --phase smoke-test
python main.py --phase all

# Visualize
python main.py --phase dashboard
```

---

**Next step**: Read [QUICKSTART.md](QUICKSTART.md) and run the tests! 🚀
