# GEPA POC Testing Guide — 5 Test Cases

This document describes the 5 test cases that prove the POC system is working correctly.

**Note**: All tests work with both AWS Bedrock and local Ollama. See [OLLAMA.md](OLLAMA.md) for running tests locally without AWS costs.

---

## Quick Test Summary

| Test | Name | What It Proves | Time | Cost |
|------|------|---|---|---|
| 1️⃣ | Database Schema & Idempotency | SQLite works, operations are atomic | <1s | $0 |
| 2️⃣ | Config Loading & Validation | Configuration system is correct | <1s | $0 |
| 3️⃣ | Synthetic Data & Oracle | Resume generation and labeling work | <1s | $0 |
| 4️⃣ | Metrics Calculation | Accuracy, precision, recall computed correctly | <1s | $0 |
| 5️⃣ | Pareto Frontier Logic | Optimization constraint logic is correct | <1s | $0 |
| 🧪 | Smoke Test | End-to-end pipeline works | ~2min | <$1 |
| 🎯 | Full POC | Achieves ≥15% improvement | ~60min | ~$41 |

---

## Test Cases Explained

### Test 1️⃣: Database Schema & Idempotency

**What it tests**: SQLite schema creation, data integrity, idempotent operations.

**Run:**
```bash
pytest tests/test_integration.py::TestDatabaseSchema -v
```

**Detailed tests:**
- **1a: Schema creation** → All 6 tables exist
- **1b: Idempotent insert** → Can insert same data multiple times without error
- **1c: Foreign key constraints** → Database enforces referential integrity
- **1d: Optimization state upsert** → State can be updated without manual delete

**Why this matters**:
- Proves checkpointing works (can resume interrupted runs)
- Proves data won't corrupt if operations repeat
- Proves all queries will have required tables

**Expected output**:
```
✅ Test 1a: Schema creation — PASSED
✅ Test 1b: Idempotent operations — PASSED
✅ Test 1c: Foreign key constraints — PASSED
✅ Test 1d: Optimization state upsert — PASSED
```

---

### Test 2️⃣: Config Loading & Validation

**What it tests**: Configuration YAML loads correctly, all required fields present.

**Run:**
```bash
pytest tests/test_integration.py::TestConfigLoading -v
```

**Detailed tests:**
- **2a: Config loads** → YAML parses to typed Config dataclass
- **2b: Required fields present** → Job description, oracle, dataset, GEPA, LLM sections exist
- **2c: Baseline prompt format** → 5 prompts present, each with name + prompt text

**Why this matters**:
- Proves config.yaml is valid
- Proves all prompts can be used for baseline
- Proves GEPA parameters are loaded correctly

**Expected output**:
```
✅ Test 2a: Config loading — PASSED
✅ Test 2b: Required fields present — PASSED
✅ Test 2c: Baseline prompt format — PASSED
```

---

### Test 3️⃣: Synthetic Data Generation & Oracle

**What it tests**: Synthetic resume generation and oracle labeling.

**Run:**
```bash
pytest tests/test_integration.py::TestSyntheticData -v
```

**Detailed tests:**
- **3a: Oracle deterministic** → Same resume always gets same label (seeded by hash)
- **3b: Oracle requires skills** → Resumes with required skills score higher
- **3c: Oracle secondary signal** → Mentoring/leadership boosts hire probability

**Why this matters**:
- Proves labels are reproducible (can't compare results if labels change)
- Proves oracle has learnable signal (GEPA has something to optimize)
- Proves oracle reflects realistic hiring (skills matter, mentoring matters)

**Expected output**:
```
✅ Test 3a: Oracle deterministic — PASSED
✅ Test 3b: Oracle requires skills — PASSED
✅ Test 3c: Oracle secondary signal — PASSED
```

---

### Test 4️⃣: Metrics Calculation & Evaluation

**What it tests**: Evaluation metrics are computed correctly.

**Run:**
```bash
pytest tests/test_integration.py::TestMetrics -v
```

**Detailed tests**:
- **4a: Perfect classifier** → 100% correct predictions = 1.0 accuracy/precision/recall
- **4b: All-negative predictor** → Predictor that always says "reject" has expected metrics
- **4c: Threshold sensitivity** → Changing threshold changes precision/recall tradeoff correctly

**Why this matters**:
- Proves accuracy calculation is correct (not inverted, not biased)
- Proves precision/recall distinguish different prediction strategies
- Proves threshold tuning actually works (not a no-op)

**Expected output**:
```
✅ Test 4a: Perfect classifier metrics — PASSED
✅ Test 4b: All-negative predictor — PASSED
✅ Test 4c: Threshold sensitivity — PASSED
```

---

### Test 5️⃣: Pareto Frontier Logic

**What it tests**: Pareto frontier optimization constraints.

**Run:**
```bash
pytest tests/test_integration.py::TestParetoFrontier -v
```

**Detailed tests**:
- **5a: Dominance logic** → A dominates B iff A≥B on all objectives and >B on at least one
- **5b: Frontier size limit** → Pool never exceeds max_size (20)
- **5c: Frontier removes dominated** → When A dominates B, B is removed from pool
- **5d: Frontier rejects dominated** → New candidate is rejected if dominated by existing candidate

**Why this matters**:
- Proves GEPA only keeps interesting candidates (Pareto frontier)
- Proves optimizer won't degenerate (e.g., "always reject" doesn't win)
- Proves frontier grows intelligently (doesn't waste memory on dominated candidates)

**Expected output**:
```
✅ Test 5a: Dominance logic — PASSED
✅ Test 5b: Frontier size limit — PASSED
✅ Test 5c: Frontier update removes dominated — PASSED
✅ Test 5d: Frontier rejects dominated — PASSED
```

---

## Running All Unit Tests

```bash
# Run all 5 test suites
pytest tests/test_integration.py -v

# Run with output capture disabled (see print statements)
pytest tests/test_integration.py -v -s

# Run specific test
pytest tests/test_integration.py::TestDatabaseSchema::test_schema_creation -v

# Run as Python script (no pytest required)
python tests/test_integration.py
```

**Expected total time**: <5 seconds
**Expected cost**: $0

---

## Smoke Test (End-to-End Validation)

Tests the full pipeline on minimal data to prove connectivity.

```bash
python main.py --phase smoke-test
```

**What it does**:
1. Generates 10 synthetic resumes
2. Runs baseline evaluation
3. Runs 3 GEPA iterations
4. Evaluates on holdout

**What it proves**:
- ✅ AWS Bedrock credentials work
- ✅ LLM calls succeed and return valid JSON
- ✅ All phases run without error
- ✅ Database persists results
- ✅ Metrics compute correctly
- ✅ GEPA loop doesn't crash

**Expected output**:
```
============================================================
SMOKE TEST
============================================================
Smoke test: creating minimal dataset (10 applicants)...
Generated 10 resumes
...
[Phase prep complete]
[Phase baseline complete]
[Phase optimize complete - 3 iterations]
[Phase evaluate complete]

✅ Smoke test passed!
```

**Expected time**: ~2 minutes
**Expected cost**: <$0.50

---

## Full POC Run (Success Validation)

Tests that system achieves the primary success criterion: ≥15% accuracy improvement.

```bash
python main.py --phase all
```

**What it does**:
1. Generates 200 synthetic resumes
2. Runs baseline evaluation
3. Runs 25 GEPA iterations
4. Evaluates on holdout set
5. Compares improvement vs baseline

**What it proves**:
- ✅ Optimization improves prompt quality
- ✅ GEPA discovers better evaluation signals
- ✅ Final prompts are human-readable
- ✅ System scales to full dataset
- ✅ Cost tracking prevents budget overrun

**Expected output**:
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

**Expected time**: ~50-60 minutes
**Expected cost**: ~$40 (within $100 cap)

---

## Test Execution Matrix

| Phase | Test Type | Command | Time (Bedrock) | Cost (Bedrock) | Time (Ollama) | Cost (Ollama) |
|-------|-----------|---------|---|---|---|---|
| Setup | Unit | `pytest tests/ -v` | <5s | $0 | <5s | $0 |
| Smoke | Integration | `python main.py --phase smoke-test` | 2m | <$1 | 2-5m | $0 |
| Validate | Success | `python main.py --phase all` | 60m | ~$40 | 30-90m* | $0 |

*Time varies with model: Mistral (30-40m), Neural-Chat (45-60m), Llama2 (60-90m)

---

## Debugging Failed Tests

### Test 1 fails: Database issues
```bash
# Check schema manually
sqlite3 data/gepa.db ".schema"

# Check counts
sqlite3 data/gepa.db "SELECT COUNT(*) FROM applicants;"
```

### Test 2 fails: Config not found
```bash
# Check file exists and is valid YAML
ls -la config.yaml
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Test 3 fails: Oracle not deterministic
```bash
# The oracle seeding should be deterministic by hash
# If failing, check oracle_hire_decision uses hash(name) for seeding
```

### Test 4 fails: Metrics wrong
```bash
# Manually check a simple case
python -c "
from src.evaluator.metrics import compute_metrics
evals = [{'applicant_id': 0, 'aggregate_score': 5.0, 'predicted_hired': 1}]
apps = [{'id': 0, 'hired': 1}]
m = compute_metrics(evals, apps)
print(f'Accuracy: {m.accuracy}')
"
```

### Test 5 fails: Pareto logic wrong
```bash
# Test dominance manually
from src.gepa.pareto import ParetoCandidate, dominates
a = ParetoCandidate(1, 0.8, 0.7)
b = ParetoCandidate(2, 0.7, 0.6)
print(f"A dominates B: {dominates(a, b)}")  # Should be True
```

### Smoke test fails: AWS credentials
```bash
# Check credentials are loaded
echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
echo "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION"

# Test Bedrock access directly
python -c "
from anthropic import AnthropicBedrock
client = AnthropicBedrock(region_name='us-west-2')
print('✅ Bedrock client initialized')
"
```

---

## Success Checklist

After running all tests, you should have:

- [ ] All 5 unit tests pass (pytest)
- [ ] Smoke test completes without errors (<$1, <2min)
- [ ] Full POC run achieves ≥15% improvement
- [ ] Database (`data/gepa.db`) contains 200 applicants
- [ ] `llm_call_log` table shows cost breakdown
- [ ] `iteration_logs` table shows Pareto pool evolution
- [ ] `optimization_state` shows `status='done'`
- [ ] Streamlit dashboard loads with all 5 pages
- [ ] Final prompt set differs from baseline

---

## Test Output Reference

### Unit Tests (passing)
```
tests/test_integration.py::TestDatabaseSchema::test_schema_creation PASSED
tests/test_integration.py::TestDatabaseSchema::test_idempotent_insert PASSED
tests/test_integration.py::TestConfigLoading::test_config_loads PASSED
...
========= 13 passed in 0.04s =========
```

### Smoke Test (passing)
```
✅ Smoke test passed!
```

### Full POC (passing)
```
✅ SUCCESS: 19.4% improvement (target: ≥15%)
```

---

## Next Steps After Testing

1. ✅ **All tests pass** → POC is correct
2. ✅ **Smoke test passes** → System works end-to-end
3. ✅ **Full run achieves ≥15%** → Hypothesis validated
4. 📊 **Review dashboard** → Understand what GEPA learned
5. 📝 **Archive prompts** → Export final prompts for production
6. 📋 **Write findings** → Document lessons learned

---

**Good luck! 🚀**
