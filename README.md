# GEPA Talent Lens POC — Testing & Setup Guide

**Proof of Concept**: Validate that GEPA (Genetic-Pareto optimization) can discover 5 evaluation prompts that outperform human-authored baselines by ≥15% on hiring decisions.

---

## Quick Start (5 minutes)

### 1. Prerequisites (Choose One)

**Option A: AWS Bedrock (Recommended for production)**
- Python 3.10+
- AWS account with Bedrock access
- `anthropic.claude-3-5-sonnet-20241022-v2:0` model enabled in your region

**Option B: Local Ollama (Free, no AWS required)**
- Python 3.10+
- [Ollama](https://ollama.ai) installed
- A model downloaded: `ollama pull mistral` (or llama2, neural-chat)

### 2. Setup

```bash
# Clone/navigate to project
cd gepa-poc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure LLM provider
cp .env.example .env

# Choose one:
# Option A: AWS Bedrock (edit .env and add AWS credentials)
# Option B: Local Ollama (just ensure ollama serve is running, edit .env with OLLAMA settings)

# Load environment
set -a
source .env
set +a
```

**For Bedrock:**
```bash
# Edit .env and add:
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-west-2
LLM_PROVIDER=bedrock
```

**For Ollama:**
```bash
# Edit .env and set:
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Make sure Ollama is running:
# In another terminal: ollama serve
# Download model: ollama pull mistral
```

### 3. Validate Setup (2 minutes)

```bash
# Test database and imports
python -c "
from src.database import get_db, init_schema
from src.config import load_config
from src.llm.client import LLMClient

# Load config
config = load_config('config.yaml')
print('✅ Config loaded')

# Init database
conn = get_db(':memory:')
init_schema(conn)
print('✅ Database schema created')

# Init LLM client (requires AWS credentials)
try:
    client = LLMClient(
        model_rollout=config.llm.model_rollout,
        model_reflection=config.llm.model_reflection,
        model_generation=config.llm.model_generation,
        max_retries=3,
        timeout_seconds=30,
        budget_hard_cap=100.0,
        budget_warn_threshold=80.0,
        max_calls_per_run=10000,
        db_conn=conn,
    )
    print('✅ LLM client initialized')
except Exception as e:
    print(f'❌ LLM client error: {e}')
    print('   Check AWS credentials and Bedrock model access')
"
```

---

## Testing Strategy

The POC includes three levels of testing:

### Level 1: Unit Tests (No LLM calls)
Test individual components in isolation without using Bedrock.

### Level 2: Integration Tests (With mocks)
Test workflows with mocked LLM responses.

### Level 3: End-to-End Tests (Real LLM calls)
Test the full pipeline with real Bedrock API calls (~$40 cost).

---

## Running Tests

### Run Unit Tests

```bash
# Metrics calculation
pytest tests/test_metrics.py -v

# Pareto frontier logic
pytest tests/test_pareto.py -v

# All unit tests
pytest tests/ -v -k "not integration and not e2e"
```

### Run Smoke Test (Recommended first)

Validates end-to-end with minimal data (10 applicants, 3 iterations, <$1).

```bash
python main.py --phase smoke-test
```

Expected output:
```
============================================================
SMOKE TEST
============================================================
Smoke test: creating minimal dataset (10 applicants)...
Generating batch 1/1 (10 resumes)...
Generated 10 resumes
Applying oracle labels...
Performing stratified train/holdout split...
Inserting into database...
Dataset created: 7 train, 3 holdout

[Phase prep complete]
[Phase baseline complete - shows train/holdout accuracy]
[Phase optimize complete - shows 3 iterations]
[Phase evaluate complete - shows improvement percentage]

✅ Smoke test passed!
```

### Run Full POC (50-60 minutes)

```bash
python main.py --phase all
```

Or run phases individually:

```bash
# Phase 1: Generate 200 resumes, split into train/holdout
python main.py --phase prep

# Phase 2: Baseline evaluation (5 human-authored prompts)
python main.py --phase baseline

# Phase 3: GEPA optimization (25 iterations)
python main.py --phase optimize

# Phase 4: Evaluate on holdout set, compare vs baseline
python main.py --phase evaluate

# Phase 5: Launch dashboard for visualization
python main.py --phase dashboard
```

### Resume Interrupted Run

If optimization is interrupted:

```bash
# Check status
sqlite3 data/gepa.db "SELECT status, current_iteration FROM optimization_state;"

# Resume from checkpoint
python main.py --phase optimize --resume
```

---

## Understanding the Outputs

### Database (`data/gepa.db`)

SQLite file containing all state:

```bash
# View applicants
sqlite3 data/gepa.db "SELECT COUNT(*), SUM(hired) FROM applicants;"
# Output: 200|70 (200 total, 70 hired)

# View optimization progress
sqlite3 data/gepa.db \
  "SELECT iteration, candidate_id, accuracy, precision_h, is_pareto \
   FROM iteration_logs \
   WHERE split='train' \
   ORDER BY iteration;"

# View cost tracking
sqlite3 data/gepa.db \
  "SELECT phase, COUNT(*) as calls, ROUND(SUM(cost_usd), 2) as cost \
   FROM llm_call_log \
   GROUP BY phase;"
```

### Key Metrics

After `--phase evaluate`, you'll see:

```
============================================================
HOLDOUT SET COMPARISON
============================================================

Baseline (human-authored):
  Accuracy: 0.640
  Precision (hired): 0.571
  Recall (hired): 0.714

Optimized (GEPA):
  Accuracy: 0.760
  Precision (hired): 0.650
  Recall (hired): 0.857

Improvement:
  Accuracy: +18.8%
  ✅ SUCCESS: 18.8% improvement (target: ≥15%)
```

**Success Criteria**: Optimized accuracy ≥15% better than baseline on holdout set.

### Dashboard

Access visualization of all results:

```bash
python main.py --phase dashboard
# Opens browser to http://localhost:8501

# Pages:
# 1. Overview - dataset stats, class distribution
# 2. Baseline Results - baseline metrics and score distribution
# 3. Optimization Progress - accuracy curve, Pareto frontier, cost tracking
# 4. Prompt Evolution - side-by-side baseline vs optimized prompts
# 5. Example Evaluations - before/after scores for sample applicants
```

---

## Troubleshooting

### Error: AWS credentials not found

```
❌ Failed to initialize Bedrock client: ...
```

**Fix:**
```bash
# Check .env is loaded
echo $AWS_ACCESS_KEY_ID
# Should print your access key

# If empty, reload environment
set -a
source .env
set +a
```

### Error: Model not enabled in region

```
❌ Failed to initialize Bedrock client: Access Denied...
```

**Fix:**
1. Go to AWS Console → Bedrock → Model access
2. Enable `Anthropic Claude 3.5 Sonnet` for your region
3. Wait 1-2 minutes for activation
4. Re-run

### Error: Budget exceeded mid-run

```
⚠️  Budget exhausted: $100.45 >= $100.00
```

**Fix:**
- Increase budget in `config.yaml`: `llm.budget_hard_cap_usd: 150.0`
- OR reduce iterations: `gepa.max_iterations: 15`
- OR reduce dataset size: `dataset.total: 100`
- Resume with: `python main.py --phase optimize --resume`

### Error: Parse error in LLM responses

```
Failed to parse resume batch 0: {invalid json}
```

**Why**: LLM occasionally adds text around JSON.

**Fix**: Automatic fallback returns neutral score. If frequent:
1. Reduce `llm.max_tokens_generation` to force concise responses
2. Add stricter JSON-forcing prompt prefix
3. Use higher model temperature for clarity

---

## Test Cases & Validation

Run the 5 validation tests:

```bash
python tests/test_integration.py -v
```

See detailed test cases in next section below.

---

## Detailed Test Cases (Proof of Correctness)

### Test Case 1: Database Schema & Idempotency
**Validates**: Schema creation, idempotent operations, data integrity.

```bash
pytest tests/test_integration.py::TestDatabaseSchema -v
```

**What it proves:**
- ✅ SQLite schema creates successfully
- ✅ Helper functions work correctly
- ✅ idempotent operations (insert twice → same data)
- ✅ Foreign key constraints enforced

### Test Case 2: Config Loading & Validation
**Validates**: Configuration system works, defaults load correctly.

```bash
pytest tests/test_integration.py::TestConfigLoading -v
```

**What it proves:**
- ✅ config.yaml parses correctly
- ✅ All required fields present
- ✅ Typed dataclasses validate structure
- ✅ Baseline prompts have correct format

### Test Case 3: Synthetic Data Generation & Oracle
**Validates**: Resume generation produces valid data, oracle is deterministic.

```bash
pytest tests/test_integration.py::TestSyntheticData -v
```

**What it proves:**
- ✅ Claude generates valid resume JSON
- ✅ Resumes contain required fields (name, experience, skills, etc.)
- ✅ Oracle labels are reproducible (same resume → same label)
- ✅ Oracle adds secondary signal (mentoring boost)
- ✅ Train/test split is stratified

### Test Case 4: Metrics Calculation & Evaluation
**Validates**: Metrics are computed correctly, edge cases handled.

```bash
pytest tests/test_integration.py::TestMetrics -v
```

**What it proves:**
- ✅ Accuracy = (correct predictions) / (total)
- ✅ Precision = TP / (TP + FP) for "hired" class
- ✅ Recall = TP / (TP + FN) for "hired" class
- ✅ F1 score correctly combines precision/recall
- ✅ Spearman ρ correlation calculated correctly
- ✅ Edge case: all-negative predictor doesn't crash

### Test Case 5: GEPA Loop & Pareto Frontier
**Validates**: GEPA optimization runs, candidates improve, Pareto is correct.

```bash
pytest tests/test_integration.py::TestGEPALoop -v
```

**What it proves:**
- ✅ Pareto dominance logic is correct
- ✅ Frontier max size enforced (≤20 candidates)
- ✅ Checkpoint/resume preserves state
- ✅ Reflection produces valid mutated prompts
- ✅ Cost tracking prevents budget overrun

---

## Smoke Test Detailed Walkthrough

**What it does**: Runs full pipeline on minimal data.

```bash
python main.py --phase smoke-test 2>&1 | tee smoke_test.log
```

**Step-by-step:**

1. **Data Generation** (30 seconds)
   ```
   Generating 10 synthetic resumes...
   Generated 10 resumes
   ```
   ✅ Validates: Resume generation works, oracle labeling works

2. **Baseline Evaluation** (15 seconds)
   ```
   Evaluating baseline on training set...
   Evaluating baseline on holdout set...
   ✅ Baseline Evaluation Results:
     Train accuracy: 0.571
     Train precision (hired): 0.500
     Holdout accuracy: 0.667
     Holdout precision (hired): 0.500
   ```
   ✅ Validates: Prompts execute, scores computed, metrics calculated

3. **GEPA Optimization** (45 seconds)
   ```
   === Iteration 1/3 ===
   Selected parent 1 (acc=0.571, prec=0.500)
   Found 3 failure cases
   Metrics: acc=0.571, prec=0.600
   Pareto pool size: 1
   
   === Iteration 2/3 ===
   Selected parent 2 (acc=0.571, prec=0.600)
   Found 2 failure cases
   Metrics: acc=0.714, prec=0.667
   Pareto pool size: 2
   
   === Iteration 3/3 ===
   Metrics: acc=0.714, prec=0.750
   Pareto pool size: 2
   ```
   ✅ Validates: GEPA loop runs, Pareto frontier updates, candidates improve

4. **Evaluation** (15 seconds)
   ```
   ============================================================
   HOLDOUT SET COMPARISON
   ============================================================
   
   Baseline: Accuracy: 0.667
   Optimized: Accuracy: 0.667
   Improvement: +0.0%
   
   ⚠️  Below target: 0.0% improvement (target: ≥15%)
   ```
   ✅ Validates: Holdout evaluation works, comparison printed
   ⚠️ Note: On small dataset (3 holdout), improvement may not meet 15% threshold
   
   **This is OK** — smoke test proves the pipeline works, not that it reaches targets.

**Total time**: ~2 minutes
**Total cost**: <$0.50
**Expected result**: All phases complete without errors ✅

---

## Full POC Run Expected Results

After `python main.py --phase all` on full 200-applicant dataset:

### Timeline
- Data prep: 5 minutes
- Baseline: 8 minutes  
- GEPA (25 iterations): 35-45 minutes
- Evaluation: 2 minutes
- **Total: ~50-60 minutes**

### Cost
- Baseline rollout (200×5): ~$2
- Seed generation: ~$0.50
- GEPA rollouts (25×150×5): ~$37
- Reflection calls (25): ~$1
- Holdout evaluation: ~$0.50
- **Total: ~$41 (within $100 cap)**

### Success Criteria
✅ **All** must pass:
1. Holdout accuracy improvement ≥15% vs baseline
2. Optimized prompts are human-readable
3. Reflection shows learning (not random changes)
4. System completes without manual intervention
5. Final prompts are exportable

### Example Final Results
```
Baseline (human-authored):
  Accuracy: 0.620
  Precision (hired): 0.571

Optimized (GEPA):
  Accuracy: 0.740
  Precision (hired): 0.690

Improvement:
  ✅ SUCCESS: 19.4% improvement (target: ≥15%)
```

---

## Debugging Commands

### View optimization status
```bash
sqlite3 data/gepa.db \
  "SELECT status, current_iteration, pareto_pool_ids FROM optimization_state;"
```

### View all prompts across iterations
```bash
sqlite3 data/gepa.db \
  "SELECT iteration, id, prompts_json FROM prompt_candidates ORDER BY iteration;"
```

### Check cost breakdown
```bash
sqlite3 data/gepa.db \
  "SELECT phase, COUNT(*) calls, ROUND(SUM(cost_usd),2) cost FROM llm_call_log GROUP BY phase;"
```

### View worst failures
```bash
sqlite3 data/gepa.db \
  "SELECT e.applicant_id, e.aggregate_score, e.predicted_hired, a.hired 
   FROM evaluations e JOIN applicants a ON e.applicant_id = a.id 
   WHERE e.candidate_id = 1 AND e.predicted_hired != a.hired
   ORDER BY ABS(e.aggregate_score - 3.0) DESC LIMIT 5;"
```

---

## Next Steps

1. **Pass smoke test** → Proves end-to-end connectivity
2. **Run full POC** → Achieves ≥15% improvement goal
3. **Review dashboard** → Understand prompt evolution
4. **Analyze failures** → Identify what GEPA learned
5. **Archive results** → Save final prompts for production POC phase

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI orchestrator, phase runner |
| `config.yaml` | Job description, baseline prompts, parameters |
| `data/gepa.db` | All state, results, logs |
| `src/database.py` | SQLite schema + 50+ helpers |
| `src/gepa/optimizer.py` | Main GEPA loop |
| `src/dashboard/app.py` | Streamlit visualization |
| `tests/test_integration.py` | Validation tests |

---

## FAQ

**Q: Can I stop and resume optimization?**
A: Yes! The system saves a checkpoint every 5 iterations. Use `--phase optimize --resume`.

**Q: What if baseline already performs well?**
A: The POC target is relative improvement (≥15%), not absolute accuracy. If baseline is 80%, target is 92%.

**Q: Can I use real resumes instead of synthetic?**
A: Yes. Drop PDF/DOCX files in `data/real/` and implement the resume parsing in `src/data_pipeline/resume_parser.py`.

**Q: How do I lower costs?**
A: Reduce `dataset.total` (e.g., 100), reduce `gepa.max_iterations` (e.g., 15), or increase `gepa.pareto_max_size` to reduce redundant evaluations.

**Q: What's the Pareto frontier for?**
A: It tracks candidates along 2 objectives (accuracy + precision_hired). This prevents the optimizer from finding a "useless" solution (e.g., "always reject" = high accuracy but zero precision).

---

## Success Checklist

Before declaring POC complete, verify:

- [ ] Smoke test passes (all phases, <$1, <2min)
- [ ] Full POC run completes (all 200 applicants, <60min, <$100)
- [ ] Holdout improvement ≥15% vs baseline
- [ ] Dashboard loads with 5 pages
- [ ] LLM call log tracks cost correctly
- [ ] Pareto frontier has ≥3 candidates
- [ ] Final prompts are human-readable
- [ ] Reflection shows learning (not random mutations)

---

**Good luck! 🚀**
