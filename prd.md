# Product Requirements Document (PRD)
## Self-Learning Talent Lens Optimization System - Proof of Concept

---

## Document Control

| Field | Value |
|-------|-------|
| **Document Version** | 1.0 |
| **Status** | Draft for Review |
| **Date** | May 18, 2026 |
| **Project Code** | POC-GEPA-TL-001 |
| **Author** | [Your Name] |
| **Stakeholders** | Engineering Lead, Product Manager, HR Tech Partner |

---

## 1. Executive Summary

### 1.1 Purpose
This PRD defines the scope, requirements, and success criteria for a Proof of Concept (POC) that validates the feasibility of using GEPA (Genetic-Pareto optimization) to automatically generate optimal Talent Lens evaluation prompts for job applicant screening.

### 1.2 POC Objective
**Prove that**: An AI-driven optimization system can discover 5 evaluation prompts that match historical hiring decisions better than manually-authored prompts, using a single client's real historical data.

### 1.3 Success Definition
The POC is successful if the optimized prompts achieve **≥15% improvement** in agreement with historical hiring decisions compared to human-authored baseline prompts.

### 1.4 Timeline & Budget
- **Duration**: 4 weeks
- **Team Size**: 2-3 engineers (1 ML engineer, 1 backend engineer, 0.5 DevOps)
- **Estimated Cost**: $2,000-$5,000 (primarily AWS/Bedrock inference costs)

---

## 2. Background & Context

### 2.1 Current State Problem
**Scenario**: A tech company receives 500 applications for Software Engineer positions. HR manually created 5 evaluation criteria:
1. Technical depth in required languages
2. Problem-solving ability
3. Cultural fit indicators
4. Communication skills from writing samples
5. Career trajectory and growth potential

**Pain Points**:
- Taking 2-3 weeks to calibrate these prompts
- Prompts drift from actual hiring decisions (hired candidates don't always score highest)
- No systematic way to improve based on feedback
- Each new role requires starting from scratch

### 2.2 Proposed Solution (POC Scope)
Build a minimal system that:
1. Takes 200 historical applicant resumes + hiring outcomes (hired/rejected)
2. Takes a job description and high-level selection criteria
3. Automatically generates 5 optimized evaluation prompts
4. Demonstrates superior performance vs. human baseline

---

## 3. POC Scope

### 3.1 In-Scope

#### Core Features
1. **Data Preparation Pipeline**
   - Ingest 200 historical resumes (PDF/DOCX format)
   - Parse into structured JSON (name, experience, skills, education, etc.)
   - Load hiring outcomes (binary: hired=1, rejected=0)
   - Split into training (150) and holdout (50) sets

2. **Baseline Evaluation System**
   - Implement 5 human-authored Talent Lens prompts
   - Run baseline prompts on all 200 applicants
   - Score: 1-5 scale + text rationale
   - Calculate agreement metrics with historical decisions

3. **GEPA Optimization Engine**
   - Implement simplified GEPA loop (using `gepa` library or custom)
   - Generate seed prompts from job description
   - Evolve prompts over 20-30 iterations
   - Track Pareto frontier of prompt candidates

4. **Evaluation & Comparison**
   - Run optimized prompts on holdout set
   - Compare metrics: accuracy, ranking correlation, F1 score
   - Generate side-by-side comparison report

5. **Simple Dashboard**
   - Show optimization progress (fitness over iterations)
   - Display final vs. baseline prompt sets
   - Show example applicant evaluations (before/after)

#### Technical Stack
- **LLM Provider**: Amazon Bedrock (Claude 3.5 Sonnet for rollouts, Opus for reflection)
- **Compute**: Local Python environment OR single EC2 instance
- **Storage**: Local filesystem OR S3 bucket
- **Framework**: Python 3.10+, `gepa` library (or `dspy.GEPA`)
- **Parsing**: PyPDF2/python-docx for resume parsing

### 3.2 Out-of-Scope (Deferred to Production)

❌ **Not Included in POC**:
- Multi-tenancy (single client only)
- Production AWS architecture (Fargate, DynamoDB, etc.)
- Real-time API endpoints
- UI for prompt editing
- Continuous learning/retraining
- Integration with existing ATS systems
- Authentication/authorization
- Cost optimization (Bedrock Batch)
- Advanced analytics dashboard
- Compliance/audit logging
- Weighted scoring (uniform weights only in POC)

### 3.3 POC Assumptions & Constraints

**Assumptions**:
1. Historical data available for at least 200 applicants
2. Ground truth labels are reasonably reliable
3. Resumes are in English, standard format
4. Access to Amazon Bedrock API
5. Single job role (e.g., "Software Engineer")

**Constraints**:
- Budget cap: $5,000 total AWS spend
- Rollout budget: Max 10,000 LLM calls per optimization run
- No PII data leaves development environment
- Manual resume parsing acceptable (no advanced NLP)

---

## 4. Functional Requirements

### 4.1 Data Ingestion Module

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1.1 | System shall accept PDF/DOCX resumes | Must Have | 95%+ parse success rate |
| FR-1.2 | System shall extract: name, experience, skills, education | Must Have | Fields populated for 90%+ resumes |
| FR-1.3 | System shall load binary hiring labels (CSV format) | Must Have | 100% label match with resumes |
| FR-1.4 | System shall perform train/test split (75/25) | Must Have | Stratified split maintains class balance |

**Input Format Example**:
```
applicants/
├── resume_001.pdf
├── resume_002.docx
└── ...
labels.csv:
applicant_id, hired
resume_001, 1
resume_002, 0
```

### 4.2 Baseline Prompt Execution Module

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-2.1 | System shall execute 5 predefined baseline prompts | Must Have | All 200 applicants scored |
| FR-2.2 | Each prompt returns score (1-5) + rationale text | Must Have | Valid JSON output 100% |
| FR-2.3 | System shall aggregate scores (simple average) | Must Have | Final score ∈ [1,5] |
| FR-2.4 | System shall calculate baseline accuracy | Must Have | Accuracy metric on holdout set |

**Example Baseline Prompt**:
```
You are evaluating a software engineer applicant.
Assess their TECHNICAL DEPTH in Python/Java.
Return:
- score: 1-5 (1=beginner, 5=expert)
- rationale: 2-3 sentences explaining your score

Applicant Profile:
{resume_json}
```

### 4.3 GEPA Optimization Module

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-3.1 | System shall generate seed prompts from job description | Must Have | 5 valid prompts created |
| FR-3.2 | System shall perform reflective mutation | Must Have | New prompts reference failure cases |
| FR-3.3 | System shall maintain Pareto frontier (≤20 candidates) | Must Have | Frontier updated each iteration |
| FR-3.4 | System shall run 20-30 optimization iterations | Must Have | Convergence or budget exhaustion |
| FR-3.5 | System shall checkpoint progress every 5 iterations | Should Have | Resume-able from checkpoint |

**GEPA Loop Pseudocode**:
```python
1. seed_candidate = generate_seed(job_description, examples=5)
2. pareto_pool = [seed_candidate]
3. for iteration in range(30):
    a. parent = select_from_pareto(pareto_pool)
    b. failures = get_worst_cases(parent, training_data)
    c. child = reflect_and_mutate(parent, failures)
    d. evaluate(child, training_data)
    e. update_pareto_pool(child)
    f. log_progress()
4. best = select_best(pareto_pool, holdout_data)
```

### 4.4 Evaluation & Reporting Module

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-4.1 | System shall compute accuracy on holdout set | Must Have | Baseline vs. optimized comparison |
| FR-4.2 | System shall compute ranking correlation (Spearman's ρ) | Should Have | Measures ranking quality |
| FR-4.3 | System shall compute precision/recall for "hired" class | Should Have | Binary classification metrics |
| FR-4.4 | System shall generate comparison report (PDF/HTML) | Must Have | Executive summary + details |
| FR-4.5 | System shall show example applicant evaluations | Should Have | 5 examples with before/after |

**Key Metrics**:
- **Accuracy**: % of applicants correctly classified (hired/rejected)
- **Spearman's ρ**: Rank correlation between predicted scores and outcomes
- **Precision/Recall**: For "hired" class prediction
- **Agreement Score**: Mean per-applicant agreement

### 4.5 Observability & Logging

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-5.1 | System shall log every LLM call (prompt, response, latency) | Must Have | Debug trace available |
| FR-5.2 | System shall track total cost (per-call and cumulative) | Must Have | Live budget monitoring |
| FR-5.3 | System shall output iteration metrics (CSV format) | Should Have | Plot fitness curves |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| NFR-1.1 | Optimization run completion time | ≤ 2 hours | Wall-clock end-to-end |
| NFR-1.2 | Resume parsing throughput | ≥ 10 resumes/min | Batch processing |
| NFR-1.3 | LLM call latency (p95) | ≤ 10 seconds | Bedrock on-demand |

### 5.2 Cost Efficiency

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| NFR-2.1 | Total POC AWS spend | ≤ $5,000 | Billing dashboard |
| NFR-2.2 | Cost per optimization run | ≤ $100 | Calculated from logs |
| NFR-2.3 | LLM calls per run | ≤ 10,000 | Counter in code |

### 5.3 Reliability

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| NFR-3.1 | Resume parsing success rate | ≥ 95% | Parsed/total ratio |
| NFR-3.2 | LLM call retry success rate | ≥ 99% | After 3 retries |
| NFR-3.3 | Checkpoint recovery success | 100% | Manual test |

### 5.4 Security (POC-level)

| ID | Requirement | Priority | Implementation |
|----|-------------|----------|----------------|
| NFR-4.1 | PII data shall not be logged to external systems | Must Have | Local-only logging |
| NFR-4.2 | Resume data shall be encrypted at rest | Should Have | S3 encryption enabled |
| NFR-4.3 | API keys shall be stored in environment variables | Must Have | No hardcoded secrets |

---

## 6. Data Requirements

### 6.1 Historical Dataset Specification

**Minimum Requirements**:
- **Volume**: 200 labeled applicants
  - Training: 150 (75%)
  - Holdout: 50 (25%)
- **Label Distribution**: Approximately balanced (40-60% hired ratio)
- **Data Fields** (per applicant):
  - Resume file (PDF/DOCX)
  - Hiring outcome (binary)
  - Application date (for temporal split)
  - Job role (if multi-role, filter to one)

**Data Quality Requirements**:
- Resumes must be machine-readable (not scanned images)
- Labels must be ground truth (actual hiring decisions)
- No duplicates
- Consistent job role across dataset

### 6.2 Job Description & Criteria Input

**Format**:
```yaml
job_title: "Senior Software Engineer"
company: "TechCorp Inc."
required_skills:
  - Python (3+ years)
  - System design
  - API development
preferred_skills:
  - AWS/cloud experience
  - Team leadership
selection_criteria: |
  We value candidates who demonstrate:
  1. Strong technical problem-solving
  2. Clear communication in writing
  3. Track record of shipping products
  4. Alignment with our collaborative culture
  5. Growth mindset and continuous learning
```

### 6.3 Synthetic Data (Fallback)

If real data unavailable, generate synthetic dataset:
- Use GPT-4/Claude to generate 200 realistic resumes
- Simulate hiring decisions with a hidden "oracle" prompt
- Ensures POC can proceed without data blockers

---

## 7. Technical Architecture (POC-Simplified)

### 7.1 System Components

```
┌─────────────────────────────────────────────────────────┐
│                     POC System                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Data Prep  │─────→│  Baseline    │               │
│  │   Pipeline   │      │  Evaluation  │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                        │
│         ↓                      ↓                        │
│  ┌──────────────────────────────────┐                 │
│  │     GEPA Optimization Loop       │                 │
│  │  ┌────────┐  ┌────────┐  ┌─────┐│                 │
│  │  │Rollout │→ │Reflect │→ │Mutate││                 │
│  │  └────────┘  └────────┘  └─────┘│                 │
│  │       ↑                      ↓    │                 │
│  │       └──────Pareto Pool────┘    │                 │
│  └──────────────────────────────────┘                 │
│         │                                              │
│         ↓                                              │
│  ┌──────────────┐                                     │
│  │  Evaluation  │                                     │
│  │  & Reporting │                                     │
│  └──────────────┘                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
           │
           ↓
    ┌─────────────┐
    │   Bedrock   │
    │ (Claude API)│
    └─────────────┘
```

### 7.2 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Language** | Python 3.10+ | Ecosystem support, GEPA library |
| **LLM API** | Amazon Bedrock | Claude 3.5 Sonnet/Opus |
| **Optimization** | `gepa` library OR custom | Reference implementation |
| **Resume Parsing** | PyPDF2, python-docx | Lightweight, sufficient for POC |
| **Data Storage** | Local filesystem / S3 | Simple, low overhead |
| **Orchestration** | Single Python script | No complex workflow needed |
| **Visualization** | Matplotlib, Plotly | Progress tracking |

### 7.3 Key Libraries

```python
# requirements.txt
anthropic>=0.18.0          # Bedrock SDK
gepa>=0.2.0                # GEPA library (if using)
PyPDF2>=3.0.0              # PDF parsing
python-docx>=0.8.11        # DOCX parsing
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical ops
scikit-learn>=1.3.0        # Metrics calculation
matplotlib>=3.7.0          # Plotting
pyyaml>=6.0                # Config files
tqdm>=4.65.0               # Progress bars
```

---

## 8. Success Metrics & Acceptance Criteria

### 8.1 Primary Success Metric

**Metric**: Improvement in holdout set accuracy over baseline

**Target**: ≥ 15% relative improvement

**Calculation**:
```
Baseline Accuracy: 65% (example)
Optimized Accuracy: 75%
Improvement: (75-65)/65 = 15.4% ✓ SUCCESS
```

### 8.2 Secondary Success Metrics

| Metric | Baseline Target | Optimized Target | Priority |
|--------|----------------|------------------|----------|
| **Spearman's ρ** | 0.40 | 0.55+ | High |
| **Precision (hired)** | 0.60 | 0.70+ | Medium |
| **Recall (hired)** | 0.70 | 0.80+ | Medium |
| **Runtime** | N/A | ≤ 2 hours | High |
| **Cost per run** | N/A | ≤ $100 | High |

### 8.3 Qualitative Acceptance Criteria

✅ **Must Demonstrate**:
1. Optimized prompts are human-readable and interpretable
2. Reflection step shows meaningful learning (not random changes)
3. Pareto frontier preserves specialist prompts
4. System completes without manual intervention
5. Final prompts are exportable for production use

✅ **Evidence Required**:
- Side-by-side comparison report (PDF)
- Iteration progress charts (fitness over time)
- Example applicant evaluations showing improvement
- Final prompt set with rationale for each lens

---

## 9. POC Deliverables

### 9.1 Code Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| **Data Pipeline** | Resume parsing + dataset prep | Python module |
| **Baseline Evaluator** | Human-authored prompts executor | Python module |
| **GEPA Engine** | Optimization loop implementation | Python module |
| **Evaluation Script** | Metrics calculation & comparison | Python script |
| **Main Orchestrator** | End-to-end runner | Python script (`main.py`) |
| **Configuration** | Job description, parameters | YAML file |
| **Requirements** | Dependencies | `requirements.txt` |

### 9.2 Documentation Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| **Setup Guide** | Installation & configuration | Markdown |
| **Usage Guide** | How to run POC end-to-end | Markdown |
| **Results Report** | Metrics, findings, recommendations | PDF/HTML |
| **Architecture Diagram** | POC system components | PNG/SVG |
| **Lessons Learned** | Challenges, surprises, insights | Markdown |

### 9.3 Results Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| **Baseline Results** | Human-authored prompt performance | JSON + report |
| **Optimized Results** | GEPA-discovered prompt performance | JSON + report |
| **Comparison Report** | Side-by-side metrics & analysis | PDF |
| **Prompt Artifacts** | Final 5 optimized prompts | YAML/JSON |
| **Iteration Logs** | Full optimization trace | CSV + plots |

---

## 10. Test Plan

### 10.1 Unit Tests

| Test ID | Component | Test Case | Success Criteria |
|---------|-----------|-----------|------------------|
| UT-1 | Resume Parser | Parse valid PDF | Extracts all fields |
| UT-2 | Resume Parser | Handle malformed PDF | Graceful error |
| UT-3 | Prompt Executor | Run single prompt | Valid JSON response |
| UT-4 | Metrics Calculator | Compute accuracy | Correct calculation |
| UT-5 | GEPA Selector | Select from Pareto | Valid parent chosen |

### 10.2 Integration Tests

| Test ID | Workflow | Test Case | Success Criteria |
|---------|----------|-----------|------------------|
| IT-1 | End-to-End | Full pipeline on 10 samples | Completes without error |
| IT-2 | Baseline Run | Execute baseline on 50 samples | Metrics generated |
| IT-3 | GEPA Loop | Run 5 optimization iterations | Frontier updates |
| IT-4 | Checkpoint | Stop and resume mid-run | State preserved |

### 10.3 Validation Tests

| Test ID | Validation | Test Case | Success Criteria |
|---------|-----------|-----------|------------------|
| VT-1 | Data Quality | Check label distribution | Balanced classes |
| VT-2 | Prompt Quality | Validate JSON output schema | 100% compliance |
| VT-3 | Cost Tracking | Monitor LLM call count | Within budget |
| VT-4 | Result Quality | Holdout set improvement | ≥15% improvement |

---

## 11. POC Execution Plan

### 11.1 Phase Breakdown

#### **Phase 1: Setup & Data Prep (Week 1)**

**Objectives**:
- Set up development environment
- Acquire/generate dataset
- Implement data pipeline

**Tasks**:
1. Set up Python environment, install dependencies
2. Configure Amazon Bedrock API access
3. Collect 200 historical resumes + labels
4. Implement resume parser
5. Create train/test split
6. Validate data quality

**Deliverables**:
- Working data pipeline
- Parsed dataset (JSON format)
- Data quality report

**Success Criteria**:
- 95%+ resumes successfully parsed
- Balanced train/test split
- All labels matched

---

#### **Phase 2: Baseline Implementation (Week 1-2)**

**Objectives**:
- Implement human-authored prompts
- Execute baseline evaluation
- Establish performance benchmark

**Tasks**:
1. Author 5 baseline Talent Lens prompts
2. Implement prompt execution wrapper (Bedrock API)
3. Run baseline on all 200 applicants
4. Calculate metrics (accuracy, correlation, F1)
5. Generate baseline report

**Deliverables**:
- 5 baseline prompts (documented)
- Baseline evaluation results
- Baseline metrics report

**Success Criteria**:
- All 200 applicants scored
- Baseline accuracy: 55-70% (realistic range)
- Metrics calculated correctly

---

#### **Phase 3: GEPA Implementation (Week 2-3)**

**Objectives**:
- Implement GEPA optimization loop
- Run optimization on training set
- Track convergence

**Tasks**:
1. Implement seed prompt generation
2. Implement rollout executor
3. Implement reflection module (meta-LLM)
4. Implement Pareto frontier selector
5. Implement mutation operator
6. Run full optimization (20-30 iterations)
7. Checkpoint progress every 5 iterations

**Deliverables**:
- Working GEPA implementation
- Optimization logs (CSV)
- Pareto frontier snapshots
- Final optimized prompt set

**Success Criteria**:
- Optimization completes in ≤2 hours
- LLM calls ≤10,000
- Pareto pool size ≤20
- Reflection shows meaningful learning

---

#### **Phase 4: Evaluation & Reporting (Week 3-4)**

**Objectives**:
- Evaluate optimized prompts on holdout
- Compare against baseline
- Generate final report

**Tasks**:
1. Run optimized prompts on holdout set
2. Calculate all metrics (accuracy, ρ, precision, recall)
3. Generate comparison visualizations
4. Create example applicant evaluations
5. Write final POC report
6. Document lessons learned

**Deliverables**:
- Holdout evaluation results
- Comparison report (PDF/HTML)
- Example evaluations (5+ samples)
- Final POC presentation deck
- Recommendations for production

**Success Criteria**:
- Optimized accuracy ≥15% improvement
- Report clearly shows value
- Prompts are human-readable
- Recommendations actionable

---

### 11.2 Resource Allocation

| Role | Allocation | Responsibilities |
|------|------------|------------------|
| **ML Engineer** | 80% (3.2 weeks) | GEPA implementation, optimization tuning |
| **Backend Engineer** | 60% (2.4 weeks) | Data pipeline, Bedrock integration, metrics |
| **DevOps/Infra** | 20% (0.8 weeks) | AWS setup, monitoring, cost tracking |
| **Product/PM** | 10% (0.4 weeks) | Requirements validation, report review |

### 11.3 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Dataset unavailable** | Medium | High | Use synthetic data generation |
| **API rate limits** | Low | Medium | Implement exponential backoff |
| **Poor baseline performance** | Low | Low | Adjust to relative improvement |
| **GEPA doesn't converge** | Medium | High | Add convergence diagnostics, tune hyperparameters |
| **Budget overrun** | Low | Medium | Hard caps in code, monitoring |
| **Parsing failures** | Medium | Low | Manual fallback for critical samples |

---

## 12. Budget Estimate

### 12.1 AWS/Bedrock Costs

| Item | Volume | Unit Cost | Total |
|------|--------|-----------|-------|
| **Baseline Evaluation** | 1,000 calls (200×5) | $0.015/call | $15 |
| **GEPA Rollouts** | 7,500 calls (25 iter×300 rollouts) | $0.015/call | $113 |
| **Reflection Calls** | 500 calls (25 iter×20 reflections) | $0.075/call (Opus) | $38 |
| **Seed Generation** | 50 calls | $0.075/call | $4 |
| **S3 Storage** | 10 GB | $0.023/GB/month | $0.23 |
| **EC2 (optional)** | t3.large × 20 hours | $0.0832/hour | $1.66 |
| **Contingency (20%)** | | | $34 |
| **TOTAL** | | | **~$206** |

### 12.2 Labor Costs (Illustrative)

| Role | Hours | Rate | Total |
|------|-------|------|-------|
| ML Engineer | 128 | $100/hr | $12,800 |
| Backend Engineer | 96 | $90/hr | $8,640 |
| DevOps | 32 | $80/hr | $2,560 |
| **TOTAL LABOR** | | | **$24,000** |

**Total POC Cost**: ~$24,200 (labor) + $206 (AWS) = **$24,406**

---

## 13. Go/No-Go Decision Criteria

### 13.1 POC Success Criteria (Go to Production)

✅ **Must Achieve**:
1. ≥15% improvement in holdout accuracy
2. Runtime ≤2 hours per optimization
3. Cost per run ≤$100
4. Prompts are interpretable and production-ready

✅ **Should Achieve**:
1. Spearman's ρ improvement ≥0.10
2. No manual intervention required during optimization
3. Reflection demonstrates learning (not random)

### 13.2 Production Readiness Checklist

After POC success, production requires:
- [ ] Multi-tenant architecture design
- [ ] Production AWS infrastructure (Fargate, DynamoDB)
- [ ] API endpoints for triggering/monitoring runs
- [ ] UI for prompt review and approval
- [ ] Cost optimization (Bedrock Batch where applicable)
- [ ] Security audit and compliance review
- [ ] Integration with existing ATS
- [ ] Monitoring and alerting
- [ ] Documentation and runbooks

---

## 14. Next Steps After POC

### 14.1 If POC Succeeds

1. **Immediate** (Week 5):
   - Present findings to stakeholders
   - Get approval for production development
   - Define production feature set

2. **Short-term** (Weeks 6-10):
   - Implement production architecture (per original doc)
   - Add multi-tenancy
   - Build operator UI

3. **Medium-term** (Weeks 11-16):
   - Pilot with 3-5 real clients
   - Gather feedback and iterate
   - Expand to more clients

### 14.2 If POC Fails

**Failure Analysis**:
- Was improvement <15%? → Re-evaluate baseline quality, try different metrics
- Did GEPA not converge? → Tune hyperparameters, increase budget
- Were prompts uninterpretable? → Add human-in-loop validation

**Pivot Options**:
1. Hybrid approach: GEPA suggests, human refines
2. Simpler optimization (grid search over templates)
3. Focus on prompt refinement tool vs. full automation

---

## 15. Appendices

### Appendix A: Sample Baseline Prompts

```yaml
lens_1:
  name: "Technical Depth"
  prompt: |
    Evaluate the applicant's technical depth in {required_skills}.
    Score 1-5 based on:
    - Years of experience
    - Breadth of technologies
    - Depth in core areas
    Return JSON: {"score": int, "rationale": string}

lens_2:
  name: "Problem-Solving"
  prompt: |
    Assess problem-solving ability from work history and projects.
    Look for evidence of:
    - Complex challenges tackled
    - Creative solutions
    - Impact delivered
    Return JSON: {"score": int, "rationale": string}

# ... (3 more lenses)
```

### Appendix B: Evaluation Metrics Formulas

**Accuracy**:
```
accuracy = (correct_predictions / total_predictions)
```

**Spearman's Rank Correlation**:
```
ρ = correlation(rank(predicted_scores), rank(actual_outcomes))
```

**Precision (Hired Class)**:
```
precision = true_positives / (true_positives + false_positives)
```

**Recall (Hired Class)**:
```
recall = true_positives / (true_positives + false_negatives)
```

### Appendix C: Example GEPA Reflection Prompt

```
You are optimizing a set of 5 applicant evaluation prompts.

Current Prompt Set:
{current_prompts}

Failure Cases (applicants where prediction was wrong):
{failure_trajectories}

For each failure, you see:
- Applicant profile
- Predicted score vs. actual outcome
- Each of the 5 prompt's scores and rationales

Task: Identify which prompt(s) are contributing to errors and propose
a specific edit to improve alignment with hiring decisions.

Return: {"prompt_index": int, "proposed_edit": string, "reasoning": string}
```

---

## 16. Sign-off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Product Owner** | | | |
| **Engineering Lead** | | | |
| **ML Lead** | | | |
| **Stakeholder** | | | |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-18 | [Your Name] | Initial draft |

---

**END OF DOCUMENT**
