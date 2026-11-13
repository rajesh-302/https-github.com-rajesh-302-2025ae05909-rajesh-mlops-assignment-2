# EduAssist AI - MLOps Assignment 2
**API Cost, Latency and Prompt Version Management with MLflow**
Course: MLOps · Student: ANDRA RAJESH · BITS ID: 2025AE05909

EduAssist AI is an academic question answering assistant that must return strict JSON.
This project calls an LLM through an API for five academic questions, measures the
latency of every request, extracts token usage, computes request-level and projected
monthly cost, registers three prompt versions in the **MLflow Prompt Registry**,
evaluates them on the same questions (including a `valid_json` metric) and promotes
the best version to the alias **`production`**.

## Repository structure
```
notebooks/2025ae05909_mlops_assignment_2.ipynb   executable notebook (Tasks 1-3)
notebooks/2025ae05909_mlops_assignment_2.html    exported, executed notebook
src/eduassist/                       Python package used by the notebook
  config.py                          settings from environment variables, paths, prompt definitions
  llm_client.py                      API call, latency measurement, token usage, retries
  metrics.py                         latency statistics, cost formulas, valid_json and schema checks
  pipeline.py                        runs the questions, MLflow runs, mlflow.genai.evaluate scorers
  prompt_registry.py                 prompt registration, diff, best-version choice, alias
  workflow.py                        Task 1-2 and Task 3 orchestration, CSV outputs
prompts/                             v1_basic.txt, v2_structured.txt, v3_json.txt (assignment text)
data/questions.json                  five academic questions + expected key concepts
scripts/                             session_setup.sh, start_mlflow.sh, export_notebook.sh, make_submission.sh
outputs/                             CSV / JSON results (see "Outputs")
screenshots/                         MLflow screenshots
report/                              final report (PDF)
```

## Requirements
* **Python 3.12** (Prayogshala lab: Miniconda Python 3.12.9)
* Libraries (pinned in `requirements.txt`): mlflow 3.18.0, openai 3.45.0, pandas 3.1.1,
  ipykernel 7.4.0, nbconvert 7.17.2, nbformat 5.11.1
* An LLM API key. OpenAI, Google Gemini and Groq are all called through the
  OpenAI-compatible Chat Completions API using the `openai` Python client.

## Environment variables
Create `.env` from the template (`cp .env.example .env`). `.env` is git-ignored and must never be committed or shared.

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `openai`, `gemini`, `groq` or `ollama` |
| `LLM_MODEL` | model name exactly as listed by the provider |
| `LLM_API_KEY` | secret API key (also accepted: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`) |
| `LLM_BASE_URL` | optional custom OpenAI-compatible endpoint (empty = provider default) |
| `LLM_TEMPERATURE` | sampling temperature (default 0) |
| `LLM_MAX_TOKENS` | maximum output tokens per request (default 400) |
| `LLM_REQUEST_DELAY` | pause between calls in seconds (keeps free tiers under rate limits) |
| `PRICE_INPUT_PER_M` | published price per 1,000,000 input tokens |
| `PRICE_OUTPUT_PER_M` | published price per 1,000,000 output tokens |
| `PRICE_CURRENCY` | currency of the prices, e.g. USD |
| `PRICE_DATE` | date the prices were read (YYYY-MM-DD) |
| `PRICE_SOURCE_URL` | provider pricing page |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` |

## Installation
```bash
git clone https://github.com/rajesh-302/https-github.com-rajesh-302-2025ae05909-rajesh-mlops-assignment-2.git
cd https-github.com-rajesh-302-2025ae05909-rajesh-mlops-assignment-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name llmops --display-name "Python (llmops)"
cp .env.example .env          # then edit .env: provider, model, key, prices, date
```
On the Prayogshala lab the environment steps are automated: `source scripts/session_setup.sh`
(virtual environment in `~/.venvs/llmops`, Jupyter kernel "Python (llmops)").

## MLflow UI
```bash
bash scripts/start_mlflow.sh
```
This starts, in the background:
`mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000`

Open **http://127.0.0.1:5000**:
* Experiments → `eduassist_llmops_assignment`
* Prompts → `eduassist-academic-qa` (versions 1–3, alias `production`)

## Steps to reproduce the experiment
1. Fill in `.env` and start MLflow (`bash scripts/start_mlflow.sh`).
2. Open `notebooks/2025ae05909_mlops_assignment_2.ipynb`, select the kernel **Python (llmops)** and run all cells.
   * Task 1–2: five requests with prompt v1 → latency table, statistics, tokens, cost, monthly projection.
   * Task 3: registers v1–v3, runs each on the same five questions in its own MLflow run, logs
     `avg_latency_seconds`, `avg_input_tokens`, `avg_output_tokens`, `avg_cost_per_request`,
     `valid_json`, compares the versions and sets the alias `production`.
   * One complete run makes 20 API calls (5 + 3 × 5).
3. Export the executed notebook: `bash scripts/export_notebook.sh`.
4. Optional: build the submission ZIP: `bash scripts/make_submission.sh`.

Re-running the notebook does not create duplicate prompt versions, but it adds new MLflow runs
and re-evaluates the alias.

## Outputs (`outputs/`)
| File | Content |
|---|---|
| `latency_tokens_cost.csv` | Task 1–2, one row per request: question ID, prompt version, model, start/end time, latency, status, input/output/total tokens, input/output/total cost |
| `latency_summary.csv` | mean, median, min, max and standard deviation of latency |
| `prompt_evaluation_results.csv` | Task 3 comparison table per prompt version (incl. `valid_json`, recommended) |
| `prompt_evaluation_requests.csv` | every Task 3 request incl. response text and validity flags |
| `prompt_diff_v1_v2.txt`, `prompt_diff_v2_v3.txt` | text diffs between prompt versions |
| `run_metadata.json` | provider, model, API version, pricing, pricing date, currency, MLflow version, run time |

## Model, API version and pricing used
* Model: [ ]  (as returned by the API: [ ])
* API: [ ]  (e.g. OpenAI Chat Completions via openai-python 3.45.0)
* Pricing: [ ] per 1M input tokens, [ ] per 1M output tokens, currency [ ], read on [YYYY-MM-DD] from [URL]

These values are also recorded automatically in `outputs/run_metadata.json` and as MLflow run parameters.

## Cost formulas
* Input cost = input tokens / 1,000,000 × input price per million
* Output cost = output tokens / 1,000,000 × output price per million
* Total cost per request = input cost + output cost
* Projected monthly cost = average cost per request × 10,000

## Security
* The API key is read only from the environment variable `LLM_API_KEY` (loaded from the
  git-ignored `.env`). It is never hard-coded, printed or logged; only non-secret settings are
  logged to MLflow.
* `.env.example` contains no secrets.
* `scripts/make_submission.sh` refuses to build the ZIP if a file contains something that looks like an API key.