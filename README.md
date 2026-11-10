# interface.ai — Time-Aware Knowledge System

A deterministic, offline, evidence-grounded Python solution to the participant knowledge-system hackathon.

## Project structure

- `my_solution/solution.py` — `build()` creates a system exposing `ingest`, `answer`, and `conflicts`.
- `my_solution/requirements.txt` — runtime dependencies.
- `my_solution/NOTES.md` — technical design, authority rules, tradeoffs.
- `harness/` — evaluation harness supplied by the organizers.
- `eval/` — organizer-provided public 40-question evaluation.
- `snapshots/README.md` — instructions for the private website data.

**Website snapshots and API keys must never be pushed to the public repository.** The original participant kit ZIP is deliberately excluded from this package.

## Local setup (Python 3.11 recommended)

1. Extract this repository ZIP and the private participant kit ZIP on your computer.
2. Copy the five `*.zip` files from the original `snapshots_zip/` into `snapshots_zip/` at the top level of THIS repository (create the folder locally). The `.gitignore` excludes this folder.
3. Open a terminal in the repository root.
4. Install dependencies: `python -m pip install -r my_solution/requirements.txt`
5. Run the official public evaluation: `python harness/run_eval.py --submission my_solution --eval eval/public_eval.jsonl`
6. Run the synthetic contract test: `python harness/run_eval.py --submission my_solution --snapshots harness/selftest/snapshots --eval harness/selftest/eval.jsonl --conflicts-gold harness/selftest/conflicts_gold.json`
7. Review reports written under `results/` (ignored by Git).

## Local verification

- Public evaluation: **0.962** on the 40 provided practice questions (not the hidden test).
- Synthetic selftest: **1.000**, including the conflict-reporting contract.
- Errors: **0**. Median individual answer latency in the practice test was below 0.02 seconds; ingestion around 10.4 seconds on the testing computer.

These figures are local measurements, not a guarantee about performance on the employer's hidden evaluation.

## Submitting

Create a **public GitHub repository**, add the contents of this directory (not the private snapshots), and commit. Run `git rev-parse HEAD` to get the exact commit hash. Email the GitHub repository URL and commit hash to `assignments@interface.ai`. Do not submit until you understand and can explain the code.
