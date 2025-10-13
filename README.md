# MACI
This repository contains an anonymized version of our Multi-LLM Adaptive Conformal Inference experiments. The entry point is `experiments/run_experiment.py`.

## Abstract

Ensuring factuality is essential for the safe use of Large Language Models (LLMs) in high-stakes domains such as medicine and law. Conformal inference provides distribution-free guarantees, but existing approaches are either overly conservative, discarding many true-claims, or rely on adaptive error rates and simple linear models that fail to capture complex group structures. To address these challenges, we reformulate conformal inference in a multiplicative filtering setting, modeling factuality as a product of claim-level scores. Our method, Multi-LLM Adaptive Conformal Inference (MACI), leverages ensembles to produce more accurate factuality-scores, which in our experiments led to higher retention, while validity is preserved through group-conditional calibration. Experiments show that MACI consistently achieves user-specified coverage with substantially higher retention and lower time cost than baselines.

## Running

Step 1) Create a fresh Conda environment (Python 3.9)

```bash
conda create -y -n maci python=3.9
```

Step 2) Install dependencies from requirements.txt

```bash
conda run -n maci \
  python -m pip install -r requirements.txt --no-input
```

Step 3) Prepare data layout (repo-relative defaults)

- Place data under `data/` in the repository root (or pass `--data-dir`).
- For MedLFQA: put files under `data/med_scores/`.
- For WikiBio: put files under `data/wiki_scores/`.

Step 4) Run a quick experiment (MedLFQA example)

```bash
conda run -n maci \
  python experiments/run_experiment.py \
  --dataset-type medlfqa \
  --conditional-groups false_claim_risk \
```

Step 5) Where outputs go

- Logs: `logs/` (repo-root-relative by default)
- Results JSON: `analysis/experiment_results/`



## CCI Attribution
Our implementation of the Conditional Conformal Inference (CCI) baseline is a direct adoption of the work from the [conformal-safety](https://github.com/jjcherian/conformal-safety.git) repository. To ensure full reproducibility, we have included a local copy of the necessary modules in the conditional-conformal/ directory. We explicitly state that the code within this directory is not the work of the MACI project. For all details, please refer to the original repository: [conformal-safety](https://github.com/jjcherian/conformal-safety.git)
