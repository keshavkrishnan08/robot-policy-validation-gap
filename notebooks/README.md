# Kaggle notebooks

Each notebook pulls the code from GitHub (`git clone`/`pull`), installs its own dependencies, and
runs one stage. Set `REPO_URL` to your fork in the first cell of each.

| Notebook | Accelerator | What it does | Time |
|----------|-------------|--------------|------|
| `00_smoke_test.ipynb` | **CPU** | preflight + Part A controlled validation + full synthetic smoke + pytest | minutes |
| `01_session1_libero.ipynb` | **GPU T4 x2** | train+metrics+rollouts for the 4 LIBERO tasks | ~1–4h |
| `02_session2_robomimic.ipynb` | **GPU T4 x2** | Robomimic tasks + full analysis + paper macros | ~1–4h |
| `03_partC_external.ipynb` | **GPU T4** | real-robot external validation (PyTorch policies) | ~3–4h |

Run `00` first (no GPU quota spent) to confirm the code is healthy on Kaggle. Then `01` → `02` → `03`.

## One-time GitHub setup (push this repo so the notebooks can pull it)
From the repo root locally:
```bash
gh repo create CORL --private --source . --remote origin   # or create on github.com and add the remote
git push -u origin main
```
Then put the repo's HTTPS URL into `REPO_URL` in each notebook's first cell. For a **private** repo on
Kaggle, add a token: `REPO_URL = "https://<TOKEN>@github.com/<user>/CORL.git"` (use a Kaggle Secret,
not a hard-coded token).

## Passing results between sessions
Results accumulate in `results/metrics.csv` / `results/rollouts.csv` via append-merge. To carry
Session-1 output into Session-2, either commit those CSVs and `git pull`, or save Session-1's
`results/` as a Kaggle dataset and copy it in (see the optional cell in `02_...`).

## Resuming after a 12h timeout
Completed tasks are saved and their checkpoints pruned, so just relaunch the session notebook — it
re-pulls and you re-run only the remaining tasks (or use `03_metrics.py --tasks <name>`).
