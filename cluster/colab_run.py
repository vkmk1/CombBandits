"""
CombBandits — Complete Colab Runner
====================================
Paste this entire file into a single Colab cell, or upload and run:
    !python colab_run.py

GPU: H100 (preferred) or A100. Enable High-RAM in runtime settings.

Runs ALL experiments:
  CPU (via ProcessPoolExecutor, uses all CPU cores):
    exp4_mind, exp5_influence_max, exp6_workshop_main,
    exp7_ablation_trust, exp9_bedrock   (Llama 4 Scout via Bedrock API)

  GPU (batched tensors on CUDA):
    exp8_scaling_d    (pure bandit simulation, all seeds in parallel)
    exp9_local        (Llama 4 Scout loaded locally, full weight tracking)

Crash recovery: every 10 tasks a checkpoint is written. Re-running
resumes from where it left off automatically.

Outputs saved to /content/CombBandits/{results,figures,metadata}/
Pushed to S3 on completion (set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
in Colab Secrets, or they'll just stay local).
"""

# ─── 0. CONFIG — fill these in ───────────────────────────────────────────────
HF_TOKEN  = ""   # HuggingFace token for Llama 4 gated model
#   get one free at https://huggingface.co/settings/tokens
#   accept license at https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
#   OR set as Colab Secret named HF_TOKEN

AWS_ACCESS_KEY_ID     = ""   # for S3 upload at the end
AWS_SECRET_ACCESS_KEY = ""   # OR set as Colab Secrets
AWS_REGION            = "us-east-1"
S3_BUCKET             = "combbandits-results-099841456154"

SKIP_EXP9_LOCAL  = False   # set True to skip local weight-tracking run
SKIP_EXP9_BEDROCK= False   # set True to skip Bedrock API run (needs AWS creds)
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, time, subprocess, threading, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Try loading secrets from Colab if not hardcoded above ────────────────────
def _colab_secret(name, fallback=""):
    try:
        from google.colab import userdata
        return userdata.get(name) or fallback
    except Exception:
        return os.environ.get(name, fallback)

if not HF_TOKEN:           HF_TOKEN           = _colab_secret("HF_TOKEN")
if not AWS_ACCESS_KEY_ID:  AWS_ACCESS_KEY_ID  = _colab_secret("AWS_ACCESS_KEY_ID")
if not AWS_SECRET_ACCESS_KEY: AWS_SECRET_ACCESS_KEY = _colab_secret("AWS_SECRET_ACCESS_KEY")

if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"]     = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"]    = AWS_REGION

# ── 1. GPU CHECK ──────────────────────────────────────────────────────────────
import torch
print("=" * 60)
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu}  ({vram:.0f} GB VRAM)")
else:
    gpu, vram = "CPU", 0
    print("WARNING: No GPU detected. exp8/exp9_local will be slow.")
print("=" * 60)

# ── 2. CLONE + INSTALL ────────────────────────────────────────────────────────
REPO = Path("/content/CombBandits")
if not REPO.exists():
    print("Cloning repo...")
    subprocess.run(["git", "clone", "https://github.com/vkmk1/CombBandits", str(REPO)], check=True)
else:
    print("Repo already cloned, pulling latest...")
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False)

os.chdir(REPO)
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.47", "accelerate", "bitsandbytes",
                "huggingface_hub", "boto3>=1.34", "scikit-learn"], check=True)
print("Install complete.\n")

# ── 3. CHECKPOINT-AWARE SAVER ─────────────────────────────────────────────────
def save_results(results, out_path, checkpoint_path=None):
    """Atomically write results JSON, then remove checkpoint if given."""
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(out_path)
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

def load_checkpoint(checkpoint_path):
    p = Path(checkpoint_path)
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            print(f"  Resumed {len(data)} completed trials from checkpoint.")
            return data
        except Exception:
            pass
    return []

# ── 4. CPU EXPERIMENTS (parallel, checkpoint-safe) ───────────────────────────
def run_cpu_experiment(exp_name, workers=None):
    """Run a single CPU experiment via the CLI. Returns path to results JSON."""
    from combbandits.engine.runner import ExperimentRunner
    cfg_path = f"configs/experiments/{exp_name}.yaml"
    out_dir  = f"results/{exp_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Use all available CPU cores split across concurrent CPU experiments
    if workers is None:
        workers = max(1, os.cpu_count() or 4)

    print(f"  [{exp_name}] starting ({workers} workers)...")
    t0 = time.time()
    runner = ExperimentRunner(cfg_path, out_dir)
    runner.run(max_workers=workers)

    elapsed = time.time() - t0
    result_path = f"{out_dir}/{exp_name}_results.json"
    print(f"  [{exp_name}] done in {elapsed:.0f}s → {result_path}")
    return result_path

def run_plots_and_metrics(exp_name):
    result_path = f"results/{exp_name}/{exp_name}_results.json"
    if not Path(result_path).exists():
        print(f"  [{exp_name}] no results to plot")
        return
    try:
        from combbandits.analysis.plots import generate_all_figures
        generate_all_figures(result_path, f"figures/{exp_name}")
        print(f"  [{exp_name}] figures saved")
    except Exception as e:
        print(f"  [{exp_name}] plot error: {e}")

# CPU experiments and worker budget
# Total Colab CPU cores with High-RAM: typically 8-16.
# exp6 is by far the biggest (14K tasks). Give it the most.
# exp9_bedrock is API-latency bound → 4 workers is fine.
cpu_core_count = os.cpu_count() or 8
CPU_EXPS = [
    ("exp4_mind",            max(1, cpu_core_count // 8)),
    ("exp5_influence_max",   max(1, cpu_core_count // 8)),
    ("exp7_ablation_trust",  max(1, cpu_core_count // 4)),
    ("exp6_workshop_main",   max(4, cpu_core_count // 2)),
]
if not SKIP_EXP9_BEDROCK and AWS_ACCESS_KEY_ID:
    CPU_EXPS.append(("exp9_bedrock", 4))
elif SKIP_EXP9_BEDROCK or not AWS_ACCESS_KEY_ID:
    print("Skipping exp9_bedrock (no AWS creds or SKIP flag set).")

print(f"\n{'='*60}")
print(f"PHASE 1: CPU experiments ({len(CPU_EXPS)} exps in parallel threads)")
print(f"CPU cores: {cpu_core_count}")
print(f"{'='*60}")

cpu_results = {}
cpu_errors  = {}

def _run_cpu_thread(name, workers):
    try:
        cpu_results[name] = run_cpu_experiment(name, workers)
    except Exception as e:
        cpu_errors[name] = str(e)
        print(f"  [{name}] FAILED: {e}")

# Launch all CPU experiments simultaneously in threads
# (each manages its own ProcessPoolExecutor internally)
cpu_threads = []
for name, w in CPU_EXPS:
    t = threading.Thread(target=_run_cpu_thread, args=(name, w), daemon=True)
    t.start()
    cpu_threads.append((name, t))
    time.sleep(1)  # stagger slightly to avoid import collisions

# Wait for all CPU threads
for name, t in cpu_threads:
    t.join()
    status = "OK" if name in cpu_results else f"FAILED: {cpu_errors.get(name)}"
    print(f"  [{name}] {status}")

print("\nGenerating CPU figures...")
for name, _ in CPU_EXPS:
    if name in cpu_results:
        run_plots_and_metrics(name)

# ── 5. GPU EXPERIMENT: exp8_scaling_d ─────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 2: exp8_scaling_d — GPU batched simulation")
print(f"{'='*60}")
try:
    import yaml
    from combbandits.gpu.batched_trial import run_batched_experiment

    with open("configs/experiments/exp8_scaling_d.yaml") as f:
        exp8_cfg = yaml.safe_load(f)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  Running on {device}...")
    t0 = time.time()
    exp8_results = run_batched_experiment(exp8_cfg, device=device)
    elapsed = time.time() - t0

    Path("results/exp8_scaling_d").mkdir(parents=True, exist_ok=True)
    out_path = "results/exp8_scaling_d/exp8_scaling_d_results.json"
    with open(out_path, "w") as f:
        json.dump(exp8_results, f, indent=2)
    print(f"  exp8 done in {elapsed:.0f}s ({len(exp8_results)} trials) → {out_path}")
    run_plots_and_metrics("exp8_scaling_d")
except Exception as e:
    print(f"  exp8 FAILED: {e}")

# ── 6. GPU EXPERIMENT: exp9_local (Llama 4 Scout + weight tracking) ──────────
if SKIP_EXP9_LOCAL:
    print("\nSkipping exp9_local (SKIP_EXP9_LOCAL=True).")
elif not HF_TOKEN:
    print("\nSkipping exp9_local: no HF_TOKEN. Set it in Colab Secrets or hardcode above.")
elif vram < 20:
    print(f"\nSkipping exp9_local: only {vram:.0f}GB VRAM (need ≥20GB for 4-bit Llama 4 Scout).")
else:
    print(f"\n{'='*60}")
    print("PHASE 3: exp9_local — Llama 4 Scout + full weight tracking")
    print(f"{'='*60}")

    # Patch HF token into config
    import yaml
    cfg_path = "configs/experiments/exp9_local.yaml"
    with open(cfg_path) as f:
        exp9_cfg = yaml.safe_load(f)
    exp9_cfg["oracles"][0]["hf_token"] = HF_TOKEN
    with open(cfg_path, "w") as f:
        yaml.dump(exp9_cfg, f)

    try:
        from combbandits.engine.runner import ExperimentRunner
        Path("results/exp9_local").mkdir(parents=True, exist_ok=True)
        Path("metadata").mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        runner = ExperimentRunner(cfg_path, "results/exp9_local")
        # workers=1: model stays loaded between trials (not reloaded per worker)
        runner.run(max_workers=1)
        elapsed = time.time() - t0
        print(f"  exp9_local done in {elapsed/60:.1f} min")
        run_plots_and_metrics("exp9_local")
    except Exception as e:
        print(f"  exp9_local FAILED: {e}")
        import traceback; traceback.print_exc()

# ── 7. WEIGHT TRACKING ANALYSIS ───────────────────────────────────────────────
db_path = Path("metadata/oracle_weights.db")
if db_path.exists():
    print(f"\n{'='*60}")
    print("PHASE 4: Weight tracking analysis")
    print(f"{'='*60}")
    try:
        import sqlite3, numpy as np, matplotlib.pyplot as plt, pandas as pd
        from sklearn.decomposition import PCA

        db = sqlite3.connect(str(db_path))
        calls  = pd.read_sql("SELECT * FROM oracle_calls  ORDER BY call_id",   db)
        groups = pd.read_sql("SELECT * FROM query_groups  ORDER BY group_id",  db)
        db.close()
        print(f"  Tracked {len(calls)} oracle calls across {len(groups)} query groups")

        Path("figures/exp9_local").mkdir(parents=True, exist_ok=True)
        primary = calls[calls["query_variant"] == 0].copy()

        # Plot 1: entropy / attention / logprob / kappa over rounds
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(primary["trial_round"], primary["output_entropy"],     alpha=0.7, color="steelblue")
        axes[0,0].set(title="Output entropy over rounds", xlabel="Round", ylabel="Entropy")
        axes[0,1].plot(primary["trial_round"], primary["attn_on_metadata"],   alpha=0.7, color="crimson")
        axes[0,1].set(title="Attention on arm metadata", xlabel="Round", ylabel="Mean attention")
        axes[1,0].plot(primary["trial_round"], primary["suggestion_logprob"], alpha=0.7, color="green")
        axes[1,0].set(title="Log-prob of suggestion",    xlabel="Round", ylabel="Σ log p(token)")
        axes[1,1].plot(groups["trial_round"],  groups["kappa"],           label="κ (output)",   color="orange")
        axes[1,1].plot(groups["trial_round"],  groups["hidden_kl_div"],   label="hidden div",   color="purple")
        axes[1,1].set(title="Output κ vs internal diversity", xlabel="Round"); axes[1,1].legend()
        plt.suptitle("Llama 4 Scout Oracle — Weight Tracking", fontsize=13)
        plt.tight_layout()
        p1 = "figures/exp9_local/weight_tracking.png"
        plt.savefig(p1, dpi=150); plt.close()
        print(f"  Saved {p1}")

        # Plot 2: hidden-state PCA trajectory
        hidden_vecs = np.array([json.loads(r) for r in primary["hidden_state_pca"]])
        if hidden_vecs.shape[0] >= 3:
            pca    = PCA(n_components=2)
            coords = pca.fit_transform(hidden_vecs)
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(coords[:,0], coords[:,1],
                            c=primary["trial_round"].values, cmap="viridis", alpha=0.7, s=20)
            plt.colorbar(sc, label="Bandit round")
            ax.set(title="Hidden state PCA — representation drift over learning",
                   xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                   ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.tight_layout()
            p2 = "figures/exp9_local/hidden_state_pca.png"
            plt.savefig(p2, dpi=150); plt.close()
            print(f"  Saved {p2}")

    except Exception as e:
        print(f"  Weight analysis failed: {e}")
        import traceback; traceback.print_exc()

# ── 8. METADATA SUMMARY ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Writing metadata summary...")
Path("metadata").mkdir(exist_ok=True)

all_results_summary = {}
for exp_name in ["exp4_mind","exp5_influence_max","exp6_workshop_main",
                 "exp7_ablation_trust","exp8_scaling_d","exp9_bedrock","exp9_local"]:
    rp = Path(f"results/{exp_name}/{exp_name}_results.json")
    if rp.exists():
        with open(rp) as f:
            r = json.load(f)
        agents = list({x["agent"] for x in r})
        all_results_summary[exp_name] = {
            "n_trials": len(r),
            "agents": agents,
            "mean_final_regret": {
                a: round(sum(x["final_regret"] for x in r if x["agent"]==a)
                         / max(1, len([x for x in r if x["agent"]==a])), 1)
                for a in agents
            }
        }

meta = {
    "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "gpu": gpu,
    "vram_gb": round(vram, 1),
    "cpu_cores": cpu_core_count,
    "llm_oracle": {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "bedrock_id": "us.meta.llama4-scout-17b-instruct-v1:0",
        "open_weights": True,
        "weights_url": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "architecture": "MoE 17B active / 109B total, 16 experts top-2",
    },
    "experiments": all_results_summary,
    "git_commit": subprocess.run(["git","rev-parse","HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
}
with open("metadata/run_info.json", "w") as f:
    json.dump(meta, f, indent=2)

subprocess.run([sys.executable, "-m", "pip", "freeze"],
               stdout=open("metadata/requirements_frozen.txt","w"), stderr=subprocess.DEVNULL)
print("  Saved metadata/run_info.json")

# ── 9. S3 UPLOAD ─────────────────────────────────────────────────────────────
if AWS_ACCESS_KEY_ID:
    print(f"\n{'='*60}")
    print("Uploading to S3...")
    try:
        import boto3
        s3 = boto3.client("s3")
        uploaded = 0
        for local_dir in ["results", "figures", "metadata"]:
            for root, _, files in os.walk(local_dir):
                for fname in files:
                    full = os.path.join(root, fname)
                    s3.upload_file(full, S3_BUCKET, full)
                    uploaded += 1
        print(f"  Uploaded {uploaded} files to s3://{S3_BUCKET}/")
    except Exception as e:
        print(f"  S3 upload failed: {e}")
else:
    print("\nNo AWS creds — skipping S3 upload.")
    print("Download results from Files panel: /content/CombBandits/{results,figures,metadata}/")

# ── 10. DONE ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
for exp, path in list(cpu_results.items()) + [
    ("exp8_scaling_d", "results/exp8_scaling_d/exp8_scaling_d_results.json"),
    ("exp9_local",     "results/exp9_local/exp9_local_results.json"),
]:
    p = Path(path) if isinstance(path, str) else path
    status = f"{p.stat().st_size//1024}KB" if p.exists() else "MISSING"
    print(f"  {exp:<25} {status}")
