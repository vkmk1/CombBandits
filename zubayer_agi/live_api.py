"""FastAPI live-data server for the bandits dashboard.

Reads the latest tier{N}_*/raw_trials.jsonl and exposes:

    GET /api/health          → {"status": "ok", "version": "..."}
    GET /api/live            → complete dashboard state (polled every 5-10s)
    GET /api/trials?since=N  → incremental trial updates

Designed to be robust: tolerates missing files, parse errors, concurrent writes.

Run:
    uvicorn live_api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

RESULTS_DIR = Path(os.environ.get(
    "RESULTS_DIR", Path(__file__).parent / "results"
))

app = FastAPI(title="CombBandits Live API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_latest_run() -> Path | None:
    """Find the most recent experiment directory (tier6/7/etc.)."""
    candidates = sorted(
        glob.glob(str(RESULTS_DIR / "tier*_*/raw_trials.jsonl"))
        + glob.glob(str(RESULTS_DIR / "*/raw_trials.jsonl")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return Path(candidates[0]).parent if candidates else None


_cache = {"path": None, "mtime": 0.0, "trials": [], "stamp": 0.0}


def load_trials(path: Path) -> list[dict]:
    """Lazy-refresh cache from raw_trials.jsonl."""
    now = time.time()
    fpath = path / "raw_trials.jsonl"
    if not fpath.exists():
        return []
    mtime = fpath.stat().st_mtime
    if _cache["path"] == path and _cache["mtime"] == mtime and (now - _cache["stamp"] < 2):
        return _cache["trials"]
    trials = []
    try:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return _cache["trials"]
    _cache.update({"path": path, "mtime": mtime, "trials": trials, "stamp": now})
    return trials


def load_summary(path: Path) -> dict:
    f = path / "summary.json"
    if not f.exists():
        return {}
    try:
        with open(f) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def compute_stats(trials: list[dict], baseline: str = "N1_corr_full") -> dict:
    """Compute rankings, paired Wilcoxon, generalization gap, regret curves."""
    if not trials:
        return {"rankings": [], "paired": [], "curves": [], "splits": {}}

    # Rankings per algo
    by_algo: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_algo[t["algo"]].append(t)

    rankings = []
    baseline_mean = None
    for algo, ts in by_algo.items():
        regrets = np.array([t["final_regret"] for t in ts])
        n = len(regrets)
        mean = float(regrets.mean())
        se = float(regrets.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        median = float(np.median(regrets))
        rankings.append({
            "algo": algo,
            "n": n,
            "mean": round(mean, 2),
            "se": round(se, 2),
            "median": round(median, 2),
        })
        if algo == baseline:
            baseline_mean = mean
    for r in rankings:
        if baseline_mean is not None and baseline_mean > 0:
            r["vs_baseline_pct"] = round(100 * (baseline_mean - r["mean"]) / baseline_mean, 2)
        else:
            r["vs_baseline_pct"] = None
    rankings.sort(key=lambda r: r["mean"])

    # Paired Wilcoxon vs baseline
    paired = []
    # Build (config_id, seed) → {algo: regret}
    by_cs: dict = defaultdict(dict)
    for t in trials:
        by_cs[(t["config_id"], t["seed"])][t["algo"]] = t["final_regret"]
    base_pairs = {cs: d[baseline] for cs, d in by_cs.items() if baseline in d}
    p_raw = []
    for algo in by_algo:
        if algo == baseline:
            continue
        pairs = [(base_pairs[cs], by_cs[cs][algo])
                 for cs in base_pairs if algo in by_cs[cs]]
        if len(pairs) < 5:
            paired.append({"algo": algo, "n": len(pairs), "wins": 0, "losses": 0,
                           "mean_adv": 0.0, "p": 1.0, "sig": False})
            p_raw.append(1.0)
            continue
        b = np.array([p[0] for p in pairs])
        a = np.array([p[1] for p in pairs])
        diffs = b - a
        wins = int((diffs > 0).sum())
        losses = int((diffs < 0).sum())
        # Wilcoxon
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(b, a)
        except Exception:
            p = 1.0
        paired.append({
            "algo": algo,
            "n": len(pairs),
            "wins": wins,
            "losses": losses,
            "mean_adv": round(float(diffs.mean()), 2),
            "p": round(float(p), 6),
        })
        p_raw.append(float(p))
    # Holm-Bonferroni
    K = len(p_raw)
    sorted_idx = np.argsort(p_raw) if p_raw else []
    holm = [False] * K
    for rank, idx in enumerate(sorted_idx):
        alpha_k = 0.05 / (K - rank)
        if p_raw[idx] < alpha_k:
            holm[idx] = True
        else:
            break
    for p, s in zip(paired, holm):
        p["sig"] = bool(s)

    # Per-split stats (for generalization-gap chart)
    split_stats: dict[str, dict] = {}
    for split in {t.get("split", "original") for t in trials}:
        sub = [t for t in trials if t.get("split", "original") == split]
        if not sub:
            continue
        by_algo_s: dict = defaultdict(list)
        for t in sub:
            by_algo_s[t["algo"]].append(t["final_regret"])
        split_stats[split] = {
            algo: round(float(np.mean(vs)), 2) for algo, vs in by_algo_s.items()
        }

    # Regret curves (sampled every 50 rounds). Aggregate across trials.
    curves_by_algo: dict = defaultdict(list)
    for t in trials:
        c = t.get("regret_curve")
        if c:
            curves_by_algo[t["algo"]].append(c)
    curves = []
    for algo, lst in curves_by_algo.items():
        if not lst:
            continue
        min_len = min(len(c) for c in lst)
        arr = np.array([c[:min_len] for c in lst])
        mean = arr.mean(axis=0).round(2).tolist()
        se_arr = (arr.std(axis=0, ddof=1) / np.sqrt(len(arr))).round(2).tolist() if len(arr) > 1 else [0.0] * min_len
        # Subsample if long
        step = max(1, min_len // 200)
        curves.append({
            "algo": algo,
            "x": list(range(50, (min_len + 1) * 50, 50))[::step],
            "mean": mean[::step],
            "se": se_arr[::step],
            "n": len(arr),
        })

    return {
        "rankings": rankings,
        "paired": paired,
        "curves": curves,
        "splits": split_stats,
    }


class LiveResponse(BaseModel):
    experiment: dict
    stats: dict
    updated_at: float


@app.get("/api/health")
def health():
    latest = find_latest_run()
    return {
        "status": "ok",
        "version": "1.0.0",
        "latest_run": latest.name if latest else None,
    }


@app.get("/api/live")
def live():
    latest = find_latest_run()
    if latest is None:
        raise HTTPException(status_code=404, detail="No experiment runs found")
    trials = load_trials(latest)
    summary = load_summary(latest)
    stats = compute_stats(trials)

    experiment = {
        "run_id": latest.name,
        "T": summary.get("T"),
        "n_seeds": summary.get("n_seeds"),
        "n_configs": summary.get("n_configs"),
        "total_trials": summary.get("total_trials") or (
            summary.get("n_seeds", 0) * summary.get("n_configs", 0) *
            len(summary.get("variants", []))
        ),
        "completed_trials": len(trials),
        "started_at": summary.get("timestamp"),
        "model": summary.get("model", "gpt-5.4"),
        "variants": summary.get("variants", []),
    }
    return {"experiment": experiment, "stats": stats, "updated_at": time.time()}


@app.get("/api/trials")
def trials_endpoint(since: int = 0, limit: int = 100):
    latest = find_latest_run()
    if latest is None:
        raise HTTPException(status_code=404, detail="No experiment runs found")
    trials = load_trials(latest)
    return {"trials": trials[since:since + limit], "total": len(trials)}
