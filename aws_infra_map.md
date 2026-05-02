# AWS + Bedrock + Vercel Infrastructure Map (CombBandits)

Reconstructed from code review on 2026-04-24. Primary reference for any new
experiment runs. AWS ONLY — ignore `cluster/della_*.sh` and `scripts/setup_della.sh`.

## 1. AWS EC2

**Launch entry point:** `bash cluster/aws_launch.sh`

- **Instance type (GPU):** `g4dn.2xlarge` (8 vCPU, 32 GB RAM, T4 16GB, 225GB NVMe)
- **Instance type (CPU-only for Bedrock-heavy runs):** `c5.4xlarge`
- **AMI:** `ami-03d653a715378d2e5` (Deep Learning Base OSS Nvidia Ubuntu 22.04, CUDA 12)
- **Region:** `us-east-1`
- **IAM instance profile:** `princetoncourses-ec2`
- **Security group:** `combbandits-sg` (auto-created on first launch; SSH from 0.0.0.0/0)
- **EBS root:** 75 GB gp3 (auto-delete on termination)
- **SSH key:** `~/.ssh/combbandits-key.pem` (EC2 key pair `combbandits-key`)
- **AWS account:** `099841456154` (user `friday-2`)

**Bootstrap flow:**

1. `cluster/aws_launch.sh` locally: rsync repo to instance (excludes `.git`, `results/`, `__pycache__`, `cache/`, `metadata/`), then launches `cluster/aws_run_experiments.sh` detached via SSH.
2. `cluster/aws_run_experiments.sh` on instance: detects NVMe mount, installs Python venv + deps, fetches HF + GH tokens from SSM Parameter Store, runs a 3-phase experiment sequence, syncs results to S3, commits + pushes to GitHub `vkmk1/CombBandits`, self-terminates via `aws ec2 terminate-instances`.

**Log convention:**
- Instance: `/home/ubuntu/experiment.log` (tee'd in real-time)
- Per-experiment: `~/log_${EXP}.log`
- Mirrored to S3 under `metadata/`

## 2. Bedrock

**Client:** `boto3.client("bedrock-runtime", region_name="us-east-1")` — uses EC2 instance role creds (no API keys in env).

**Models actually used:**
- `us.meta.llama4-scout-17b-instruct-v1:0` — cross-region inference profile; current production primary
- `anthropic.claude-3-5-haiku-20241022-v1:0` — older default in `llm_oracle.py` (may be deprecated; prefer Haiku 4.5 if available)

**Invocation:** Bedrock Converse API (`client.converse(...)`), not direct InvokeModel. Temp 0.7, max_tokens 256. Prompts from `_build_prompt()`. Response parsed as JSON first, regex-integer fallback.

**Caching:** `CachedOracle` wraps `LLMOracle` with SQLite at `cache/oracle_<exp>/oracle_cache.db`. Key = SHA256(context JSON + quantized mu_hat). Query schedule is O(sqrt(T)).

**Known gaps:**
- No retry/backoff on throttle or service-unavailable. Single failure kills trial.
- No fallback model chain.

## 3. S3 storage

**Bucket:** `combbandits-results-099841456154` (us-east-1).

**Actual current top-level prefixes:** `dashboard-live/`, `live/`, `smoke_test/`, `tier4_full/`

**Convention for new runs:** Each named run gets its own top-level prefix:
- `tier4_full/` — April 2026 tier-4 runs
- `live/` — streaming live-results for dashboard
- `smoke_test/` — local smoke tests

For exp10-exp13: use **`tier5_rerun/`** (or `rerun_2026_04_24/`) as the top-level prefix. Inside: `results/`, `figures/`, `metadata/` subdirectories.

**SSM params (require instance role `ssm:GetParameter` with decryption):**
- `/combbandits/hf_token`
- `/combbandits/github_token`

## 4. Vercel dashboard

**Framework:** Next.js 14.2.5, React 18.3.1, Tailwind, Recharts at `dashboard-v2/`.

**Deployment:** Connected to Vercel project `prj_61ztjYC9jCUdm1hrXEOs4Mn9lpy4` (team `team_44MxE0InKhKl2KEkWMdV8tqk`). Auto-deploys from GitHub.

**Data pipeline (BROKEN):**
- Frontend `useSWR("/api/live")` with 6s refresh
- `next.config.js` rewrites `/api/:path*` → `NEXT_PUBLIC_API_URL/api/:path*`
- Default `NEXT_PUBLIC_API_URL = http://54.87.245.227:8000` — dead IP, no backend
- Dashboard always shows "No data / error"

**Fix strategy (preferred):** Replace live API with a static `dashboard-v2/public/data.json`.
- Build script `scripts/build_dashboard_data.py`: reads results from S3 or local `results/`, writes `data.json` with shape `{ experiment, updated_at, stats: { rankings, curves, paired, splits } }`.
- Frontend change: swap `fetch("/api/live")` → `fetch("/data.json")`.

**Expected data shape:** See `dashboard-v2/app/page.tsx` — `LiveResponse` interface.

**Colors:** `dashboard-v2/lib/colors.ts` — `ALGO_COLORS` dict has 16 existing entries. Add new ones for `escb`, `club`, `randomcorr_rbf`, `tsllm_1call`. Also update `algoDisplay()` and `isBaseline()`.

**Deploy:**
- Automatic: git push to main → Vercel webhook
- Manual: `cd dashboard-v2 && vercel deploy --prod` (requires `vercel` CLI installed + `vercel login` one-time, or `VERCEL_TOKEN` env var)

## 5. Canonical "run experiment on AWS" recipe

```bash
# Preflight
ls ~/.ssh/combbandits-key.pem                         # must exist
aws sts get-caller-identity                           # must work
aws ssm get-parameter --name /combbandits/hf_token --with-decryption --region us-east-1
aws ssm get-parameter --name /combbandits/github_token --with-decryption --region us-east-1
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1  # need >= 8 vCPUs

# Launch (edit aws_run_experiments.sh first to set the experiment list)
bash cluster/aws_launch.sh

# Monitor (IP printed by aws_launch.sh)
ssh -i ~/.ssh/combbandits-key.pem ubuntu@$PUBLIC_IP tail -f ~/experiment.log

# Poll for completion via S3
aws s3 ls s3://combbandits-results-099841456154/tier5_rerun/ --recursive | tail -20

# After termination (instance self-terminates), pull results
bash cluster/aws_download.sh
```

## 6. What's missing to run exp10-exp13

1. Agent implementations in `src/combbandits/agents/`:
   - `escb.py`, `club.py`, `randomcorr_rbf.py`, `tsllm_1call.py`
   - Register in `agents/__init__.py`
2. Configs in `configs/experiments/`:
   - `exp10_randomcorr_rbf.yaml`, `exp11_escb_club.yaml`, `exp12_tsllm_1call.yaml`, `exp13_endogenous_oracle.yaml`
3. `aws_run_experiments.sh` phase-2 list updated with exp10-13
4. `src/combbandits/oracle/llm_oracle.py`: exponential backoff on Bedrock throttles
5. `scripts/verify_bedrock.sh`: preflight check for model availability
6. `scripts/build_dashboard_data.py`: generates static `data.json`
7. `dashboard-v2/lib/colors.ts`: colors + display names for 4 new agents
8. Frontend: swap from `/api/live` to `/data.json`
9. Vercel CLI: `npm install -g vercel && vercel login` (interactive, one-time)

## 7. Dashboard update flow for new baselines

1. Edit `dashboard-v2/lib/colors.ts`:
   - Add `ESCB`, `CLUB`, `RandomCorr_RBF`, `TS_LLM_1call` to `ALGO_COLORS`
   - Update `algoDisplay()` with human-readable strings
   - Update `isBaseline()` if ESCB/CLUB should be marked as baselines
2. Run `python scripts/build_dashboard_data.py` to regenerate `dashboard-v2/public/data.json`
3. `cd dashboard-v2 && vercel deploy --prod`
4. `git commit -am "dashboard: add exp10-13 baselines"; git push`
