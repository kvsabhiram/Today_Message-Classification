# Deployment — ToDoZee Classifier on AWS GPU (CI/CD)

Serves the Qwen2.5-3B + LoRA v11 classifier as a FastAPI service on a GPU EC2 box,
deployed automatically by GitHub Actions.

## Architecture

```
git push (main)
   └─> GitHub Actions
         ├─ checkout + git lfs pull (adapter weights, 228 MB)
         ├─ docker build (CUDA 12.6, base model baked in)
         ├─ push image -> Amazon ECR (ap-south-1)
         └─ SSM RunCommand -> EC2 g5.xlarge: docker pull + run --gpus all
                                   └─> FastAPI on :5011  (/health /classify /batch /tasks)
```

## Target hardware

- **Instance:** `g5.xlarge` — NVIDIA A10G (Ampere, 24 GB), native bf16, ~$1.0/hr in Mumbai.
- **Region:** `ap-south-1` (Mumbai).
- **AMI:** Ubuntu 22.04 (or AWS Deep Learning AMI to skip the driver install).
- Model footprint: ~6 GB weights + ~2 GB overhead → fits comfortably in 24 GB.

> g4dn.xlarge (T4) is cheaper but lacks native bf16 — it would require changing
> `torch.bfloat16` to `torch.float16` in `inference.py`.

## One-time AWS setup (Terraform — recommended)

All infra is codified in `terraform/` (ECR, g5 EC2 + IAM instance role, security
group, and the GitHub Actions OIDC role). State is kept in **S3 + DynamoDB**
(DynamoDB provides the lock so concurrent applies can't corrupt state).

### Step 0 — create the remote-state backend (run once)

```bash
cd terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars   # set a globally-unique bucket name
terraform init && terraform apply              # creates the S3 bucket + lock table
```

Then put the same bucket/table names into `terraform/backend.tf` (replace
`todozee-tfstate-CHANGE-ME-12345`), or pass them via `-backend-config` at init.

### Step 1 — provision the infrastructure

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # edit as needed
terraform init                                  # migrates state into S3
terraform apply
```

The EC2 user-data runs `deploy/ec2_bootstrap.sh` automatically (NVIDIA driver,
Docker, NVIDIA container toolkit, AWS CLI, SSM agent). After apply, read the
outputs — they map directly to the GitHub secrets below:

```bash
terraform output instance_id              # -> EC2_INSTANCE_ID
terraform output github_actions_role_arn  # -> AWS_ROLE_ARN
terraform output app_url                  # base URL to test
```

> If your account already has a GitHub OIDC provider, set
> `create_github_oidc_provider = false` (only one provider per account is allowed).
> For zero driver hassle, set `ami_id` to an AWS Deep Learning Base GPU AMI.

### Manual alternative (no Terraform)

Launch a `g5.xlarge` (Ubuntu 22.04, ≥60 GB gp3) with an IAM role granting
`AmazonSSMManagedInstanceCore` + `AmazonEC2ContainerRegistryReadOnly`, open
`5011/tcp`, then run `sudo bash deploy/ec2_bootstrap.sh`. Verify:
`docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi`.

## GitHub repository secrets

| Secret | Value | Source |
|---|---|---|
| `AWS_ROLE_ARN` | OIDC role GitHub assumes for **deploy** (scoped to `main`) | `terraform output github_actions_role_arn` |
| `AWS_PLAN_ROLE_ARN` | Read-only OIDC role for **PR `terraform plan`** | `terraform output github_actions_plan_role_arn` |
| `EC2_INSTANCE_ID` | e.g. `i-0abc123...` of the g5 box | `terraform output instance_id` |

(Region `ap-south-1`, ECR repo `todozee-classifier` are set in the workflow `env:` block.)

## CI workflows

| Workflow | Trigger | Does |
|---|---|---|
| `terraform-plan.yml` | PR touching `terraform/**` | fmt + validate + `plan -lock=false` (read-only role), posts a sticky plan comment on the PR |
| `deploy.yml` | push to `main` / manual | build → ECR → SSM deploy to g5 → health check |

So infra changes are reviewed via the plan comment on the PR, then applied
(currently `terraform apply` is run manually after merge — say the word if you want
an auto-apply-on-merge job too).

## Deploy

- **Automatic:** push to `main` (changes to code/Dockerfile/`output_v11/`).
- **Manual:** Actions tab → *Build & Deploy (GPU / EC2)* → Run workflow.

The pipeline creates the ECR repo if missing, builds, pushes, then SSM-deploys and
health-checks `/health` on the instance.

## Verify

```bash
curl http://<EC2_PUBLIC_IP>:5011/health
curl -X POST http://<EC2_PUBLIC_IP>:5011/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"remind me to call mom at 6pm"}'
```

## Notes / gotchas

- **Adapter weights** (`output_v11/adapter_model.safetensors`) are Git LFS. CI runs
  `git lfs pull`; ensure the object exists in the remote (`git lfs push origin main`).
- **First container start** loads + merges the model (~1–3 min). Healthcheck
  `start-period` is 180s; the pipeline polls `/health` for up to 5 min.
- **Cost control:** stop the instance when idle, or move to ECS/Auto Scaling later
  (the same image is reused).
- **Local test (on a GPU box):**
  ```bash
  docker build -t todozee .
  docker run --rm --gpus all -p 5011:5011 todozee
  ```
