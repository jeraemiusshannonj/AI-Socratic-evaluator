# Deploying

Three Cloud Run services, all scale-to-zero.

| Service | Public | Holds credentials | Purpose |
|---|---|---|---|
| `socratic-ui` | yes | no | Streamlit front end; speaks HTTP to the API and nothing else |
| `socratic-api` | yes, rate-limited | Mongo URI, Vertex SA | Retrieval, the tutor, the taxonomy |
| `socratic-sandbox` | **no** | **none** | Executes untrusted student code |

The split exists for one reason. `/api/run` executes arbitrary Python
supplied over HTTP. Running that in the same process as a Mongo connection
string and a Vertex-capable service account means any student submission
can read both. Running it in a service with no credentials, no IAM roles
and no way to be called from the internet means a hostile submission gets
an empty container and a 60-second timeout.

The API **refuses to start** without `SANDBOX_URL` when it detects it is
deployed (Cloud Run sets `K_SERVICE`). Falling back to in-process
execution is right on a laptop and wrong on a public URL, and nothing at
the call site distinguishes them — so the failure is a refusal to boot
rather than a quietly unsafe service.

---

## First time

```bash
export GCP_PROJECT=socratic-evaluator       # your real project id
export REGION=us-central1
export MONGO_URI='mongodb+srv://...'         # the full string, with password

./deploy/deploy.sh setup
```

That enables the APIs, creates three service accounts, grants the API
exactly two roles and the other two none, creates the Artifact Registry
repository, and stores the Mongo URI in Secret Manager.

PowerShell equivalents for the exports:

```powershell
$env:GCP_PROJECT = "socratic-evaluator"
$env:REGION      = "us-central1"
$env:MONGO_URI   = "mongodb+srv://..."
```

`deploy.sh` is bash. On Windows run it from Git Bash, or from WSL, or just
run the gcloud commands inside it by hand — they are ordinary and
commented.

## Deploying

```bash
./deploy/deploy.sh all
```

Order matters and the script enforces it: sandbox, then API (which needs
the sandbox URL), then UI (which needs the API URL). Each is also
deployable alone — `./deploy/deploy.sh api`.

Afterwards:

```bash
curl -s "$(gcloud run services describe socratic-api --region=$REGION --format='value(status.url)')/api/health"
```

`"ok": true` means Mongo is reachable, the corpus is populated, the vector
index is queryable, and `execution.mode` is `sandbox-service`. If
`execution.mode` says `in-process` on a deployed service, something is
wrong and the health check will say so.

---

## What this protects against, and what it does not

**Enforced:**

- The sandbox cannot be called by anyone but the API. `--no-allow-unauthenticated`
  means Google's edge rejects a request without a valid identity token
  carrying `roles/run.invoker`, before the container is reached. Only
  `socratic-api@` has that binding.
- The sandbox holds no credentials. No Mongo URI, no Vertex access, and a
  service account with zero IAM roles. Its image contains two Python files
  and FastAPI — no database driver, no cloud SDK.
- Student code runs as a non-root user, in a fresh temp directory, in
  isolated mode (`-I`, so a file beside the server is not importable),
  with a stripped environment, output truncation, a wall-clock timeout and
  POSIX resource limits.
- Memory, CPU, concurrency and instance caps bound what a flood costs.
  `--concurrency=1` means one container never runs two hostile programs
  at once.
- The Mongo URI comes from Secret Manager, not an environment variable
  baked into the image.
- The public API is rate-limited per IP and per day, because an endpoint
  that calls Gemini is otherwise a way for a stranger to spend the budget.
  The budget alert notifies; it does not stop anything.

**Not enforced, and worth knowing:**

- **The sandbox has outbound internet access.** Cloud Run allows egress by
  default, and blocking it needs a VPC connector with no NAT route —
  which is not in the free tier. Student code can therefore make outbound
  requests. It has nothing worth exfiltrating, but it could be used as a
  traffic source. Closing this properly means `--vpc-egress=all-traffic`
  with a connector, and paying for it.
- **`--ingress=internal` is deliberately not set.** It restricts callers to
  VPC networks in the project, and Cloud Run-to-Cloud Run calls do not
  traverse a VPC without a connector — so it would break the API's calls
  while appearing to tighten things. IAM is doing the work here, not the
  network.
- **The API is public.** It is rate-limited rather than authenticated,
  which is a demo-grade choice. Anything real would put an identity in
  front of it.

---

## Cost

Everything is scale-to-zero, so an idle deployment costs nothing for
compute. Three things do cost:

- **Vertex AI calls.** Billed per token regardless of where the code runs.
  This is the only meaningful cost and it was being paid before any of
  this was deployed.
- **Artifact Registry storage** for the images, a few hundred MB.
- **Cloud Build minutes** per deploy, within a free monthly allowance at
  this frequency.

Atlas M0 is free. Confirm current Cloud Run free-tier figures on Google's
pricing page rather than trusting any number written here — they change.

`min-instances=0` on every service is not optional. A warm container bills
for existing, whether or not anyone is using it, and burns the allowance
in days.

---

## Teardown

```bash
for s in socratic-ui socratic-api socratic-sandbox; do
  gcloud run services delete "$s" --region="$REGION" --quiet
done
```

Service accounts, the secret and the registry survive, so a redeploy skips
`setup`. To remove everything, delete the project.
