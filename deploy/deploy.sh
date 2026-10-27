#!/usr/bin/env bash
#
# Deploy the three Cloud Run services.
#
#   ./deploy/deploy.sh setup      one-off: APIs, service accounts, secret, registry
#   ./deploy/deploy.sh all        build and deploy everything
#   ./deploy/deploy.sh sandbox    one service at a time
#
# Read deploy/README.md first. It explains what this protects against and,
# more usefully, what it does not.
#
# Idempotent: re-running updates in place.

set -euo pipefail

PROJECT="${GCP_PROJECT:-socratic-evaluator}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-socratic}"

API_SVC="socratic-api"
UI_SVC="socratic-ui"
SANDBOX_SVC="socratic-sandbox"

API_SA="socratic-api@${PROJECT}.iam.gserviceaccount.com"
UI_SA="socratic-ui@${PROJECT}.iam.gserviceaccount.com"
SANDBOX_SA="socratic-sandbox@${PROJECT}.iam.gserviceaccount.com"

MONGO_SECRET="mongo-uri"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"
TAG="$(date +%Y%m%d-%H%M%S)"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

require_gcloud() {
  command -v gcloud >/dev/null || { echo "gcloud not found"; exit 1; }
  gcloud config set project "$PROJECT" >/dev/null 2>&1
}

url_of() {
  gcloud run services describe "$1" --region="$REGION" \
    --format='value(status.url)' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# One-off setup
# ---------------------------------------------------------------------------
enable_apis() {
  say "Enabling APIs"
  gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com
}

make_sa() {
  local name="$1" display="$2"
  if gcloud iam service-accounts describe \
       "${name}@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    note "service account ${name} exists"
  else
    gcloud iam service-accounts create "$name" --display-name="$display"
  fi
}

service_accounts() {
  say "Service accounts"
  make_sa socratic-api     "Socratic API"
  make_sa socratic-ui      "Socratic UI"
  # Granted nothing, deliberately and permanently. A compromise of the
  # sandbox inherits an identity that cannot call a single Google API.
  make_sa socratic-sandbox "Socratic sandbox (no permissions by design)"

  say "IAM"
  # The API calls Gemini and reads one secret. That is the whole list.
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${API_SA}" \
    --role="roles/aiplatform.user" --condition=None >/dev/null
  note "api     -> roles/aiplatform.user"

  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${API_SA}" \
    --role="roles/secretmanager.secretAccessor" --condition=None >/dev/null
  note "api     -> roles/secretmanager.secretAccessor"
  note "ui      -> no roles (it only speaks HTTP to the API)"
  note "sandbox -> no roles (intentionally; see README)"
}

registry() {
  say "Artifact Registry"
  if gcloud artifacts repositories describe "$REPO" \
       --location="$REGION" >/dev/null 2>&1; then
    note "repository ${REPO} exists"
  else
    gcloud artifacts repositories create "$REPO" \
      --repository-format=docker --location="$REGION" \
      --description="Socratic evaluator images"
  fi
}

secret() {
  say "Secret Manager"
  if gcloud secrets describe "$MONGO_SECRET" >/dev/null 2>&1; then
    note "secret ${MONGO_SECRET} exists"
    note "to rotate: printf '%s' \"\$MONGO_URI\" | gcloud secrets versions add ${MONGO_SECRET} --data-file=-"
  else
    [ -n "${MONGO_URI:-}" ] || {
      echo "   MONGO_URI is not set. Run:"
      echo "     \$env:MONGO_URI='mongodb+srv://...'   (PowerShell)"
      echo "     export MONGO_URI='mongodb+srv://...'  (bash)"
      exit 1
    }
    printf '%s' "$MONGO_URI" \
      | gcloud secrets create "$MONGO_SECRET" --data-file=- \
          --replication-policy=automatic
    note "created ${MONGO_SECRET}"
  fi
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build() {
  local dockerfile="$1" name="$2"
  local image="${REGISTRY}/${name}:${TAG}"
  say "Building ${name} from ${dockerfile}"
  gcloud builds submit --config=deploy/cloudbuild.yaml \
    --substitutions="_DOCKERFILE=${dockerfile},_IMAGE=${image}" .
  echo "$image"
}

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
deploy_sandbox() {
  local image
  image="$(build Dockerfile.sandbox sandbox | tail -1)"

  say "Deploying ${SANDBOX_SVC}"
  # --no-allow-unauthenticated is the boundary that actually holds here.
  # Google's edge rejects a caller without a valid identity token carrying
  # run.invoker, before any request reaches the container.
  #
  # Not --ingress=internal: that means "from VPC networks in this project",
  # and Cloud Run to Cloud Run calls do not traverse a VPC unless you pay
  # for a connector. Setting it would simply break the API's calls while
  # looking more secure.
  #
  # --concurrency=1 so one container never runs two hostile programs at
  # once; --max-instances bounds what a flood costs.
  gcloud run deploy "$SANDBOX_SVC" \
    --image="$image" \
    --region="$REGION" \
    --service-account="$SANDBOX_SA" \
    --no-allow-unauthenticated \
    --min-instances=0 \
    --max-instances=5 \
    --concurrency=1 \
    --cpu=1 \
    --memory=512Mi \
    --timeout=60s

  local sandbox_url
  sandbox_url="$(url_of "$SANDBOX_SVC")"
  note "$sandbox_url"

  gcloud run services add-iam-policy-binding "$SANDBOX_SVC" \
    --region="$REGION" \
    --member="serviceAccount:${API_SA}" \
    --role="roles/run.invoker" >/dev/null
  note "only ${API_SA} may invoke it"
}

deploy_api() {
  local sandbox_url image
  sandbox_url="$(url_of "$SANDBOX_SVC")"
  [ -n "$sandbox_url" ] || {
    echo "   Deploy the sandbox first. The API refuses to start without"
    echo "   SANDBOX_URL rather than run student code in its own process."
    exit 1
  }

  image="$(build Dockerfile.api api | tail -1)"

  say "Deploying ${API_SVC}"
  gcloud run deploy "$API_SVC" \
    --image="$image" \
    --region="$REGION" \
    --service-account="$API_SA" \
    --allow-unauthenticated \
    --min-instances=0 \
    --max-instances=3 \
    --cpu=1 \
    --memory=1Gi \
    --timeout=300s \
    --set-secrets="MONGO_URI=${MONGO_SECRET}:latest" \
    --set-env-vars="GCP_PROJECT=${PROJECT},GCP_LOCATION=global,MONGO_DB=socratic,SANDBOX_URL=${sandbox_url}"

  note "$(url_of "$API_SVC")"
}

deploy_ui() {
  local api_url image
  api_url="$(url_of "$API_SVC")"
  [ -n "$api_url" ] || { echo "   Deploy the API first."; exit 1; }

  image="$(build Dockerfile.ui ui | tail -1)"

  say "Deploying ${UI_SVC}"
  gcloud run deploy "$UI_SVC" \
    --image="$image" \
    --region="$REGION" \
    --service-account="$UI_SA" \
    --allow-unauthenticated \
    --min-instances=0 \
    --max-instances=2 \
    --cpu=1 \
    --memory=512Mi \
    --timeout=300s \
    --set-env-vars="API_BASE_URL=${api_url}"

  note "$(url_of "$UI_SVC")"
}

summary() {
  say "Deployed"
  for svc in "$SANDBOX_SVC" "$API_SVC" "$UI_SVC"; do
    printf '   %-20s %s\n' "$svc" "$(url_of "$svc" || echo 'not deployed')"
  done
  echo
  note "Health check:"
  note "  curl -s $(url_of "$API_SVC")/api/health"
  echo
  note "All three have min-instances=0, so an idle deployment costs nothing."
}

main() {
  require_gcloud
  case "${1:-all}" in
    setup)   enable_apis; service_accounts; registry; secret ;;
    sandbox) deploy_sandbox ;;
    api)     deploy_api ;;
    ui)      deploy_ui ;;
    all)     deploy_sandbox; deploy_api; deploy_ui; summary ;;
    urls)    summary ;;
    *) echo "usage: $0 [setup|all|sandbox|api|ui|urls]"; exit 1 ;;
  esac
}

main "$@"
