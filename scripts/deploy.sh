#!/usr/bin/env bash
#
# Universal, interactive deploy for the Hedonism Wines Assistant.
#
# Provisions a bare server from scratch: inspects what's installed, installs
# Docker/UFW if missing, builds the serving image LOCALLY and ships it over SSH
# (no Docker Hub pull of the app), transfers the local Qdrant index and the
# bottle pictures (baked into the image), drops .env.prod, and stands up Caddy
# with automatic HTTPS.
#
# Usage:
#   make deploy                 # interactive
#   AUTO_YES=1 make deploy      # don't ask, just go (CI / re-deploys)
#   SERVER_IP=1.2.3.4 SSH_USER=ubuntu make deploy   # override any var below
#
# Every step prints what it will do and is idempotent — safe to re-run.
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override any of these via the environment)
# ---------------------------------------------------------------------------
SERVER_IP="${SERVER_IP:-186.246.10.124}"
SSH_USER="${SSH_USER:-root}"
DOMAIN="${DOMAIN:-hedonism-wines.era-lands.ru}"
REMOTE_DIR="${REMOTE_DIR:-/opt/hedonism}"
IMAGE="${IMAGE:-hedonism-api:prod}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
CADDY_IMAGE="${CADDY_IMAGE:-caddy:2}"
# Tiny helper image used to untar the index into the remote volume. Shipped too
# so the server never pulls from a registry.
ALPINE_IMAGE="${ALPINE_IMAGE:-alpine:3.20}"
LOCAL_QDRANT_VOLUME="${LOCAL_QDRANT_VOLUME:-hedonism-wines-assistant_qdrant_storage}"
REMOTE_QDRANT_VOLUME="${REMOTE_QDRANT_VOLUME:-hedonism_qdrant_storage}"
# Docker platform of the SERVER. Everything we build/pull/save must target this,
# not the local machine's arch — an arm64 Mac otherwise ships images the amd64
# server can't exec. Auto-detected from the server in step 4 when left empty.
TARGET_PLATFORM="${TARGET_PLATFORM:-}"
AUTO_YES="${AUTO_YES:-}"

# Repo root = parent of this script's dir.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# SSH connection multiplexing so we don't re-auth on every call.
SSH_CTL="${TMPDIR:-/tmp}/hedonism-deploy-ssh-$$"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${SSH_CTL}" -o ControlPersist=120 -o ConnectTimeout=15)
REMOTE="${SSH_USER}@${SERVER_IP}"

cleanup() { ssh "${SSH_OPTS[@]}" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pretty output + helpers
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
  C_BLUE=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_OFF=""
fi
step()  { printf '\n%s==>%s %s\n' "$C_BLUE"  "$C_OFF" "$*"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '%s  ok%s %s\n' "$C_GREEN" "$C_OFF" "$*"; }
warn()  { printf '%swarn%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
die()   { printf '%s err%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }

confirm() {
  # confirm "question" -> 0 if yes. AUTO_YES skips the prompt.
  [ -n "$AUTO_YES" ] && return 0
  local reply
  printf '%s  ?%s %s [y/N] ' "$C_YELLOW" "$C_OFF" "$1" >&2
  read -r reply </dev/tty || true
  [[ "$reply" =~ ^[Yy]$ ]]
}

# Remote command runner. Auto-prefixes sudo when the SSH user isn't root.
RSUDO=""
rsh()  { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
rsudo(){ ssh "${SSH_OPTS[@]}" "$REMOTE" "$RSUDO $*"; }

have()  { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Local preflight
# ---------------------------------------------------------------------------
step "1/12  Local preflight"
have docker || die "docker not found locally."
docker info >/dev/null 2>&1 || die "Docker daemon not running. Start Docker Desktop and retry."
for f in Dockerfile.prod docker-compose.prod.yml Caddyfile .env.prod.example; do
  [ -f "$f" ] || die "missing $f in $REPO_DIR"
done
[ -f data/sparse_encoder.json ] || die "missing data/sparse_encoder.json (run 'make index' locally first)."
bottles_n=$(find src/hedonism_assistant/api/static/bottles -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
[ "${bottles_n:-0}" -gt 0 ] || die "no bottle images under src/.../static/bottles (run 'python data/import_images.py')."
docker volume inspect "$LOCAL_QDRANT_VOLUME" >/dev/null 2>&1 \
  || die "local Qdrant volume '$LOCAL_QDRANT_VOLUME' not found — bring the index up locally first (make index)."
ok "image inputs present (${bottles_n} bottle jpgs, sparse encoder, qdrant volume)."

# .env.prod: create from template, fill secrets.
if [ ! -f .env.prod ]; then
  step "      .env.prod not found — creating from template"
  cp .env.prod.example .env.prod
  printf '    Paste your OpenRouter API key (sk-or-...): ' >&2
  read -rs OPENROUTER_KEY </dev/tty; echo >&2
  [ -n "$OPENROUTER_KEY" ] || die "OpenRouter key is required."
  GEN_PW="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)"
  # Portable in-place sed (BSD/macOS + GNU).
  sed -i.bak "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=${OPENROUTER_KEY}|" .env.prod
  sed -i.bak "s|^AUTH_PASSWORD=.*|AUTH_PASSWORD=${GEN_PW}|" .env.prod
  rm -f .env.prod.bak
  ok ".env.prod written."
  printf '\n%s    AUTH_PASSWORD = %s%s   (save this — gates the UI/API)\n\n' "$C_GREEN" "$GEN_PW" "$C_OFF"
else
  ok ".env.prod already exists (leaving as-is)."
fi
GEN_PW="$(grep -E '^AUTH_PASSWORD=' .env.prod | cut -d= -f2-)"

# ---------------------------------------------------------------------------
# 2. DNS check (non-fatal)
# ---------------------------------------------------------------------------
step "2/12  DNS for ${DOMAIN}"
resolved=""
if have dig; then resolved="$(dig +short "$DOMAIN" A | tail -n1)"
elif have getent; then resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | tail -n1)"
fi
if [ "$resolved" = "$SERVER_IP" ]; then
  ok "${DOMAIN} -> ${SERVER_IP}"
else
  warn "${DOMAIN} resolves to '${resolved:-nothing}', expected ${SERVER_IP}."
  warn "Caddy can't get a Let's Encrypt cert until the A-record points here. Continuing anyway."
fi

# ---------------------------------------------------------------------------
# 3. SSH reachability
# ---------------------------------------------------------------------------
step "3/12  SSH to ${REMOTE}"
if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$REMOTE" true 2>/dev/null; then
  warn "Non-interactive SSH failed; trying once interactively (you may be prompted)."
  ssh "${SSH_OPTS[@]}" "$REMOTE" true || die "cannot SSH to $REMOTE. Set up a key (ssh-copy-id) and retry."
fi
[ "$(rsh 'id -u')" = "0" ] || RSUDO="sudo"
ok "connected${RSUDO:+ (using sudo for privileged steps)}."

# ---------------------------------------------------------------------------
# 4. Remote inspection + provisioning (Docker)
# ---------------------------------------------------------------------------
step "4/12  Docker on the server"
os_pretty="$(rsh '. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo unknown')"
info "remote OS: ${os_pretty}"
# Auto-detect the server's CPU arch so we build/pull images for IT, not for the
# local machine (an arm64 Mac would otherwise ship images amd64 can't run).
if [ -z "$TARGET_PLATFORM" ]; then
  case "$(rsh 'uname -m')" in
    x86_64|amd64)  TARGET_PLATFORM="linux/amd64" ;;
    aarch64|arm64) TARGET_PLATFORM="linux/arm64" ;;
    *) warn "unknown remote arch '$(rsh 'uname -m')'; defaulting to linux/amd64"; TARGET_PLATFORM="linux/amd64" ;;
  esac
fi
info "target platform: ${TARGET_PLATFORM}"
if [ "$TARGET_PLATFORM" != "linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')" ]; then
  warn "server arch differs from this machine — images build/pull under emulation (slower)."
fi
if rsh 'command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1'; then
  ok "docker + compose present: $(rsh 'docker --version')"
else
  warn "Docker (or the compose plugin) is missing on the server."
  confirm "Install Docker Engine via get.docker.com on ${SERVER_IP}?" \
    || die "Docker required. Install it manually and re-run."
  rsh 'curl -fsSL https://get.docker.com | sh' >/dev/null
  rsudo 'systemctl enable --now docker'
  ok "installed: $(rsh 'docker --version')"
fi

# ---------------------------------------------------------------------------
# 5. Remote firewall (UFW) — open SSH first to avoid lockout
# ---------------------------------------------------------------------------
step "5/12  Firewall"
if rsh 'command -v ufw >/dev/null 2>&1'; then
  if confirm "Configure UFW (allow 22, 80, 443; enable)?"; then
    rsudo 'ufw allow 22/tcp'  >/dev/null || true
    rsudo 'ufw allow 80/tcp'  >/dev/null || true
    rsudo 'ufw allow 443/tcp' >/dev/null || true
    rsudo 'ufw --force enable' >/dev/null || true
    ok "UFW: 22/80/443 open."
  else
    info "skipped UFW."
  fi
else
  info "ufw not installed; skipping (ensure 22/80/443 are reachable some other way)."
fi

# ---------------------------------------------------------------------------
# 6. Build the image locally + pull base images so they can be saved
# ---------------------------------------------------------------------------
step "6/12  Build ${IMAGE} for ${TARGET_PLATFORM} (bakes embed model + sparse encoder + bottles)"
docker build --platform "$TARGET_PLATFORM" -f Dockerfile.prod -t "$IMAGE" .
ok "built $(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')"
info "pulling base images for ${TARGET_PLATFORM} so they ship from here (server pulls nothing)..."
docker pull --platform "$TARGET_PLATFORM" "$QDRANT_IMAGE" >/dev/null
docker pull --platform "$TARGET_PLATFORM" "$CADDY_IMAGE"  >/dev/null
docker pull --platform "$TARGET_PLATFORM" "$ALPINE_IMAGE" >/dev/null
ok "have ${QDRANT_IMAGE}, ${CADDY_IMAGE}, ${ALPINE_IMAGE} for ${TARGET_PLATFORM}."

# ---------------------------------------------------------------------------
# 7. Ship images over SSH (docker save | gzip | docker load)
# ---------------------------------------------------------------------------
step "7/12  Ship images to ${SERVER_IP} (this is the slow part, ~1 GB)"
PV=cat; have pv && PV="pv"
docker save "$IMAGE" "$QDRANT_IMAGE" "$CADDY_IMAGE" "$ALPINE_IMAGE" \
  | gzip \
  | $PV \
  | ssh "${SSH_OPTS[@]}" "$REMOTE" 'gunzip | docker load'
ok "images loaded on the server."

# ---------------------------------------------------------------------------
# 8 + 9. Transfer the Qdrant index (stream the volume straight into a fresh
#        remote volume, no temp files)
# ---------------------------------------------------------------------------
step "8/12  Export local Qdrant index (brief local qdrant stop for a consistent copy)"
docker compose stop qdrant >/dev/null 2>&1 || true
trap 'docker compose start qdrant >/dev/null 2>&1 || true; cleanup' EXIT

step "9/12  Restore index into remote volume ${REMOTE_QDRANT_VOLUME}"
# Ensure no remote qdrant holds the volume, then recreate it fresh.
rsh "cd '$REMOTE_DIR' 2>/dev/null && docker compose -f docker-compose.prod.yml rm -sf qdrant" >/dev/null 2>&1 || true
rsh "docker volume rm '$REMOTE_QDRANT_VOLUME'" >/dev/null 2>&1 || true
rsh "docker volume create '$REMOTE_QDRANT_VOLUME'" >/dev/null
# Local side reads the volume with the same (server-arch) alpine — emulated here
# if arches differ, but it's only tarring data bytes so that's fine. The remote
# side runs the alpine natively (it matches the server).
docker run --platform "$TARGET_PLATFORM" --rm -v "${LOCAL_QDRANT_VOLUME}:/data:ro" "$ALPINE_IMAGE" tar czf - -C /data . \
  | ssh "${SSH_OPTS[@]}" "$REMOTE" "docker run --rm -i -v '${REMOTE_QDRANT_VOLUME}:/data' '${ALPINE_IMAGE}' tar xzf - -C /data"
docker compose start qdrant >/dev/null 2>&1 || true
trap cleanup EXIT
ok "index transferred."

# ---------------------------------------------------------------------------
# 10. Sync config to the server
# ---------------------------------------------------------------------------
step "10/12 Sync compose / Caddyfile / .env.prod to ${REMOTE_DIR}"
rsh "mkdir -p '$REMOTE_DIR'"
scp "${SSH_OPTS[@]}" docker-compose.prod.yml Caddyfile .env.prod "${REMOTE}:${REMOTE_DIR}/" >/dev/null
ok "config in place."

# ---------------------------------------------------------------------------
# 11. Bring the stack up
# ---------------------------------------------------------------------------
step "11/12 Start the stack"
rsh "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml up -d"
rsh "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml ps"

# ---------------------------------------------------------------------------
# 12. Verify
# ---------------------------------------------------------------------------
step "12/12 Verify (waiting for api health + Caddy certificate)"
healthy=""
for i in $(seq 1 24); do
  if curl -fsS --max-time 5 "https://${DOMAIN}/health" >/dev/null 2>&1; then
    healthy="https"; break
  fi
  sleep 10
done
if [ "$healthy" = "https" ]; then
  ok "https://${DOMAIN}/health is up with a valid certificate."
else
  warn "https health check didn't pass yet. Checking the stack directly..."
  rsh "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml ps"
  rsh "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml logs --tail=20 caddy" || true
  warn "If DNS only just propagated, give Caddy a minute and re-check https://${DOMAIN}/health."
fi

cat <<EOF

${C_GREEN}Deploy complete.${C_OFF}
  URL            https://${DOMAIN}
  AUTH_PASSWORD  ${GEN_PW}
  API auth       Authorization: Bearer ${GEN_PW}   (or header X-Auth-Password)
  Logs           ssh ${REMOTE} "cd ${REMOTE_DIR} && docker compose -f docker-compose.prod.yml logs -f"
EOF
