# VoiceCoach — Production Operations

Everything needed to deploy, debug, and recover the production instance of this app.
See [DESIGN.md](DESIGN.md) for application architecture and [README.md](README.md) for
local dev setup. This document is about the **server**, not the code.

---

## Table of Contents

1. [Access](#1-access)
2. [Infrastructure Map](#2-infrastructure-map)
3. [voassess Container Details](#3-voassess-container-details)
4. [Nginx Routing](#4-nginx-routing)
5. [Deployment Procedure](#5-deployment-procedure)
6. [Rollback Procedure](#6-rollback-procedure)
7. [Shared Auth Backend (PocketBase)](#7-shared-auth-backend-pocketbase)
8. [Cloudflare / CDN Caching](#8-cloudflare--cdn-caching)
9. [Disk Space Management](#9-disk-space-management)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Command Reference](#11-command-reference)

---

## 1. Access

| What | Value |
|---|---|
| Server | `root@91.99.144.77` — SSH, key-based |
| Container runtime | Podman (rootful, no restart-policy automation by default) |
| Live domain | `https://voassess.auravo.ai` — proxied through **Cloudflare** |
| Auth backend | `https://pb.auravo.ai` (PocketBase, shared across apps — see §7) |
| GitHub repo | `git@github.com:talkinglabs/voassess.git`, branch `main` |

No app-specific secrets live in this repo or on this server for voassess itself — it needs
no API keys (Whisper and grammar checking are both local). The only external credential in
play is the shared PocketBase Google OAuth2 config (§7).

---

## 2. Infrastructure Map

Single server hosting multiple apps as separate Podman containers, fronted by one shared
`router` (nginx) container that terminates TLS and reverse-proxies by hostname. All
containers below run on the same host; the ones relevant to voassess's auth are on the
`voca` Podman network so they can resolve each other by container name.

| Container | Image | Role |
|---|---|---|
| `router` | `nginx:alpine` | Reverse proxy / TLS termination for all `*.auravo.ai`, `*.talkinglabs.in`, etc. domains |
| `voasses` | `localhost/voassess:0.2` | **This app.** Note the container name has no double-`s` typo fix applied — it's `voasses`, not `voassess`. The nginx upstream depends on this exact name. |
| `auth` | `localhost/pocketbase:v0.36.5` | Shared PocketBase auth backend for every app on this server |
| `postgres` | `postgres:18.3-trixie` | Shared Postgres, used by other apps (not voassess) |
| `stalwart` | `stalwartlabs/stalwart` | Mail server |
| `voca`, `ic-backend`, `auravo-web`, `talking-english-backend`, `voice-clone`, `vaicoach-backend`, `tl-coach-backend` | various | Other apps on this box, unrelated to voassess but share the server's disk, and some share the `auth` PocketBase instance |

voassess itself has **no database or cache dependency on any other container** — its only
runtime dependency is the filesystem volume for SQLite + recordings. Its *auth* depends on
the shared `auth` container, but that's client-side (browser talks directly to
`pb.auravo.ai`); the FastAPI backend itself never calls PocketBase.

---

## 3. voassess Container Details

```
Container name : voasses
Image          : localhost/voassess:0.2
Network        : voca            (needed so nginx's `voasses:5050` DNS lookup resolves)
Port            : 5050 internal only — no host port published; nginx reaches it via the
                  podman network, not localhost
Volume         : /root/voassess  (host)  →  /app/data  (container)
Restart policy : none — if the container dies it stays down until manually restarted
```

The volume is the **only** persistent state: `data/assessments.db` (SQLite) and
`data/recordings/` (raw + converted audio, never auto-deleted — see DESIGN.md §17). Nothing
else in `/app` inside the container is meant to persist; the app source is baked into the
image at build time, not bind-mounted.

### Image build

There's no long-lived build directory kept on the server — a fresh one is created per
deploy (see §5). The Dockerfile (reconstructed from the existing image's history, since no
Dockerfile shipped in the original repo):

```dockerfile
FROM python:3.14-slim-trixie

RUN apt update && apt install -y --no-install-recommends ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --break-system-packages -r /app/requirements.txt

WORKDIR /app

CMD ["python", "app.py"]
```

**No Java/JRE is installed.** This means `language_tool_python` is unavailable in
production and the app always runs with `GRAMMAR_OK = False`, falling back to the crude
repeated-word regex (see DESIGN.md §14 and §17). This is expected, current, unchanged
behavior — not a bug to fix.

---

## 4. Nginx Routing

Config file: `/root/router/nginx/conf.d/voassess.conf` (on the server, inside the `router`
container's mounted config — edit on the host, nginx picks it up on reload/restart of that
container).

```
upstream voassess_backend {
    server voasses:5050 resolve;   # DNS lookup by container name on the `voca` network
}

server_name voassess.auravo.ai;
proxy_pass http://voassess_backend;

client_max_body_size 100m;
proxy_connect_timeout 30s;
proxy_send_timeout    60s;
proxy_read_timeout    60s;
```

Access/error logs: `/var/log/nginx/voassess_access.log` / `voassess_error.log` inside the
`router` container (`podman exec router sh -c "tail -f /var/log/nginx/voassess_access.log"`).

TLS certs are the shared wildcard `auravo.ai.cert.pem` / `.key.pem`, not per-app.

---

## 5. Deployment Procedure

There is no CI/CD — deploys are manual, done by building the image directly **on the
server** (not locally and transferred — the image is large, ~20GB with torch/whisper
dependencies, and the server is arm64, so building where it'll run avoids cross-arch and
transfer-time problems entirely).

```bash
# 1. Push code changes to GitHub first (source of truth)
git push origin main

# 2. Sync source to a fresh build directory on the server (data/ dir is excluded —
#    it's the persistent volume, must never be touched by a deploy)
ssh root@91.99.144.77 "rm -rf /root/voassess-build && mkdir -p /root/voassess-build"
rsync -az --exclude 'venv/' --exclude 'data/' --exclude '__pycache__/' --exclude '.git/' \
  --exclude '.DS_Store' --exclude '*.pptx' \
  ./ root@91.99.144.77:/root/voassess-build/

# 3. Write/confirm the Dockerfile is present in the build dir (see §3), then build with a
#    bumped tag — never overwrite the currently-running tag, so rollback stays trivial
ssh root@91.99.144.77 "cd /root/voassess-build && podman build -t localhost/voassess:0.3 ."

# 4. Smoke-test the new image on a throwaway port BEFORE touching live traffic
ssh root@91.99.144.77 "podman run --rm -d --name voassess-test -p 15050:5050 \
  -v /root/voassess:/app/data localhost/voassess:0.3"
# wait ~10-15s for Whisper model load, then:
ssh root@91.99.144.77 "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:15050/app"
ssh root@91.99.144.77 "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:15050/assessment/manifest"
ssh root@91.99.144.77 "podman stop voassess-test"

# 5. Swap the live container — rename the old one out of the way instead of deleting it
#    (instant rollback target, see §6), then start the new one under the exact name/
#    network/mount the nginx config and data volume depend on
ssh root@91.99.144.77 "podman rename voasses voasses-old-0.2 && podman stop voasses-old-0.2"
ssh root@91.99.144.77 "podman run -d --name voasses --network voca \
  -v /root/voassess:/app/data localhost/voassess:0.3"

# 6. Verify on the live public domain (through nginx + Cloudflare, not the internal port)
curl -s -o /dev/null -w '%{http_code}' https://voassess.auravo.ai/app
curl -s https://voassess.auravo.ai/history | python3 -c \
  "import json,sys; print(len(json.load(sys.stdin)), 'rows — confirms data volume intact')"
```

**After every deploy that touches `script.js`, `style.css`, `index.html`, or
`assessment.js`: purge the Cloudflare cache.** See §8 — this is not optional, the site will
silently serve the old JS/CSS to every visitor otherwise.

If disk space runs out mid-build (`no space left on device`), see §9 before retrying.

---

## 6. Rollback Procedure

Every deploy renames the previous container instead of removing it, so rollback is just:

```bash
ssh root@91.99.144.77 "podman stop voasses && podman rename voasses voasses-broken"
ssh root@91.99.144.77 "podman rename voasses-old-0.2 voasses && podman start voasses"
```

(substitute the actual previous tag's container name — check `podman ps -a | grep voasses`
for what's available). Old images are *not* deleted by the deploy process either, only
pruned manually (§9), so you can also rebuild any prior tag if the renamed container itself
was cleaned up.

---

## 7. Shared Auth Backend (PocketBase)

Full details in the [pocketbase-auth skill] — summarized here for what's specific to
production troubleshooting.

- **Instance**: `pb.auravo.ai`, container `auth` (`localhost/pocketbase:v0.36.5`), network IP
  on the `voca` network, data at `/root/voca/pb_data/` on the host.
- **voassess's role**: pure client — `login.js`/`auth-check.js` talk to `pb.auravo.ai`
  directly from the browser. The FastAPI backend has zero involvement in auth and holds no
  PocketBase credentials.
- **Superuser admin credentials**: not stored in this app. Found via
  `POCKETBASE_ADMIN_EMAIL` / `POCKETBASE_ADMIN_PASSWORD` in `/root/talking-english/.env` on
  the server (that app happens to hold the bootstrap admin credentials — a historical
  artifact, not a documented convention). **Password rotates** — if admin API calls
  suddenly 400 with an auth error, check whether a sibling app's `.env` has drifted stale
  after a rotation.
- **Google OAuth2 provider config**: lives on the PocketBase `users` collection itself
  (Admin UI → Collections → `users` → Options → OAuth2 providers → Google), *not* in any
  app's `.env`. Client ID is visible publicly via
  `GET https://pb.auravo.ai/api/collections/users/auth-methods` (safe, no secret exposed);
  the client secret is write-only through the API — PocketBase never echoes it back, so
  there's no way to verify it was saved correctly except a live login test.
- **Editing the OAuth2 config via API** (needed when Cloudflare/terminal access to the
  Admin UI isn't convenient): PocketBase's API is behind Cloudflare, and Python's default
  `urllib`/`requests` User-Agent gets blocked with **HTTP 403 / Cloudflare error 1010**. Set
  `User-Agent: curl/8.7.1` (or any non-flagged UA) on the request to get through.
- **Redirect URI**: the OAuth2 popup flow always redirects back through
  `https://pb.auravo.ai/api/oauth2-redirect` — **never** the calling app's own domain. Only
  that one URL needs to be registered in Google Cloud Console's "Authorized redirect URIs,"
  regardless of how many apps share this PocketBase instance.

---

## 8. Cloudflare / CDN Caching

`voassess.auravo.ai` (and the other `*.auravo.ai` domains) are proxied through Cloudflare.
Cloudflare caches static assets (`.js`, `.css`, images) at the edge **independently of the
origin** — deploying a new container does not invalidate anything Cloudflare already cached.

**Symptom**: after a deploy, the live site behaves like the old version — new features
don't appear, clicking things does nothing — even though the container is confirmed to have
the correct files on disk.

**Confirm it's this** (not a bad deploy):
```bash
curl -sI https://voassess.auravo.ai/script.js | grep -iE 'cf-cache-status|age'
# cf-cache-status: HIT  +  a large `age` value (seconds since cached) = confirmed stale edge cache
```
Cross-check against the container's actual file, to rule out a bad deploy first:
```bash
ssh root@91.99.144.77 "podman exec voasses stat -c '%y' /app/script.js"
# if this mtime is recent (matches your deploy) but the curl above shows an old `age`,
# it's Cloudflare, not the deploy
```

**Fix**: Cloudflare dashboard → Caching → Configuration → Purge Cache → purge by URL for
`script.js`, `assessment.js`, `style.css`, `/app` (or Purge Everything — low risk for this
app's traffic level). No server-side credential is available to automate this currently; do
it manually until cache-busting is added to the asset URLs.

---

## 9. Disk Space Management

The server has a single 75G root volume shared by every app's container images. Podman
image layers accumulate fast — repeated builds across all the apps on this box (not just
voassess) can silently eat tens of GB in dangling/unused layers.

**Symptom during a build**: `no space left on device` partway through `pip install` (the
Whisper/torch dependency layer is large, several hundred MB of wheels).

```bash
df -h /                          # check current usage
podman system df                 # breakdown — look at "Images" RECLAIMABLE column
podman image prune -a -f         # removes only images with ZERO containers using them —
                                  # safe, won't touch anything running (any app, not just
                                  # voassess), but affects shared image storage so confirm
                                  # with whoever owns other apps on this box first if in doubt
podman builder prune -f          # clear leftover build cache from a failed build
```

Freed ~9GB from ~180 dangling images the one time this happened (2026-08-16) — check current
numbers before assuming the same scale.

---

## 10. Troubleshooting Guide

### "Nothing happens" / a UI feature silently doesn't work after deploying
→ Almost always Cloudflare edge cache serving stale JS/CSS. See §8 first, before assuming
the deploy itself failed.

### Google Sign-In fails with "Failed to fetch OAuth2 token"
This is a **token exchange** failure (PocketBase ↔ Google, server-to-server), distinct from
a redirect URI problem. Diagnose by checking how far the flow got:

```bash
ssh root@91.99.144.77 "podman exec router sh -c \
  \"grep -E 'auth-with-oauth2|oauth2-redirect' /var/log/nginx/auth_access.log | tail -20\""
```

- If you see `GET /api/oauth2-redirect ... 307` (Google successfully redirected back with a
  code) **followed by** `POST .../auth-with-oauth2 ... 400` → the authorize step is fine,
  the **client secret** PocketBase has stored doesn't match the client ID's actual secret in
  Google Cloud Console (commonly: secret was regenerated in Google Console, or a client
  ID/secret pair got mismatched during an edit). Fix in PocketBase Admin UI → `users`
  collection → OAuth2 providers → Google → update Client secret.
- If Google itself shows an error page (`redirect_uri_mismatch`) and you never even see a
  `GET /api/oauth2-redirect` in the logs → the redirect URI isn't registered for the current
  Client ID in Google Cloud Console. Confirm the current client ID first:
  ```bash
  curl -s https://pb.auravo.ai/api/collections/users/auth-methods | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['oauth2']['providers'][0]['authURL'].split('client_id=')[1].split('&')[0])"
  ```
  then verify that exact ID's Console entry has `https://pb.auravo.ai/api/oauth2-redirect`
  in Authorized redirect URIs.
- This break affects **every app** sharing this PocketBase instance simultaneously (it's
  one shared `users` collection config) — if voassess's Google login is broken, check
  whether other apps (voca, vaicoach, talking-english, etc.) are also affected before
  assuming it's voassess-specific.

### Local dev: `[Errno 48] address already in use` on port 5050
A previous `python app.py` run (yours or a leftover test instance) is still bound.
```bash
lsof -nP -iTCP:5050 -sTCP:LISTEN     # find the PID
kill <PID>
```

### Container crash-loops with `panic: registerMigrations: ... types.d.ts`
Seen once on the `auth` (PocketBase) container, 2026-07-11 — self-resolved after ~1 minute
and a couple of restart attempts, hasn't recurred. If seen again: it's the JS migration
runtime choking on a stray `types.d.ts` file it's trying to interpret as a migration —
check `pb_migrations/` on the host for anything that shouldn't be there.

### `.env` value looks wrong / admin auth fails after a "known good" password
Admin passwords for the shared PocketBase rotate periodically and sibling apps' `.env`
files can drift stale (confirmed happened to `talking-english-backend` on 2026-07-18).
Don't assume any single app's copy of the credential is current — verify with a live
`auth-with-password` call before trusting it.

---

## 11. Command Reference

```bash
# SSH in
ssh root@91.99.144.77

# Container status / logs
podman ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
podman logs --tail 50 --timestamps voasses
podman exec voasses <cmd>                       # inspect files/state inside the running container

# Restart (picks up nothing new — same image; use this only after a crash, not for deploys)
podman restart voasses

# nginx logs (inside the shared router container)
podman exec router sh -c "tail -f /var/log/nginx/voassess_access.log"
podman exec router sh -c "tail -f /var/log/nginx/voassess_error.log"

# Public health checks
curl -s -o /dev/null -w '%{http_code}\n' https://voassess.auravo.ai/app
curl -s https://voassess.auravo.ai/assessment/manifest | python3 -m json.tool
curl -s https://voassess.auravo.ai/history | python3 -c "import json,sys; print(len(json.load(sys.stdin)), 'sessions')"

# Disk / image housekeeping
df -h /
podman system df
podman image prune -a -f
```
