# First deployment: lightweight PolePilot on Amazon Linux 2023

This profile serves the homepage, recorded real-model inspections, survey suggestions, reviews, exports and Andhra Pradesh demo map. It deliberately excludes live inference, videos, PostgreSQL and the authenticated workspace. It is a candidate for a small initial VM, subject to its actual free memory and existing workloads. The full model stack should use a larger VM.

## Read-only server checks first

```sh
free -h
uname -m
docker version
docker compose version
docker ps --format 'table {{.Names}}\t{{.Ports}}'
ss -ltnp
```

The server shown already has Docker containers. Do not stop or replace them. If ports80/443 belong to an existing reverse proxy, add PolePilot to that proxy instead of starting this profile's proxy service; deployment needs to be adapted after inspecting the existing setup. Do not apply global Docker cleanup commands.

Obtain the EC2 **public IPv4** from the AWS console; the172.31.x.x hostname is a private address. SSH normally uses ec2-user for Amazon Linux, with the instance's key pair and an inbound SSH rule for the connecting IP. The AWS console terminal remains an alternative if SSH is not configured.

## Transfer and start after the checks

Use a dedicated directory such as `/opt/polepilot-demo` and transfer the prepared demo bundle. If using the complete repository, it also works with this profile; the demo image excludes model weights and videos. The public site is not configured until secrets, DNS and networking are ready.

Create a secret locally on the server (no need to send its value in chat):

```sh
umask 077
python3 - <<'PY'
import secrets
from pathlib import Path
path = Path('.env.demo')
with path.open('x') as output:
    output.write('DOMAIN=surveyai.live\nAPP_SECRET_KEY=' + secrets.token_hex(32) + '\n')
PY
docker compose --env-file .env.demo -f compose.demo.yaml config --quiet
docker compose --env-file .env.demo -f compose.demo.yaml up -d --build
docker compose --env-file .env.demo -f compose.demo.yaml ps
```

If Compose reports `build requires buildx 0.17.0 or later` on Docker25 with an older plugin, this Dockerfile also supports that engine's legacy builder:

```sh
DOCKER_BUILDKIT=0 docker build -t polepilot-demo-app -f Dockerfile.demo .
docker compose --env-file .env.demo -f compose.demo.yaml up -d --no-build
```

This compatibility path was successfully used on the supplied AmazonLinux VM. Prefer a supported matching Docker/Compose/buildx installation for future builds.

The secret creation refuses to overwrite an existing file. Preserve it across restarts and migration to keep session signing stable.

## DNS and AWS networking

With Namecheap BasicDNS, set the apex A record (`@`) to the confirmed EC2 publicIPv4. Configure DNS at the authoritative provider if different nameservers are used. Leave existing mail records alone. Permit incoming TCP80/443 in the EC2 security group; PostgreSQL/app ports stay private. Caddy obtains HTTPS after DNS and network access work. The shared Caddyfile serves the apex domain; www requires a separate hostname/redirect configuration.

```sh
curl --fail https://surveyai.live/health/ready
```

Verify homepage, `/demo`, `/map`, a replay, review persistence and exports before sharing. This profile has a128MB app-container limit and64MB proxy limit; those are protective caps, not a total-host memory estimate or capacity guarantee. Image builds and other running applications need additional memory. The supplied server snapshot reports911MiB total,384MiB available and1GiB swap in use while Superset is running. Check actual usage before building or starting; a successful deployment on that host has not yet been established.

## Move to another VM later

Copy the bundle or repository and `.env.demo`, and back up/restore the `polepilot-demo_demo_data` named volume with the application stopped so SQLite is consistent. Bring up the new stack and verify it before moving DNS to the new publicIP. Keep the old server available during DNS propagation. Caddy may obtain certificates on the new server once reachable; avoid repeated unnecessary certificate issuance.

If enabling the full workspace on the new VM, use the full `compose.yaml`, provision PostgreSQL/admin access and validate each advanced workflow. Demo SQLite reviews do not automatically become production PostgreSQL assets. No conversion is implied by switching profiles.

Prepared locally; Docker image build and remote deployment still need verification on the actual server.
