# Deploy PolePilot on surveyai.live

The domain can remain registered at Namecheap. Run the application on a Linux VPS with Docker Compose; CPU inference and PostgreSQL need a persistent backend. This package does not purchase hosting, modify DNS, or publish the site.

## Prepare the server

Use a Linux VPS with at least 2 vCPU and 4 GB RAM for evaluation; 8 GB gives more room for video/model workloads. Actual throughput depends on CPU and input resolution; no hosting load benchmark has been performed. Allow inbound TCP 80/443 and SSH only as needed. Do not expose PostgreSQL or port 8000 publicly.

Copy this directory, including `models/` and `static/demo_samples/`, to the VPS. Keep `.env`, databases and training exports out of public repositories. Existing company records are not needed for the evaluator; the Docker database starts fresh.

```sh
cp .env.example .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Generate two separate random secrets and put them into `.env` as `APP_SECRET_KEY` and `POSTGRES_PASSWORD`. Keep `DOMAIN=surveyai.live`, `WORKSPACE_ENABLED=true`, `APP_ENV=production`. Keep public live inference disabled initially or set `LIVE_INFERENCE_ENABLED=true` after checking server capacity. Optional Azure/PlantNet keys enable only the corresponding workspace integrations.

```sh
docker compose config --quiet
docker compose up -d --build
docker compose exec app python manage.py create-admin --username your-admin-name
docker compose ps
docker compose logs --tail=100 app
```

Administrator provisioning prompts for a password; no credentials are embedded in source. Sign in at `/login`. Workers start on the first video submission, including under Gunicorn.

## Namecheap DNS

In Namecheap → Domain List → surveyai.live → Manage → Advanced DNS, if using Namecheap BasicDNS:

| Type | Host | Value |
| --- | --- | --- |
| A | @ | VPS public IPv4 address |
| CNAME | www | surveyai.live |

Use the authoritative DNS provider if custom nameservers are configured. Remove conflicting parking/redirect records for the same host. Add an AAAA record only if the VPS has working public IPv6. Do not change mail/MX records.

The default Caddy configuration serves the apex **surveyai.live**. The optional www DNS record alone does not enable the www hostname in Caddy. To serve it too, add `www.surveyai.live { redir https://surveyai.live{uri} permanent }` to `deploy/Caddyfile` and reload Caddy after DNS resolves. Caddy obtains and renews HTTPS certificates when the domain points to the server and ports 80/443 are reachable. HTTPS is required for secure sessions and camera/geolocation on a public domain.

## Verify before sharing

```sh
curl --fail https://surveyai.live/health/live
curl --fail https://surveyai.live/health/ready
```

Open the landing and `/demo` from a separate browser. Replay all three samples, save a review, export JSON/CSV, verify only entered coordinates create markers, and reset. Sign in with the newly provisioned admin and verify a test image/video, survey, PDF/Excel export and annotation export in the fresh database. Configure optional providers separately and verify their actual outputs before advertising them as enabled.

## Map service

The public demo uses bundled Leaflet 1.9.4 with OpenStreetMap Standard tiles. The dedicated `/map` view loads the basemap when visible and includes a separately labeled fictional eight-pole Andhra Pradesh layer for presentations. Actual session locations come from entered coordinates; fictional poles are not written into inspection records or their GeoJSON export. It needs no API key. Show visible OSM attribution, preserve browser referrers and caching, and do not add bulk/offline tile downloads. Community tiles have no availability guarantee; use an appropriate commercial/self-hosted tile provider if traffic grows. Coordinates and GeoJSON remain usable when external maps are unavailable. The app does not send uploaded photographs to OSM.

HERE is an alternative if its current account terms suit the project; it is not integrated or configured in this build. Review current limits and pricing at https://www.here.com/get-started/pricing before choosing a paid or metered provider.

## Operations and limitations

- Keep one Gunicorn process: evaluation/video queues are process-local. Restarted evaluation jobs fail with a retry message; legacy video jobs do not survive restart. Use a durable external queue before horizontal scaling.
- Back up PostgreSQL, app_data and training_data volumes before updates. Never run `docker compose down -v` against data you intend to retain.
- Public recorded runs remain usable when live inference is disabled. Set `WORKSPACE_ENABLED=false` to publish only the evaluator if the full workspace is not ready.
- The full workspace retains legacy CDN styling and some legacy storage paths. Its public hosting checks must be completed on the actual VPS; local isolated API tests do not certify all legacy authorization/data flows.
- Models and demo photographs are included under the company's permission described by the project owner; do not assign an open-source license without confirming it.
- Docker Desktop's engine was not running during preparation. Compose configuration was validated, but image build, TLS issuance, DNS changes, live map tiles and full PostgreSQL workflows must be verified on the deployment machine.
