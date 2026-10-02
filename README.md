# AquaVision ML Service

A backend-facing FastAPI service for underwater image enhancement with a 1,928,483-parameter PyTorch U-Net. The API returns JPEG/PNG bytes at the input's EXIF-corrected dimensions. It provides color/contrast enhancement; it does not provide super-resolution.

Existing deployment: https://aquavision-ml-service-cxf5.onrender.com

**The changes in this checkout are not deployed yet.** The existing deployment was checked on 2026-10-03 and exposes `/health` and `/enhance`. The contract below describes this checkout's version 0.2.0.

## Run locally

Python 3.14 is used by Docker and CI. Versions were tested locally with Python 3.14.8 and PyTorch 2.14.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
# macOS (CPU/MPS wheel):
python -m pip install torch==2.14.1 -r requirements.txt -r requirements-dev.txt
# Linux CPU instead:
# python -m pip install -r requirements-cpu.txt -r requirements-dev.txt

export AQUAVISION_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export AQUAVISION_DEVICE=cpu
export TORCH_NUM_THREADS=1
# Lower-memory profile; default 384 retains the old tile context:
export INFERENCE_TILE_SIZE=192
python -m uvicorn main:app --host 0.0.0.0 --port 10000 --workers 1
```

Environment variables must be exported or configured by your hosting platform. `.env.example` documents them; the service does not automatically load `.env` files. Set the same API key in the calling backend. Keep it out of browser code, source control, and logs.

`best.pt` is the trusted local training checkpoint (epoch 19). To use a smaller artifact locally:

```bash
python export_weights.py best.pt inference.pt
export MODEL_PATH="$PWD/inference.pt"
```

The export retains weights, epoch and the source checkpoint SHA-256. Docker exports it during its build and copies only inference weights into the final runtime image. The loader uses `weights_only=True`; missing/incompatible weights leave `/ready` at 503 and cannot produce successful inference responses.

## API contract for the backend engineer

| Endpoint | Behavior |
|---|---|
| `GET /` | Service and documentation links |
| `GET /health` | Process liveness, always 200 while responding; includes model status |
| `GET /ready` | 200 when model and API key are configured; otherwise 503 |
| `GET /info` | Actual model version, limits, device and tile size |
| `GET /docs` | Interactive OpenAPI documentation with API-key authorization |
| `POST /v1/enhance` | Preferred enhancement endpoint |
| `POST /enhance` | Same handler; compatible path retained |

Both POST endpoints require:

- `X-API-Key: <backend-only secret>`.
- `Content-Type: multipart/form-data`, exactly one upload under field `file`.
- A valid, single-frame JPEG or PNG. Actual decoded format is checked, rather than trusting the filename or MIME type.

Optional checksum headers are `X-Model-Input-SHA256` and the legacy `X-SHA256-Checksum`. Hash the original upload bytes; if both headers are supplied, both must match.

```bash
curl --fail-with-body --max-time 180 \
  -H "X-API-Key: $AQUAVISION_API_KEY" \
  -F "file=@underwater.jpg" \
  http://localhost:10000/v1/enhance \
  --output enhanced.jpg
```

After rollout, replace the localhost base URL with the deployed URL. The 180-second timeout is an initial integration setting, not a measured Render SLA. Tune it using deployment measurements.

Successful responses contain raw `image/jpeg` or `image/png` bytes in the input format. PNG alpha/grayscale input becomes RGB. EXIF orientation is applied and metadata is stripped; a rotated phone image can therefore swap width and height. Do not assume the output's stored dimensions match the unrotated input header.

Response headers:

- `X-Request-ID`: generated per request, also included in error JSON and server logs.
- `X-Model-Version`: pipeline revision, tile/overlap configuration and source checkpoint hash prefix.
- `X-Inference-Ms`: model enhancement time, excluding upload parsing and output encoding.
- `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

Errors use a stable JSON shape:

```json
{"error": "invalid_image", "message": "Upload a valid JPEG or PNG image.", "request_id": "..."}
```

| Status | Error codes / handling |
|---|---|
| 400 | `missing_file`, `empty_file`, `invalid_image`, `unsupported_format`, `checksum_mismatch`, `invalid_request`; correct the input |
| 401 | `unauthorized`; fix the backend's secret/header |
| 413 | `file_too_large`, `image_too_large`; resize/compress before resubmission |
| 503 | `service_busy`, `compute_unavailable`, `model_unavailable`, `service_unconfigured`; inspect readiness/configuration |
| 500 | `inference_failed`; correlate `X-Request-ID` with service logs |

`service_busy` and `compute_unavailable` include `Retry-After: 5`. Use bounded retries with backoff for busy responses; do not repeatedly retry an image that exhausts memory. Avoid automatic immediate retries on an inference timeout: the original worker may still be running. This service has no job queue or idempotency store.

## Limits and resource behavior

| Environment variable | Default | Meaning |
|---|---|---|
| `AQUAVISION_API_KEY` | unset | Required; absence disables enhancement and readiness |
| `AQUAVISION_DEVICE` | `cpu` | `cpu`, `cuda`, `mps`, or explicit `auto` |
| `TORCH_NUM_THREADS` | `1` | Positive CPU thread count |
| `INFERENCE_TILE_SIZE` | `384` | Multiple of 8 between 72 and 384; overlap stays 64 |
| `MAX_FILE_SIZE_MB` | `20` | Upload file cap in MiB |
| `MAX_IMAGE_PIXELS` | `2073600` | Conservative 1080p pixel budget |
| `MAX_IMAGE_DIMENSION` | `4096` | Maximum width or height, also limits row-buffer memory |
| `MODEL_PATH` | local `best.pt` | Docker sets `/app/inference.pt` |
| `CORS_ORIGINS` | empty | Optional comma-separated explicit browser origins |
| `PORT` | Docker `10000` | HTTP listening port |

The request-body cap includes 64 KiB of multipart overhead. It is enforced while receiving the body even without `Content-Length`; individual file bytes are separately capped. Pixel/dimension limits are checked before decoding the whole image. Animated PNGs are rejected.

Input/output images remain byte images. Only individual tiles and a row band use float buffers, preserving the original feathered blending. Memory still grows with input/output pixels and row width; it is bounded by the configured limits, not constant for arbitrary images.

One upload is admitted per process before multipart parsing. Inference, decode and encode run in a worker thread so health checks stay responsive. Additional inference requests fail fast with 503 rather than accumulating a queue. A worker-owned lock prevents overlapping inference even after caller cancellation. Keep one Uvicorn worker; multiple workers duplicate models and bypass the per-process limit.

The default 384px profile reproduces the previous tile context. The 192px profile lowers activation memory and can alter enhancement at tile borders/context. Its version header differs. Review representative underwater samples before selecting it. The Blueprint explicitly selects 192 for the demo; this is not an automatic change to your existing Render service.

## Tests and benchmarks

```bash
python -m unittest discover -s tests -v
python -m pip check
# Run each size in a fresh process for meaningful peak RSS:
python benchmark.py --width 800 --height 600 --tile 192 --runs 3
python benchmark.py --width 1920 --height 1080 --tile 192 --runs 3
# Optional real image, not supplied with this repository:
python benchmark.py --image underwater.jpg --tile 192 --runs 3
```

Tests cover the binary API contract, authentication, readiness failures, validation/checksums, streaming limits, concurrent requests, orientation, feathered stitching and actual checkpoint inference/export. CI runs these offline on Linux; no test contacts the deployed API. The root `test_api.py` is an older manual live-service script and is not part of this suite.

The benchmark uses the OS process high-water RSS (`resource.getrusage`) including native allocations. It reports cold/warm times, model load time, device and parameters. GPU timing is synchronized; CUDA allocation is reported separately. Solid-color inputs measure resources, not image quality. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for measured local results and deployment limitations.

## Deploy the existing Render service

1. Configure the caller to send `X-API-Key` using a shared random secret. The old service ignores the extra header, which allows preparing the caller before rollout.
2. In the existing Render service's environment, set that same `AQUAVISION_API_KEY`, `AQUAVISION_DEVICE=cpu`, `TORCH_NUM_THREADS=1`, `INFERENCE_TILE_SIZE=192` and the documented upload/pixel/dimension limits. Select 384 instead if preserving tile context is required and resources permit it.
3. Deploy this repository's Dockerfile and set the health check path to `/ready`. Render uses successful HTTP status codes for deployment health ([official health-check documentation](https://render.com/docs/health-checks)).
4. Check `/ready` and `/info`, then test valid JPEG/PNG, an invalid key, pixel-limit rejection and simultaneous requests. Benchmark peak container/cgroup memory and latency at the permitted maximum; process RSS alone does not capture every cgroup allocation.
5. Only increase image limits or promise latency after those measurements. Free Render services can sleep when idle ([official free-service documentation](https://render.com/docs/free)); plan for cold starts.

`render.yaml` is a reproducible template for a new Blueprint, not a mechanism for updating an existing unmanaged service. Do not create a duplicate service accidentally. Docker was not built locally because no Docker executable is available on this workstation; the Linux build and CPU wheel must be verified in CI/Render.

The audited repository URL and this checkout's Git remote differ. Confirm which repository/branch the existing Render service tracks before publishing changes.

## Responsibility boundary and remaining work

This project owns model loading, preprocessing, enhancement, API validation, deployment and backend handoff. Application users/auth, credits, database, storage, LLMs and frontend integration belong to the main backend/application team.

The HD/SR helpers in `hd_pipeline.py` are experimental and not called by the API. They still use full-image floats, lack bundled SR weights, and must not be advertised as production 4K enhancement. No new SR dependency is added.

Remaining work needs deployment access and representative labeled underwater images: verify the Linux memory budget, review the 192px profile visually, establish golden input/output samples with quality metrics, and select a latency target with the backend team. Use a larger CPU instance or evaluate GPU only if those measurements justify it.
