# Measured resource results — 2026-10-03

These are local **macOS ARM64 CPU** measurements, not Render performance claims. Python 3.14.8, PyTorch 2.14.1, one CPU thread, weights-only checkpoint from epoch 19. Each resolution/profile ran in a fresh process. Inputs were synthetic, solid-color RGB images; one cold inference followed by one warm run. Timing is illustrative, not a statistically stable SLA. Larger runs below exceed the default API pixel budget and were called directly through the model for capacity exploration.

| Resolution | Tile size | Model load (s) | Cold (s) | Warm (s) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| 800×600 | 384 | 0.015 | 1.553 | 1.475 | 961.31 |
| 1920×1080 | 384 | 0.017 | 7.168 | 5.947 | 1005.45 |
| 800×600 | 192 | 0.014 | 1.757 | 1.725 | 410.14 |
| 1920×1080 | 192 | 0.015 | 7.110 | 6.993 | 441.75 |
| 2560×1440 | 192 | 0.014 | 12.868 | 12.817 | 478.39 |
| 3840×2160 | 192 | 0.015 | 29.782 | 29.459 | 557.50 |
| 4000×3000 | 192 | 0.015 | 41.439 | 41.500 | 620.61 |

RSS comes from OS high-water memory (`resource.getrusage`), including PyTorch/native allocations. The recorded initial process peak was about 206 MiB; weights-only model loading raised it to about 222 MiB. CPU activation/runtime allocations dominate small-image memory. The original 800×600 implementation measured 1,058.42 MiB with its unnecessary torchvision import and training checkpoint. At the unchanged 384px tile context, the new pipeline measured about 961 MiB.

The 192px profile lowers activation memory substantially. It is explicitly selected in the Render template, while the application default remains 384 to preserve the previous tile context. Smaller tiles can change enhancement near/context around tile borders; they are not a promise of identical image quality. Model-version headers include the selected tile size.

The default 2,073,600-pixel budget permits 1080p but rejects 1440p, 4K and 12MP photographs. The exploratory results support keeping that limit until the Linux deployment is measured. Full-image input/output and row-band memory still grow with pixels/width, and file/output encoding adds memory beyond model-only runs.

## Complete API check

The real lifespan/model was also exercised through the ASGI client with exported weights and 192px tiles: readiness 200, JPEG 32×24 output 200, PNG 1920×1080 output 200, matching binary media types/dimensions, and missing-key rejection 401. The 1080p PNG request took **7.175 seconds** end to end, with **7,119.88 ms** reported model inference. Peak process RSS was **461.81 MiB**, including the in-process test client, upload handling, decode and encode. This remains a synthetic local measurement, not a Linux cgroup or live-network benchmark.

## Regression evidence

A seeded 800×600 RGB input was run through the original and new implementations with the actual checkpoint and 384px tiles: maximum pixel difference 0, mean difference 0. Both runs used the same local PyTorch 2.14.1 environment; the deployed runtime's library version was not inspected. This is a numerical stitching regression, not underwater quality evaluation. The permanent test suite also compares awkward tile edges, narrow/tiny images and 192px feathering against the original accumulator, and compares actual checkpoint output with the original tile path.

The original checkpoint is 23,229,339 bytes (22.15 MiB), containing weights plus optimizer/scheduler state. A weights-only export was 7,761,755 bytes (about 7.40 MiB); minor serialization-size variation depends on the export filename. Both original/exported artifacts preserve epoch 19 and the same model version through the source SHA-256.

## Deployment measurements still required

No Docker executable is available locally, and the sandbox prevents binding a localhost TCP listener. Automated checks use Starlette's in-process ASGI client, including real model startup and JPEG/PNG output. CI includes Linux tests and a Docker build, but it has not been run remotely for this unpushed checkout.

Before increasing limits or treating the service as production-ready, measure the deployed Linux container with representative JPEG/PNG images, maximum accepted dimensions/payloads and concurrent calls. Record **container/cgroup peak memory**, upload/decode/encode overhead, cold starts and sustained CPU latency. These local values cannot establish that Render's configured memory budget is met.

No labeled underwater validation set was supplied. Review real enhancement samples and retain approved golden images/quality metrics before adopting the 192px profile or claiming better visual quality. HD/SR was not enabled.
