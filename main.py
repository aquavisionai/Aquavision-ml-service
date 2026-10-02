"""AquaVision's backend-facing image enhancement API (one model, one worker)."""
import hashlib
import hmac
import io
import logging
import os
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from PIL import Image, UnidentifiedImageError
from torch import OutOfMemoryError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from inference import AquaVisionModel

logger = logging.getLogger("aquavision")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
SERVICE_VERSION = "0.2.0"
API_KEY = os.getenv("AQUAVISION_API_KEY", "")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
# Conservative demo limit; increase only after benchmarking the deployed container.
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "2073600"))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
if min(MAX_FILE_SIZE, MAX_IMAGE_PIXELS, MAX_IMAGE_DIMENSION) < 1:
    raise ValueError("Upload and image limits must be positive.")
model_instance: AquaVisionModel | None = None
# ponytail: one inference per process; use a job queue if measured demand needs more throughput.
request_slot = threading.Lock()
inference_slot = threading.Lock()  # Still held by the worker if its HTTP caller disconnects.


def api_error(status_code, code, message, headers=None):
    return HTTPException(status_code, {"error": code, "message": message}, headers=headers)


def error_response(scope, status_code, code, message, headers=None):
    return JSONResponse({"error": code, "message": message, "request_id": scope.get("request_id")},
                        status_code=status_code, headers=headers)


class RequestGuard:
    """Authenticate and admit one bounded upload before multipart parsing starts."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        scope["request_id"] = uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        admitted = False
        is_enhance = scope["method"] == "POST" and scope["path"].rstrip("/") in {"/enhance", "/v1/enhance"}

        async def send_headers(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = scope["request_id"]
                headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
            await send(message)

        async def reject(status, code, message, headers=None):
            await error_response(scope, status, code, message, headers)(scope, receive, send_headers)

        try:
            if is_enhance:
                if not API_KEY:
                    return await reject(503, "service_unconfigured", "The ML service API key is not configured.")
                headers = Headers(scope=scope)
                supplied = headers.get("X-API-Key", "")
                if not hmac.compare_digest(supplied.encode("utf-8"), API_KEY.encode("utf-8")):
                    return await reject(401, "unauthorized", "A valid X-API-Key header is required.")
                if model_instance is None:
                    return await reject(503, "model_unavailable", "The ML model is not ready.")
                length = headers.get("content-length")
                if length is not None:
                    try:
                        length = int(length)
                        if length < 0:
                            raise ValueError
                    except ValueError:
                        return await reject(400, "invalid_request", "Invalid Content-Length header.")
                    if length > MAX_FILE_SIZE + 65536:
                        return await reject(413, "file_too_large", "The upload exceeds the request size limit.")
                admitted = request_slot.acquire(blocking=False)
                if not admitted:
                    return await reject(503, "service_busy", "The ML service is processing another image.",
                                        {"Retry-After": "5"})
            received = 0

            async def bounded_receive():
                nonlocal received
                message = await receive()
                if is_enhance and message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > MAX_FILE_SIZE + 65536:
                        raise api_error(413, "file_too_large", "The upload exceeds the request size limit.")
                return message

            await self.app(scope, bounded_receive, send_headers)
        finally:
            if admitted:
                request_slot.release()
            logger.info("request_completed request_id=%s method=%s path=%r status=%s duration_ms=%.2f",
                        scope["request_id"], scope["method"], scope["path"], status_code,
                        (time.perf_counter() - started) * 1000)


@asynccontextmanager
async def lifespan(app):
    global model_instance
    if not API_KEY:
        logger.error("service_unconfigured: set AQUAVISION_API_KEY; readiness and enhancement return 503")
    checkpoint = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("best.pt"))))
    try:
        model_instance = await run_in_threadpool(AquaVisionModel, checkpoint_path=checkpoint)
    except Exception:
        logger.exception("model_load_failed")
        model_instance = None
    yield
    model_instance = None
    logger.info("model_unloaded")


app = FastAPI(title="AquaVision ML Service", version=SERVICE_VERSION,
              description="Same-resolution underwater image enhancement for the AquaVision backend.",
              lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["X-API-Key", "X-Model-Input-SHA256", "X-SHA256-Checksum", "Content-Type"],
                       expose_headers=["X-Request-ID", "X-Model-Version", "X-Inference-Ms"])
app.add_middleware(RequestGuard)
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": "invalid_request", "message": str(exc.detail)}
    return error_response(request.scope, exc.status_code, detail["error"], detail["message"], exc.headers)


@app.get("/")
async def root():
    return {"service": "aquavision-ml-service", "service_version": SERVICE_VERSION,
            "docs": "/docs", "health": "/health", "readiness": "/ready", "info": "/info"}


@app.get("/health")
async def health_check():
    loaded = model_instance is not None
    return {"status": "ok", "service": "aquavision-ml-service", "model_loaded": loaded,
            "device": model_instance.device if loaded else None,
            "epoch": model_instance.epoch if loaded else None}


@app.get("/ready")
async def readiness():
    ready = model_instance is not None and bool(API_KEY)
    return JSONResponse({"status": "ready" if ready else "not_ready", "model_loaded": model_instance is not None,
                         "api_key_configured": bool(API_KEY)}, status_code=200 if ready else 503)


@app.get("/info")
async def service_info():
    return {"service": "aquavision-ml-service", "service_version": SERVICE_VERSION,
            "model_version": model_instance.model_version if model_instance else None,
            "device": model_instance.device if model_instance else None,
            "max_file_size_bytes": MAX_FILE_SIZE, "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_image_dimension": MAX_IMAGE_DIMENSION, "supported_formats": ["JPEG", "PNG"],
            "tile_size": model_instance.tile_size if model_instance else None,
            "authentication": "X-API-Key", "max_concurrent_inferences": 1,
            "enhancement": "same-resolution U-Net; EXIF orientation applied"}


def process_image(file, checksums, request_id):
    # The thread owns this lock until it exits, including on caller cancellation.
    if not inference_slot.acquire(blocking=False):
        raise api_error(503, "service_busy", "The ML service is processing another image.", {"Retry-After": "5"})
    image = None
    try:
        model = model_instance
        if model is None:
            raise api_error(503, "model_unavailable", "The ML model is not ready.")
        try:
            contents = file.file.read(MAX_FILE_SIZE + 1)
        except OSError:
            raise api_error(400, "invalid_request", "Failed to read the uploaded image.") from None
        if not contents:
            raise api_error(400, "empty_file", "The uploaded image is empty.")
        if len(contents) > MAX_FILE_SIZE:
            raise api_error(413, "file_too_large", "The image exceeds the file size limit.")
        for checksum in checksums:
            if checksum is not None and not hmac.compare_digest(hashlib.sha256(contents).hexdigest().encode(),
                                                               checksum.strip().lower().encode()):
                raise api_error(400, "checksum_mismatch", "Image SHA-256 checksum mismatch.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(contents)) as probe:
                    fmt = probe.format
                    if fmt not in {"JPEG", "PNG"}:
                        raise api_error(400, "unsupported_format", "Only JPEG and PNG images are supported.")
                    if probe.width * probe.height > MAX_IMAGE_PIXELS or max(probe.size) > MAX_IMAGE_DIMENSION:
                        raise api_error(413, "image_too_large", "The image exceeds the pixel or dimension limit.")
                    if getattr(probe, "n_frames", 1) != 1:
                        raise api_error(400, "unsupported_format", "Animated images are not supported.")
                    probe.verify()
                image = Image.open(io.BytesIO(contents))
                image.load()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise api_error(413, "image_too_large", "The decoded image is too large.") from None
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise api_error(400, "invalid_image", "Upload a valid JPEG or PNG image.") from None
        del contents
        started = time.perf_counter()
        enhanced = model.enhance(image)
        inference_ms = (time.perf_counter() - started) * 1000
        with enhanced, io.BytesIO() as output:
            enhanced.save(output, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
            result = output.getvalue()
        logger.info("inference_completed request_id=%s dimensions=%sx%s input_bytes=%s output_bytes=%s inference_ms=%.2f device=%s",
                    request_id, image.width, image.height, file.size, len(result), inference_ms, model.device)
        return Response(result, media_type="image/jpeg" if fmt == "JPEG" else "image/png",
                        headers={"X-Model-Version": model.model_version,
                                 "X-Inference-Ms": f"{inference_ms:.2f}"})
    except HTTPException:
        raise
    except (MemoryError, OutOfMemoryError):
        logger.exception("compute_unavailable request_id=%s", request_id)
        raise api_error(503, "compute_unavailable", "Insufficient memory to enhance this image.", {"Retry-After": "5"}) from None
    except Exception:
        logger.exception("inference_failed request_id=%s", request_id)
        raise api_error(500, "inference_failed", "Image enhancement failed.") from None
    finally:
        if image is not None:
            image.close()
        inference_slot.release()


upload_schema = {"requestBody": {"required": True, "content": {"multipart/form-data": {
    "schema": {"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary"}}}}}}}
image_responses = {200: {"description": "Enhanced image in the input format", "content": {
    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
    "image/png": {"schema": {"type": "string", "format": "binary"}}}},
    **{code: {"description": description} for code, description in {
        400: "Invalid upload or checksum", 401: "Invalid API key", 413: "Image exceeds limits",
        500: "Enhancement failed", 503: "Service unavailable or busy"}.items()}}


@app.post("/enhance", response_class=Response, responses=image_responses, openapi_extra=upload_schema)
@app.post("/v1/enhance", response_class=Response, responses=image_responses, openapi_extra=upload_schema)
async def enhance_image(request: Request, api_key: str | None = Security(key_header),
                        x_model_input_sha256: str | None = Header(None),
                        x_sha256_checksum: str | None = Header(None)):
    """Send exactly one JPEG/PNG as multipart field file; receive binary image bytes."""
    async with request.form(max_files=1, max_fields=0) as form:
        file = form.get("file")
        if not isinstance(file, UploadFile) or not file.filename:
            raise api_error(400, "missing_file", "Provide an image in the multipart field 'file'.")
        return await run_in_threadpool(process_image, file, (x_model_input_sha256, x_sha256_checksum),
                                       request.scope["request_id"])
