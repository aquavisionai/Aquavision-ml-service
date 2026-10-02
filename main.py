"""
main.py — FastAPI microservice for AquaVision PyTorch model inference.
"""

import hashlib
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

try:
    from dotenv import load_dotenv
    _root_env = Path(__file__).parent.parent / ".env"
    if _root_env.exists():
        load_dotenv(dotenv_path=_root_env)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from .inference import AquaVisionModel
except ImportError:
    from inference import AquaVisionModel

model_instance: Optional[AquaVisionModel] = None
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB max payload size limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Model Lifespan Manager: Loads AquaVision PyTorch model once on startup.
    Keeps model in memory across requests without per-request reloads.
    """
    global model_instance
    print("[AquaVision ML Service] Initializing PyTorch model engine...")
    checkpoint_path = Path(__file__).parent / "best.pt"

    try:
        model_instance = AquaVisionModel(checkpoint_path=checkpoint_path)
        print(f"[AquaVision ML Service] Model loaded successfully on device '{model_instance.device}' (epoch={model_instance.epoch}).")
    except Exception as e:
        print(f"[AquaVision ML Service] Failed to initialize model checkpoint: {e}")
        model_instance = None

    yield

    print("[AquaVision ML Service] Shutting down ML service and clearing model memory.")
    model_instance = None


app = FastAPI(
    title="AquaVision ML Microservice",
    description="Dedicated PyTorch Underwater Image Enhancement Inference Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration: strict explicit origins (no wildcard credentialed CORS)
raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5000,http://127.0.0.1:5000")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint verifying service operational status and model readiness.
    """
    is_loaded = model_instance is not None
    return {
        "status": "ok" if is_loaded else "degraded",
        "service": "aquavision-ml-service",
        "model_loaded": is_loaded,
        "device": model_instance.device if is_loaded else None,
        "epoch": model_instance.epoch if is_loaded else None,
    }


@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    x_model_input_sha256: Optional[str] = Header(None, alias="X-Model-Input-SHA256"),
    x_sha256_checksum: Optional[str] = Header(None, alias="X-SHA256-Checksum"),
):
    """
    Underwater Image Enhancement Endpoint:
    1. Ingests raw JPEG/PNG image binary payload.
    2. Enforces maximum upload payload size limits.
    3. Validates SHA-256 integrity checksum if header is provided.
    4. Executes PyTorch model inference (native or tiled based on resolution).
    5. Returns enhanced image bytes with original dimensions preserved.
    """
    if model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_53_SERVICE_UNAVAILABLE,
            detail="ML Inference Engine not available or model initialization failed.",
        )

    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in request upload payload.",
        )

    # Read binary bytes into memory (no permanent disk storage)
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file stream.",
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum size limit of {MAX_FILE_SIZE // (1024 * 1024)}MB.",
        )

    # SHA-256 Integrity Verification (supports both X-Model-Input-SHA256 & legacy X-SHA256-Checksum)
    target_checksum = x_model_input_sha256 or x_sha256_checksum
    if target_checksum:
        calculated_sha = hashlib.sha256(contents).hexdigest()
        if calculated_sha.lower() != target_checksum.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image integrity verification failed: SHA-256 checksum mismatch.",
            )

    # Decode and validate image with PIL
    try:
        input_img = Image.open(io.BytesIO(contents))
        input_img.verify()  # Verify byte integrity
        input_img = Image.open(io.BytesIO(contents))  # Re-open stream after verify
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image data. File must be a valid JPEG or PNG image.",
        )

    img_format = (input_img.format or "PNG").upper()
    if img_format not in ("JPEG", "JPG", "PNG"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{img_format}'. Allowed formats: JPEG, PNG.",
        )

    # Execute PyTorch Model Inference
    try:
        enhanced_img = model_instance.enhance(input_img)
    except Exception as e:
        print(f"[AquaVision ML Service] Error during model enhancement: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during image enhancement processing.",
        )

    # Encode output buffer preserving original format and dimensions
    out_buffer = io.BytesIO()
    save_format = "JPEG" if img_format in ("JPEG", "JPG") else "PNG"
    media_type = "image/jpeg" if save_format == "JPEG" else "image/png"

    if save_format == "JPEG":
        enhanced_img.save(out_buffer, format=save_format, quality=95)
    else:
        enhanced_img.save(out_buffer, format=save_format)

    out_buffer.seek(0)
    return Response(content=out_buffer.getvalue(), media_type=media_type)
