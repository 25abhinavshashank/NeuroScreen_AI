from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.logging import get_logger
from backend.app.schemas.video import VideoAsdPredictionResult, VideoObservationResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/video", tags=["Video"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB
ALLOWED_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}


async def _save_upload_to_temp(file: UploadFile) -> Path:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'. Expected one of {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video file too large (max 100MB).")
            tmp.write(chunk)

    return tmp_path


@router.post("/analyze", response_model=VideoObservationResult)
async def analyze_video_endpoint(file: UploadFile = File(...)) -> VideoObservationResult:
    """Extract descriptive face/gaze/head-pose observation metrics from
    an uploaded video (file or a webcam recording sent as a file blob).

    This does NOT predict autism and its output is never combined with
    the screening model's result -- see VideoObservationResult's own
    disclaimer. For the experimental (low-accuracy) video classifier,
    see POST /predict-asd instead.
    """

    tmp_path = await _save_upload_to_temp(file)

    try:
        from backend.app.video.analyzer import analyze_video

        result = analyze_video(str(tmp_path))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/predict-asd", response_model=VideoAsdPredictionResult)
async def predict_video_asd_endpoint(file: UploadFile = File(...)) -> VideoAsdPredictionResult:
    """Runs the experimental clip-level video ASD classifier on an
    uploaded video. Measured test accuracy is 59.1% -- barely above
    the 50% chance baseline -- so `disclaimer` on the response is not
    boilerplate: this is a research artifact, not a reliable predictor,
    and it is never combined with the questionnaire-based screening
    result. See backend/app/ml/video_asd/inference.py for how the
    per-window probabilities are computed.
    """

    tmp_path = await _save_upload_to_temp(file)

    try:
        from backend.app.ml.video_asd.inference import predict_video_asd

        result = predict_video_asd(str(tmp_path))
        return VideoAsdPredictionResult(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video ASD prediction failed")
        raise HTTPException(status_code=500, detail=f"Video ASD prediction failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
