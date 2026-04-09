"""
errors.py – Structured error handling and API error responses
"""

import logging
import traceback
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger("vision-api.errors")


# ─── Error codes (machine-readable) ───────────────────────────────────────────

class ErrorCode(str, Enum):
    # Input errors (4xx)
    IMAGE_DECODE_FAILED   = "IMAGE_DECODE_FAILED"
    FILE_TOO_LARGE        = "FILE_TOO_LARGE"
    NO_FACE_DETECTED      = "NO_FACE_DETECTED"
    INVALID_BASE64        = "INVALID_BASE64"
    INVALID_PARAMETER     = "INVALID_PARAMETER"
    FACE_ID_NOT_FOUND     = "FACE_ID_NOT_FOUND"
    FACE_ID_EXPIRED       = "FACE_ID_EXPIRED"
    SESSION_NOT_FOUND     = "SESSION_NOT_FOUND"
    TOO_FEW_CLUSTERS      = "TOO_FEW_CLUSTERS"
    UNKNOWN_METHOD        = "UNKNOWN_METHOD"
    MISSING_PARAMETER     = "MISSING_PARAMETER"
    EMBEDDING_DIM_MISMATCH = "EMBEDDING_DIM_MISMATCH"
    QUEUE_FULL            = "QUEUE_FULL"
    JOB_NOT_FOUND         = "JOB_NOT_FOUND"
    JOB_ACTIVE            = "JOB_ACTIVE"

    # Server errors (5xx)
    MODEL_LOAD_FAILED     = "MODEL_LOAD_FAILED"
    INFERENCE_FAILED      = "INFERENCE_FAILED"
    CLUSTERING_FAILED     = "CLUSTERING_FAILED"
    DB_ERROR              = "DB_ERROR"
    INTERNAL_ERROR        = "INTERNAL_ERROR"


class APIError(BaseModel):
    """Standard error envelope returned on all error responses."""
    error: bool = True
    code: ErrorCode
    message: str
    detail: Optional[Any] = None


class VisionAPIException(Exception):
    """Base exception that maps to a structured HTTP error response."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        detail: Any = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=APIError(
                code=self.code,
                message=self.message,
                detail=self.detail,
            ).model_dump(),
        )


# ─── Convenience constructors ─────────────────────────────────────────────────

def err_image_decode() -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.IMAGE_DECODE_FAILED,
        "Could not decode the uploaded image. "
        "Ensure the file is a valid JPEG, PNG, WEBP, BMP, or TIFF.")


def err_file_too_large(max_mb: int) -> VisionAPIException:
    return VisionAPIException(413, ErrorCode.FILE_TOO_LARGE,
        f"Uploaded file exceeds the {max_mb} MB limit.")


def err_no_face(path: str = "") -> VisionAPIException:
    return VisionAPIException(422, ErrorCode.NO_FACE_DETECTED,
        "No faces could be detected in the image. "
        "Try a higher-resolution image or a different detector backend.",
        detail={"image": path} if path else None)


def err_invalid_b64() -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.INVALID_BASE64,
        "The provided string is not valid base64.")


def err_face_not_found(face_id: str) -> VisionAPIException:
    return VisionAPIException(404, ErrorCode.FACE_ID_NOT_FOUND,
        f"Face ID '{face_id}' was not found in the cache.",
        detail={"face_id": face_id})


def err_face_expired(face_id: str) -> VisionAPIException:
    return VisionAPIException(410, ErrorCode.FACE_ID_EXPIRED,
        f"Face ID '{face_id}' exists but has expired. Re-embed the image.",
        detail={"face_id": face_id})


def err_session_not_found(session_id: str) -> VisionAPIException:
    return VisionAPIException(404, ErrorCode.SESSION_NOT_FOUND,
        f"Group session '{session_id}' not found or has expired.",
        detail={"session_id": session_id})


def err_too_few_clusters(n: int) -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.TOO_FEW_CLUSTERS,
        f"Need at least 2 clusters to group, got {n}.",
        detail={"received": n})


def err_unknown_method(method: str) -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.UNKNOWN_METHOD,
        f"Unknown grouping method '{method}'. "
        "Supported: 'hdbscan', 'agglomerative', 'kmeans'.",
        detail={"received": method, "supported": ["hdbscan", "agglomerative", "kmeans"]})


def err_missing_param(param: str, reason: str) -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.MISSING_PARAMETER,
        f"Parameter '{param}' is required: {reason}",
        detail={"param": param})


def err_dim_mismatch(expected: int, got: int, face_id: str) -> VisionAPIException:
    return VisionAPIException(400, ErrorCode.EMBEDDING_DIM_MISMATCH,
        f"Embedding dimension mismatch for face_id '{face_id}': "
        f"expected {expected}, got {got}. All embeddings must use the same model.",
        detail={"face_id": face_id, "expected": expected, "got": got})


def err_inference(exc: Exception) -> VisionAPIException:
    log.error("Inference error: %s\n%s", exc, traceback.format_exc())
    return VisionAPIException(500, ErrorCode.INFERENCE_FAILED,
        "Face analysis failed. Check server logs for details.",
        detail={"exception": type(exc).__name__, "message": str(exc)})


def err_clustering(exc: Exception) -> VisionAPIException:
    log.error("Clustering error: %s\n%s", exc, traceback.format_exc())
    return VisionAPIException(500, ErrorCode.CLUSTERING_FAILED,
        "Clustering algorithm failed. Check server logs for details.",
        detail={"exception": type(exc).__name__, "message": str(exc)})


def err_db(exc: Exception) -> VisionAPIException:
    log.error("DB error: %s\n%s", exc, traceback.format_exc())
    return VisionAPIException(500, ErrorCode.DB_ERROR,
        "A database error occurred.",
        detail={"exception": type(exc).__name__})


# ─── FastAPI exception handlers ───────────────────────────────────────────────

async def vision_exception_handler(request: Request, exc: VisionAPIException):
    return exc.to_response()


async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(exc.detail),
        ).model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s %s: %s\n%s",
              request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=APIError(
            code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred. Please check server logs.",
            detail={"exception": type(exc).__name__},
        ).model_dump(),
    )
