"""MinIO client for model checkpoints, FAISS indices, and teacher snapshots."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from config import get_minio_config

logger = logging.getLogger(__name__)


class MinIOClient:
    """Thin wrapper around the MinIO Python SDK."""

    def __init__(self) -> None:
        cfg = get_minio_config()
        self._client = Minio(
            endpoint=cfg["endpoint"],
            access_key=cfg["access_key"],
            secret_key=cfg["secret_key"],
            secure=cfg.get("secure", False),
        )
        self._buckets = cfg["buckets"]
        self._ensure_buckets()

    def _ensure_buckets(self) -> None:
        for bucket in self._buckets.values():
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info("Created MinIO bucket: %s", bucket)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def upload_checkpoint(self, local_path: str | Path, run_id: str, epoch: int) -> str:
        """Upload a PyTorch checkpoint. Returns the object name."""
        object_name = f"{run_id}/epoch_{epoch:04d}.pt"
        self._client.fput_object(
            self._buckets["checkpoints"], object_name, str(local_path)
        )
        logger.info("Uploaded checkpoint: %s/%s", self._buckets["checkpoints"], object_name)
        return object_name

    def download_checkpoint(self, run_id: str, epoch: int, local_path: str | Path) -> Path:
        """Download a checkpoint to local_path. Returns the local path."""
        object_name = f"{run_id}/epoch_{epoch:04d}.pt"
        self._client.fget_object(
            self._buckets["checkpoints"], object_name, str(local_path)
        )
        return Path(local_path)

    def list_checkpoints(self, run_id: str) -> list[str]:
        """List all checkpoint object names for a run."""
        objects = self._client.list_objects(
            self._buckets["checkpoints"], prefix=f"{run_id}/"
        )
        return [obj.object_name for obj in objects]

    # ------------------------------------------------------------------
    # FAISS indices
    # ------------------------------------------------------------------

    def upload_faiss_index(
        self,
        local_path: str | Path,
        model_name: str,
        scoring_method: str,
    ) -> str:
        """Upload a FAISS index file. Returns the object name."""
        object_name = f"{model_name}/{scoring_method}.index"
        self._client.fput_object(
            self._buckets["faiss"], object_name, str(local_path)
        )
        # Also upload the companion .ids.npy file
        ids_path = str(local_path) + ".ids.npy"
        if Path(ids_path).exists():
            self._client.fput_object(
                self._buckets["faiss"], object_name + ".ids.npy", ids_path
            )
        logger.info("Uploaded FAISS index: %s/%s", self._buckets["faiss"], object_name)
        return object_name

    def download_faiss_index(
        self,
        model_name: str,
        scoring_method: str,
        local_path: str | Path,
    ) -> Path:
        object_name = f"{model_name}/{scoring_method}.index"
        self._client.fget_object(self._buckets["faiss"], object_name, str(local_path))
        ids_object = object_name + ".ids.npy"
        try:
            self._client.fget_object(
                self._buckets["faiss"], ids_object, str(local_path) + ".ids.npy"
            )
        except S3Error:
            pass
        return Path(local_path)

    # ------------------------------------------------------------------
    # Teacher snapshots
    # ------------------------------------------------------------------

    def upload_teacher_snapshot(
        self,
        local_path: str | Path,
        run_id: str,
        step: int,
    ) -> str:
        object_name = f"{run_id}/teacher_step_{step:08d}.pt"
        self._client.fput_object(
            self._buckets["teachers"], object_name, str(local_path)
        )
        return object_name

    # ------------------------------------------------------------------
    # Cross-distillation checkpoints
    # ------------------------------------------------------------------

    def upload_cross_distill(
        self,
        local_path: str | Path,
        student_name: str,
        teacher_name: str,
    ) -> str:
        object_name = f"{teacher_name}_to_{student_name}/checkpoint.pt"
        self._client.fput_object(
            self._buckets["cross_distill"], object_name, str(local_path)
        )
        return object_name
