"""Airflow-facing wrappers: sensor class + thin PythonOperator callables.

All business logic lives in pipeline_logic.py (no Airflow imports there).
This file only contains:
  - RawTableWatermarkSensor  (needs BaseSensorOperator)
  - Thin wrappers so PythonOperator task_ids match the expected names
"""
from __future__ import annotations

import logging

from airflow.sensors.base import BaseSensorOperator

from airflow.dags.pipeline_logic import (
    RAW_TABLES,
    check_watermark,
    update_trigger_log,
    run_ingest     as task_ingest,
    run_validate   as task_validate,
    run_featurize  as task_featurize,
    run_split      as task_split,
    run_tune       as task_tune,
    run_train      as task_train,
    run_cross_distill as task_cross_distill,
    run_evaluate   as task_evaluate,
    run_precompute as task_precompute,
    run_visualize  as task_visualize,
    run_finalize   as task_finalize,
    check_new_data_for_daily as task_ingest_if_new_data,
)

logger = logging.getLogger(__name__)


class RawTableWatermarkSensor(BaseSensorOperator):
    """Fires when ANY raw.* table has rows newer than the last recorded watermark."""

    def __init__(self, tables: list[str] = RAW_TABLES, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tables = tables

    def poke(self, context) -> bool:          # type: ignore[override]
        has_new, _ = check_watermark(self.tables)
        return has_new
