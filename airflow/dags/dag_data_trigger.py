"""DAG: movielens_data_trigger

Fires when the watermark sensor detects new rows in any raw.* table.
Runs the full pipeline for ALL registered arch×loss combinations.

  sensor → ingest → validate → featurize → split
                                              ├─ tune_A_L → train_A_L ─┐
                                              └─ tune_B_L → train_B_L ─┤
                                                                         ▼
                                              cross_distill → evaluate → precompute
                                                                              └── visualize → finalize
"""
from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.dags.common import (
    RAW_TABLES, RawTableWatermarkSensor,
    task_ingest, task_validate, task_featurize, task_split,
    task_tune, task_train, task_cross_distill,
    task_evaluate, task_precompute, task_visualize, task_finalize,
    update_trigger_log,
)
from airflow.dags.pipeline_logic import get_active_combinations
from training.registry import ComponentRegistry

_combos = ComponentRegistry().get_enabled_combinations()


def _on_failure(ctx):
    update_trigger_log((ctx["dag_run"].conf or {}).get("trigger_id"), "failed")


with DAG(
    dag_id="movielens_data_trigger",
    description="Full pipeline triggered by new data in any raw.* table.",
    schedule_interval=None,
    start_date=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["recsys", "movielens", "data-trigger"],
    default_args={
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=5),
        "on_failure_callback": _on_failure,
    },
    params={
        "trigger_type": "data",
        "trigger_id":   "",
        "losses":       "all",
        "architectures":"all",
    },
) as dag:

    sensor     = RawTableWatermarkSensor(
        task_id="watermark_sensor",
        tables=RAW_TABLES,
        poke_interval=60,
        timeout=7200,
        mode="reschedule",
    )
    ingest     = PythonOperator(task_id="ingest",    python_callable=task_ingest)
    validate   = PythonOperator(task_id="validate",  python_callable=task_validate)
    featurize  = PythonOperator(task_id="featurize", python_callable=task_featurize)
    split      = PythonOperator(task_id="split",     python_callable=task_split)

    train_tasks = []
    for arch_entry, loss_entry in _combos:
        aname, lname = arch_entry.name, loss_entry.name
        tune_t  = PythonOperator(
            task_id=f"tune_{aname}_{lname}",
            python_callable=task_tune,
            op_kwargs={"arch_name": aname, "loss_name": lname},
            execution_timeout=dt.timedelta(hours=4),
        )
        train_t = PythonOperator(
            task_id=f"train_{aname}_{lname}",
            python_callable=task_train,
            op_kwargs={"arch_name": aname, "loss_name": lname},
            execution_timeout=dt.timedelta(hours=12),
        )
        train_tasks.append(train_t)
        split >> tune_t >> train_t

    cross_distill_t = PythonOperator(task_id="cross_distill",  python_callable=task_cross_distill,
                                     execution_timeout=dt.timedelta(hours=6))
    evaluate_t      = PythonOperator(task_id="evaluate",        python_callable=task_evaluate)
    precompute_t    = PythonOperator(task_id="precompute",      python_callable=task_precompute,
                                     execution_timeout=dt.timedelta(hours=3))
    visualize_t     = PythonOperator(task_id="visualize",       python_callable=task_visualize)
    finalize_t      = PythonOperator(task_id="finalize",        python_callable=task_finalize)

    sensor >> ingest >> validate >> featurize >> split
    for train_t in train_tasks:
        train_t >> cross_distill_t
    cross_distill_t >> evaluate_t >> precompute_t >> visualize_t >> finalize_t
