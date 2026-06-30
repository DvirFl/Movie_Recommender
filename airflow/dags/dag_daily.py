"""DAG: movielens_daily

Runs once per day at midnight UTC.
Ingest is skipped when no new data has arrived since the last watermark,
but featurize onward always runs so models stay fresh.

  check_new_data ─┬─► ingest ─┐
                  │            └─► validate ─► featurize ─► split
                  └────────────────────────────────────►
                                    (ingest skipped, validate still runs)
  split ─► tune_* ─► train_* ─► cross_distill ─► evaluate ─► precompute ─► visualize ─► finalize
"""
from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.dags.common import (
    task_ingest, task_validate, task_featurize, task_split,
    task_tune, task_train, task_cross_distill,
    task_evaluate, task_precompute, task_visualize, task_finalize,
    task_ingest_if_new_data, update_trigger_log,
)
from training.registry import ComponentRegistry

_combos = ComponentRegistry().get_enabled_combinations()


def _on_failure(ctx):
    update_trigger_log((ctx["dag_run"].conf or {}).get("trigger_id"), "failed")


with DAG(
    dag_id="movielens_daily",
    description="Daily full pipeline. Ingest skipped when no new data.",
    schedule_interval="0 0 * * *",
    start_date=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["recsys", "movielens", "scheduled"],
    default_args={
        "retries": 1,
        "retry_delay": dt.timedelta(minutes=10),
        "on_failure_callback": _on_failure,
    },
    params={
        "trigger_type": "schedule",
        "trigger_id":   "",
        "losses":       "all",
        "architectures":"all",
    },
) as dag:

    # ShortCircuitOperator: True → run ingest; False → skip ingest gracefully
    check_new_data = ShortCircuitOperator(
        task_id="check_new_data",
        python_callable=task_ingest_if_new_data,
        ignore_downstream_trigger_rules=False,
    )
    ingest    = PythonOperator(task_id="ingest",    python_callable=task_ingest,
                               trigger_rule="all_success")
    validate  = PythonOperator(task_id="validate",  python_callable=task_validate,
                               trigger_rule="none_failed_min_one_success")
    featurize = PythonOperator(task_id="featurize", python_callable=task_featurize,
                               trigger_rule="none_failed_min_one_success")
    split     = PythonOperator(task_id="split",     python_callable=task_split)

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

    # Both paths (ingest ran, or ingest skipped) converge at validate
    check_new_data >> ingest >> validate
    check_new_data >> validate          # short-circuit bypass
    validate >> featurize >> split
    for train_t in train_tasks:
        train_t >> cross_distill_t
    cross_distill_t >> evaluate_t >> precompute_t >> visualize_t >> finalize_t
