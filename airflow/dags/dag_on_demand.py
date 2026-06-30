"""DAG: movielens_on_demand

Triggered exclusively via POST /trigger or the Airflow REST API.
Conf payload controls which arch×loss combinations run:

    {
        "trigger_type":   "on_demand",
        "trigger_id":     "<uuid>",
        "losses":         ["TimedecayMSELoss"] | "all",
        "architectures":  ["TwoTower"]          | "all",
        "run_from":       "featurize"
    }

Always starts from featurize (per spec).
Unselected combinations are short-circuited at the guard task so the
DAG graph stays static and fully visible in the Airflow UI.

  featurize → split
      ├─ guard_A_L → tune_A_L → train_A_L ─┐
      └─ guard_B_L → tune_B_L → train_B_L ─┤  (skipped if not requested)
                                              ▼
              cross_distill → evaluate → precompute → visualize → finalize
"""
from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow.dags.common import (
    task_featurize, task_split,
    task_tune, task_train, task_cross_distill,
    task_evaluate, task_precompute, task_visualize, task_finalize,
    update_trigger_log,
)
from airflow.dags.pipeline_logic import should_run_combination
from training.registry import ComponentRegistry

_all_combos = ComponentRegistry().get_enabled_combinations()


def _on_failure(ctx):
    update_trigger_log((ctx["dag_run"].conf or {}).get("trigger_id"), "failed")


def _make_guard(arch_name: str, loss_name: str):
    """Factory returning a ShortCircuitOperator callable for one combination."""
    def guard(**context) -> bool:
        conf = (context["dag_run"].conf or {})
        result = should_run_combination(arch_name, loss_name, conf)
        if not result:
            import logging
            logging.getLogger(__name__).info(
                "Skipping %s×%s — not in requested combinations.", arch_name, loss_name
            )
        return result
    guard.__name__ = f"guard_{arch_name}_{loss_name}"
    return guard


with DAG(
    dag_id="movielens_on_demand",
    description="On-demand pipeline via POST /trigger. Runs featurize onward for requested combos.",
    schedule_interval=None,
    start_date=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    catchup=False,
    max_active_runs=3,
    tags=["recsys", "movielens", "on-demand"],
    default_args={
        "retries": 0,
        "on_failure_callback": _on_failure,
    },
    params={
        "trigger_type":  "on_demand",
        "trigger_id":    "",
        "losses":        "all",
        "architectures": "all",
        "run_from":      "featurize",
    },
) as dag:

    featurize = PythonOperator(task_id="featurize", python_callable=task_featurize)
    split     = PythonOperator(task_id="split",     python_callable=task_split)

    train_tasks = []
    for arch_entry, loss_entry in _all_combos:
        aname, lname = arch_entry.name, loss_entry.name
        key = f"{aname}_{lname}"

        guard_t = ShortCircuitOperator(
            task_id=f"guard_{key}",
            python_callable=_make_guard(aname, lname),
            ignore_downstream_trigger_rules=True,
        )
        tune_t = PythonOperator(
            task_id=f"tune_{key}",
            python_callable=task_tune,
            op_kwargs={"arch_name": aname, "loss_name": lname},
            execution_timeout=dt.timedelta(hours=4),
        )
        train_t = PythonOperator(
            task_id=f"train_{key}",
            python_callable=task_train,
            op_kwargs={"arch_name": aname, "loss_name": lname},
            execution_timeout=dt.timedelta(hours=12),
        )
        train_tasks.append(train_t)
        split >> guard_t >> tune_t >> train_t

    cross_distill_t = PythonOperator(
        task_id="cross_distill",
        python_callable=task_cross_distill,
        trigger_rule="none_failed_min_one_success",
        execution_timeout=dt.timedelta(hours=6),
    )
    evaluate_t   = PythonOperator(task_id="evaluate",   python_callable=task_evaluate)
    precompute_t = PythonOperator(task_id="precompute", python_callable=task_precompute,
                                  execution_timeout=dt.timedelta(hours=3))
    visualize_t  = PythonOperator(task_id="visualize",  python_callable=task_visualize)
    finalize_t   = PythonOperator(task_id="finalize",   python_callable=task_finalize)

    featurize >> split
    for train_t in train_tasks:
        train_t >> cross_distill_t
    cross_distill_t >> evaluate_t >> precompute_t >> visualize_t >> finalize_t
