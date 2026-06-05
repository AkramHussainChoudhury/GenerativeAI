# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 01_data_fetcher.py
# Fetches all data from Databricks system tables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import statistics

w            = WorkspaceClient()
WAREHOUSE_ID = dbutils.secrets.get("rightsizing", "warehouse_id")

# ── Core SQL runner ───────────────────────────────────────
def run_sql(query: str) -> list[dict]:
    stmt = w.statement_execution.execute_statement(
        warehouse_id = WAREHOUSE_ID,
        statement    = query,
        wait_timeout = "60s"
    )
    if stmt.status.state != StatementState.SUCCEEDED:
        raise Exception(f"SQL failed: {stmt.status.error.message}")
    cols = [c.name for c in stmt.manifest.schema.columns]
    rows = stmt.result.data_array or []
    return [dict(zip(cols, row)) for row in rows]

# ── Fetch last 10 completed runs by job name ──────────────
def fetch_last_10_runs(job_name: str) -> list[dict]:
    rows = run_sql(f"""
        SELECT
            run_id,
            job_id,
            run_name,
            cluster_id,
            CAST(start_time AS STRING)  AS start_time,
            CAST(end_time   AS STRING)  AS end_time,
            ROUND(
              (UNIX_TIMESTAMP(end_time) -
               UNIX_TIMESTAMP(start_time)) / 60.0, 1
            )                           AS duration_mins,
            result_state,
            trigger_type
        FROM system.lakeflow.job_runs
        WHERE run_name    = '{job_name}'
          AND result_state IN (
                'SUCCEEDED','FAILED',
                'TIMED_OUT','MAXIMUM_CONCURRENT_RUNS_REACHED'
              )
          AND end_time IS NOT NULL
        ORDER BY start_time DESC
        LIMIT 10
    """)
    if not rows:
        raise ValueError(
            f"No completed runs found for job '{job_name}'. "
            f"Check the exact name — it is case-sensitive."
        )
    print(f"  Found {len(rows)} completed runs for '{job_name}'")
    return rows

# ── Fetch cluster config ──────────────────────────────────
def fetch_cluster_config(cluster_id: str) -> dict:
    rows = run_sql(f"""
        SELECT
            cluster_id,
            cluster_name,
            cluster_source,
            node_type_id              AS worker_node_type,
            driver_node_type_id       AS driver_node_type,
            num_workers,
            autoscale_min_workers,
            autoscale_max_workers,
            spark_version,
            COALESCE(
              num_workers,
              autoscale_max_workers,
              1
            )                         AS effective_workers
        FROM system.compute.clusters
        WHERE cluster_id = '{cluster_id}'
        ORDER BY change_time DESC
        LIMIT 1
    """)
    if not rows:
        raise ValueError(f"No cluster found for cluster_id={cluster_id}")

    config  = rows[0]
    specs   = get_vm_specs(config["worker_node_type"])
    workers = int(config["effective_workers"])

    config["total_cores"]     = specs["vcpu"]   * workers
    config["total_memory_gb"] = specs["mem_gb"] * workers
    config["dbu_per_hour"]    = specs["dbu"]    * workers
    config["vm_type"]         = specs["type"]
    return config

# ── Fetch utilisation for one run ─────────────────────────
def fetch_utilisation_for_run(cluster_id: str,
                               start_time: str,
                               end_time: str) -> dict:
    rows = run_sql(f"""
        SELECT
            ROUND(AVG(cpu_user_percent +
                      cpu_system_percent), 1)  AS avg_cpu_pct,
            ROUND(MAX(cpu_user_percent +
                      cpu_system_percent), 1)  AS peak_cpu_pct,
            ROUND(AVG(mem_used_percent),  1)   AS avg_mem_pct,
            ROUND(MAX(mem_used_percent),  1)   AS peak_mem_pct,
            ROUND(SUM(disk_used_gb),      2)   AS disk_spill_gb,
            COUNT(DISTINCT node_id)            AS active_nodes
        FROM system.compute.node_timeline
        WHERE cluster_id = '{cluster_id}'
          AND timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
    """)
    if not rows or rows[0]["avg_cpu_pct"] is None:
        print(f"    ⚠️  No node_timeline data for cluster {cluster_id}")
        return {
            "avg_cpu_pct": None, "peak_cpu_pct": None,
            "avg_mem_pct": None, "peak_mem_pct": None,
            "disk_spill_gb": 0,  "active_nodes": 0
        }
    return rows[0]

# ── Fetch cost for one run ────────────────────────────────
def fetch_cost_for_run(run_id: int, cluster_id: str) -> dict:
    rows = run_sql(f"""
        SELECT
            ROUND(SUM(u.usage_quantity), 3)                      AS total_dbus,
            ROUND(SUM(u.usage_quantity * p.pricing.default), 2)  AS cost_usd
        FROM system.billing.usage u
        JOIN system.billing.list_prices p
          ON  u.sku_name     = p.sku_name
          AND u.workspace_id = p.workspace_id
        WHERE u.usage_metadata.job_run_id = '{run_id}'
           OR u.custom_tags['RunId']      = '{run_id}'
    """)
    if not rows or rows[0]["total_dbus"] is None:
        return {"total_dbus": 0.0, "cost_usd": 0.0}
    return rows[0]

# ── Aggregate metrics across all 10 runs ─────────────────
def _percentile(data: list, pct: int) -> float:
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = min(int(len(sorted_data) * pct / 100), len(sorted_data) - 1)
    return round(sorted_data[idx], 1)

def aggregate_runs(runs: list[dict]) -> dict:
    per_run_data = []
    cpu_avgs, cpu_peaks  = [], []
    mem_avgs, mem_peaks  = [], []
    spill_totals         = []
    costs, durations     = [], []
    failed_count         = 0

    print(f"\n  Fetching metrics for {len(runs)} runs...")

    for i, run in enumerate(runs):
        run_id     = run["run_id"]
        cluster_id = run["cluster_id"]
        duration   = float(run.get("duration_mins") or 0)

        metrics = fetch_utilisation_for_run(
            cluster_id, run["start_time"], run["end_time"]
        )
        cost = fetch_cost_for_run(run_id, cluster_id)

        if run["result_state"] != "SUCCEEDED":
            failed_count += 1

        per_run_data.append({
            "run_id":        run_id,
            "start_time":    run["start_time"],
            "result_state":  run["result_state"],
            "duration_mins": duration,
            "avg_cpu_pct":   metrics.get("avg_cpu_pct"),
            "avg_mem_pct":   metrics.get("avg_mem_pct"),
            "disk_spill_gb": float(metrics.get("disk_spill_gb") or 0),
            "cost_usd":      float(cost.get("cost_usd") or 0)
        })

        if metrics.get("avg_cpu_pct") is not None:
            cpu_avgs.append(float(metrics["avg_cpu_pct"]))
            cpu_peaks.append(float(metrics["peak_cpu_pct"] or 0))
            mem_avgs.append(float(metrics["avg_mem_pct"]))
            mem_peaks.append(float(metrics["peak_mem_pct"] or 0))
            spill_totals.append(float(metrics.get("disk_spill_gb") or 0))

        costs.append(float(cost.get("cost_usd") or 0))
        durations.append(duration)

        print(f"    Run {i+1}/{len(runs)}  "
              f"CPU: {metrics.get('avg_cpu_pct')}%  "
              f"Mem: {metrics.get('avg_mem_pct')}%  "
              f"Spill: {metrics.get('disk_spill_gb',0)} GB  "
              f"Cost: ${cost.get('cost_usd')}  "
              f"[{run['result_state']}]")

    def safe(lst, fn, d=1):
        return round(fn(lst), d) if lst else 0

    return {
        "avg_cpu_pct":       safe(cpu_avgs,  statistics.mean),
        "median_cpu_pct":    safe(cpu_avgs,  statistics.median),
        "p90_cpu_pct":       _percentile(cpu_avgs, 90),
        "max_cpu_pct":       safe(cpu_peaks, max),
        "min_cpu_pct":       safe(cpu_avgs,  min),
        "avg_mem_pct":       safe(mem_avgs,  statistics.mean),
        "median_mem_pct":    safe(mem_avgs,  statistics.median),
        "p90_mem_pct":       _percentile(mem_avgs, 90),
        "max_mem_pct":       safe(mem_peaks, max),
        "any_spill":         any(s > 0 for s in spill_totals),
        "spill_run_count":   sum(1 for s in spill_totals if s > 0),
        "max_spill_gb":      safe(spill_totals, max, 2),
        "avg_cost_usd":      safe(costs,    statistics.mean, 2),
        "total_cost_usd":    safe(costs,    sum, 2),
        "avg_duration_mins": safe(durations, statistics.mean, 1),
        "max_duration_mins": safe(durations, max, 1),
        "total_runs":        len(runs),
        "failed_runs":       failed_count,
        "failure_rate_pct":  round(failed_count / len(runs) * 100, 1),
        "runs_with_metrics": len(cpu_avgs),
        "per_run_data":      per_run_data
    }

print("✅ Data fetcher loaded")