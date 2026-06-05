# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 04_agent.py
# Master entry point — run this notebook
# Input:  job_name (string)
# Output: right-sizing report in Delta + MLflow + HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Load all notebooks ────────────────────────────────────
%run ./00_vm_catalogue
%run ./01_data_fetcher
%run ./02_rule_engine
%run ./03_llm_narrative

import uuid, mlflow
from datetime import datetime

# ── Create output table if not exists ────────────────────
spark.sql("""
    CREATE TABLE IF NOT EXISTS platform.rightsizing.job_reports (
        report_id            STRING,
        job_name             STRING,
        cluster_id           STRING,
        report_time          TIMESTAMP,
        runs_analysed        INT,
        worker_node_type     STRING,
        driver_node_type     STRING,
        num_workers          INT,
        autoscale_min        INT,
        autoscale_max        INT,
        total_cores          INT,
        total_memory_gb      DOUBLE,
        avg_cpu_pct          DOUBLE,
        median_cpu_pct       DOUBLE,
        p90_cpu_pct          DOUBLE,
        max_cpu_pct          DOUBLE,
        avg_mem_pct          DOUBLE,
        p90_mem_pct          DOUBLE,
        max_mem_pct          DOUBLE,
        spill_runs           INT,
        failed_runs          INT,
        failure_rate_pct     DOUBLE,
        avg_duration_mins    DOUBLE,
        avg_cost_usd         DOUBLE,
        total_cost_usd       DOUBLE,
        verdict              STRING,
        priority             STRING,
        reason               STRING,
        recommended_node     STRING,
        recommended_workers  INT,
        projected_cost_usd   DOUBLE,
        savings_per_run_usd  DOUBLE,
        annual_savings_usd   DOUBLE,
        savings_pct          DOUBLE,
        findings             STRING,
        recommendations      STRING,
        risk_flags           STRING,
        full_report_md       STRING
    )
    USING DELTA
    TBLPROPERTIES ("delta.enableChangeDataFeed" = "true")
""")

# ── Master agent function ─────────────────────────────────
def run_rightsizing_agent(job_name: str) -> dict:

    print(f"\n{'='*60}")
    print(f"  CLUSTER RIGHT-SIZING AGENT")
    print(f"  Job: {job_name}")
    print(f"{'='*60}\n")

    # 1. Fetch last 10 runs
    print("  [1/5] Fetching last 10 completed runs...")
    runs       = fetch_last_10_runs(job_name)
    latest_run = runs[0]
    cluster_id = latest_run["cluster_id"]

    # 2. Fetch cluster config from most recent run
    print("\n  [2/5] Fetching cluster configuration...")
    cluster = fetch_cluster_config(cluster_id)
    print(f"        VM:      {cluster.get('worker_node_type')}")
    print(f"        Workers: {cluster.get('effective_workers')}")
    print(f"        Cores:   {cluster.get('total_cores')}")
    print(f"        Memory:  {cluster.get('total_memory_gb')} GB")

    # 3. Aggregate metrics across all runs
    print("\n  [3/5] Aggregating metrics across all runs...")
    aggregated = aggregate_runs(runs)
    print(f"\n        CPU   — avg: {aggregated['avg_cpu_pct']}%  "
          f"p90: {aggregated['p90_cpu_pct']}%  "
          f"max: {aggregated['max_cpu_pct']}%")
    print(f"        Mem   — avg: {aggregated['avg_mem_pct']}%  "
          f"p90: {aggregated['p90_mem_pct']}%  "
          f"max: {aggregated['max_mem_pct']}%")
    print(f"        Spill — {aggregated['spill_run_count']} of "
          f"{aggregated['total_runs']} runs")
    print(f"        Cost  — avg: ${aggregated['avg_cost_usd']}  "
          f"total: ${aggregated['total_cost_usd']}")
    print(f"        Fails — {aggregated['failed_runs']} of "
          f"{aggregated['total_runs']} runs "
          f"({aggregated['failure_rate_pct']}%)")

    # 4. Rule engine — verdict and savings
    print("\n  [4/5] Computing verdict and savings...")
    verdict_result = compute_verdict(aggregated, cluster)
    savings        = compute_savings(cluster, verdict_result,
                                      aggregated, runs)
    print(f"        Verdict:  {verdict_result['verdict']}")
    print(f"        Priority: {verdict_result['priority']}")
    print(f"        Reason:   {verdict_result['reason']}")
    print(f"        Rec VM:   {savings['recommended_node']} "
          f"x {savings['recommended_workers']} workers")
    print(f"        Saving:   ${savings['savings_per_run_usd']}/run  |  "
          f"${savings['annual_savings_usd']}/year")

    # 5. LLM narrative
    print("\n  [5/5] Generating LLM narrative...")
    narrative  = generate_narrative(
        job_name, cluster, aggregated, verdict_result, savings
    )
    report_md  = format_report_md(
        job_name, cluster, aggregated,
        verdict_result, savings, narrative, runs
    )

    # 6. Save to Delta
    report_id = str(uuid.uuid4())
    row = {
        "report_id":           report_id,
        "job_name":            job_name,
        "cluster_id":          cluster_id,
        "report_time":         datetime.now(),
        "runs_analysed":       aggregated["total_runs"],
        "worker_node_type":    cluster.get("worker_node_type"),
        "driver_node_type":    cluster.get("driver_node_type"),
        "num_workers":         int(cluster.get("effective_workers", 0)),
        "autoscale_min":       cluster.get("autoscale_min_workers"),
        "autoscale_max":       cluster.get("autoscale_max_workers"),
        "total_cores":         cluster.get("total_cores"),
        "total_memory_gb":     cluster.get("total_memory_gb"),
        "avg_cpu_pct":         aggregated["avg_cpu_pct"],
        "median_cpu_pct":      aggregated["median_cpu_pct"],
        "p90_cpu_pct":         aggregated["p90_cpu_pct"],
        "max_cpu_pct":         aggregated["max_cpu_pct"],
        "avg_mem_pct":         aggregated["avg_mem_pct"],
        "p90_mem_pct":         aggregated["p90_mem_pct"],
        "max_mem_pct":         aggregated["max_mem_pct"],
        "spill_runs":          aggregated["spill_run_count"],
        "failed_runs":         aggregated["failed_runs"],
        "failure_rate_pct":    aggregated["failure_rate_pct"],
        "avg_duration_mins":   aggregated["avg_duration_mins"],
        "avg_cost_usd":        savings["current_avg_cost_usd"],
        "total_cost_usd":      savings["current_total_cost_usd"],
        "verdict":             verdict_result["verdict"],
        "priority":            verdict_result["priority"],
        "reason":              verdict_result["reason"],
        "recommended_node":    savings["recommended_node"],
        "recommended_workers": savings["recommended_workers"],
        "projected_cost_usd":  savings["projected_avg_cost_usd"],
        "savings_per_run_usd": savings["savings_per_run_usd"],
        "annual_savings_usd":  savings["annual_savings_usd"],
        "savings_pct":         savings["savings_pct"],
        "findings":            str(narrative.get("findings", [])),
        "recommendations":     str(narrative.get("recommendations", [])),
        "risk_flags":          str(narrative.get("risk_flags", [])),
        "full_report_md":      report_md
    }

    spark.createDataFrame([row]) \
         .write.format("delta") \
         .mode("append") \
         .saveAsTable("platform.rightsizing.job_reports")

    # 7. Log to MLflow
    with mlflow.start_run(run_name=f"rightsizing_{job_name}"):
        mlflow.log_metrics({
            "avg_cpu_pct":        float(aggregated["avg_cpu_pct"] or 0),
            "p90_cpu_pct":        float(aggregated["p90_cpu_pct"] or 0),
            "avg_mem_pct":        float(aggregated["avg_mem_pct"] or 0),
            "p90_mem_pct":        float(aggregated["p90_mem_pct"] or 0),
            "spill_runs":         float(aggregated["spill_run_count"]),
            "failed_runs":        float(aggregated["failed_runs"]),
            "avg_cost_usd":       float(savings["current_avg_cost_usd"]),
            "savings_per_run_usd":float(savings["savings_per_run_usd"]),
            "annual_savings_usd": float(savings["annual_savings_usd"]),
            "savings_pct":        float(savings["savings_pct"])
        })
        mlflow.set_tags({
            "job_name":   job_name,
            "verdict":    verdict_result["verdict"],
            "priority":   verdict_result["priority"],
            "cluster_id": cluster_id
        })
        mlflow.log_text(report_md, "report.md")

    # 8. Display report in notebook
    print(f"\n{'='*60}")
    print(f"  ✅ REPORT COMPLETE")
    print(f"  Report ID: {report_id}")
    print(f"  Verdict:   {verdict_result['verdict']}")
    print(f"  Priority:  {verdict_result['priority']}")
    print(f"  Saving:    ${savings['savings_per_run_usd']}/run  |  "
          f"${savings['annual_savings_usd']}/year")
    print(f"{'='*60}\n")

    displayHTML(report_md.replace("\n", "<br>"))
    return {"report_id": report_id,
            "verdict":   verdict_result["verdict"],
            "savings":   savings}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── RUN THE AGENT ─────────────────────────────────────────
# Replace with your exact job name as shown in Workflows UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_NAME = "your_job_name_here"

report = run_rightsizing_agent(JOB_NAME)