# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 02_rule_engine.py
# Deterministic verdict + savings calculation
# No LLM used here — pure Python logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_verdict(aggregated: dict, cluster: dict) -> dict:

    p90_cpu    = float(aggregated.get("p90_cpu_pct",   0) or 0)
    p90_mem    = float(aggregated.get("p90_mem_pct",   0) or 0)
    avg_cpu    = float(aggregated.get("avg_cpu_pct",   0) or 0)
    avg_mem    = float(aggregated.get("avg_mem_pct",   0) or 0)
    max_cpu    = float(aggregated.get("max_cpu_pct",   0) or 0)
    any_spill  = aggregated.get("any_spill", False)
    spill_runs = int(aggregated.get("spill_run_count", 0))
    fail_rate  = float(aggregated.get("failure_rate_pct", 0))
    workers    = int(cluster.get("effective_workers", 1))
    node_type  = cluster.get("worker_node_type", "unknown")
    total_runs = int(aggregated.get("total_runs", 1))

    # ── UNDER-PROVISIONED ─────────────────────────────────
    if (p90_cpu > 80
            or p90_mem > 80
            or spill_runs > (total_runs * 0.2)
            or fail_rate > 30):

        verdict  = "UNDER_PROVISIONED"
        priority = "IMMEDIATE"

        if any_spill or p90_mem > 80:
            recommended_node    = _upgrade_to_memory_optimised(node_type)
            recommended_workers = workers
            reason = "Memory pressure or disk spill detected"
        else:
            recommended_node    = node_type
            recommended_workers = min(int(workers * 1.5), 50)
            reason = f"CPU saturation — p90 CPU at {p90_cpu}%"

    # ── OVER-PROVISIONED ──────────────────────────────────
    elif p90_cpu < 35 and p90_mem < 50:

        verdict  = "OVER_PROVISIONED"
        priority = "IMMEDIATE" if avg_cpu < 20 else "NEXT_SPRINT"

        utilisation_ratio   = max(p90_cpu, p90_mem) / 100
        recommended_workers = max(2, round(workers * utilisation_ratio * 1.4))
        recommended_node    = _downsize_vm_if_possible(
            node_type, recommended_workers, workers
        )
        reason = (f"p90 CPU {p90_cpu}% and p90 mem {p90_mem}% "
                  f"across {total_runs} runs — well below thresholds")

    # ── OPTIMAL ───────────────────────────────────────────
    else:
        verdict             = "OPTIMAL"
        priority            = "MONITOR"
        recommended_node    = node_type
        recommended_workers = workers
        reason              = "Utilisation within healthy range"

    return {
        "verdict":             verdict,
        "priority":            priority,
        "recommended_node":    recommended_node,
        "recommended_workers": recommended_workers,
        "reason":              reason
    }


def compute_savings(cluster: dict,
                     verdict_result: dict,
                     aggregated: dict,
                     runs: list[dict]) -> dict:

    current_node    = cluster["worker_node_type"]
    current_workers = int(cluster["effective_workers"])
    avg_duration    = float(aggregated.get("avg_duration_mins", 1))
    avg_cost        = float(aggregated.get("avg_cost_usd", 0))
    total_cost      = float(aggregated.get("total_cost_usd", 0))
    rec_node        = verdict_result["recommended_node"]
    rec_workers     = verdict_result["recommended_workers"]

    proj_cost_per_run = estimate_cost(rec_node, rec_workers, avg_duration)
    savings_per_run   = max(0, round(avg_cost - proj_cost_per_run, 2))
    savings_pct       = round(savings_per_run / avg_cost * 100, 1) \
                        if avg_cost > 0 else 0

    # Estimate run frequency from date range of last 10 runs
    try:
        from datetime import datetime
        dates     = [r["start_time"][:10] for r in runs]
        first     = datetime.strptime(min(dates), "%Y-%m-%d")
        last      = datetime.strptime(max(dates), "%Y-%m-%d")
        days_span = max((last - first).days, 1)
        runs_per_day  = round(len(runs) / days_span, 1)
        annual_runs   = int(runs_per_day * 365)
    except Exception:
        runs_per_day = 1
        annual_runs  = 365

    return {
        "current_node":           current_node,
        "current_workers":        current_workers,
        "current_avg_cost_usd":   round(avg_cost, 2),
        "current_total_cost_usd": round(total_cost, 2),
        "recommended_node":       rec_node,
        "recommended_workers":    rec_workers,
        "projected_avg_cost_usd": proj_cost_per_run,
        "savings_per_run_usd":    savings_per_run,
        "savings_pct":            savings_pct,
        "runs_per_day":           runs_per_day,
        "annual_runs_estimate":   annual_runs,
        "annual_savings_usd":     round(savings_per_run * annual_runs, 0)
    }

print("✅ Rule engine loaded")