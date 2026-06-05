# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 03_llm_narrative.py
# LLM call via Databricks Model Serving
# Llama 3.3 70B — no external API key needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import mlflow.deployments
import json

llm_client = mlflow.deployments.get_deploy_client("databricks")

PRIMARY_ENDPOINT  = "databricks-meta-llama-3-3-70b-instruct"
FALLBACK_ENDPOINT = "databricks-mixtral-8x7b-instruct"

# ── Raw LLM call with fallback ────────────────────────────
def call_llm(prompt: str,
             max_tokens: int = 900,
             temperature: float = 0.1) -> str:

    messages = [{"role": "user", "content": prompt}]

    for endpoint in [PRIMARY_ENDPOINT, FALLBACK_ENDPOINT]:
        try:
            print(f"   Calling {endpoint}...")
            response = llm_client.predict(
                endpoint = endpoint,
                inputs   = {
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature
                }
            )
            raw_text = response["choices"][0]["message"]["content"]
            print(f"   ✅ Response received ({len(raw_text)} chars)")
            return raw_text

        except Exception as e:
            print(f"   ⚠️  {endpoint} failed: {e}")
            if endpoint == FALLBACK_ENDPOINT:
                raise RuntimeError(
                    f"Both LLM endpoints failed. Last error: {e}"
                )
            print(f"   Trying fallback...")
            continue

# ── Safe JSON parser ──────────────────────────────────────
def parse_llm_json(raw_text: str) -> dict:
    cleaned = (raw_text.strip()
               .lstrip("```json")
               .lstrip("```")
               .rstrip("```")
               .strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON parse failed: {e}")
        return {
            "findings":        ["Could not parse LLM response"],
            "recommendations": ["Review raw metrics manually"],
            "risk_flags":      [],
            "summary":         raw_text[:500]
        }

# ── Smoke test ────────────────────────────────────────────
def test_llm_connection():
    raw    = call_llm('Reply with exactly: {"status":"ok"}', max_tokens=50)
    result = parse_llm_json(raw)
    print(f"✅ LLM smoke test passed: {result}")
    return result

# ── Generate narrative ────────────────────────────────────
def generate_narrative(job_name: str,
                        cluster: dict,
                        aggregated: dict,
                        verdict_result: dict,
                        savings: dict) -> dict:

    per_run_rows = "\n".join([
        f"  Run {i+1}: {r['start_time'][:10]}  "
        f"CPU {r.get('avg_cpu_pct','?')}%  "
        f"Mem {r.get('avg_mem_pct','?')}%  "
        f"Spill {r.get('disk_spill_gb',0)} GB  "
        f"Cost ${r.get('cost_usd',0)}  "
        f"[{r.get('result_state')}]"
        for i, r in enumerate(aggregated.get("per_run_data", []))
    ])

    prompt = f"""
You are a senior Databricks cost engineer writing a
right-sizing report for a data engineering team.
Be specific with numbers. Reference run-level patterns.
No generic advice. No preamble.

─── JOB PROFILE ──────────────────────────────────────────
Job Name:        {job_name}
Runs Analysed:   {aggregated['total_runs']}
Failed Runs:     {aggregated['failed_runs']} ({aggregated['failure_rate_pct']}%)
Avg Duration:    {aggregated['avg_duration_mins']} mins
Max Duration:    {aggregated['max_duration_mins']} mins

─── CURRENT CLUSTER ──────────────────────────────────────
Worker VM:       {cluster.get('worker_node_type')}
Worker Count:    {cluster.get('effective_workers')}
Total Cores:     {cluster.get('total_cores')}
Total Memory:    {cluster.get('total_memory_gb')} GB
Avg Cost/Run:    ${savings['current_avg_cost_usd']}
Total (10 runs): ${savings['current_total_cost_usd']}

─── UTILISATION ACROSS {aggregated['total_runs']} RUNS ───────────────────
CPU avg:         {aggregated['avg_cpu_pct']}%
CPU median:      {aggregated['median_cpu_pct']}%
CPU p90:         {aggregated['p90_cpu_pct']}%  ← primary signal
CPU max ever:    {aggregated['max_cpu_pct']}%
Memory avg:      {aggregated['avg_mem_pct']}%
Memory p90:      {aggregated['p90_mem_pct']}%
Memory max:      {aggregated['max_mem_pct']}%
Spill runs:      {aggregated['spill_run_count']} of {aggregated['total_runs']} runs

─── PER-RUN BREAKDOWN ────────────────────────────────────
{per_run_rows}

─── VERDICT ──────────────────────────────────────────────
Verdict:         {verdict_result['verdict']}
Reason:          {verdict_result['reason']}
Priority:        {verdict_result['priority']}
Recommended VM:  {savings['recommended_node']}
Rec. Workers:    {savings['recommended_workers']}
Savings/Run:     ${savings['savings_per_run_usd']} ({savings['savings_pct']}%)
Annual Savings:  ${savings['annual_savings_usd']}

Respond ONLY in this JSON — no preamble, no markdown fences:
{{
  "findings": [
    "Finding 1 — specific numbers and run references",
    "Finding 2 — specific numbers and run references",
    "Finding 3 — specific numbers and run references"
  ],
  "recommendations": [
    "Recommendation 1 — exact config change and expected impact",
    "Recommendation 2 — exact config change and expected impact"
  ],
  "risk_flags": [
    "Any reliability concerns — failures, spill, duration variance"
  ],
  "summary": "Two sentence executive summary."
}}
"""

    raw    = call_llm(prompt, max_tokens=900, temperature=0.1)
    result = parse_llm_json(raw)
    return result

# ── Format markdown report ────────────────────────────────
def format_report_md(job_name: str,
                      cluster: dict,
                      aggregated: dict,
                      verdict_result: dict,
                      savings: dict,
                      narrative: dict,
                      runs: list[dict]) -> str:

    from datetime import datetime

    EMOJI = {
        "OVER_PROVISIONED":  "⚠️",
        "UNDER_PROVISIONED": "🔴",
        "OPTIMAL":           "✅"
    }
    emoji = EMOJI.get(verdict_result["verdict"], "❓")

    per_run_table = "\n".join([
        f"| {i+1} | {r['start_time'][:10]} | "
        f"{r.get('avg_cpu_pct','?')}% | "
        f"{r.get('avg_mem_pct','?')}% | "
        f"{r.get('disk_spill_gb',0)} GB | "
        f"${r.get('cost_usd',0)} | "
        f"{r.get('result_state')} |"
        for i, r in enumerate(aggregated.get("per_run_data", []))
    ])

    findings_text = "\n".join(
        f"- {f}" for f in narrative.get("findings", [])
    )
    recs_text = "\n".join(
        f"{i+1}. {r}"
        for i, r in enumerate(narrative.get("recommendations", []))
    )
    risks_text = "\n".join(
        f"- {r}" for r in narrative.get("risk_flags", [])
    ) or "- None identified"

    return f"""
# Cluster Right-Sizing Report
**Job:** {job_name}  
**Runs Analysed:** {aggregated['total_runs']}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## {emoji} Verdict: {verdict_result['verdict']}
**Priority:** {verdict_result['priority']}  
**Reason:** {verdict_result['reason']}

---

## Configuration

| | Current | Recommended |
|---|---|---|
| Worker VM | `{savings['current_node']}` | `{savings['recommended_node']}` |
| Workers | {savings['current_workers']} | {savings['recommended_workers']} |
| Avg Cost / Run | ${savings['current_avg_cost_usd']} | ${savings['projected_avg_cost_usd']} |
| Savings / Run | | ${savings['savings_per_run_usd']} ({savings['savings_pct']}%) |
| **Annual Savings** | | **${savings['annual_savings_usd']}** |

---

## Utilisation Summary ({aggregated['total_runs']} runs)

| Metric | Avg | Median | p90 | Max |
|---|---|---|---|---|
| CPU | {aggregated['avg_cpu_pct']}% | {aggregated['median_cpu_pct']}% | {aggregated['p90_cpu_pct']}% | {aggregated['max_cpu_pct']}% |
| Memory | {aggregated['avg_mem_pct']}% | {aggregated['median_mem_pct']}% | {aggregated['p90_mem_pct']}% | {aggregated['max_mem_pct']}% |
| Disk Spill | {aggregated['spill_run_count']} of {aggregated['total_runs']} runs had spill | | | {aggregated['max_spill_gb']} GB |

---

## Per-Run Breakdown

| Run | Date | CPU Avg | Mem Avg | Spill | Cost | Result |
|---|---|---|---|---|---|---|
{per_run_table}

---

## Findings
{findings_text}

---

## Recommendations
{recs_text}

---

## Risk Flags
{risks_text}

---

## Summary
{narrative.get('summary', '')}
"""

print("✅ LLM narrative loaded")