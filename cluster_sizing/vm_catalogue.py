# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 00_vm_catalogue.py
# Reference table for VM specs and cost estimates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VM_CATALOGUE = {
    # ── AWS ───────────────────────────────────────────────
    # General purpose
    "m5.xlarge":          {"vcpu": 4,  "mem_gb": 16,  "dbu": 0.75, "type": "general", "cloud": "aws"},
    "m5.2xlarge":         {"vcpu": 8,  "mem_gb": 32,  "dbu": 1.5,  "type": "general", "cloud": "aws"},
    "m5.4xlarge":         {"vcpu": 16, "mem_gb": 64,  "dbu": 3.0,  "type": "general", "cloud": "aws"},
    # Compute optimised
    "c5.2xlarge":         {"vcpu": 8,  "mem_gb": 16,  "dbu": 1.5,  "type": "compute", "cloud": "aws"},
    "c5.4xlarge":         {"vcpu": 16, "mem_gb": 32,  "dbu": 3.0,  "type": "compute", "cloud": "aws"},
    # Memory optimised
    "r5.2xlarge":         {"vcpu": 8,  "mem_gb": 64,  "dbu": 1.5,  "type": "memory",  "cloud": "aws"},
    "r5.4xlarge":         {"vcpu": 16, "mem_gb": 128, "dbu": 3.0,  "type": "memory",  "cloud": "aws"},
    # Storage optimised
    "i3.2xlarge":         {"vcpu": 8,  "mem_gb": 61,  "dbu": 1.5,  "type": "storage", "cloud": "aws"},
    "i3.4xlarge":         {"vcpu": 16, "mem_gb": 122, "dbu": 3.0,  "type": "storage", "cloud": "aws"},

    # ── Azure ─────────────────────────────────────────────
    # General purpose
    "Standard_D4s_v3":    {"vcpu": 4,  "mem_gb": 16,  "dbu": 0.75, "type": "general", "cloud": "azure"},
    "Standard_D8s_v3":    {"vcpu": 8,  "mem_gb": 32,  "dbu": 1.5,  "type": "general", "cloud": "azure"},
    "Standard_D16s_v3":   {"vcpu": 16, "mem_gb": 64,  "dbu": 3.0,  "type": "general", "cloud": "azure"},
    # Memory optimised
    "Standard_E8s_v3":    {"vcpu": 8,  "mem_gb": 64,  "dbu": 1.5,  "type": "memory",  "cloud": "azure"},
    "Standard_E16s_v3":   {"vcpu": 16, "mem_gb": 128, "dbu": 3.0,  "type": "memory",  "cloud": "azure"},
    # Compute optimised
    "Standard_F8s_v2":    {"vcpu": 8,  "mem_gb": 16,  "dbu": 1.5,  "type": "compute", "cloud": "azure"},
    "Standard_F16s_v2":   {"vcpu": 16, "mem_gb": 32,  "dbu": 3.0,  "type": "compute", "cloud": "azure"},
}

# Update this to your contract DBU rate
DBU_PRICE_USD = 0.22

def get_vm_specs(node_type: str) -> dict:
    return VM_CATALOGUE.get(node_type, {
        "vcpu": 4, "mem_gb": 16, "dbu": 1.0,
        "type": "general", "cloud": "unknown"
    })

def estimate_cost(node_type: str,
                   num_workers: int,
                   duration_mins: float) -> float:
    specs  = get_vm_specs(node_type)
    hours  = duration_mins / 60
    return round(specs["dbu"] * num_workers * hours * DBU_PRICE_USD, 4)

def _upgrade_to_memory_optimised(current_node: str) -> str:
    upgrades = {
        "m5.xlarge":         "r5.2xlarge",
        "m5.2xlarge":        "r5.2xlarge",
        "m5.4xlarge":        "r5.4xlarge",
        "c5.2xlarge":        "r5.2xlarge",
        "c5.4xlarge":        "r5.4xlarge",
        "Standard_D4s_v3":   "Standard_E8s_v3",
        "Standard_D8s_v3":   "Standard_E8s_v3",
        "Standard_D16s_v3":  "Standard_E16s_v3",
        "Standard_F8s_v2":   "Standard_E8s_v3",
        "Standard_F16s_v2":  "Standard_E16s_v3",
    }
    return upgrades.get(current_node, current_node)

def _downsize_vm_if_possible(current_node: str,
                               new_workers: int,
                               old_workers: int) -> str:
    reduction_pct = (old_workers - new_workers) / max(old_workers, 1)
    if reduction_pct < 0.4:
        return current_node
    downgrades = {
        "m5.4xlarge":        "m5.2xlarge",
        "m5.2xlarge":        "m5.xlarge",
        "r5.4xlarge":        "r5.2xlarge",
        "i3.4xlarge":        "i3.2xlarge",
        "Standard_D16s_v3":  "Standard_D8s_v3",
        "Standard_D8s_v3":   "Standard_D4s_v3",
        "Standard_E16s_v3":  "Standard_E8s_v3",
        "Standard_F16s_v2":  "Standard_F8s_v2",
    }
    return downgrades.get(current_node, current_node)

print("✅ VM catalogue loaded")