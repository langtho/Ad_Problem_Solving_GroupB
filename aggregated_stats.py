#!/usr/bin/env python3
import os
import glob
import json
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "benchmark_results"
OUTPUT_HTML = "aggregated_benchmark_dashboard.html"

# Strategy labels matching sol.cpp ENUM definitions
STRATEGY_NAMES = {
    "init": {
        0: "BASELINE_EXACT_NOISY",
        1: "PURE_RANDOM",
        2: "NOISY_PRIORITY_ADD",
        3: "DENSITY_REQUEST_SORT",
        4: "ACTIVE_FRONT_SHUFFLE",
        5: "HYBRID_COMPOSITE"
    },
    "cx": {
        0: "BASELINE_OX1",
        1: "ACTIVE_PREFIX_OX1",
        2: "POSITION_BASED_POS",
        3: "PARTIALLY_MAPPED_PMX",
        4: "CYCLE_CROSSOVER_CX"
    },
    "mut": {
        0: "BASELINE_SWAP",
        1: "ACTIVE_PREFIX_SWAP",
        2: "ACTIVE_PREFIX_INVERSION",
        3: "ACTIVE_PREFIX_INSERTION",
        4: "BOUNDARY_SWAP"
    },
    "ls": {
        0: "NONE",
        1: "FIRST_IMPROVEMENT_SWAP",
        2: "BEST_IMPROVEMENT_ACTIVE_INSERT",
        3: "ACTIVE_WINDOW_2OPT_SCAN",
        4: "REQUEST_FREQUENCY_GREEDY"
    },
    "dec": {
        0: "STATIC_GREEDY",
        1: "SOFT_SLACK_REPAIRED",
        2: "DEMAND_CACHE_CENTRIC",
        3: "FIXED_THRESHOLD_PROFILE",
        4: "CO_EVOLVED_THRESHOLDS"
    }
}


def load_and_normalize_data():
    json_files = glob.glob(os.path.join(RESULTS_DIR, "fitness_plot_*.json"))
    if not json_files:
        print(f"No JSON files found in directory: '{RESULTS_DIR}/'")
        return []

    records = []
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}")

    # Compute max fitness per instance to scale scores from 0.0 to 100.0%
    instance_max_fitness = defaultdict(float)
    for r in records:
        inst = r.get("instance", "unknown").split("_init")[0]
        fitness = r.get("best_fitness_achieved", 0.0)
        if fitness > instance_max_fitness[inst]:
            instance_max_fitness[inst] = fitness

    # Apply per-instance relative normalization
    for r in records:
        inst = r.get("instance", "unknown").split("_init")[0]
        max_fit = instance_max_fitness[inst]
        if max_fit > 0:
            r["norm_fitness_pct"] = (r.get("best_fitness_achieved", 0.0) / max_fit) * 100.0
            r["norm_gen0_best_pct"] = (r.get("gen_0_best_score", 0.0) / max_fit) * 100.0
        else:
            r["norm_fitness_pct"] = 0.0
            r["norm_gen0_best_pct"] = 0.0

    return records


def aggregate_by_block(records, id_param_name, name_mapping):
    groups = defaultdict(list)
    for r in records:
        strat_id = r.get(id_param_name, 0)
        groups[strat_id].append(r)

    labels = []
    norm_fitness = []
    gen0_best = []
    cx_succ = []
    mut_succ = []
    ls_succ = []
    ls_delta = []
    eval_latency = []
    cache_util = []

    for strat_id, group in sorted(groups.items()):
        n = len(group)
        labels.append(name_mapping.get(strat_id, f"ID_{strat_id}"))

        norm_fitness.append(sum(r["norm_fitness_pct"] for r in group) / n)
        gen0_best.append(sum(r["norm_gen0_best_pct"] for r in group) / n)
        cx_succ.append(sum(r.get("crossover_success_rate", 0.0) for r in group) / n * 100.0)
        mut_succ.append(sum(r.get("mutation_success_rate", 0.0) for r in group) / n * 100.0)
        ls_succ.append(sum(r.get("local_search_success_rate", 0.0) for r in group) / n * 100.0)
        ls_delta.append(sum(r.get("local_search_avg_delta", 0.0) for r in group) / n)
        eval_latency.append(sum(r.get("avg_eval_microseconds", 0.0) for r in group) / n)
        cache_util.append(sum(r.get("cache_utilization_ratio", 0.0) for r in group) / n * 100.0)

    return {
        "labels": labels,
        "norm_fitness": norm_fitness,
        "gen0_best": gen0_best,
        "cx_succ": cx_succ,
        "mut_succ": mut_succ,
        "ls_succ": ls_succ,
        "ls_delta": ls_delta,
        "eval_latency": eval_latency,
        "cache_util": cache_util
    }


def generate_dashboard():
    records = load_and_normalize_data()
    if not records:
        return

    # Aggregate block data
    init_data = aggregate_by_block(records, "init_strategy_id", STRATEGY_NAMES["init"])
    cx_data   = aggregate_by_block(records, "crossover_strategy_id", STRATEGY_NAMES["cx"])
    mut_data  = aggregate_by_block(records, "mutation_strategy_id", STRATEGY_NAMES["mut"])
    ls_data   = aggregate_by_block(records, "local_search_strategy_id", STRATEGY_NAMES["ls"])
    dec_data  = aggregate_by_block(records, "decoding_strategy_id", STRATEGY_NAMES["dec"])

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Block 1: Initializations (Gen 0 vs Final Fitness)",
            "Block 2: Crossover Operators (Success Rate vs Final Fitness)",
            "Block 3: Mutation Operators (Success Rate vs Final Fitness)",
            "Block 4: Local Search (Success Rate & Average Delta)",
            "Block 5: Decoder Latency (μs/eval) vs Final Fitness",
            "Block 5: Cache Utilization Ratio (%) by Decoder"
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )

    # Palette
    c_blue = "#1f77b4"
    c_cyan = "#17becf"
    c_green = "#2ca02c"
    c_orange = "#ff7f0e"
    c_purple = "#9467bd"
    c_red = "#d62728"

    # --- Panel 1: Initialization ---
    fig.add_trace(
        go.Bar(x=init_data["labels"], y=init_data["gen0_best"], name="Gen 0 Best (% Max)", marker_color=c_cyan),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=init_data["labels"], y=init_data["norm_fitness"], name="Final Fitness (% Max)", marker_color=c_blue),
        row=1, col=1
    )

    # --- Panel 2: Crossover ---
    fig.add_trace(
        go.Bar(x=cx_data["labels"], y=cx_data["cx_succ"], name="CX Success Rate (%)", marker_color=c_green),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=cx_data["labels"], y=cx_data["norm_fitness"], name="Final Fitness (%)",
                   mode="lines+markers", line=dict(color=c_orange, width=3)),
        row=1, col=2
    )

    # --- Panel 3: Mutation ---
    fig.add_trace(
        go.Bar(x=mut_data["labels"], y=mut_data["mut_succ"], name="Mut Success Rate (%)", marker_color="#e377c2"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=mut_data["labels"], y=mut_data["norm_fitness"], name="Final Fitness (%)",
                   mode="lines+markers", line=dict(color=c_blue, width=3)),
        row=2, col=1
    )

    # --- Panel 4: Local Search ---
    fig.add_trace(
        go.Bar(x=ls_data["labels"], y=ls_data["ls_succ"], name="LS Success Rate (%)", marker_color=c_purple),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=ls_data["labels"], y=ls_data["ls_delta"], name="LS Avg Delta Gain", marker_color=c_red),
        row=2, col=2
    )

    # --- Panel 5: Decoder Latency vs Fitness ---
    fig.add_trace(
        go.Bar(x=dec_data["labels"], y=dec_data["eval_latency"], name="Eval Latency (μs)", marker_color="#7f7f7f"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dec_data["labels"], y=dec_data["norm_fitness"], name="Final Fitness (%)",
                   mode="lines+markers", line=dict(color=c_blue, width=3)),
        row=3, col=1
    )

    # --- Panel 6: Decoder Cache Utilization ---
    fig.add_trace(
        go.Bar(x=dec_data["labels"], y=dec_data["cache_util"], name="Cache Utilization (%)", marker_color="#bcbd22"),
        row=3, col=2
    )

    fig.update_layout(
        title_text="Multi-Block Algorithm Benchmark Analytics (Cross-Instance Normalized)",
        barmode="group",
        height=1350,
        showlegend=False,
        template="plotly_white"
    )

    fig.update_yaxes(title_text="% Score", row=1, col=1)
    fig.update_yaxes(title_text="Rate / Score %", row=1, col=2)
    fig.update_yaxes(title_text="Rate / Score %", row=2, col=1)
    fig.update_yaxes(title_text="Rate / Delta", row=2, col=2)
    fig.update_yaxes(title_text="μs / % Score", row=3, col=1)
    fig.update_yaxes(title_text="Utilization %", range=[0, 105], row=3, col=2)

    fig.write_html(OUTPUT_HTML)
    print(f"Aggregated summary dashboard saved to: {OUTPUT_HTML}")


if __name__ == "__main__":
    generate_dashboard()