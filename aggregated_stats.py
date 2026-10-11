import os
import csv
import statistics
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

# Standardized Hash Code categories based on file naming patterns
INSTANCE_CATEGORIES = {
    "me_at_the_zoo": "Standard (Sparse)",
    "kittens": "Standard (Balanced)",
    "trending_today": "Standard (Dense)",
    "videos_worth_spreading": "Standard (Constrained)",
    "custom_dejavu42": "Custom Graph",
    "custom_universallambda42": "Custom Graph",
    "custom_universallambda42_asymmetric": "Custom Graph",
    "custom_medium_flat_dense_dejavu_13_seed54": "Custom Graph",
    "instance1": "Hidden Eval",
    "instance2": "Hidden Eval"
}

STRATEGY_MAPS = {
    "init": {0: "BASELINE", 1: "RANDOM", 2: "NOISY_ADD", 3: "DENSITY", 4: "ACTIVE_FRONT", 5: "HYBRID"},
    "cx": {0: "BASELINE", 1: "ACTIVE_OX1", 2: "POS", 3: "PMX", 4: "CX", 5: "ACTIVE_PPX", 6: "ACTIVE_SET"},
    "mut": {0: "BASELINE", 1: "ACTIVE_SWAP", 2: "ACTIVE_INV", 3: "ACTIVE_INS", 4: "BOUNDARY", 5: "PROMOTE", 6: "GRAPH_REPAIR"},
    "ls": {0: "NONE", 1: "FI_SWAP", 2: "BI_INSERT", 3: "ACTIVE_2OPT", 4: "FREQ_GREEDY"},
    "dec": {0: "STATIC", 1: "SOFT_SLACK", 2: "DEMAND", 3: "FIXED_THRESH", 4: "CO_EVOLVED"}
}

def load_round_data(csv_file):
    records = []
    if not os.path.exists(csv_file): return records
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try: row[k] = float(v) if '.' in v else int(v)
                except ValueError: pass

            # Map instance to its category
            inst_name = row.get("instance", "")
            row["category"] = INSTANCE_CATEGORIES.get(inst_name, "Other")
            records.append(row)
    return records

def aggregate_by_strategy_and_category(records, map_dict):
    # grouped[strategy][category] = [list of records]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        strat_name = map_dict.get(r["tested_id"], str(r["tested_id"]))
        cat = r["category"]
        grouped[strat_name]["Overall"].append(r)
        grouped[strat_name][cat].append(r)
    return grouped

def generate_grouped_plot(title, filename, grouped_data, metrics, y_labels):
    strategies = list(grouped_data.keys())
    # Extract unique categories present in the data, ensuring "Overall" is first
    categories = ["Overall"] + sorted({cat for strat in grouped_data.values() for cat in strat.keys() if cat != "Overall"})

    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=y_labels, shared_xaxes=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, metric in enumerate(metrics):
        for j, cat in enumerate(categories):
            y_vals = []
            for strat in strategies:
                vals = [run.get(metric, 0.0) for run in grouped_data[strat][cat]]
                y_vals.append(statistics.mean(vals) if vals else 0.0)

            # Only show the legend for the first subplot to avoid duplicates
            show_leg = (i == 0)
            fig.add_trace(go.Bar(
                x=strategies,
                y=y_vals,
                name=cat,
                marker_color=colors[j % len(colors)],
                showlegend=show_leg
            ), row=i+1, col=1)

    fig.update_layout(
        height=400 * len(metrics),
        title_text=title,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Invert Y axis for Optimality Gap so smaller bars (better) look logical
    for i, label in enumerate(y_labels):
        if "Gap" in label:
            fig.update_yaxes(autorange="reversed", row=i+1, col=1)

    fig.write_html(filename)
    print(f"Generated {filename}")

def main():
    # Slide 1: Initialization Setup
    f_init = glob.glob("tournament_Round1_init.csv")
    if f_init:
        data = aggregate_by_strategy_and_category(load_round_data(f_init[0]), STRATEGY_MAPS["init"])
        generate_grouped_plot("Round 1: Initialization (Diversity vs Fitness)", "plot_slide1_init.html", data,
                              ["optimality_gap_pct", "gen_0_diversity_d0"],
                              ["Final Optimality Gap (%) [Lower is Better]", "Genotypic Diversity (d0)"])

    # Slide 2: Crossover Setup
    f_cx = glob.glob("tournament_Round2_cx.csv")
    if f_cx:
        data = aggregate_by_strategy_and_category(load_round_data(f_cx[0]), STRATEGY_MAPS["cx"])
        generate_grouped_plot("Round 2: Crossover (Success vs Deleterious Drop)", "plot_slide2_cx.html", data,
                              ["crossover_success_rate", "cx_deleterious_avg_drop"],
                              ["Success Rate (%)", "Avg Deleterious Fitness Drop"])

    # Slide 3: Mutation Setup (Highlight Custom Operator)
    f_mut = glob.glob("tournament_Round3_mut.csv")
    if f_mut:
        data = aggregate_by_strategy_and_category(load_round_data(f_mut[0]), STRATEGY_MAPS["mut"])
        generate_grouped_plot("Round 3: Mutation (Custom Graph Repair Validation)", "plot_slide3_mut.html", data,
                              ["optimality_gap_pct", "mutation_success_rate"],
                              ["Final Optimality Gap (%) [Lower is Better]", "Mutation Success Rate (%)"])

    # Slide 4: Local Search Setup (Highlight Custom Operator)
    f_ls = glob.glob("tournament_Round4_ls.csv")
    if f_ls:
        data = aggregate_by_strategy_and_category(load_round_data(f_ls[0]), STRATEGY_MAPS["ls"])
        generate_grouped_plot("Round 4: Local Search (Delta Gains vs CPU Speed)", "plot_slide4_ls.html", data,
                              ["local_search_avg_delta", "avg_eval_microseconds"],
                              ["Avg Local Search Fitness Gain", "Avg Evaluation Time (μs)"])

    # Slide 5: Decoding Setup
    f_dec = glob.glob("tournament_Round5_dec.csv")
    if f_dec:
        data = aggregate_by_strategy_and_category(load_round_data(f_dec[0]), STRATEGY_MAPS["dec"])
        generate_grouped_plot("Round 5: Decoding (Utilization)", "plot_slide5_dec.html", data,
                              ["optimality_gap_pct", "cache_utilization_ratio"],
                              ["Final Optimality Gap (%) [Lower is Better]", "Cache Utilization Ratio (%)"])

if __name__ == "__main__":
    main()