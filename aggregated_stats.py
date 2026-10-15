import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = "benchmark_results"
LOGS_DIR = "benchmark_results"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
HTML_FILE = os.path.join(BASE_DIR, "benchmark_report.html")

os.makedirs(PLOTS_DIR, exist_ok=True)

INSTANCE_TAGS = {
    "me_at_the_zoo": ["Small-Scale", "Constrained-Cache"],
    "INSTANCED": ["Small-Scale", "Constrained-Cache"],
    "regional_instance": ["Small-Scale", "Abundant-Cache", "Low-Contention"],
    "zipF_instance": ["Small-Scale", "Abundant-Cache", "Skewed-Distribution", "Low-Contention"],
    "supra": ["Small-Scale", "Low-Contention"],
    "medium_flat_dense_dejavu": ["Medium-Scale", "Constrained-Cache", "High-Contention"],
    "medium_flat_sparse_lambda": ["Medium-Scale", "Constrained-Cache", "High-Contention"],
    "medium_mixed_lambda": ["Medium-Scale", "Constrained-Cache", "High-Contention"],
    "medium_skew_dense_dejavu": ["Medium-Scale", "Constrained-Cache", "High-Contention", "Skewed-Distribution"],
    "medium_skew_sparse_lambda": ["Medium-Scale", "Constrained-Cache", "High-Contention", "Skewed-Distribution"],
    "3860_246_84901_66": ["Medium-Scale", "Abundant-Cache"],
    "4990_246_84901_50": ["Medium-Scale", "Abundant-Cache"],
    "instance1": ["Medium-Scale", "High-Contention"],
    "instance2": ["Medium-Scale", "Abundant-Cache"],
    "INSATANC2D": ["Medium-Scale", "Constrained-Cache", "High-Contention"],
    "trending_today": ["Massive-Scale", "Abundant-Cache"],
    "videos_worth_spreading": ["Massive-Scale", "Constrained-Cache"],
    "kittens": ["Massive-Scale", "Abundant-Cache"],
    "custom_dejavu42": ["Massive-Scale", "Constrained-Cache", "High-Contention", "Skewed-Distribution"],
    "custom_universallambda42": ["Massive-Scale", "Constrained-Cache", "High-Contention"],
    "custom_universallambda42_asymmetric": ["Massive-Scale", "Constrained-Cache"],
    "10000_500_100000_100": ["Massive-Scale", "Abundant-Cache"],
    "6970_311_100000_69": ["Massive-Scale", "Abundant-Cache"],
    "big": ["Massive-Scale", "Low-Contention"],
    "dense": ["Massive-Scale", "Constrained-Cache"],
    "dense2": ["Massive-Scale", "Constrained-Cache"],
    "realistic_large_clustered": ["Massive-Scale", "Abundant-Cache"],
    "realistic_large_random": ["Massive-Scale", "Abundant-Cache"]
}

ALL_TAGS = [
    "ALL",
    "Constrained-Cache",
    "Abundant-Cache",
    "Small-Scale",
    "Medium-Scale",
    "Massive-Scale",
    "High-Contention",
    "Skewed-Distribution"
]

STRATEGY_NAMES = {
    "init": {0: "BASELINE", 1: "RANDOM", 2: "NOISY_ADD", 3: "DENSITY", 4: "ACTIVE_FRONT", 5: "HYBRID"},
    "cx": {0: "BASELINE", 1: "ACTIVE_OX1", 2: "POS", 3: "PMX", 4: "CX", 5: "ACTIVE_PPX", 6: "ACTIVE_SET"},
    "mut": {0: "BASELINE", 1: "ACTIVE_SWAP", 2: "ACTIVE_INV", 3: "ACTIVE_INS", 4: "BOUNDARY", 5: "PROMOTE", 6: "GRAPH_REPAIR"},
    "ls": {0: "NONE", 1: "FI_SWAP", 2: "BI_INSERT", 3: "ACTIVE_2OPT", 4: "FREQ_GREEDY"},
    "dec": {0: "STATIC", 1: "SOFT_SLACK", 2: "DEMAND", 3: "FIXED_THRESH", 4: "CO_EVOLVED"}
}

def clean_instance_name(raw_name):
    base = os.path.basename(raw_name)
    base = base.replace(".in", "").strip()
    if "fitness_plot_" in base:
        base = base.replace("fitness_plot_", "")
    return base.split("_init")[0]

def get_instance_tags(inst_name):
    clean = clean_instance_name(inst_name)
    return INSTANCE_TAGS.get(clean, [])

def load_data():
    records = []
    skipped_zero_fitness = 0

    candidate_dirs = [
        LOGS_DIR,
        os.path.join(BASE_DIR, "raw_logs"),
        "raw_logs",
        "logs",
        "."
    ]

    file_list = []
    for cdir in candidate_dirs:
        found = glob.glob(os.path.join(cdir, "fitness_plot_*.json"))
        if found:
            file_list = found
            print(f"📁 Found {len(file_list)} JSON log file(s) in: {cdir}")
            break

    if not file_list:
        print("❌ Could not find any 'fitness_plot_*.json' files in candidate directories.")
        return records

    for fp in file_list:
        try:
            with open(fp, "r") as f:
                d = json.load(f)
                d["file_path"] = fp

                if d.get("best_fitness_achieved", 0.0) <= 0.0:
                    skipped_zero_fitness += 1
                    continue

                raw_inst = d.get("instance", "")
                clean_name = clean_instance_name(raw_inst)
                if not clean_name:
                    clean_name = clean_instance_name(fp)

                tags = get_instance_tags(clean_name)
                if not tags:
                    continue

                d["clean_instance"] = clean_name
                d["tags"] = tags
                records.append(d)
        except Exception:
            continue

    if skipped_zero_fitness > 0:
        print(f"Skipped {skipped_zero_fitness} zero-fitness runs (timeouts/crashes).")
    print(f" Loaded {len(records)} valid runs for analysis.")
    return records

def normalize_records(records):
    inst_bfv = defaultdict(float)

    # 1. Determine Empirical Best-Found Value (BFV) across all runs for each instance
    for r in records:
        inst = r["clean_instance"]
        fit = r.get("best_fitness_achieved", 0.0)
        if fit > inst_bfv[inst]:
            inst_bfv[inst] = fit

    # 2. Normalize against the local instance empirical maximum
    for r in records:
        inst = r["clean_instance"]
        bfv = inst_bfv[inst]
        fit = r.get("best_fitness_achieved", 0.0)
        r["bfv"] = bfv
        r["norm_fitness"] = (fit / bfv * 100.0) if bfv > 0 else 0.0
        g0_best = r.get("gen_0_best_score", 0.0)
        r["norm_gen_0_best"] = (g0_best / bfv * 100.0) if bfv > 0 else 0.0

    return records

def generate_plots(records):
    generated = []
    palette = {
        "box": "#CBD5E1",
        "beneficial": "#2563EB",
        "neutral": "#94A3B8",
        "deleterious": "#EF4444",
        "teal": "#0D9488",
        "amber": "#D97706",
        "purple": "#7C3AED",
        "green": "#16A34A"
    }

    for tag in ALL_TAGS:
        sub = records if tag == "ALL" else [r for r in records if tag in r["tags"]]
        if not sub:
            continue
        tag_slug = tag.replace(" ", "_").replace("/", "-")

        # 1. Global Normalized Fitness Across Blocks
        for blk in ["init", "cx", "mut", "ls", "dec"]:
            key_map = {
                "init": "init_strategy_id",
                "cx": "crossover_strategy_id",
                "mut": "mutation_strategy_id",
                "ls": "local_search_strategy_id",
                "dec": "decoding_strategy_id"
            }
            key = key_map[blk]
            op_scores = defaultdict(list)
            for r in sub:
                val = r.get(key)
                if val is not None and val in STRATEGY_NAMES[blk]:
                    op_scores[STRATEGY_NAMES[blk][val]].append(r["norm_fitness"])

            if not op_scores:
                continue

            labels = list(op_scores.keys())
            data = [op_scores[l] for l in labels]

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor=palette["box"]))
            ax.set_title(f"{blk.upper()} - Normalized Quality [{tag}]")
            ax.set_ylabel("% of Best-Found Value (BFV)")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            fname = f"{tag_slug}_{blk}_fitness.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": blk, "file": fname,
                "title": f"{blk.upper()} Fitness Distribution",
                "desc": f"Normalized solution quality (% BFV) for {blk.upper()} operators across [{tag}] instances."
            })

        # 2. Initialization Dynamics (Seed Fitness vs Diversity)
        init_sub = [r for r in sub if r.get("gen_0_diversity_d0", 0.0) > 0.0]
        if init_sub:
            strat_div = defaultdict(lambda: {"d0": [], "fit": [], "pheno": []})
            for r in init_sub:
                sid = r.get("init_strategy_id", 0)
                sname = STRATEGY_NAMES["init"].get(sid, f"ID_{sid}")
                strat_div[sname]["d0"].append(r.get("gen_0_diversity_d0", 0.0))
                strat_div[sname]["fit"].append(r.get("norm_gen_0_best", 0.0))
                strat_div[sname]["pheno"].append(r.get("phenotypic_diversity_d0", 0))

            labels = list(strat_div.keys())
            d0_means = [np.mean(strat_div[l]["d0"]) for l in labels]
            fit_means = [np.mean(strat_div[l]["fit"]) for l in labels]
            pheno_means = [np.mean(strat_div[l]["pheno"]) for l in labels]

            fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
            ax2 = ax1.twinx()
            ax1.bar(np.arange(len(labels)) - 0.2, fit_means, width=0.4, color=palette["teal"], label="Gen 0 Best Fitness")
            ax2.scatter(np.arange(len(labels)) + 0.2, d0_means, s=np.array(pheno_means) * 15 + 40, color=palette["purple"], label="D0 Rank Diversity")

            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=25, ha="right")
            ax1.set_ylabel("Gen 0 Best Fitness (% BFV)", color=palette["teal"])
            ax2.set_ylabel("D0 Active Rank Diversity", color=palette["purple"])
            ax1.set_title(f"Init Trade-off: Seed Fitness vs Diversity [{tag}]")
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()

            fname = f"{tag_slug}_init_tradeoff.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "init", "file": fname,
                "title": "Init: Seed Fitness vs Population Diversity",
                "desc": f"Gen 0 starting quality (% BFV) versus diversity (D0) on [{tag}] instances."
            })

        # 3. Crossover Operator Mechanics
        cx_sub = [r for r in sub if "crossover_success_rate" in r]
        if cx_sub:
            strat_agg = defaultdict(lambda: {"succ": [], "neut": [], "del": [], "drop": []})
            for r in cx_sub:
                sid = r.get("crossover_strategy_id", 0)
                sname = STRATEGY_NAMES["cx"].get(sid, f"ID_{sid}")
                s = r.get("crossover_success_rate", 0.0)
                n = r.get("crossover_neutral_rate", 0.0)
                d = max(0.0, 1.0 - (s + n))
                strat_agg[sname]["succ"].append(s)
                strat_agg[sname]["neut"].append(n)
                strat_agg[sname]["del"].append(d)
                strat_agg[sname]["drop"].append(r.get("cx_deleterious_avg_drop", 0.0))

            labels = list(strat_agg.keys())
            succ_means = [np.mean(strat_agg[l]["succ"]) * 100 for l in labels]
            neut_means = [np.mean(strat_agg[l]["neut"]) * 100 for l in labels]
            del_means = [np.mean(strat_agg[l]["del"]) * 100 for l in labels]
            drop_means = [np.mean(strat_agg[l]["drop"]) for l in labels]

            fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
            ax2 = ax1.twinx()
            ax1.bar(labels, succ_means, label="Beneficial", color=palette["beneficial"])
            ax1.bar(labels, neut_means, bottom=succ_means, label="Neutral", color=palette["neutral"])
            ax1.bar(labels, del_means, bottom=np.array(succ_means) + np.array(neut_means), label="Deleterious", color=palette["deleterious"])
            ax2.plot(labels, drop_means, color="#7F1D1D", marker="x", linewidth=2.0, label="Deleterious Drop")

            ax1.set_ylabel("Offspring Percentage (%)")
            ax2.set_ylabel("Mean Fitness Drop")
            ax1.set_title(f"Crossover Action Split & Destructive Severity [{tag}]")
            ax1.legend(loc="upper left")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            fname = f"{tag_slug}_cx_mechanics.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "cx", "file": fname,
                "title": "Crossover Action Breakdown & Destructive Drop",
                "desc": f"Mechanics breakdown (Beneficial, Neutral, Deleterious) and severity drop on [{tag}] instances."
            })

        # 4. Mutation Mechanics
        mut_sub = [r for r in sub if "mutation_success_rate" in r]
        if mut_sub:
            mut_agg = defaultdict(lambda: {"succ": [], "neut": [], "del": [], "drop": []})
            for r in mut_sub:
                sid = r.get("mutation_strategy_id", 0)
                sname = STRATEGY_NAMES["mut"].get(sid, f"ID_{sid}")
                s = r.get("mutation_success_rate", 0.0)
                n = r.get("mutation_neutral_rate", 0.0)
                d = max(0.0, 1.0 - (s + n))
                mut_agg[sname]["succ"].append(s)
                mut_agg[sname]["neut"].append(n)
                mut_agg[sname]["del"].append(d)
                mut_agg[sname]["drop"].append(r.get("mut_deleterious_avg_drop", 0.0))

            labels = list(mut_agg.keys())
            succ_means = [np.mean(mut_agg[l]["succ"]) * 100 for l in labels]
            neut_means = [np.mean(mut_agg[l]["neut"]) * 100 for l in labels]
            del_means = [np.mean(mut_agg[l]["del"]) * 100 for l in labels]
            drop_means = [np.mean(mut_agg[l]["drop"]) for l in labels]

            fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
            ax2 = ax1.twinx()
            ax1.bar(labels, succ_means, label="Beneficial", color=palette["beneficial"])
            ax1.bar(labels, neut_means, bottom=succ_means, label="Neutral", color=palette["neutral"])
            ax1.bar(labels, del_means, bottom=np.array(succ_means) + np.array(neut_means), label="Deleterious", color=palette["deleterious"])
            ax2.plot(labels, drop_means, color="#7F1D1D", marker="o", linewidth=2.0, label="Deleterious Drop")

            ax1.set_ylabel("Mutation Output (%)")
            ax2.set_ylabel("Mean Fitness Drop")
            ax1.set_title(f"Mutation Mechanics & Disruption Severity [{tag}]")
            ax1.legend(loc="upper left")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            fname = f"{tag_slug}_mut_mechanics.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "mut", "file": fname,
                "title": "Mutation Action Breakdown & Destructive Drop",
                "desc": f"Distribution of beneficial, neutral, and deleterious mutation operations across [{tag}]."
            })

        # 5. Local Search Neighborhoods
        ls_sub = [r for r in sub if "local_search_success_rate" in r]
        if ls_sub:
            ls_agg = defaultdict(lambda: {"succ": [], "cap": [], "delta": []})
            for r in ls_sub:
                sid = r.get("local_search_strategy_id", 0)
                sname = STRATEGY_NAMES["ls"].get(sid, f"ID_{sid}")
                ls_agg[sname]["succ"].append(r.get("local_search_success_rate", 0.0) * 100)
                ls_agg[sname]["cap"].append(r.get("ls_cap_hit_rate", 0.0) * 100)
                ls_agg[sname]["delta"].append(r.get("local_search_avg_delta", 0.0))

            labels = list(ls_agg.keys())
            succ_means = [np.mean(ls_agg[l]["succ"]) for l in labels]
            cap_means = [np.mean(ls_agg[l]["cap"]) for l in labels]
            delta_means = [np.mean(ls_agg[l]["delta"]) for l in labels]

            fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
            ax2 = ax1.twinx()
            x = np.arange(len(labels))
            ax1.bar(x - 0.2, succ_means, width=0.4, label="Success Rate (%)", color=palette["green"])
            ax1.bar(x + 0.2, cap_means, width=0.4, label="Cap Hit Rate (%)", color=palette["amber"])
            ax2.plot(x, delta_means, color=palette["purple"], marker="s", linewidth=2.0, label="Avg Gain Delta")

            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=25, ha="right")
            ax1.set_ylabel("Rate (%)")
            ax2.set_ylabel("Fitness Gain Delta")
            ax1.set_title(f"Local Search Efficiency & Neighborhood Exhaustion [{tag}]")
            ax1.legend(loc="upper left")
            plt.tight_layout()

            fname = f"{tag_slug}_ls_efficiency.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "ls", "file": fname,
                "title": "Local Search Efficiency vs Evaluation Cap",
                "desc": f"Success rate, cap exhaustion rate, and average gain for Local Search variants on [{tag}]."
            })

        # 6. Decoding Paradigms
        dec_sub = [r for r in sub if "cache_utilization_ratio" in r]
        if dec_sub:
            dec_agg = defaultdict(lambda: {"util": [], "us": [], "slack": []})
            for r in dec_sub:
                sid = r.get("decoding_strategy_id", 0)
                sname = STRATEGY_NAMES["dec"].get(sid, f"ID_{sid}")
                dec_agg[sname]["util"].append(r.get("cache_utilization_ratio", 0.0) * 100)
                dec_agg[sname]["us"].append(r.get("avg_eval_microseconds", 0.0))
                dec_agg[sname]["slack"].append(r.get("residual_slack_mb", 0.0))

            labels = list(dec_agg.keys())
            util_means = [np.mean(dec_agg[l]["util"]) for l in labels]
            us_means = [np.mean(dec_agg[l]["us"]) for l in labels]
            slack_means = [np.mean(dec_agg[l]["slack"]) for l in labels]

            fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
            ax2 = ax1.twinx()
            ax1.bar(labels, util_means, color=palette["teal"], alpha=0.85, width=0.45)
            ax2.plot(labels, us_means, color=palette["amber"], marker="o", linewidth=2.5, markersize=7)
            ax1.set_ylabel("Cache Utilization (%)", color=palette["teal"])
            ax2.set_ylabel("Mean Eval Duration (μs)", color=palette["amber"])
            ax1.set_title(f"Decoding: Utilization vs Evaluation Overhead [{tag}]")
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            fname = f"{tag_slug}_dec_efficiency.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "dec", "file": fname,
                "title": "Decoder Utilization vs Speed",
                "desc": f"Trade-off between packing efficiency and microsecond runtime on [{tag}] instances."
            })

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.bar(labels, slack_means, color="#64748B", width=0.45)
            ax.set_ylabel("Average Residual Slack (MB)")
            ax.set_title(f"Decoder Residual Capacity Slack [{tag}]")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            fname_slack = f"{tag_slug}_dec_slack.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname_slack))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "dec", "file": fname_slack,
                "title": "Decoder Residual Slack (Unused MB)",
                "desc": f"Average unused megabytes remaining across saturated caches on [{tag}]."
            })

        # 7. Evolutionary Throughput
        throughput_agg = defaultdict(lambda: {"gens": [], "evals": []})
        for r in sub:
            gens = r.get("total_generations", 0)
            evals = r.get("total_evaluations", 0)
            if gens > 0:
                inst = r["clean_instance"]
                throughput_agg[inst]["gens"].append(gens)
                throughput_agg[inst]["evals"].append(evals)

        if throughput_agg:
            inst_labels = list(throughput_agg.keys())
            mean_gens = [np.mean(throughput_agg[k]["gens"]) for k in inst_labels]
            mean_evals = [np.mean(throughput_agg[k]["evals"]) for k in inst_labels]

            fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
            ax2 = ax1.twinx()
            x = np.arange(len(inst_labels))
            ax1.bar(x, mean_gens, color=palette["purple"], alpha=0.75, width=0.4, label="Completed Generations")
            ax2.plot(x, mean_evals, color=palette["amber"], marker="o", linewidth=2.0, label="Total Evals")

            ax1.set_xticks(x)
            ax1.set_xticklabels(inst_labels, rotation=35, ha="right")
            ax1.set_ylabel("Completed Generations in 45s", color=palette["purple"])
            ax2.set_ylabel("Total Evals", color=palette["amber"])
            ax1.set_title(f"Evolutionary Turnover & Evaluation Throughput [{tag}]")
            plt.tight_layout()

            fname_tp = f"{tag_slug}_throughput.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname_tp))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "global", "file": fname_tp,
                "title": "Evolutionary Turnover & Throughput",
                "desc": f"Generations completed within the 45-second execution ceiling for [{tag}]."
            })

        # 8. Convergence Profiles (Empirical Step Curves)
        history_runs = [r for r in sub if r.get("targets_hit")]
        if history_runs:
            fig, ax = plt.subplots(figsize=(9.0, 4.5))
            sample_runs = history_runs[:6]
            for r in sample_runs:
                history = r["targets_hit"]
                evals = [pt["eval"] for pt in history]
                bfv = r.get("bfv", r["best_fitness_achieved"])
                fits = [(pt["fitness"] / bfv * 100.0) if bfv > 0 else 0.0 for pt in history]
                cfg_label = f"{r['clean_instance']} (Init {r.get('init_strategy_id', 0)})"
                ax.step(evals, fits, where="post", label=cfg_label, alpha=0.85)

            ax.set_xlabel("Number of Evaluations")
            ax.set_ylabel("% of Best-Found Value (BFV)")
            ax.set_title(f"Target Velocity Convergence Profiles [{tag}]")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()

            fname_conv = f"{tag_slug}_convergence.svg"
            fig.savefig(os.path.join(PLOTS_DIR, fname_conv))
            plt.close(fig)
            generated.append({
                "tag": tag, "block": "global", "file": fname_conv,
                "title": "Target Velocity Convergence Profiles",
                "desc": f"Convergence trajectory tracking target thresholds hit over evaluations on [{tag}]."
            })

    return generated

def generate_html(plot_meta):
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Genetic Algorithm Telemetry & Benchmark Dashboard</title>
<style>
  :root {
    --bg-main: #0F172A;
    --bg-card: #1E293B;
    --border-color: #334155;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --accent: #38BDF8;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    background: var(--bg-main);
    color: var(--text-primary);
  }
  header {
    background: #0B1120;
    padding: 24px 36px;
    border-bottom: 1px solid var(--border-color);
  }
  header h1 {
    margin: 0 0 6px 0;
    font-size: 22px;
    letter-spacing: -0.02em;
    color: var(--accent);
  }
  header p {
    margin: 0;
    font-size: 14px;
    color: var(--text-secondary);
  }
  .controls-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    padding: 16px 36px;
    background: #162032;
    border-bottom: 1px solid var(--border-color);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .control-group label {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }
  select {
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-card);
    color: var(--text-primary);
    cursor: pointer;
    outline: none;
  }
  select:focus {
    border-color: var(--accent);
  }
  .summary-bar {
    padding: 12px 36px;
    font-size: 13px;
    color: var(--text-secondary);
    border-bottom: 1px solid #1E293B;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 24px;
    padding: 30px 36px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
  }
  .card-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
  }
  .badge {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    background: #0284C7;
    color: white;
  }
  .badge-block {
    background: #475569;
    margin-left: 6px;
  }
  .card-body {
    padding: 16px;
    flex: 1;
    display: flex;
    flex-direction: column;
  }
  .card img {
    width: 100%;
    height: auto;
    border-radius: 6px;
    background: white;
  }
  .card-desc {
    margin: 12px 0 0 0;
    font-size: 13px;
    line-height: 1.4;
    color: var(--text-secondary);
  }
</style>
</head>
<body>
<header>
  <h1>Genetic Algorithm Telemetry & Benchmark Dashboard</h1>
  <p>Standardized Normalization against Empirical Best-Found Values (BFV)</p>
</header>
<div class="controls-bar">
  <div class="control-group">
    <label for="tagFilter">Analytical Tag / Dimension</label>
    <select id="tagFilter" onchange="filterDashboard()">
      <option value="ALL">All Categories Combined</option>
      <option value="Constrained-Cache">Constrained-Cache (Cap Ratio ≤ 0.15x)</option>
      <option value="Abundant-Cache">Abundant-Cache (Cap Ratio ≥ 1.0x)</option>
      <option value="High-Contention">High-Contention (High Traffic/Cache)</option>
      <option value="Skewed-Distribution">Skewed-Distribution (Zipf / Power-Law)</option>
      <option value="Small-Scale">Small-Scale (Prototypes)</option>
      <option value="Medium-Scale">Medium-Scale (Production Standard)</option>
      <option value="Massive-Scale">Massive-Scale (Enterprise CDN)</option>
    </select>
  </div>
  <div class="control-group">
    <label for="blockFilter">Ablation Component</label>
    <select id="blockFilter" onchange="filterDashboard()">
      <option value="ALL">All Strategy Blocks</option>
      <option value="init">Initialization Dynamics (Seed vs Diversity)</option>
      <option value="cx">Crossover Mechanics & Severity</option>
      <option value="mut">Mutation Mechanics & Disruption</option>
      <option value="ls">Local Search & Cap Exhaustion</option>
      <option value="dec">Decoding (Utilization, Speed, Slack)</option>
      <option value="global">Throughput & Convergence Profiles</option>
    </select>
  </div>
</div>
<div class="summary-bar" id="statusCounter">Displaying all telemetry plots</div>
<div class="grid" id="plotGrid">
"""
    for p in plot_meta:
        card = f"""  <div class="card" data-tag="{p['tag']}" data-block="{p['block']}">
    <div class="card-header">
      <h3>{p['title']}</h3>
      <div>
        <span class="badge">{p['tag']}</span>
        <span class="badge badge-block">{p['block'].upper()}</span>
      </div>
    </div>
    <div class="card-body">
      <img src="plots/{p['file']}" loading="lazy" alt="{p['title']}" />
      <p class="card-desc">{p['desc']}</p>
    </div>
  </div>\n"""
        html_template += card

    html_template += """</div>
<script>
function filterDashboard() {
  const selectedTag = document.getElementById('tagFilter').value;
  const selectedBlk = document.getElementById('blockFilter').value;
  const cards = document.querySelectorAll('.card');
  let visibleCount = 0;

  cards.forEach(card => {
    const cardTag = card.getAttribute('data-tag');
    const cardBlk = card.getAttribute('data-block');
    const matchTag = (selectedTag === 'ALL' || cardTag === selectedTag);
    const matchBlk = (selectedBlk === 'ALL' || cardBlk === selectedBlk);

    if (matchTag && matchBlk) {
      card.style.display = 'flex';
      visibleCount++;
    } else {
      card.style.display = 'none';
    }
  });

  const counter = document.getElementById('statusCounter');
  counter.textContent = `Displaying ${visibleCount} plot(s) for Tag [${selectedTag}] and Component [${selectedBlk.toUpperCase()}]`;
}
</script>
</body>
</html>"""

    with open(HTML_FILE, "w") as f:
        f.write(html_template)

def main():
    records = load_data()
    if not records:
        print("No valid JSON logs found in candidate directories.")
        return
    norm_records = normalize_records(records)
    meta = generate_plots(norm_records)
    generate_html(meta)
    print(f"Aggregated reports and interactive dashboard written to {HTML_FILE}")

if __name__ == "__main__":
    main()