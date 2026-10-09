import subprocess
import os
import glob
import pathlib
import json
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed

BEST_KNOWN_VALUES = {
    "instances/instances/custom_medium_flat_dense_dejavu_13_seed54.in": 1022556,
    "instances/instances/kittens.in": 1022556,
    "instances/instances/me_at_the_zoo.in": 507906,
    "instances/instances/trending_today.in": 499991,
    "instances/instances/videos_worth_spreading.in": 611597,
    "instances/instances/custom_dejavu42.in": 196131,
    "instances/instances/custom_universallambda42.in": 1342673,
    "instances/instances/custom_universallambda42_asymmetric.in": 975514,
    "instances/instances/instance1.in": 1498093,
    "instances/instances/instance2.in": 1971053,
}

EXECUTABLE = "./cmake-build-debug/solver"
OUTPUT_DIR = "benchmark_results"
SUMMARY_CSV = "full_benchmark_summary.csv"

INIT_MAP = {
    0: "BASELINE_EXACT_NOISY",
    1: "PURE_RANDOM",
    2: "NOISY_PRIORITY_ADD",
    3: "DENSITY_REQUEST_SORT",
    4: "ACTIVE_FRONT_SHUFFLE",
    5: "HYBRID_COMPOSITE"
}

CX_MAP = {
    0: "BASELINE_OX1",
    1: "ACTIVE_PREFIX_OX1",
    2: "POSITION_BASED_POS",
    3: "PARTIALLY_MAPPED_PMX",
    4: "CYCLE_CROSSOVER_CX"
}

MUT_MAP = {
    0: "BASELINE_SWAP",
    1: "ACTIVE_PREFIX_SWAP",
    2: "ACTIVE_PREFIX_INVERSION",
    3: "ACTIVE_PREFIX_INSERTION",
    4: "BOUNDARY_SWAP"
}

LS_MAP = {
    0: "NONE",
    1: "FIRST_IMPROVEMENT_SWAP",
    2: "BEST_IMPROVEMENT_ACTIVE_INSERT",
    3: "ACTIVE_WINDOW_2OPT_SCAN",
    4: "REQUEST_FREQUENCY_GREEDY"
}

DEC_MAP = {
    0: "STATIC_GREEDY",
    1: "SOFT_SLACK_REPAIRED",
    2: "DEMAND_CACHE_CENTRIC",
    3: "FIXED_THRESHOLD_PROFILE",
    4: "CO_EVOLVED_THRESHOLDS"
}

DEFAULT_CONFIG = {
    "init": 2,  # NOISY_PRIORITY_ADD
    "cx": 1,    # ACTIVE_PREFIX_OX1
    "mut": 2,   # ACTIVE_PREFIX_INVERSION
    "ls": 1,    # FIRST_IMPROVEMENT_SWAP
    "dec": 0    # STATIC_GREEDY
}

def generate_ablation_experiments():
    """Generates strategy combinations sweeping each block while locking all other blocks."""
    experiments = []

    # Block 1: Initializations
    for init_id in INIT_MAP.keys():
        cfg = DEFAULT_CONFIG.copy()
        cfg["init"] = init_id
        cfg["ablation_block"] = "Block1_Init"
        experiments.append(cfg)

    # Block 2: Crossover Strategies
    for cx_id in CX_MAP.keys():
        cfg = DEFAULT_CONFIG.copy()
        cfg["cx"] = cx_id
        cfg["ablation_block"] = "Block2_Crossover"
        experiments.append(cfg)

    # Block 3: Mutation Strategies
    for mut_id in MUT_MAP.keys():
        cfg = DEFAULT_CONFIG.copy()
        cfg["mut"] = mut_id
        cfg["ablation_block"] = "Block3_Mutation"
        experiments.append(cfg)

    # Block 4: Local Search Strategies
    for ls_id in LS_MAP.keys():
        cfg = DEFAULT_CONFIG.copy()
        cfg["ls"] = ls_id
        cfg["ablation_block"] = "Block4_LocalSearch"
        experiments.append(cfg)

    # Block 5: Decoding Strategies
    for dec_id in DEC_MAP.keys():
        cfg = DEFAULT_CONFIG.copy()
        cfg["dec"] = dec_id
        cfg["ablation_block"] = "Block5_Decoder"
        experiments.append(cfg)

    # Deduplicate configurations
    seen = set()
    unique_experiments = []
    for exp in experiments:
        key = (exp["init"], exp["cx"], exp["mut"], exp["ls"], exp["dec"])
        if key not in seen:
            seen.add(key)
            unique_experiments.append(exp)

    return unique_experiments


def parse_solution_lines(solution_source):
    if isinstance(solution_source, list):
        return [line.strip() for line in solution_source if line.strip()]
    if isinstance(solution_source, str):
        if "\n" in solution_source or (not os.path.exists(solution_source) and len(solution_source.split()) > 1):
            return [line.strip() for line in solution_source.strip().split("\n") if line.strip()]
        if os.path.exists(solution_source):
            with open(solution_source, "r") as f:
                return [line.strip() for line in f.read().strip().split("\n") if line.strip()]
    return []


def extract_cache_contents(sol_lines, c_count):
    cache_contents = {i: set() for i in range(c_count)}
    if not sol_lines:
        return cache_contents

    start_idx = 0
    for idx, line in enumerate(sol_lines):
        if line.isdigit() and int(line) == c_count:
            start_idx = idx + 1
            break
        elif line.isdigit() and idx == 0:
            start_idx = 1
            break

    for i in range(start_idx, min(start_idx + c_count, len(sol_lines))):
        parts = [int(p) for p in sol_lines[i].split() if p.isdigit()]
        if parts:
            c_id = parts[0]
            if c_id in cache_contents:
                cache_contents[c_id] = set(parts[1:])

    return cache_contents


def extract_progress_data(run_name, output_dir):
    search_dirs = [output_dir, ".", "./cmake-build-debug"]
    target_json = None

    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        exact_match = os.path.join(directory, f"fitness_plot_{run_name}.json")
        if os.path.exists(exact_match):
            target_json = exact_match
            break

        for path in glob.glob(os.path.join(directory, "*.json")):
            stem = pathlib.Path(path).stem
            if run_name in stem:
                target_json = path
                break
        if target_json:
            break

    if not target_json:
        return [], []

    try:
        with open(target_json, "r") as f:
            data = json.load(f)

        targets_hit = data.get("targets_hit", [])
        if not targets_hit:
            return [], []

        history = targets_hit[0] if isinstance(targets_hit[0], list) else targets_hit
        evals = [entry["eval"] for entry in history if isinstance(entry, dict) and "eval" in entry]
        percentages = [(i + 1) / 50.0 * 100.0 for i in range(len(evals))]
        return evals, percentages
    except Exception:
        return [], []


def evaluate_and_plot(input_file, solution_data, out_file=None, run_name=None):
    try:
        with open(input_file, "r") as f:
            in_data = f.read().split()
    except FileNotFoundError:
        return

    if not in_data:
        return

    it = iter(in_data)
    v_count = int(next(it))
    e_count = int(next(it))
    r_count = int(next(it))
    c_count = int(next(it))
    cache_capacity = int(next(it))

    video_sizes = [int(next(it)) for _ in range(v_count)]

    endpoints = {}
    for e_id in range(e_count):
        l_d = int(next(it))
        k = int(next(it))
        connections = {}
        for _ in range(k):
            c_id = int(next(it))
            l_c = int(next(it))
            connections[c_id] = l_c
        endpoints[e_id] = {"L_D": l_d, "connections": connections}

    requests = []
    for _ in range(r_count):
        v = int(next(it))
        e = int(next(it))
        n = int(next(it))
        requests.append((v, e, n))

    sol_lines = parse_solution_lines(solution_data)
    cache_contents = extract_cache_contents(sol_lines, c_count)

    space_pct = [0.0] * c_count
    time_saved = [0] * c_count
    video_presence = [0] * v_count
    data_dc = [0] * e_count
    data_cache = {c: [0] * e_count for c in range(c_count)}

    for c in range(c_count):
        used = sum(video_sizes[v] for v in cache_contents[c] if v < len(video_sizes))
        space_pct[c] = (used / cache_capacity) * 100 if cache_capacity > 0 else 0.0

    for c in range(c_count):
        for v in cache_contents[c]:
            if v < len(video_presence):
                video_presence[v] += 1

    for v, e, n in requests:
        ep = endpoints[e]
        l_d = ep["L_D"]
        min_l = l_d
        best_c = -1

        for c, l_c in ep["connections"].items():
            if v in cache_contents[c] and l_c < min_l:
                min_l = l_c
                best_c = c

        data_vol = video_sizes[v] * n
        if best_c != -1:
            time_saved[best_c] += n * (l_d - min_l)
            data_cache[best_c][e] += data_vol
        else:
            data_dc[e] += data_vol

    evals, percentages = [], []
    if run_name:
        evals, percentages = extract_progress_data(run_name, OUTPUT_DIR)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
        subplot_titles=(
            f"Benchmark Progress: Targets Cleared vs Evaluations ({run_name})" if run_name else "Benchmark Progress",
            "Space Used per Cache (%)",
            "Time Saved per Cache",
            "Cache Presence per Video",
            "Data Volume per Endpoint (MB)",
        ),
    )

    if evals and percentages:
        fig.add_trace(
            go.Scatter(x=evals, y=percentages, mode="lines+markers", name="Progress",
                       line=dict(color="#1f77b4", width=2), marker=dict(size=6)),
            row=1, col=1
        )

    fig.add_trace(go.Bar(x=list(range(c_count)), y=space_pct, marker_color="steelblue"), row=2, col=1)
    fig.add_trace(go.Bar(x=list(range(c_count)), y=time_saved, marker_color="seagreen"), row=2, col=2)
    fig.add_trace(go.Bar(x=list(range(v_count)), y=video_presence, marker_color="firebrick"), row=3, col=1)

    if sum(data_dc) > 0:
        fig.add_trace(go.Bar(x=list(range(e_count)), y=data_dc, marker_color="black"), row=3, col=2)

    for c in range(c_count):
        if sum(data_cache[c]) > 0:
            fig.add_trace(go.Bar(x=list(range(e_count)), y=data_cache[c]), row=3, col=2)

    fig.update_layout(barmode="stack", height=1100, showlegend=False)
    target_file = out_file if out_file is not None else "dashboard.html"
    fig.write_html(target_file)

import subprocess
import os
import pathlib
import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

def execute_solver(exp, instance_path, best_known, instance_name):
    run_name = f"{instance_name}_init{exp['init']}_cx{exp['cx']}_mut{exp['mut']}_ls{exp['ls']}_dec{exp['dec']}"
    dashboard_file = os.path.join(OUTPUT_DIR, f"dashboard_{run_name}.html")
    out_txt = os.path.join(OUTPUT_DIR, f"output_{run_name}.txt")
    out_json = os.path.join(OUTPUT_DIR, f"fitness_plot_{run_name}.json")

    if os.path.exists(dashboard_file) and os.path.exists(out_json):
        return out_json, None # Data already exists

    cmd = [
        EXECUTABLE, str(best_known), instance_name,
        str(exp["init"]), str(exp["cx"]), str(exp["mut"]),
        str(exp["ls"]), str(exp["dec"])
    ]

    try:
        with open(instance_path, "r") as infile:
            result = subprocess.run(cmd, stdin=infile, capture_output=True, text=True, timeout=240)
            if result.returncode == 0:
                with open(out_txt, "w") as f:
                    f.write(result.stdout)
                # Return the rendering payload rather than rendering inside the worker
                return out_json, (instance_path, result.stdout, dashboard_file, run_name)
    except subprocess.TimeoutExpired:
        pass

    return None, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiments = generate_ablation_experiments()
    summary_records = []
    plot_queue = []

    print(f"Dispatching {len(experiments) * len(BEST_KNOWN_VALUES)} jobs across {os.cpu_count()} cores...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for instance_path, best_known in BEST_KNOWN_VALUES.items():
            if not os.path.exists(instance_path): continue
            instance_name = pathlib.Path(instance_path).stem
            for exp in experiments:
                futures.append(executor.submit(execute_solver, exp, instance_path, best_known, instance_name))

        for future in as_completed(futures):
            json_path, plot_payload = future.result()
            if json_path and os.path.exists(json_path):
                with open(json_path, "r") as jf:
                    summary_records.append(json.load(jf))
            if plot_payload:
                plot_queue.append(plot_payload)

    # Process Plotly graphs sequentially at the end to prevent IPC memory locking
    print(f"Generating {len(plot_queue)} HTML dashboards...")
    for payload in plot_queue:
        evaluate_and_plot(*payload)

    if summary_records:
        with open(SUMMARY_CSV, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(summary_records[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_records)


if __name__ == "__main__":
    main()