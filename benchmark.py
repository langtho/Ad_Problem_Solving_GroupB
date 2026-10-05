import subprocess
import os
import glob
import pathlib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BEST_KNOWN_VALUES = {
    #"instances/instances/kittens.in": 1022556,
    #"instances/instances/me_at_the_zoo.in": 507906,
    #"instances/instances/trending_today.in": 499991,
    #"instances/instances/videos_worth_spreading.in": 611597,
    #"instances/instances/custom_dejavu42.in": 196131,
    #"instances/instances/custom_universallambda42.in": 1342673,
    #"instances/instances/custom_universallambda42_asymmetric.in": 975514,
    "instances/instances/instance1.in": 1498093,
    "instances/instances/instance2.in": 1971053,
}

EXECUTABLE = "./cmake-build-debug/solver"
OUTPUT_DIR = "benchmark_results"


def parse_solution_lines(solution_source):
    """Normalize solution input from raw stdout string, a file path, or line list."""
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
    """Extract cache allocations by finding the nbr_caches line printed by sol.cpp."""
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


def get_placements(solution_source, c_count=None):
    lines = parse_solution_lines(solution_source)
    placements = set()
    if not lines:
        return placements

    start_idx = 1 if lines[0].isdigit() else 0
    for line in lines[start_idx:]:
        parts = [int(p) for p in line.split() if p.isdigit()]
        if parts:
            c_id = parts[0]
            for v_id in parts[1:]:
                placements.add((c_id, v_id))
    return placements


def calc_similarity(sol1, sol2):
    s1 = get_placements(sol1)
    s2 = get_placements(sol2)
    union_size = len(s1 | s2)
    if union_size == 0:
        return 100.0
    return (len(s1 & s2) / union_size) * 100.0


def extract_progress_data(instance_name, output_dir):
    """Locate benchmark JSON history in output directory or current workspace."""
    search_dirs = [output_dir, ".", "./cmake-build-debug"]
    target_json = None

    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        exact_match = os.path.join(directory, f"{instance_name}.json")
        if os.path.exists(exact_match):
            target_json = exact_match
            break

        for path in glob.glob(os.path.join(directory, "*.json")):
            stem = pathlib.Path(path).stem
            if instance_name in stem:
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


def evaluate_and_plot(input_file, solution_data, out_file=None, instance_name=None):
    try:
        with open(input_file, "r") as f:
            in_data = f.read().split()
    except FileNotFoundError:
        print(f"Error: input file '{input_file}' not found")
        return

    if not in_data:
        print("Error: empty input file")
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
    if instance_name:
        evals, percentages = extract_progress_data(instance_name, OUTPUT_DIR)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=(
            f"Benchmark Progress: Targets Cleared vs Evaluations ({instance_name})"
            if instance_name
            else "Benchmark Progress",
            "Space Used per Cache (%)",
            "Time Saved per Cache",
            "Cache Presence per Video",
            "Data Volume per Endpoint (MB)",
        ),
    )

    if evals and percentages:
        fig.add_trace(
            go.Scatter(
                x=evals,
                y=percentages,
                mode="lines+markers",
                name="Progress",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
                hovertemplate="Evals: %{x}<br>Targets Cleared: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No progress JSON data found for this instance",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=list(range(c_count)),
            y=space_pct,
            marker_color="steelblue",
            hovertemplate="Cache %{x}<br>Space: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(range(c_count)),
            y=time_saved,
            marker_color="seagreen",
            hovertemplate="Cache %{x}<br>Time: %{y}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=list(range(v_count)),
            y=video_presence,
            marker_color="firebrick",
            hovertemplate="Video %{x}<br>Caches: %{y}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    if sum(data_dc) > 0:
        fig.add_trace(
            go.Bar(
                x=list(range(e_count)),
                y=data_dc,
                marker_color="black",
                hovertemplate="Endpoint %{x}<br>Source: DC<br>Volume: %{y} MB<extra></extra>",
            ),
            row=3,
            col=2,
        )

    for c in range(c_count):
        if sum(data_cache[c]) > 0:
            fig.add_trace(
                go.Bar(
                    x=list(range(e_count)),
                    y=data_cache[c],
                    hovertemplate=f"Endpoint %{{x}}<br>Source: Cache {c}<br>Volume: %{{y}} MB<extra></extra>",
                ),
                row=3,
                col=2,
            )

    fig.update_layout(
        barmode="stack",
        height=1100,
        showlegend=False,
        hovermode="closest",
    )

    fig.update_xaxes(title_text="Evaluations", row=1, col=1)
    fig.update_yaxes(title_text="% Cleared", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Cache ID", row=2, col=1)
    fig.update_xaxes(title_text="Cache ID", row=2, col=2)
    fig.update_xaxes(title_text="Video ID", row=3, col=1)
    fig.update_xaxes(title_text="Endpoint ID", row=3, col=2)

    target_file = out_file if out_file is not None else "dashboard.html"
    fig.write_html(target_file)
    print(f"Interactive dashboard saved to {target_file}")


def main():
    if not os.path.exists(EXECUTABLE):
        print(f"Error: Executable '{EXECUTABLE}' not found. Please build the project first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for instance_path, best_known in BEST_KNOWN_VALUES.items():
        if not os.path.exists(instance_path):
            print(f"Skipping {instance_path} (file not found)")
            continue

        instance_name = pathlib.Path(instance_path).stem
        print(f"Running benchmark for: {instance_name} (Best Known: {best_known})")

        try:
            with open(instance_path, "r") as infile:
                result = subprocess.run(
                    [EXECUTABLE, str(best_known), instance_name],
                    stdin=infile,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1200,
                )

                if result.returncode == 0:
                    print(f"Successfully finished {instance_name}")

                    # Archive stdout to text file for reference
                    output_file_path = os.path.join(OUTPUT_DIR, f"output_{instance_name}.txt")
                    with open(output_file_path, "w") as outfile:
                        outfile.write(result.stdout)

                    # Pass in-memory stdout directly to the evaluation & HTML generator
                    dashboard_file = os.path.join(OUTPUT_DIR, f"dashboard_{instance_name}.html")
                    evaluate_and_plot(instance_path, result.stdout, dashboard_file, instance_name)
                else:
                    print(f"Error running {instance_name}:")
                    print(result.stderr)

        except subprocess.TimeoutExpired:
            print(f"Timeout expired for {instance_name}")
        except Exception as e:
            print(f"An error occurred while running {instance_name}: {e}")


if __name__ == "__main__":
    main()