import subprocess
import os
import glob
import pathlib
import json
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE = os.path.join(SCRIPT_DIR, "cmake-build-release", "solver")
# Change BASE_DIR to use os.path.join with SCRIPT_DIR
BASE_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")
LOGS_DIR = os.path.join(BASE_DIR, "raw_logs")
SWEEPS_DIR = os.path.join(BASE_DIR, "sweeps")

SEEDS = [42, 137, 999]
MAX_TIME_SECONDS = 60.0
SUBPROCESS_TIMEOUT = 75.0

GRID = {
    "pop_size": [-1.0, 40.0],
    "cx_frac": [-1.0, 0.5],
    "mut_rate": [-1.0, 0.25],
    "k_mult": [-1.0, 1.0],
    "slack": [-1.0, 1.05],
    "elite": [-1, 4],
    "noise": [-1.0, 0.75],
    "stall": [-1, 500]
}

# Helper to match C++ ostringstream float formatting
def fmt(val):
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)

def execute_solver(cfg, instance_path, instance_name, seed):
    # Use fmt() so Python strings exactly match the C++ output filenames
    run_name = (f"{instance_name}_pop{fmt(cfg['pop_size'])}_cx{fmt(cfg['cx_frac'])}"
                f"_mut{fmt(cfg['mut_rate'])}_k{fmt(cfg['k_mult'])}_slk{fmt(cfg['slack'])}"
                f"_eli{cfg['elite']}_nsc{fmt(cfg['noise'])}_stl{cfg['stall']}_seed{seed}")

    out_txt = os.path.join(LOGS_DIR, f"out_{run_name}.txt")
    out_json = os.path.join(LOGS_DIR, f"fit_{run_name}.json")

    cmd = [
        EXECUTABLE, "0", instance_name,
        str(cfg["pop_size"]), str(cfg["cx_frac"]),
        str(cfg["mut_rate"]), str(cfg["k_mult"]),
        str(cfg["slack"]), str(cfg["elite"]),
        str(cfg["noise"]), str(cfg["stall"]),
        str(seed), str(MAX_TIME_SECONDS)
    ]

    config_key = (f"pop{fmt(cfg['pop_size'])}_cx{fmt(cfg['cx_frac'])}_mut{fmt(cfg['mut_rate'])}"
                  f"_k{fmt(cfg['k_mult'])}_slk{fmt(cfg['slack'])}_eli{cfg['elite']}"
                  f"_nsc{fmt(cfg['noise'])}_stl{cfg['stall']}")

    fallback_data = {
        "instance": instance_name,
        "config_key": config_key,
        "best_fitness_achieved": 0.0
    }

    try:
        with open(instance_path, "r") as infile:
            res = subprocess.run(
                cmd,
                stdin=infile,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                cwd=os.path.dirname(EXECUTABLE)
            )

            if res.returncode == 0:
                with open(out_txt, "w") as f:
                    f.write(res.stdout)
    except Exception:
        return fallback_data

    if os.path.exists(out_json):
        with open(out_json, "r") as jf:
            data = json.load(jf)
            data["config_key"] = fallback_data["config_key"]
            return data

    return fallback_data

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SWEEPS_DIR, exist_ok=True)

    # NO leading slashes here either
    instances_dir = os.path.join(SCRIPT_DIR, "instances", "promo")
    all_instances = glob.glob(os.path.join(instances_dir, "*.in"))

    if not all_instances:
        print(f"ERROR: No .in files found in {instances_dir}")
        return

    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = []
    for instance_path in all_instances:
        instance_name = pathlib.Path(instance_path).stem
        for cfg in combinations:
            for seed in SEEDS:
                tasks.append((cfg, instance_path, instance_name, seed))

    raw_results = []
    workers = max(1, os.cpu_count() - 2)
    print(f"Launching Sweep 3 with {len(tasks)} tasks across {workers} cores...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute_solver, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                raw_results.append(res)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(tasks)} tasks.")

    csv_file = os.path.join(SWEEPS_DIR, "round3_hyperparameter_search.csv")
    if raw_results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(raw_results)
        print(f"Sweep complete! Results saved to {csv_file}")

if __name__ == "__main__":
    main()