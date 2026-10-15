import subprocess
import os
import glob
import pathlib
import json
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

EXECUTABLE = os.path.abspath("./cmake-build-release/solver")
BASE_DIR = os.path.abspath("benchmark_results")
LOGS_DIR = os.path.join(BASE_DIR, "raw_logs")
SWEEPS_DIR = os.path.join(BASE_DIR, "sweeps")

SEEDS = [42, 137, 999, 2024, 7]
MAX_TIME_SECONDS = 30.0
SUBPROCESS_TIMEOUT = 45.0

GRID = {
    "init": [3, 5],
    "cx": [1, 6],
    "mut": [3, 4, 5],
    "ls": [0, 2],
    "dec": [0, 1]
}

def execute_solver(cfg, instance_path, instance_name, seed):
    run_name = f"{instance_name}_i{cfg['init']}_c{cfg['cx']}_m{cfg['mut']}_l{cfg['ls']}_d{cfg['dec']}_s{seed}"
    out_txt = os.path.join(LOGS_DIR, f"out_{run_name}.txt")
    out_json = os.path.join(LOGS_DIR, f"fit_{run_name}.json")

    cmd = [
        EXECUTABLE, "0", instance_name,
        str(cfg["init"]), str(cfg["cx"]), str(cfg["mut"]),
        str(cfg["ls"]), str(cfg["dec"]), str(seed), str(MAX_TIME_SECONDS)
    ]

    fallback_data = {
        "instance": instance_name,
        "config_key": f"i{cfg['init']}_c{cfg['cx']}_m{cfg['mut']}_l{cfg['ls']}_d{cfg['dec']}",
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

    instances_dir = os.path.abspath("instances/promo")
    all_instances = glob.glob(os.path.join(instances_dir, "*.in"))

    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = []
    for instance_path in all_instances:
        instance_name = pathlib.Path(instance_path).stem
        for cfg in combinations:
            for seed in SEEDS:
                tasks.append((cfg, instance_path, instance_name, seed))

    raw_results = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
        futures = {executor.submit(execute_solver, *task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                raw_results.append(res)

    csv_file = os.path.join(SWEEPS_DIR, "round2_grid_search.csv")
    if raw_results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(raw_results)

if __name__ == "__main__":
    main()