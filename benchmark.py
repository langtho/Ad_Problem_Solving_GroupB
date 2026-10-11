import subprocess
import os
import glob
import pathlib
import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- TOGGLE THIS FLAG ---
TEST_MODE = False  # Set to False for your definitive main benchmark run
# ------------------------

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

EXECUTABLE = "./cmake-build-release/solver"

# --- FOLDER STRUCTURE ---
BASE_DIR = "benchmark_results"
LOGS_DIR = os.path.join(BASE_DIR, "raw_logs")
SWEEPS_DIR = os.path.join(BASE_DIR, "sweeps")

if TEST_MODE:
    SEEDS = [42, 137]
    MAX_TIME_SECONDS = 15.0
    SUBPROCESS_TIMEOUT = 30.0
else:
    SEEDS = [42, 137, 999, 2024, 7]
    MAX_TIME_SECONDS = 45.0
    SUBPROCESS_TIMEOUT = 60.0

STRATEGY_MAPS = {
    "init": {0: "BASELINE", 1: "RANDOM", 2: "NOISY_ADD", 3: "DENSITY", 4: "ACTIVE_FRONT", 5: "HYBRID"},
    "cx": {0: "BASELINE", 1: "ACTIVE_OX1", 2: "POS", 3: "PMX", 4: "CX", 5: "ACTIVE_PPX", 6: "ACTIVE_SET"},
    "mut": {0: "BASELINE", 1: "ACTIVE_SWAP", 2: "ACTIVE_INV", 3: "ACTIVE_INS", 4: "BOUNDARY", 5: "PROMOTE", 6: "GRAPH_REPAIR"},
    "ls": {0: "NONE", 1: "FI_SWAP", 2: "BI_INSERT", 3: "ACTIVE_2OPT", 4: "FREQ_GREEDY"},
    "dec": {0: "STATIC", 1: "SOFT_SLACK", 2: "DEMAND", 3: "FIXED_THRESH", 4: "CO_EVOLVED"}
}

def execute_solver(cfg, instance_path, best_known, instance_name, seed):
    run_name = f"{instance_name}_init{cfg['init']}_cx{cfg['cx']}_mut{cfg['mut']}_ls{cfg['ls']}_dec{cfg['dec']}_seed{seed}"
    out_txt = os.path.join(LOGS_DIR, f"output_{run_name}.txt")
    out_json = os.path.join(LOGS_DIR, f"fitness_plot_{run_name}.json")

    cmd = [
        EXECUTABLE, str(best_known), instance_name,
        str(cfg["init"]), str(cfg["cx"]), str(cfg["mut"]),
        str(cfg["ls"]), str(cfg["dec"]), str(seed), str(MAX_TIME_SECONDS)
    ]

    fallback_data = {
        "instance": instance_name,
        "config_key": f"i{cfg['init']}_c{cfg['cx']}_m{cfg['mut']}_l{cfg['ls']}_d{cfg['dec']}",
        "tested_id": cfg[cfg["testing_block"]],
        "best_fitness_achieved": 0.0,
        "crossover_success_rate": 0.0,
        "cx_deleterious_avg_drop": 0.0,
        "mutation_success_rate": 0.0,
        "local_search_avg_delta": 0.0,
        "avg_eval_microseconds": 0.0,
        "cache_utilization_ratio": 0.0,
        "gen_0_diversity_d0": 0.0
    }

    if not (os.path.exists(out_json) and os.path.exists(out_txt)):
        if not os.path.exists(EXECUTABLE):
            print(f" ERROR: Executable not found at {EXECUTABLE}")
            return None

        try:
            with open(instance_path, "r") as infile:
                res = subprocess.run(cmd, stdin=infile, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT)
                if res.returncode == 0:
                    with open(out_txt, "w") as f: f.write(res.stdout)
                else:
                    print(f"\n⚠️ C++ CRASH CAUGHT: {run_name} failed. Assigning 0.0 fitness.")
                    return fallback_data
        except subprocess.TimeoutExpired:
            print(f"\n⏳ TIMEOUT CAUGHT: {run_name} exceeded {SUBPROCESS_TIMEOUT}s. Assigning 0.0 fitness.")
            return fallback_data
        except FileNotFoundError:
            print(f"\n FILE NOT FOUND: Cannot open {instance_path}")
            return None

    if os.path.exists(out_json):
        with open(out_json, "r") as jf:
            data = json.load(jf)
            data["config_key"] = fallback_data["config_key"]
            data["tested_id"] = fallback_data["tested_id"]
            return data

    return fallback_data

def run_evaluation_sweep(sweep_name, testing_block, base_cfg, available_ids, all_instances):
    # Group results into dedicated round subfolders under sweeps/
    round_dir = os.path.join(SWEEPS_DIR, f"round_{sweep_name}")
    os.makedirs(round_dir, exist_ok=True)

    print(f"\n=== Starting Round: {sweep_name.upper()} (Sweeping {testing_block.upper()}) ===")
    configs_to_test = []

    for test_id in available_ids:
        cfg = base_cfg.copy()
        cfg[testing_block] = test_id
        cfg["testing_block"] = testing_block
        configs_to_test.append(cfg)

    tasks = []
    for instance_path in all_instances:
        bkv = BEST_KNOWN_VALUES.get(instance_path, 0.0)
        instance_name = pathlib.Path(instance_path).stem
        for cfg in configs_to_test:
            for seed in SEEDS:
                tasks.append((cfg, instance_path, bkv, instance_name, seed))

    total_tasks = len(tasks)
    completed = 0
    print(f"📊 Total evaluations queued for {sweep_name}: {total_tasks}")

    raw_results = []

    # Safe parallel execution using ProcessPoolExecutor with Release build
    safe_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=safe_workers) as executor:
        futures = {executor.submit(execute_solver, task[0], task[1], task[2], task[3], task[4]): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            task_info = futures[future]
            cfg = task_info[0]
            tested_val = cfg[cfg["testing_block"]]

            # Print on a new line so you can see the history scroll
            print(f"[{completed}/{total_tasks}] Completed: {task_info[3]} (Seed {task_info[4]} | {cfg['testing_block'].upper()}={tested_val})", flush=True)

            res = future.result()
            if res:
                raw_results.append(res)

    csv_file = os.path.join(round_dir, f"sweep_{sweep_name}_{testing_block}.csv")
    if raw_results:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(raw_results)
        print(f"\n Data saved to {csv_file}")
    else:
        print(f"\n CRITICAL: No data generated for {sweep_name}.")

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SWEEPS_DIR, exist_ok=True)

    if TEST_MODE:
        test_instances = [
            "instances/promo/me_at_the_zoo.in",
            "instances/promo/trending_today.in",
            "instances/promo/videos_worth_spreading.in",
            "instances/promo/custom_dejavu42.in"
        ]
        all_instances = [inst for inst in test_instances if os.path.exists(inst)]
        if not all_instances:
            all_instances = glob.glob("instances/promo/*.in")[:4]
    else:
        all_instances = glob.glob("instances/promo/*.in")

    if not all_instances:
        print(" ERROR: No instance files found! Check your folder paths.")
        return

    base_cfg = {"init": 1, "cx": 0, "mut": 0, "ls": 0, "dec": 0}
    print(f"Base Config Initialized: {base_cfg}")

    run_evaluation_sweep("1_init", "init", base_cfg, list(STRATEGY_MAPS["init"].keys()), all_instances)
    run_evaluation_sweep("2_crossover", "cx", base_cfg, list(STRATEGY_MAPS["cx"].keys()), all_instances)
    run_evaluation_sweep("3_mutation", "mut", base_cfg, list(STRATEGY_MAPS["mut"].keys()), all_instances)
    run_evaluation_sweep("4_localsearch", "ls", base_cfg, list(STRATEGY_MAPS["ls"].keys()), all_instances)
    run_evaluation_sweep("5_decoding", "dec", base_cfg, list(STRATEGY_MAPS["dec"].keys()), all_instances)

    print(f"\n🎉 ALL SWEEPS COMPLETE SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()