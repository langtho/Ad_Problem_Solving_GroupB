"""Runs the benchmark of block 1 (initialisation), block 5 (decoder) and the threshold study.

    py experiments\\run_blocks.py                          # blocks 1 + 5, essential instances, 5 seeds
    py experiments\\run_blocks.py --study thresholds       # sensitivity of decoder 3 thresholds
    py experiments\\run_blocks.py --seeds 42 --instances me_at_the_zoo

Configurations (init cx mut ls dec [tau_density tau_reserve]):
    blocks:     block 1 = init 0..5, the rest 0 ; block 5 = dec 0..4, the rest 0
                (0 0 0 0 0 is the shared baseline, run once)
    thresholds: decoder 3 with one threshold varied at a time, the other at 0

Each run gets its own folder <runs>/<instance>/<config>_s<seed>/ (the solver writes its JSON
in the current folder, so parallel runs do not overwrite each other). One line per run is
appended to the CSV: solver JSON + score.py + solution_stats.py (+ thresholds of decoder 4).
Runs already in the CSV are skipped, so the script can be restarted at any time.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import score  # noqa: E402
import solution_stats  # noqa: E402



def promo(name):
    """Promo instances live in instances/promo/ (shared repo); fall back to the old local folder."""
    path = f"instances/promo/{name}.in"
    return path if (ROOT / path).exists() else f"instances/promo_instances/{name}.in"


# name -> (path, best known value or None). Best known values: BEST_KNOWN_VALUES of benchmark.py.
INSTANCES = OrderedDict([
    ("me_at_the_zoo", ("instances/instances/me_at_the_zoo.in", 507906)),
    ("custom_small_flat_sparse_lambda_02_seed43", ("instances/instances/custom_small_flat_sparse_lambda_02_seed43.in", None)),
    ("custom_small_skew_dense_lambda_08_seed49", ("instances/instances/custom_small_skew_dense_lambda_08_seed49.in", None)),
    ("custom_small_flat_dense_dejavu_03_seed44", ("instances/instances/custom_small_flat_dense_dejavu_03_seed44.in", None)),
    ("custom_small_skew_sparse_dejavu_05_seed46", ("instances/instances/custom_small_skew_sparse_dejavu_05_seed46.in", None)),
    ("custom_medium_skew_dense_lambda_18_seed59", ("instances/instances/custom_medium_skew_dense_lambda_18_seed59.in", None)),
    ("instance1", (promo("instance1"), 1498093)),
    ("instance2", (promo("instance2"), 1971053)),
    ("instance3", (promo("instance3"), None)),
    # large instances, not in the essential set (too long for the deadline)
    ("test", ("instances/instances/test.in", None)),
    ("instance4", (promo("instance4"), None)),
    ("trending_today", ("instances/instances/trending_today.in", 499991)),
    ("videos_worth_spreading", ("instances/instances/videos_worth_spreading.in", 611597)),
    ("custom_universallambda42", (promo("custom_universallambda42"), 1342673)),
    ("custom_dejavu42", (promo("custom_dejavu42"), 196131)),
    ("custom_universallambda42_asymmetric", (promo("custom_universallambda42_asymmetric"), 975514)),
    ("kittens", ("instances/instances/kittens.in", 1022556)),
])
# Small and representative: 1 official, 4 small customs (lambda/dejavu x flat/skew x sparse/dense),
# 1 medium custom, 3 promo instances (instance3 has huge videos: scale test for the thresholds)
ESSENTIAL = list(INSTANCES)[:9]
THRESHOLD_INSTANCES = ["me_at_the_zoo", "custom_small_skew_dense_lambda_08_seed49", "instance1", "instance3"]


def block_configs():
    configs = OrderedDict()  # (init, cx, mut, ls, dec, tau_density, tau_reserve) -> block
    configs[(0, 0, 0, 0, 0, None, None)] = "baseline"
    for i in range(1, 6):
        configs[(i, 0, 0, 0, 0, None, None)] = "init"
    for d in range(1, 5):
        configs[(0, 0, 0, 0, d, None, None)] = "dec"
    return configs


def threshold_configs():
    configs = OrderedDict()
    for tau_density in (0, 0.01, 0.15, 1, 10, 100, 1000):
        configs[(0, 0, 0, 0, 3, tau_density, 0)] = "thr_density"
    for tau_reserve in (0.05, 0.1, 0.2, 0.3):
        configs[(0, 0, 0, 0, 3, 0, tau_reserve)] = "thr_reserve"
    return configs


JSON_FIELDS = [
    "active_prefix_k", "gen_0_best_score", "gen_0_mean_score", "gen_0_diversity_d0",
    "crossover_success_rate", "crossover_neutral_rate", "mutation_success_rate", "mutation_neutral_rate",
    "avg_eval_microseconds", "cache_utilization_ratio", "total_evaluations", "total_generations",
    "best_fitness_achieved", "total_time_seconds",
]
STAT_FIELDS = [
    "score_check", "upper_bound_score", "pct_upper_bound", "fill_mean", "fill_min", "fill_max",
    "copies", "distinct_videos_placed", "duplication", "nb_candidates", "pct_candidates_placed",
    "pct_requests_served", "mean_gain_ms", "useless_copies", "pct_useless_copies",
    "pct_capacity_useless", "pct_top10_missing", "pct_potential_missing",
]
COLUMNS = (["run_id", "instance", "size", "demand", "popularity", "density", "best_known",
            "block", "init", "cx", "mut", "ls", "dec", "tau_density", "tau_reserve", "seed", "status",
            "wall_seconds", "score", "pct_best_known", "gen_0_pct_best_known",
            "thr_gen0_density", "thr_gen0_reserve", "thr_final_density", "thr_final_reserve"]
           + JSON_FIELDS + STAT_FIELDS)


def family(name):
    """Family attributes read from the instance name."""
    if name in ("me_at_the_zoo", "trending_today", "videos_worth_spreading", "kittens", "test"):
        size = "official"
    elif name.startswith("instance"):
        size = "promo"
    elif name.startswith("custom_small"):
        size = "small"
    elif name.startswith("custom_medium"):
        size = "medium"
    else:
        size = "custom"
    demand = "dejavu" if "dejavu" in name else "lambda" if "lambda" in name else ""
    popularity = next((p for p in ("flat", "skew", "mixed") if f"_{p}_" in name), "")
    density = next((d for d in ("sparse", "dense") if f"_{d}_" in name), "")
    return size, demand, popularity, density


def config_label(config):
    base = "-".join(map(str, config[:5]))
    return base if config[5] is None else f"{base}-t{config[5]}-{config[6]}"


def run_id(name, config, seed):
    return f"{name}|{config_label(config)}|{seed}"


timed_out = set()  # (instance, config) that already hit the timeout: other seeds are skipped


def run_solver(exe, runs_dir, name, config, seed, timeout):
    """Runs one configuration; returns (status, wall seconds, run folder)."""
    path, best_known = INSTANCES[name]
    folder = runs_dir / name / f"{config_label(config)}_s{seed}"
    if (name, config) in timed_out:
        return "skipped_after_timeout", 0.0, folder
    folder.mkdir(parents=True, exist_ok=True)
    args = [str(exe), str(best_known or 0), "run", *map(str, config[:5]), str(seed)]
    if config[5] is not None:
        args += [str(config[5]), str(config[6])]
    start = time.perf_counter()
    try:
        with open(ROOT / path, "rb") as stdin, open(folder / "sol.out", "wb") as stdout, \
                open(folder / "stderr.txt", "wb") as stderr:
            result = subprocess.run(args, stdin=stdin, stdout=stdout, stderr=stderr,
                                    cwd=folder, timeout=timeout)
        status = "ok" if result.returncode == 0 else f"exit_{result.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        timed_out.add((name, config))
    return status, time.perf_counter() - start, folder


class InstanceCache:
    """Keeps the last parsed instances (parsing the large ones takes a few seconds)."""

    def __init__(self, size=3):
        self.size = size
        self.items = OrderedDict()

    def get(self, name):
        if name not in self.items:
            instance = score.read_instance(ROOT / INSTANCES[name][0])
            self.items[name] = solution_stats.prepare(instance)
            if len(self.items) > self.size:
                self.items.popitem(last=False)
        self.items.move_to_end(name)
        return self.items[name]


def make_row(name, config, block, seed, status, wall, folder, cache):
    best_known = INSTANCES[name][1]
    size, demand, popularity, density = family(name)
    row = {
        "run_id": run_id(name, config, seed), "instance": name, "size": size, "demand": demand,
        "popularity": popularity, "density": density, "best_known": best_known or "",
        "block": block, "init": config[0], "cx": config[1], "mut": config[2], "ls": config[3],
        "dec": config[4], "tau_density": "" if config[5] is None else config[5],
        "tau_reserve": "" if config[6] is None else config[6], "seed": seed, "status": status,
        "wall_seconds": round(wall, 3),
    }
    if status != "ok":
        return row

    json_files = list((folder / "benchmark_results").glob("fitness_plot_*.json"))
    if json_files:
        data = json.loads(json_files[0].read_text())
        for field in JSON_FIELDS:
            row[field] = data.get(field, "")
        if best_known:
            row["gen_0_pct_best_known"] = round(100.0 * data["gen_0_best_score"] / best_known, 3)
    for line in (folder / "stderr.txt").read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in ("thresholds_gen0", "thresholds_final"):
            key = "gen0" if parts[0] == "thresholds_gen0" else "final"
            row[f"thr_{key}_density"], row[f"thr_{key}_reserve"] = parts[1], parts[2]

    prepared = cache.get(name)
    caches = score.read_solution((folder / "sol.out").read_text())
    errors = score.check_solution(prepared["instance"], caches)
    if errors:
        row["status"] = "invalid"
        return row
    official = score.score_solution(prepared["instance"], caches)
    row["score"] = official
    if best_known:
        row["pct_best_known"] = round(100.0 * official / best_known, 3)
    for key, value in solution_stats.compute_stats(prepared, caches).items():
        row[key] = round(value, 4) if isinstance(value, float) else value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["blocks", "thresholds"], default="blocks")
    parser.add_argument("--exe", default=str(HERE / "solver_fast.exe"))
    parser.add_argument("--csv", default=None, help="default: experiments/<study>_results.csv")
    parser.add_argument("--runs-dir", default=str(HERE / "runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--instances", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retry", action="store_true", help="also rerun runs that failed or timed out")
    args = parser.parse_args()

    configs = block_configs() if args.study == "blocks" else threshold_configs()
    instances = args.instances or (ESSENTIAL if args.study == "blocks" else THRESHOLD_INSTANCES)
    seeds = args.seeds or ([42, 1, 2, 3, 4] if args.study == "blocks" else [42, 1, 2])
    exe = Path(args.exe).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    csv_path = Path(args.csv or HERE / f"{args.study}_results.csv")

    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok" or not args.retry:
                    done.add(row["run_id"])

    # Seed by seed (first seed = complete first picture), small instances first
    todo = [(name, config, seed) for seed in seeds for name in instances for config in configs
            if run_id(name, config, seed) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {csv_path.name}), exe = {exe.name}", flush=True)

    new_file = not csv_path.exists()
    cache = InstanceCache()
    with open(csv_path, "a", newline="") as f, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(run_solver, exe, runs_dir, *job, args.timeout): job for job in todo}
        for count, future in enumerate(as_completed(futures), 1):
            name, config, seed = futures[future]
            status, wall, folder = future.result()
            row = make_row(name, config, configs[config], seed, status, wall, folder, cache)
            writer.writerow(row)
            f.flush()
            print(f"[{count}/{len(todo)}] {name} {config_label(config)} seed {seed}: {row['status']} "
                  f"{wall:.1f}s score={row.get('score', '-')}", flush=True)


if __name__ == "__main__":
    main()
