import subprocess
import os
import glob
import pathlib
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE = os.path.join(SCRIPT_DIR, "cmake-build-release", "solver")
INSTANCES_DIR = os.path.join(SCRIPT_DIR, "instances", "promo")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

SEED = 42
MAX_TIME_SECONDS = 230.0
SUBPROCESS_TIMEOUT = MAX_TIME_SECONDS + 15.0


def evaluate_solution(instance_path, output_text):
    """Independently parses the problem instance and verifies the C++ stdout solution."""
    with open(instance_path, "r") as f:
        tokens = f.read().split()
    if not tokens:
        return 0, False

    it = iter(tokens)
    V = int(next(it))
    E = int(next(it))
    R = int(next(it))
    C = int(next(it))
    X = int(next(it))

    video_sizes = [int(next(it)) for _ in range(V)]

    dc_latency = [0] * E
    connections = [{} for _ in range(E)]
    for e in range(E):
        dc_latency[e] = int(next(it))
        k = int(next(it))
        for _ in range(k):
            c = int(next(it))
            lat = int(next(it))
            connections[e][c] = lat

    requests = []
    for _ in range(R):
        v = int(next(it))
        e = int(next(it))
        n = int(next(it))
        requests.append((v, e, n))

    # Parse C++ Hash Code output format
    lines = [line.strip() for line in output_text.strip().splitlines() if line.strip()]
    if not lines:
        return 0, False

    allocated = [set() for _ in range(C)]
    for line in lines[1:]:
        parts = [int(x) for x in line.split()]
        if not parts:
            continue
        c_id = parts[0]
        for v_id in parts[1:]:
            allocated[c_id].add(v_id)

    # Verify hard capacity constraints per cache
    for c_id in range(C):
        used_mb = sum(video_sizes[v] for v in allocated[c_id])
        if used_mb > X:
            return 0, False

    # Compute official Google Hash Code score (microseconds / ms saved * 1000)
    total_saved = 0
    total_reqs = 0
    for v, e, n in requests:
        total_reqs += n
        best_lat = dc_latency[e]
        for c_id, lat in connections[e].items():
            if v in allocated[c_id] and lat < best_lat:
                best_lat = lat
        total_saved += n * (dc_latency[e] - best_lat)

    score = (total_saved * 1000) // total_reqs if total_reqs > 0 else 0
    return score, True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    instance_files = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.in")))

    if not instance_files:
        print(f"No .in files found in {INSTANCES_DIR}")
        return

    print(f"Running production solver sequentially on {len(instance_files)} instances...")
    print("-" * 75)
    print(f"{'Instance':<30} | {'Score':>12} | {'Status':<8} | {'Time (s)':>8}")
    print("-" * 75)

    grand_total = 0

    # Sequential execution avoids thread oversubscription while OpenGA uses all cores
    for inst_path in instance_files:
        inst_name = pathlib.Path(inst_path).stem
        out_file = os.path.join(OUTPUT_DIR, f"{inst_name}.out")
        cmd = [EXECUTABLE, str(SEED), str(MAX_TIME_SECONDS)]

        t0 = time.perf_counter()
        try:
            with open(inst_path, "r") as infile:
                res = subprocess.run(
                    cmd,
                    stdin=infile,
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT,
                    cwd=os.path.dirname(EXECUTABLE)
                )
            elapsed = time.perf_counter() - t0

            if res.returncode == 0 and res.stdout.strip():
                with open(out_file, "w") as f:
                    f.write(res.stdout)
                score, valid = evaluate_solution(inst_path, res.stdout)
                status = "VALID" if valid else "INVALID"
                if valid:
                    grand_total += score
                print(f"{inst_name:<30} | {score:>12,d} | {status:<8} | {elapsed:>8.2f}")
            else:
                print(f"{inst_name:<30} | {'0':>12} | {'ERROR':<8} | {elapsed:>8.2f}")

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"{inst_name:<30} | {'0':>12} | {'TIMEOUT':<8} | {elapsed:>8.2f}")

    print("-" * 75)
    print(f"{'GRAND TOTAL SCORE':<30} | {grand_total:>12,d}")
    print("-" * 75)


if __name__ == "__main__":
    main()