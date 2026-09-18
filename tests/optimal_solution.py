import argparse
import subprocess
from pathlib import Path


def read_instance(path):
    with path.open(encoding="ascii") as instance_file:
        values = iter(map(int, instance_file.read().split()))
        video_count = next(values)
        endpoint_count = next(values)
        request_count = next(values)
        cache_count = next(values)
        capacity = next(values)
        video_sizes = [next(values) for _ in range(video_count)]

        endpoint_latencies = []
        for _ in range(endpoint_count):
            datacenter_latency = next(values)
            connection_count = next(values)
            connections = {}
            for _ in range(connection_count):
                cache_id = next(values)
                connections[cache_id] = next(values)
            endpoint_latencies.append((datacenter_latency, connections))

        requests = [[0] * video_count for _ in range(endpoint_count)]
        for _ in range(request_count):
            video_id = next(values)
            endpoint_id = next(values)
            requests[endpoint_id][video_id] += next(values)

    return video_sizes, endpoint_latencies, requests, cache_count, capacity


def score_solution(video_sizes, endpoint_latencies, requests, cache_masks):
    total_saved = 0
    total_requests = sum(sum(endpoint_requests) for endpoint_requests in requests)
    for endpoint_id, (datacenter_latency, connections) in enumerate(endpoint_latencies):
        for video_id, request_count in enumerate(requests[endpoint_id]):
            if request_count == 0:
                continue
            best_latency = datacenter_latency
            for cache_id, cache_latency in connections.items():
                if cache_masks[cache_id] & (1 << video_id):
                    best_latency = min(best_latency, cache_latency)
            total_saved += (datacenter_latency - best_latency) * request_count
    return (total_saved * 1000) // total_requests if total_requests else 0


def feasible_masks(video_sizes, capacity):
    masks = []
    for mask in range(1 << len(video_sizes)):
        used_capacity = sum(
            size for video_id, size in enumerate(video_sizes) if mask & (1 << video_id)
        )
        if used_capacity <= capacity:
            masks.append(mask)
    return masks


def find_optimum(video_sizes, endpoint_latencies, requests, cache_count, capacity):
    if len(video_sizes) > 20 or cache_count > 8:
        raise ValueError("instance trop grande pour une recherche exacte")

    options = feasible_masks(video_sizes, capacity)
    best_score = -1
    best_masks = None

    def search(cache_id, cache_masks):
        nonlocal best_score, best_masks
        if cache_id == cache_count:
            score = score_solution(video_sizes, endpoint_latencies, requests, cache_masks)
            if score > best_score:
                best_score = score
                best_masks = cache_masks[:]
            return
        for mask in options:
            cache_masks.append(mask)
            search(cache_id + 1, cache_masks)
            cache_masks.pop()

    search(0, [])
    return best_score, best_masks


def parse_solver_output(output, cache_count):
    lines = [line.split() for line in output.splitlines() if line.split()]
    solution_lines = lines[-(cache_count + 1):]
    if len(solution_lines) != cache_count + 1 or int(solution_lines[0][0]) != cache_count:
        raise ValueError("sortie du solveur introuvable")

    cache_masks = [0] * cache_count
    for line in solution_lines[1:]:
        cache_id = int(line[0])
        for video_id in map(int, line[1:]):
            cache_masks[cache_id] |= 1 << video_id
    return cache_masks


def main():
    parser = argparse.ArgumentParser(description="Compare le solveur genetique a un optimum exact.")
    parser.add_argument("instance", type=Path)
    parser.add_argument("--solver", type=Path, help="executable a comparer")
    arguments = parser.parse_args()

    data = read_instance(arguments.instance)
    video_sizes, endpoint_latencies, requests, cache_count, capacity = data
    optimal_score, optimal_masks = find_optimum(*data)
    print(f"{arguments.instance}: optimum exact = {optimal_score}")

    if arguments.solver:
        completed = subprocess.run(
            [str(arguments.solver)],
            input=arguments.instance.read_bytes(),
            capture_output=True,
            check=True,
        )
        solver_masks = parse_solver_output(completed.stdout.decode(), cache_count)
        solver_score = score_solution(video_sizes, endpoint_latencies, requests, solver_masks)
        print(f"score du solveur = {solver_score}")
        print(f"ecart = {optimal_score - solver_score}")

    print("configuration optimale:")
    for cache_id, mask in enumerate(optimal_masks):
        videos = [str(video_id) for video_id in range(len(video_sizes)) if mask & (1 << video_id)]
        print(f"{cache_id}: {' '.join(videos)}")


if __name__ == "__main__":
    main()