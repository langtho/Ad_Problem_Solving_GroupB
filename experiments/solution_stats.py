"""Statistics that describe a solution of Hash Code 2017 "Streaming Videos".

    py experiments\\solution_stats.py <instance.in> <solution.out>

Reuses the parser, the checker and the scorer of score.py (independent from the
C++ solver), then computes:

- fill rate of the caches (mean / min / max over all caches)
- duplication: number of copies per distinct video placed
- share of the candidate videos placed in at least one cache
  (candidate = fits in a cache and can save time: same definition as greedy.cpp)
- share of the requests served by a cache (weighted by number of requests)
- mean time saved per request (ms) and share of the upper bound
  (upper bound = every video in the best cache of every endpoint, capacities ignored)
- useless copies: copies whose removal would not change the score
- most demanded videos missing: among the 10 % candidates with the largest
  potential gain, share that is in no cache, and share of the total potential
  gain carried by the candidates that are in no cache
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402


def prepare(instance):
    """Precompute what does not depend on the solution (done once per instance)."""
    capacity = instance["capacity"]
    sizes = instance["video_sizes"]
    endpoints = instance["endpoints"]

    # Duplicate (video, endpoint) lines add up, as in the statement
    demand = defaultdict(int)
    for video, endpoint, count in instance["requests"]:
        demand[(video, endpoint)] += count
    total_requests = sum(demand.values())

    potential = defaultdict(int)  # time the video can save if it is in the best cache everywhere
    upper_bound_saved = 0
    for (video, endpoint), count in demand.items():
        datacenter_latency, connections = endpoints[endpoint]
        if sizes[video] > capacity or not connections:
            continue
        gain = max(0, datacenter_latency - connections[0][1]) * count  # connections sorted by latency
        potential[video] += gain
        upper_bound_saved += gain
    candidates = {video for video, gain in potential.items() if gain > 0}

    return {
        "instance": instance,
        "latency": [dict(connections) for _, connections in endpoints],  # endpoint -> {cache: latency}
        "demand": demand,
        "total_requests": total_requests,
        "potential": potential,
        "candidates": candidates,
        "upper_bound_score": (upper_bound_saved * 1000) // total_requests if total_requests else 0,
    }


def compute_stats(prepared, caches):
    instance = prepared["instance"]
    sizes = instance["video_sizes"]
    capacity = instance["capacity"]
    endpoints = instance["endpoints"]
    cache_count = instance["cache_count"]
    total_requests = prepared["total_requests"]
    candidates = prepared["candidates"]
    potential = prepared["potential"]

    # Fill rate of every cache (caches not described in the output are empty)
    fills = []
    for cache_id in range(cache_count):
        used = sum(sizes[video] for video in caches.get(cache_id, ()))
        fills.append(used / capacity if capacity else 0.0)

    copies = sum(len(videos) for videos in caches.values())
    placed = set().union(*caches.values()) if caches else set()

    # For every request: best and second best latency among the caches holding the video
    # (only the caches that hold the video are looked at: fast on instances with many caches)
    holders = defaultdict(list)
    for cache_id, videos in caches.items():
        for video in videos:
            holders[video].append(cache_id)
    latency = prepared["latency"]
    saved = 0
    served = 0
    marginal = defaultdict(int)  # (cache, video) -> score lost if this copy is removed
    for (video, endpoint), count in prepared["demand"].items():
        datacenter_latency = endpoints[endpoint][0]
        lat = latency[endpoint]
        holding = sorted((lat[c], c) for c in holders.get(video, ()) if c in lat and lat[c] < datacenter_latency)
        if not holding:
            continue
        served += count
        best_latency, best_cache = holding[0]  # connections are sorted by latency
        saved += (datacenter_latency - best_latency) * count
        second_latency = holding[1][0] if len(holding) > 1 else datacenter_latency
        if second_latency > best_latency:
            marginal[(best_cache, video)] += (second_latency - best_latency) * count

    useless = [(cache_id, video) for cache_id, videos in caches.items() for video in videos
               if marginal.get((cache_id, video), 0) == 0]
    useless_mb = sum(sizes[video] for _, video in useless)

    # Candidates with the largest potential gain
    ranked = sorted(candidates, key=lambda video: potential[video], reverse=True)
    top = ranked[:max(1, len(ranked) // 10)] if ranked else []
    top_missing = sum(1 for video in top if video not in placed)
    total_potential = sum(potential[video] for video in candidates)
    missing_potential = sum(potential[video] for video in candidates if video not in placed)

    score_value = (saved * 1000) // total_requests if total_requests else 0
    upper_bound = prepared["upper_bound_score"]
    return {
        "score_check": score_value,
        "upper_bound_score": upper_bound,
        "pct_upper_bound": 100.0 * score_value / upper_bound if upper_bound else 0.0,
        "fill_mean": 100.0 * sum(fills) / len(fills) if fills else 0.0,
        "fill_min": 100.0 * min(fills) if fills else 0.0,
        "fill_max": 100.0 * max(fills) if fills else 0.0,
        "copies": copies,
        "distinct_videos_placed": len(placed),
        "duplication": copies / len(placed) if placed else 0.0,
        "nb_candidates": len(candidates),
        "pct_candidates_placed": 100.0 * len(placed & candidates) / len(candidates) if candidates else 0.0,
        "pct_requests_served": 100.0 * served / total_requests if total_requests else 0.0,
        "mean_gain_ms": saved / total_requests if total_requests else 0.0,
        "useless_copies": len(useless),
        "pct_useless_copies": 100.0 * len(useless) / copies if copies else 0.0,
        "pct_capacity_useless": 100.0 * useless_mb / (cache_count * capacity) if cache_count and capacity else 0.0,
        "pct_top10_missing": 100.0 * top_missing / len(top) if top else 0.0,
        "pct_potential_missing": 100.0 * missing_potential / total_potential if total_potential else 0.0,
    }


def main():
    if len(sys.argv) != 3:
        print("usage: py solution_stats.py <instance.in> <solution.out>")
        sys.exit(2)
    instance = score.read_instance(sys.argv[1])
    caches = score.read_solution(Path(sys.argv[2]).read_text())
    errors = score.check_solution(instance, caches)
    if errors:
        print("INVALID solution:")
        for error in errors:
            print("  " + error)
        sys.exit(1)
    official = score.score_solution(instance, caches)
    stats = compute_stats(prepare(instance), caches)
    print(f"score (score.py) = {official}")
    for key, value in stats.items():
        print(f"{key:24s} {value:.2f}" if isinstance(value, float) else f"{key:24s} {value}")


if __name__ == "__main__":
    main()
