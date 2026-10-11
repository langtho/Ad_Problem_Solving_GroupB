"""Independent checker and scorer for Hash Code 2017 "Streaming Videos".

It does not reuse any code from the solver, so it can be used to verify the
solver's output:

    python experiments/score.py instances/instances/me_at_the_zoo.in results/me_at_the_zoo_run1.out

Score (as in the statement): for every request description, the time saved is
(datacenter latency - best latency to a connected cache holding the video)
multiplied by the number of requests. The score is the total time saved,
multiplied by 1000 and divided (integer division) by the total number of
requests.
"""

import sys
from pathlib import Path


def read_instance(path):
    values = iter(map(int, Path(path).read_text(encoding="ascii").split()))
    video_count = next(values)
    endpoint_count = next(values)
    request_count = next(values)
    cache_count = next(values)
    capacity = next(values)
    video_sizes = [next(values) for _ in range(video_count)]

    endpoints = []
    for _ in range(endpoint_count):
        datacenter_latency = next(values)
        connection_count = next(values)
        connections = [(next(values), next(values)) for _ in range(connection_count)]
        # (cache_id, latency) sorted by latency: the first cache holding the
        # video is then the best one
        connections.sort(key=lambda connection: connection[1])
        endpoints.append((datacenter_latency, connections))

    requests = [(next(values), next(values), next(values)) for _ in range(request_count)]

    return {
        "video_sizes": video_sizes,
        "endpoints": endpoints,
        "requests": requests,
        "cache_count": cache_count,
        "capacity": capacity,
    }


def read_solution(text):
    """Parse the output format: N, then N lines "cache_id video_id ..."."""
    lines = [line.split() for line in text.splitlines() if line.split()]
    if not lines:
        raise ValueError("empty solution")
    described = int(lines[0][0])
    if len(lines) - 1 != described:
        raise ValueError(f"{described} cache descriptions announced, {len(lines) - 1} found")

    caches = {}
    for line in lines[1:]:
        cache_id = int(line[0])
        if cache_id in caches:
            raise ValueError(f"cache {cache_id} described twice")
        caches[cache_id] = set(map(int, line[1:]))
    return caches


def check_solution(instance, caches):
    """Return a list of constraint violations (empty list = valid solution)."""
    errors = []
    video_count = len(instance["video_sizes"])
    for cache_id, videos in caches.items():
        if not 0 <= cache_id < instance["cache_count"]:
            errors.append(f"unknown cache {cache_id}")
            continue
        unknown = [video for video in videos if not 0 <= video < video_count]
        if unknown:
            errors.append(f"cache {cache_id}: unknown videos {unknown[:5]}")
            continue
        used = sum(instance["video_sizes"][video] for video in videos)
        if used > instance["capacity"]:
            errors.append(f"cache {cache_id}: {used} MB used > capacity {instance['capacity']} MB")
    return errors


def score_solution(instance, caches):
    total_saved = 0
    total_requests = 0
    for video_id, endpoint_id, count in instance["requests"]:
        datacenter_latency, connections = instance["endpoints"][endpoint_id]
        total_requests += count
        for cache_id, latency in connections:
            if video_id in caches.get(cache_id, ()):
                total_saved += (datacenter_latency - min(latency, datacenter_latency)) * count
                break
    return (total_saved * 1000) // total_requests if total_requests else 0


def cache_usage(instance, caches):
    """Average fill rate of the caches (0..1)."""
    if instance["cache_count"] == 0 or instance["capacity"] == 0:
        return 0.0
    used = sum(
        sum(instance["video_sizes"][video] for video in videos) for videos in caches.values()
    )
    return used / (instance["cache_count"] * instance["capacity"])


def main():
    if len(sys.argv) != 3:
        print("usage: python score.py <instance.in> <solution.out>")
        sys.exit(2)
    instance = read_instance(sys.argv[1])
    caches = read_solution(Path(sys.argv[2]).read_text())
    errors = check_solution(instance, caches)
    if errors:
        print("INVALID solution:")
        for error in errors:
            print("  " + error)
        sys.exit(1)
    print(f"score = {score_solution(instance, caches)}")
    print(f"cache usage = {cache_usage(instance, caches):.1%}")


if __name__ == "__main__":
    main()
