from pathlib import Path
import random


ROOT = Path(__file__).parent / "generated"


def write_instance(path, video_sizes, endpoint_latencies, requests, capacity):
    endpoint_count = len(endpoint_latencies)
    cache_ids = [
        cache_id
        for _, connections in endpoint_latencies
        for cache_id, _ in connections
    ]
    cache_count = max(cache_ids, default=-1) + 1
    request_count = len(requests)

    lines = [
        f"{len(video_sizes)} {endpoint_count} {request_count} {cache_count} {capacity}",
        " ".join(map(str, video_sizes)),
    ]

    for endpoint_latency, connections in endpoint_latencies:
        lines.append(f"{endpoint_latency} {len(connections)}")
        lines.extend(f"{cache_id} {latency}" for cache_id, latency in connections)

    lines.extend(f"{video_id} {endpoint_id} {count}" for video_id, endpoint_id, count in requests)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def make_tiny():
    write_instance(
        ROOT / "tiny.in",
        [10, 20],
        [(100, [(0, 20)])],
        [(0, 0, 5), (1, 0, 3)],
        30,
    )


def make_capacity():
    write_instance(
        ROOT / "capacity.in",
        [10, 20, 30, 40],
        [
            (100, [(0, 10), (1, 60)]),
            (120, [(0, 80), (1, 20)]),
        ],
        [(0, 0, 100), (1, 0, 50), (2, 1, 80), (3, 1, 30)],
        50,
    )


def make_disconnected():
    write_instance(
        ROOT / "disconnected.in",
        [15, 25, 35],
        [
            (100, []),
            (150, [(1, 30)]),
        ],
        [(0, 0, 10), (1, 1, 20), (2, 1, 5)],
        40,
    )


def make_large():
    rng = random.Random(20260918)
    video_count = 800
    endpoint_count = 80
    cache_count = 40
    request_count = 4000
    capacity = 6000

    video_sizes = [rng.randint(1, 1000) for _ in range(video_count)]
    endpoint_latencies = []
    for _ in range(endpoint_count):
        endpoint_latency = rng.randint(500, 5000)
        connected = rng.sample(range(cache_count), rng.randint(8, 25))
        connections = [
            (cache_id, rng.randint(1, endpoint_latency - 1))
            for cache_id in connected
        ]
        endpoint_latencies.append((endpoint_latency, connections))

    requests = [
        (rng.randrange(video_count), rng.randrange(endpoint_count), rng.randint(1, 10000))
        for _ in range(request_count)
    ]
    write_instance(ROOT / "large.in", video_sizes, endpoint_latencies, requests, capacity)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    make_tiny()
    make_capacity()
    make_disconnected()
    make_large()


if __name__ == "__main__":
    main()