import subprocess
import os
import glob
import pathlib
import json
import matplotlib.pyplot as plt


BEST_KNOWN_VALUES = {
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

EXECUTABLE = "./cmake-build-debug/solver"
OUTPUT_DIR = "benchmark_results"

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
        output_file_path = os.path.join(OUTPUT_DIR, f"output_{instance_name}.txt")

        try:
            with open(instance_path, "r") as infile, open(output_file_path, "w") as outfile:
                result = subprocess.run(
                    [EXECUTABLE, str(best_known),instance_name],
                    stdin=infile,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1200  # 20 minutes timeout per instance as a safeguard
                )

                if result.returncode == 0:
                    print(f"Successfully finished {instance_name}")
                else:
                    print(f"Error running {instance_name}:")
                    print(result.stderr)

        except subprocess.TimeoutExpired:
            print(f"Timeout expired for {instance_name}")
        except Exception as e:
            print(f"An error occurred while running {instance_name}: {e}")

    generate_plots()

def generate_plots():
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    if not json_files:
        print("No JSON results found for plotting.")
        return

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            targets_hit_list = data.get("targets_hit", [])
            if not targets_hit_list:
                continue

            if isinstance(targets_hit_list, list) and len(targets_hit_list) > 0 and isinstance(targets_hit_list[0], list):
                history = targets_hit_list[0]
            else:
                history = targets_hit_list

            if not history:
                continue

            evals = [entry["eval"] for entry in history]
            percentages = [(i + 1) / 50.0 * 100.0 for i in range(len(history))]

            instance_name = data.get("instance", pathlib.Path(json_file).stem)

            plt.figure(figsize=(8, 5))
            plt.plot(evals, percentages, marker='o', label=instance_name, color='b')
            plt.xlabel("Number of Evaluations")
            plt.ylabel("Percentage of Targets Cleared (%)")
            plt.title(f"Benchmark Progress: {instance_name}")
            plt.ylim(0, 105)
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(OUTPUT_DIR, f"plot_{instance_name}.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            plt.close()

        except Exception as e:
            print(f"Failed to parse or plot {json_file}: {e}")

if __name__ == "__main__":
    main()