import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def generate_sweep2_plots(csv_path="benchmark_results/sweeps/round2_grid_search.csv", out_dir="benchmark_results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df['best_fitness_achieved'] > 0.0].copy()

    df['max_fit'] = df.groupby('instance')['best_fitness_achieved'].transform('max')
    df['normalized_fitness'] = df['best_fitness_achieved'] / df['max_fit']

    pattern = r'i(\d+)_c(\d+)_m(\d+)_l(\d+)_d(\d+)'
    extracted = df['config_key'].str.extract(pattern)
    extracted.columns = ['Init', 'Crossover', 'Mutation', 'LocalSearch', 'Decoding']
    df = pd.concat([df, extracted], axis=1)

    plt.figure(figsize=(12, 8))
    top_configs = df.groupby('config_key')['normalized_fitness'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top_configs.values, y=top_configs.index, palette="viridis")
    plt.title("Top 15 Architectures (Average Normalized Fitness)")
    plt.xlabel("Normalized Fitness (1.0 = Best in Instance)")
    plt.ylabel("Configuration Key")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_top_architectures.png"))
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
    fig.suptitle("Main Effects: Impact of Individual Operators on Normalized Fitness")

    operators = ['Init', 'Crossover', 'Mutation', 'LocalSearch', 'Decoding']
    for i, op in enumerate(operators):
        sns.boxplot(ax=axes[i], x=op, y='normalized_fitness', data=df, palette="Set2")
        axes[i].set_title(op)
        axes[i].set_ylabel("Normalized Fitness" if i == 0 else "")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_main_effects.png"))
    plt.close()

    if 'phenotypic_diversity_final' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='phenotypic_diversity_final', y='normalized_fitness', hue='Init', data=df, alpha=0.7)
        plt.title("Final Diversity vs. Normalized Fitness (Colored by Init Strategy)")
        plt.xlabel("Final Phenotypic Diversity")
        plt.ylabel("Normalized Fitness")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot3_diversity_vs_fitness.png"))
        plt.close()

if __name__ == "__main__":
    generate_sweep2_plots()