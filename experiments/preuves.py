"""Shows the results and proofs of blocks 1 and 5 in the terminal, and draws the curves.

    py experiments\\preuves.py                      # summary of all results + curves (PNG)
    py experiments\\preuves.py --demo me_at_the_zoo # live run: solver -> score.py -> statistics
    py experiments\\preuves.py --open               # same as the first, then opens the curves folder

Reads the files produced by the experiments (nothing is recomputed, except with --demo):
    experiments/blocks_results.csv      450 runs, blocks 1 and 5 (9 instances x 10 configs x 5 seeds)
    experiments/blocks_curves.csv       same 450 runs re-run with curves (reproducibility check)
    experiments/thresholds_results.csv  132 runs, thresholds of decoder 3
    experiments/analysis/before_after.csv, instance_features.csv
    results/<instance>_ga.log.csv, _ga_perm.log.csv   generation-by-generation curves (old program)
    experiments/runs_curves/.../stderr.txt            generation-by-generation curves (variants)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "analysis" / "courbes"
sys.path.insert(0, str(HERE))
import score  # noqa: E402
import solution_stats  # noqa: E402
from run_blocks import ESSENTIAL, INSTANCES  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def title(text):
    print("\n" + "=" * 78 + f"\n {text}\n" + "=" * 78)


def table(header, rows, widths):
    line = " | ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line + "\n" + "-" * len(line))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def f1(x):
    return "-" if x is None else f"{x:.1f}".replace(".", ",")


def f2(x):
    return "-" if x is None else f"{x:.2f}".replace(".", ",")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summary():
    deck = json.loads((HERE / "analysis" / "deck_data.json").read_text(encoding="utf-8"))

    title("1. AVANT -> APRES (11 instances, graine 42, re-notees par score.py)")
    ba = read(HERE / "analysis" / "before_after.csv")
    by = {(r["instance"], r["method"]): r for r in ba}
    insts = list(dict.fromkeys(r["instance"] for r in ba))
    rows = []
    for i in insts:
        def show(m):
            r = by.get((i, m))
            return "-" if not r else (f"{r['pct_best_known']} %" if r["pct_best_known"] else r["score"])
        rows.append([i[:34], show("ga"), show("greedy"), show("ga_perm")])
    table(["instance", "GA matrice (avant)", "glouton seul", "GA ordres+glouton"], rows, [34, 18, 13, 17])
    m = deck["before_after"]["means"]
    print()
    table(["statistique (moyenne)", "avant", "apres"], [
        ["copies inutiles", f"{f1(m['ga']['pct_useless_copies'])} %", f"{f1(m['ga_perm']['pct_useless_copies'])} %"],
        ["capacite gaspillee", f"{f1(m['ga']['pct_capacity_useless'])} %", f"{f1(m['ga_perm']['pct_capacity_useless'])} %"],
        ["requetes servies par un cache", f"{f1(m['ga']['pct_requests_served'])} %", f"{f1(m['ga_perm']['pct_requests_served'])} %"],
        ["videos utiles en cache", f"{f1(m['ga']['pct_candidates_placed'])} %", f"{f1(m['ga_perm']['pct_candidates_placed'])} %"],
        ["top 10 % demandees absentes", f"{f1(m['ga']['pct_top10_missing'])} %", f"{f1(m['ga_perm']['pct_top10_missing'])} %"],
        ["remplissage", f"{f1(m['ga']['fill_mean'])} %", f"{f1(m['ga_perm']['fill_mean'])} %"],
    ], [30, 10, 10])

    for block, label, names in (("init", "2. BLOC 1 - INITIALISATION", ["0 utilite x bruit (Rihem)", "1 hasard pur", "2 bruit additif",
                                                                        "3 densite (controle)", "4 melange du debut", "5 hybride"]),
                                ("dec", "3. BLOC 5 - DECODEUR", ["0 glouton par tours (Rihem)", "1 marge 15 % + reparation",
                                                               "2 centre caches", "3 seuils fixes", "4 seuils co-evolues"])):
        title(f"{label} ({len(deck['instances'])} instances x {len(deck['seeds'])} graines ; variante 0 = 100)")
        rows = []
        for v, n in enumerate(names):
            x = deck[block][str(v)]
            rows.append([n, f2(x["rel_mean"]), f"{f1(x['rel_min'])} / {f1(x['rel_max'])}", f2(x["rank_mean"]),
                         len(x["wins"]), len(x["losses"]), f"+{f2(x['ga_gain_pct'])} %",
                         f"{f1(x['fill_mean'])} %", f2(x["duplication"]), f"{f1(x['pct_requests_served'])} %"])
        table(["variante", "score rel.", "min / max", "rang", "gagne", "perd", "gain GA", "remplis.", "copies", "req. servies"],
              rows, [27, 10, 13, 5, 5, 4, 9, 9, 6, 12])
        print("gagne / perd = nombre d'instances ou la difference avec la variante 0 est significative (> 2 erreurs type)")

    title("4. SEUILS DE THOMAS (decodeur 3, tau_densite ; seuil 0 = 100 ; 3 graines)")
    t = deck["thresholds"]
    taus = list(next(iter(t.values()))["density"])
    table(["instance"] + taus, [[i[:30]] + [f1(t[i]["density"][x]) for x in taus] for i in t], [30] + [7] * len(taus))

    title("5. COMPLEXITE : formule contre temps mesure (pente log-log ~ 1 = proportionnel)")
    table(["decodeur", "formule", "pente", "correlation"],
          [[k, v["formula"], f2(v["slope"]), f2(v["corr"])] for k, v in deck["complexity"].items()], [24, 40, 6, 11])

    title("6. REPRODUCTIBILITE")
    a = {r["run_id"]: r for r in read(HERE / "blocks_results.csv")}
    if (HERE / "blocks_curves.csv").exists():
        b = read(HERE / "blocks_curves.csv")
        same = sum(1 for r in b if r["run_id"] in a and a[r["run_id"]]["score"] == r["score"])
        print(f"2e lancement des memes runs : {same} scores identiques sur {len(b)}")
    print(f"runs du bloc 1/5 : {sum(1 for r in a.values() if r['status'] == 'ok')} reussis sur {len(a)}")


def curves():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    # old GA vs current GA, generation by generation
    names = [p.name[: -len("_ga.log.csv")] for p in sorted((ROOT / "results").glob("*_ga.log.csv"))]
    names = [n for n in names if n in INSTANCES and n != "test"]
    cols = 4
    fig, axes = plt.subplots((len(names) + cols - 1) // cols, cols, figsize=(16, 3.2 * ((len(names) + cols - 1) // cols)))
    for ax, n in zip(axes.flat, names):
        for method, color, label in (("ga", "#B0443F", "avant (GA matrice)"), ("ga_perm", "#0E7C86", "après (GA ordres + glouton)")):
            rows = read(ROOT / "results" / f"{n}_{method}.log.csv")
            g = [int(r["generation"]) for r in rows]
            ax.plot(g, [float(r["best_saved"]) for r in rows], color=color, lw=2, label=f"{label} : meilleur")
            ax.plot(g, [float(r["avg_saved"]) for r in rows], color=color, lw=1, ls="--", label=f"{label} : moyenne")
        if INSTANCES[n][1]:
            ax.axhline(INSTANCES[n][1], color="grey", ls=":", lw=1, label="meilleure valeur connue")
        ax.set_title(n, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Score génération par génération : avant (rouge) / après (bleu), graine 42")
    fig.tight_layout()
    path = OUT / "convergence_avant_apres.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    made.append(path)

    # block 1: curves of the 6 initialisations, averaged over instances and seeds
    data = json.loads((HERE / "analysis" / "page_data.json").read_text(encoding="utf-8"))["variant_curves"]
    for block, labels, fname in (("init", ["0 utilité × bruit", "1 hasard", "2 bruit additif", "3 densité", "4 mélange début", "5 hybride"], "bloc1_courbes.png"),
                                 ("dec", ["0 glouton par tours", "1 marge 15 %", "2 centré caches", "3 seuils fixes", "4 seuils co-évolués"], "bloc5_courbes.png")):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
        for v, label in enumerate(labels):
            key = f"{v}-0-0-0-0" if block == "init" else f"0-0-0-0-{v}"
            if key not in data[block]:
                continue
            c = data[block][key]["_mean"]
            a1.plot(c["best"], lw=2.5 if v == 0 else 1.5, label=label)
            a2.plot(c["mean"], lw=2.5 if v == 0 else 1.5, label=label)
        a1.set_title("meilleur individu")
        a2.set_title("moyenne de la population")
        for a in (a1, a2):
            a.set_xlabel("génération")
            a.set_ylabel("score relatif (final de la variante 0 = 100)")
            a.grid(alpha=.3)
        a1.legend(fontsize=8)
        fig.suptitle(f"{'Bloc 1' if block == 'init' else 'Bloc 5'} : évolution moyenne sur {len(ESSENTIAL)} instances × {data['n_seeds']} graines")
        fig.tight_layout()
        path = OUT / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        made.append(path)

    title("7. COURBES ET GRAPHIQUES")
    for p in made + sorted((HERE / "analysis").glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  rapports détaillés : experiments\\analysis\\report.md, extra_report.md, before_after.md")


def demo(name, seed):
    exe = HERE / "solver_fast_curves.exe"
    if not exe.exists():
        print(f"{exe.name} est introuvable : la démo en direct a besoin de ce solveur compilé (non fourni dans le dépôt). "
              "Le résumé et les courbes marchent sans lui : py experiments\\preuves.py")
        return
    path, best = INSTANCES[name]
    title(f"DEMO EN DIRECT : {name} (variante 0 = solution de Rihem, graine {seed})")
    with tempfile.TemporaryDirectory() as tmp:
        start = time.perf_counter()
        with open(ROOT / path, "rb") as stdin:
            res = subprocess.run([str(exe), str(best or 0), "demo", "0", "0", "0", "0", "0", str(seed)],
                                 stdin=stdin, capture_output=True, cwd=tmp)
        wall = time.perf_counter() - start
        gens = [l.split() for l in res.stderr.decode(errors="replace").splitlines() if l.startswith("gen ")]
        print(f"solveur : {wall:.1f} s, {len(gens)} générations\n")
        print("génération |   meilleur |    moyenne")
        shown = gens[:: max(1, len(gens) // 12)]
        if gens and shown[-1] is not gens[-1]:
            shown.append(gens[-1])
        for g in shown:
            print(f"{g[1]:>10} | {float(g[2]):>10,.0f} | {float(g[3]):>10,.0f}".replace(",", " "))
        sol = res.stdout.decode()
    instance = score.read_instance(ROOT / path)
    caches = score.read_solution(sol)
    errors = score.check_solution(instance, caches)
    official = score.score_solution(instance, caches)
    print(f"\nvérification score.py : {'VALIDE' if not errors else 'INVALIDE'} ; score = {official:,}".replace(",", " ")
          + (f" ({100 * official / best:.2f} % de la meilleure valeur connue {best:,})".replace(",", " ") if best else ""))
    stats = solution_stats.compute_stats(solution_stats.prepare(instance), caches)
    old = ROOT / "results" / f"{name}_ga.out"
    old_stats = None
    if old.exists():
        text = "\n".join(l for l in old.read_text().splitlines() if not l.startswith("Generation"))
        old_caches = score.read_solution(text)
        old_stats = solution_stats.compute_stats(solution_stats.prepare(instance), old_caches)
        old_stats["score"] = score.score_solution(instance, old_caches)
    print()
    rows = [["score", f"{official:,}".replace(",", " "), f"{old_stats['score']:,}".replace(",", " ") if old_stats else "-"]]
    for k, label in (("fill_mean", "remplissage %"), ("duplication", "copies par vidéo"), ("pct_requests_served", "requêtes servies %"),
                     ("pct_candidates_placed", "vidéos utiles en cache %"), ("pct_useless_copies", "copies inutiles %"),
                     ("pct_top10_missing", "top 10 % absentes %")):
        rows.append([label, f2(stats[k]), f2(old_stats[k]) if old_stats else "-"])
    table(["statistique", "maintenant", "ancien GA (results/)"], rows, [26, 12, 20])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", metavar="INSTANCE", help="lance le solveur en direct sur cette instance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open", action="store_true", help="ouvre le dossier des courbes")
    args = parser.parse_args()
    if args.demo:
        demo(args.demo, args.seed)
        return
    summary()
    curves()
    if args.open and os.name == "nt":
        os.startfile(OUT)


if __name__ == "__main__":
    main()
