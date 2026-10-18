# Présentation de l'algorithme génétique

Ce document accompagne la présentation `Genetic Algorithms Pitch.pptx`. Il relie chaque étape de l'algorithme génétique au code du projet.

## Représentation d'une solution

**Fichier :** `../global.h`, structure `Solution` (lignes 23-26)

```cpp
struct Solution {
    std::vector<std::vector<bool>> results;
    std::vector<int> used_capacity;
};
```

Une solution est une matrice `caches x videos`. `results[c][v]` vaut `true` lorsque la vidéo `v` est placée dans le cache `c`. `used_capacity` permet de suivre la capacité utilisée dans chaque cache.

## 1. Initialisation de la population

**Fichier :** `evolAlgo.cpp`, `init_genes` (lignes 72-84)

```cpp
void init_genes(Solution& s, const ProblemData& problem,
                const std::function<double(void)>& rnd) {
    s.results.assign(problem.nbr_caches,
                     std::vector<bool>(problem.nbr_videos, false));
    s.used_capacity.assign(problem.nbr_caches, 0);
    for (int c = 0; c < problem.nbr_caches; ++c) {
        int start_idx = static_cast<int>(rnd() * problem.nbr_videos)
                        % problem.nbr_videos;
        for (int step = 0; step < problem.nbr_videos; ++step) {
            int v = (start_idx + step) % problem.nbr_videos;
            if (can_add_video(s, problem, c, v) && rnd() < 0.25) {
                add_video(s, problem, c, v);
            }
        }
    }
}
```

Pour chaque cache, les vidéos sont parcourues à partir d'un indice de départ aléatoire. Une vidéo est ajoutée avec une probabilité de `0.25`, seulement si elle respecte la capacité restante. La population initiale est donc générée aléatoirement, tout en restant valide.

## 2. Évaluation

**Fichier :** `../main.cpp`, `eval_time_saved` (lignes 86-113)

```cpp
int eval_time_saved(const ProblemData& problemData,
                    const vector<vector<bool>>& results) {
    long long total_saved = 0;
    long long total_requests = 0;

    for (int e = 0; e < problemData.nbr_endpoints; ++e) {
        for (int v = 0; v < problemData.nbr_videos; ++v) {
            int req_count = problemData.requests[e][v];
            if (req_count <= 0) continue;

            total_requests += req_count;
            int best_latency = problemData.endpoints[e];

            for (int c = 0; c < problemData.nbr_caches; ++c) {
                bool cache_connected = (problemData.network[e][c] > 0);
                bool video_in_cache = results[c][v];
                if (cache_connected && video_in_cache) {
                    best_latency = min(best_latency,
                                       problemData.network[e][c]);
                }
            }

            int saved = problemData.endpoints[e] - best_latency;
            total_saved += saved * req_count;
        }
    }

    return (total_saved * 1000) / total_requests;
}
```

La fonction compare, pour chaque requête, la latence du datacenter avec la meilleure latence disponible parmi les caches connectés contenant la vidéo. Le score est le temps économisé moyen, multiplié par `1000`. Plus le score est élevé, meilleure est la solution.

Dans `evolAlgo.cpp`, cette fonction est branchée sur la librairie (lignes 42-45) :

```cpp
ga.eval_solution = [&problem](const Solution& s, double& score) -> bool {
    score = static_cast<double>(eval_time_saved(problem, s.results));
    return true;
};
```

La librairie minimise un coût. Le projet convertit donc le score à maximiser en coût négatif avec `calculate_SO_total_fitness` (lignes 47-49) :

```cpp
ga.calculate_SO_total_fitness =
    [](const GA_Engine::thisChromosomeType& c) -> double {
        return -c.middle_costs;
    };
```

## 3. Condition d'arrêt

**Fichier :** `evolAlgo.cpp`, paramètre `generation_max` (ligne 31)

```cpp
ga.generation_max = 50;
```

L'exécution s'arrête après 50 générations maximum. La librairie vérifie cette limite dans `../openGA.hpp`, notamment dans la boucle de résolution et la condition d'arrêt. Dans cette version, aucune détection de stagnation personnalisée n'est ajoutée dans le projet.

## 4. Sélection

La sélection n'est pas réécrite dans notre code : elle est fournie par `../openGA.hpp`. Elle reste néanmoins modifiable, car `openGA.hpp` est un fichier source inclus dans le projet.

### Sélection des parents

**Fichier :** `../openGA.hpp`, `select_parent` (lignes 1580-1587)

```cpp
int select_parent(const thisGenerationType &g) {
    int N_max = int(g.chromosomes.size());
    double r = random01();
    int position = 0;
    while (position < N_max &&
           g.selection_chance_cumulative[position] < r) {
        position++;
    }
    return position;
}
```

La fonction tire un nombre aléatoire et choisit une position dans une distribution cumulative. Les meilleurs individus ont une probabilité plus élevée, mais les autres peuvent encore être sélectionnés.

### Calcul de la probabilité selon le rang

**Fichier :** `../openGA.hpp`, `generate_selection_chance` (lignes 1126-1142)

```cpp
void generate_selection_chance(thisGenerationType &gen,
                               const vector<int> &rank) {
    double chance_cumulative = 0.0;
    unsigned int N = (unsigned int)gen.chromosomes.size();
    gen.selection_chance_cumulative.clear();
    gen.selection_chance_cumulative.reserve(N);

    for (unsigned int i = 0; i < N; i++) {
        chance_cumulative += 1.0 / sqrt(double(rank[i] + 1));
        gen.selection_chance_cumulative.push_back(chance_cumulative);
    }

    for (unsigned int i = 0; i < N; i++) {
        gen.selection_chance_cumulative[i] =
            gen.selection_chance_cumulative[i] /
            gen.selection_chance_cumulative[population - 1];
    }
}
```

C'est une sélection par rang non linéaire : le rang est utilisé plutôt que la valeur brute du score. Le terme `1 / sqrt(rank + 1)` favorise les meilleurs individus.

### Élites et sélection de la génération suivante

**Fichier :** `../openGA.hpp`, `select_population_SO` (lignes 1015-1057)

```cpp
void select_population_SO(const thisGenerationType &g,
                          thisGenerationType &g2) {
    if (generation_step <= 0) {
        g2 = g;
        return;
    }

    vector<int> blocked;
    for (int i = 0; i < elite_count; i++) {
        g2.chromosomes.push_back(g.chromosomes[g.sorted_indices[i]]);
        blocked.push_back(g.sorted_indices[i]);
    }

    for (int i = 0; i < int(population) - elite_count; i++) {
        int j;
        bool allowed;
        do {
            allowed = true;
            j = select_parent(g);
            for (int k = 0; k < int(blocked.size()) && allowed; k++) {
                if (blocked[k] == j) allowed = false;
            }
        } while (!allowed);
        g2.chromosomes.push_back(g.chromosomes[j]);
        blocked.push_back(g.sorted_indices[j]);
    }
}
```

Les meilleurs individus sont copiés directement grâce à l'élitisme. Le reste de la population est rempli avec la sélection par rang. Pour l'instant, nous utilisons donc la sélection de la librairie telle quelle ; il suffirait de modifier ou remplacer ces fonctions dans `../openGA.hpp` pour essayer une sélection par tournoi, roulette ou autre stratégie.

## 5. Crossover

**Fichier :** `evolAlgo.cpp`, `crossover_solution` (lignes 123-141)

```cpp
Solution crossover_solution(const Solution& p1, const Solution& p2,
                           const ProblemData& problem,
                           const std::function<double(void)>& rnd) {
    Solution child;
    child.results.assign(problem.nbr_caches,
                         std::vector<bool>(problem.nbr_videos, false));
    child.used_capacity.assign(problem.nbr_caches, 0);

    for (int c = 0; c < problem.nbr_caches; ++c) {
        for (int v = 0; v < problem.nbr_videos; ++v) {
            bool bit = (rnd() < 0.5) ? p1.results[c][v]
                                     : p2.results[c][v];
            if (bit && can_add_video(child, problem, c, v)) {
                add_video(child, problem, c, v);
            }
        }
    }
    return child;
}
```

Pour chaque couple cache/vidéo, le gène vient aléatoirement du parent 1 ou du parent 2. C'est donc un crossover uniforme. Le test de capacité empêche de créer un enfant invalide.

## 6. Mutation

**Fichier :** `evolAlgo.cpp`, `mutate_solution` (lignes 110-121)

```cpp
Solution mutate_solution(const Solution& original,
                         const ProblemData& problem,
                         const std::function<double(void)>& rnd) {
    Solution mutated = original;
    int cache_idx = static_cast<int>(rnd() * problem.nbr_caches)
                    % problem.nbr_caches;
    int video_idx = static_cast<int>(rnd() * problem.nbr_videos)
                    % problem.nbr_videos;

    if (mutated.results[cache_idx][video_idx]) {
        remove_video(mutated, problem, cache_idx, video_idx);
    } else if (can_add_video(mutated, problem, cache_idx, video_idx)) {
        add_video(mutated, problem, cache_idx, video_idx);
    }
    return mutated;
}
```

La mutation choisit une case au hasard dans la matrice. Si la vidéo est présente, elle est retirée ; sinon elle est ajoutée si la capacité le permet. Cela correspond à un flip d'un seul bit.

La librairie applique cette mutation avec la probabilité définie par `ga.mutation_rate`.

## 7. Hyperparamètres

**Fichier :** `evolAlgo.cpp`, `run_evol_algo` (lignes 30-36)

```cpp
ga.population = 30;
ga.generation_max = 50;
ga.crossover_fraction = 0.7;
ga.mutation_rate = 0.2;
ga.elite_count = 2;
```

| Paramètre | Valeur actuelle | Rôle |
|---|---:|---|
| `population` | `30` | Nombre d'individus par génération |
| `generation_max` | `50` | Nombre maximal de générations |
| `crossover_fraction` | `0.7` | Fraction de nouveaux individus créés par crossover |
| `mutation_rate` | `0.2` | Probabilité d'appliquer une mutation à un enfant |
| `elite_count` | `2` | Nombre de meilleurs individus conservés directement |
| probabilité d'ajout initial | `0.25` | Probabilité d'ajouter une vidéo lors de l'initialisation |
| probabilité de choisir le parent 1 au crossover | `0.5` | Choix uniforme entre les deux parents |

Ces valeurs sont, pour l'instant, arbitraires. Elles ont été choisies pour obtenir une première version fonctionnelle de l'algorithme, mais elles n'ont pas encore été optimisées par expérimentation systématique. On pourrait comparer plusieurs valeurs en mesurant le score final, le temps d'exécution et la stabilité des résultats.

## 8. Exécution et sortie

Le point d'entrée est `../main.cpp`, qui lit l'instance puis appelle l'algorithme (lignes 69-71) :

```cpp
Solution sol = run_evol_algo(problemData);
```

Le solveur affiche ensuite le nombre de caches puis, pour chaque cache, la liste des vidéos placées :

```text
<nbr de caches>
<id cache 0> <vidéo ...>
<id cache 1> <vidéo ...>
...
```

La fonction `show_generation_summary` de `evolAlgo.cpp` affiche aussi, à chaque génération, un résumé de la forme :

```text
Generation <numéro> | Best saved: <meilleur score> | Avg saved: <score moyen>
```

`Best saved` est le meilleur temps économisé trouvé dans la génération et `Avg saved` est la moyenne de la génération. À la fin, la solution de la dernière génération est convertie au format des caches.

La commande prévue dans le projet est :

```bat
solver.exe < tests\generated\tiny.in > tests\generated\tiny.out
```

L'instance `tiny.in` est conçue pour atteindre le score attendu `80000` lorsque les deux vidéos sont placées dans l'unique cache. Comme l'algorithme est aléatoire, le résultat peut varier d'une exécution à l'autre. Dans cet environnement Windows, le fichier `exec` fourni est un exécutable Linux et n'a pas pu être lancé ; une exécution réelle peut être faite avec `solver.exe` après compilation CMake ou dans WSL.

## Résumé du flux

```text
Initialisation
    -> Évaluation
    -> Sélection par rang + élitisme
    -> Crossover uniforme
    -> Mutation bit-flip
    -> Évaluation de la nouvelle génération
    -> Répétition jusqu'à 50 générations
```

La logique spécifique au problème se trouve surtout dans `evolAlgo.cpp` et `../main.cpp`. La mécanique générique de sélection est actuellement fournie par `openGA.hpp`, mais elle reste accessible et modifiable dans le projet.
