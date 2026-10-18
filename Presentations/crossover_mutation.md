# Crossovers et mutations du `SequenceSolver`

Ce document décrit le fonctionnement de chaque mutation (et, plus brièvement, de chaque crossover) de [sequenceSolver.cpp](../sequenceSolver.cpp), puis compare les six mutations (section 4) et les sept crossovers (section 5) sur la moitié des instances de `../instances/promo`.

## 1. Contexte : ce qu'un opérateur modifie réellement

**Encodage.** Un individu (`VideoOrder`) est une permutation des vidéos candidates. Une vidéo est candidate si elle tient dans un cache et si au moins une requête peut gagner du temps grâce à elle.

**Décodage.** Le décodeur glouton (`GreedyDecoder::decode_static_greedy`) parcourt la permutation dans l'ordre. Pour chaque vidéo, il la place dans le cache qui rapporte le plus de temps gagné, s'il reste de la place. Il refait ensuite des passes sur les vidéos déjà placées pour leur trouver un second cache, jusqu'à ce que plus rien ne bouge.

**Conséquence : la fenêtre active.** Les vidéos du début de la permutation sont servies en premier et remplissent les caches. Une fois les caches pleins, les vidéos suivantes ne sont plus placées : les échanger entre elles ne change pas la solution, donc pas le score. Une mutation qui n'agit que dans cette queue est *neutre*.

**Estimation de la fenêtre.** Le code n'a pas la taille exacte de cette fenêtre. Il utilise `active_k = max(2, nb_candidats / 4)`, soit le premier quart de la permutation. C'est une approximation fixe, pas une mesure (voir la section 4).

**Quand la mutation s'applique.** Dans openGA, la mutation n'est appliquée qu'aux enfants issus d'un crossover, avec une probabilité `mutation_rate = 0.2`. Avec une population de 30 et `crossover_fraction = 0.7`, cela fait environ 21 enfants et donc 4 mutations par génération.

Dans les exemples ci-dessous, la permutation a 8 vidéos et `active_k = 4`. La barre `|` marque la limite de la fenêtre active :

```
A B C D | E F G H
```

## 2. Les mutations

Toutes les mutations copient l'individu puis modifient la copie. Aucune ne peut produire une permutation invalide.

| ID | Nom | Zone touchée | Change l'ensemble des vidéos actives ? | Coût de la modification | Coût total |
|---|---|---|---|---|---|
| 0 | `BASELINE_SWAP` | toute la permutation | parfois | O(1) | O(n) |
| 1 | `ACTIVE_PREFIX_SWAP` | fenêtre active | non | O(1) | O(n) |
| 2 | `ACTIVE_PREFIX_INVERSION` | fenêtre active | non | O(k) | O(n) |
| 3 | `ACTIVE_PREFIX_INSERTION` | fenêtre active | non | O(n) | O(n) |
| 4 | `BOUNDARY_SWAP` | une vidéo de la fenêtre, une vidéo juste après | oui, toujours | O(1) | O(n) |
| 5 | `PROMOTE_UNPLACED` | une vidéo non placée, remontée devant une vidéo placée | oui, c'est son but | O(n) | O(n) |

Les mutations 0 à 4 s'appuient sur l'estimation `active_k`. La mutation 5 s'appuie sur le résultat réel du décodeur.

**Complexité.** On note `n` le nombre de vidéos candidates (la longueur de la permutation) et `k = active_k`. Chaque mutation commence par copier l'individu, ce qui coûte O(n) en temps et en mémoire. Le coût total est donc O(n) pour toutes, quelle que soit la modification faite ensuite ; la colonne « coût de la modification » isole cette dernière. Dans tous les cas, ce coût est négligeable devant celui du décodage qui suit : le décodeur parcourt, pour chaque vidéo, toutes ses requêtes et tous les caches reliés aux endpoints concernés, et il le fait sur plusieurs passes.

### 2.0 `BASELINE_SWAP` - échange uniforme

1. Tirer deux positions `i` et `j` uniformément dans `[0, n)`.
2. Échanger les deux vidéos.

```
A B C D | E F G H      i = 1, j = 6
A G C D | E F B H
```

C'est la mutation de référence : elle ignore la fenêtre active. Avec `active_k = n/4`, les deux positions tombent dans la queue 9 fois sur 16, et dans ce cas l'échange ne concerne que des vidéos au-delà de la fenêtre estimée. Dans les 6 cas sur 16 où une position est dans la fenêtre et l'autre dehors, elle fait entrer une vidéo de la queue, éventuellement tirée de très loin. Si `i == j`, rien ne change.

**Complexité :** O(1) pour l'échange (deux tirages, un `swap`), O(n) avec la copie.

### 2.1 `ACTIVE_PREFIX_SWAP` - échange dans la fenêtre

1. Tirer `i` et `j` uniformément dans `[0, active_k)`.
2. Échanger les deux vidéos.

```
A B C D | E F G H      i = 0, j = 2
C B A D | E F G H
```

Les deux vidéos restent dans la fenêtre : seul leur ordre de passage change. L'effet est donc indirect : la vidéo qui passe en premier choisit son cache avant l'autre. Si les deux vidéos ne se disputent aucun cache, l'échange est neutre. Si `i == j` (probabilité `1/active_k`), rien ne change.

**Complexité :** O(1) pour l'échange, O(n) avec la copie.

### 2.2 `ACTIVE_PREFIX_INVERSION` - inversion d'un segment

1. Tirer `a` et `b` dans `[0, active_k)`, avec `a <= b`.
2. Renverser l'ordre des vidéos entre `a` et `b` inclus.

```
A B C D | E F G H      a = 0, b = 2
C B A D | E F G H

A B C D | E F G H      a = 1, b = 3
A D C B | E F G H
```

C'est la plus grosse perturbation des trois mutations « dans la fenêtre » : elle déplace jusqu'à `active_k` vidéos d'un coup, contre deux pour un échange. Elle ne fait entrer aucune vidéo de la queue. Si `a == b`, rien ne change.

**Complexité :** O(b - a) pour le renversement, soit O(k) au pire et environ k/3 vidéos déplacées en moyenne ; O(n) avec la copie.

### 2.3 `ACTIVE_PREFIX_INSERTION` - déplacement d'une vidéo

1. Tirer `from` et `to` dans `[0, active_k)`.
2. Retirer la vidéo en position `from`.
3. La réinsérer en position `to` ; les vidéos intermédiaires se décalent d'un cran.

```
A B C D | E F G H      from = 3, to = 0
D A B C | E F G H
```

Une seule vidéo change de rang, les autres gardent leur ordre relatif. C'est la perturbation la plus fine : elle revient à « promouvoir » ou « rétrograder » une vidéo. Elle ne fait entrer aucune vidéo de la queue. Si `from == to`, rien ne change.

**Complexité :** O(n) pour le déplacement. Le `erase` décale toutes les vidéos situées après `from` et le `insert` toutes celles situées après `to`, queue comprise : comme `from` et `to` sont dans le premier quart, chaque opération déplace au moins les trois quarts du vecteur. C'est la seule mutation fondée sur `active_k` dont la modification elle-même est linéaire. Un `std::rotate` entre `from` et `to` ferait le même déplacement en O(|from - to|), donc O(k).

### 2.4 `BOUNDARY_SWAP` - échange à la frontière

1. Poser `L = min(n - 1, active_k)`, la limite de la fenêtre.
2. Tirer `i` dans `[0, L)` : une vidéo active.
3. Tirer `j` dans `[L, L + min(active_k, n - L))` : une vidéo dans la bande située juste après la fenêtre, de même largeur qu'elle.
4. Échanger les deux vidéos.

```
A B C D | E F G H      i = 2, j = 5
A B F D | E C G H
```

C'est la seule mutation qui modifie à coup sûr l'ensemble des vidéos de la fenêtre : une vidéo sort, une autre entre. La vidéo entrante vient de la bande voisine et non du fond de la queue, donc d'une zone où les vidéos ont encore une priorité correcte. Elle ne réordonne jamais deux vidéos déjà actives.

**Complexité :** O(1) pour l'échange, O(n) avec la copie.

### 2.5 `PROMOTE_UNPLACED` - remontée d'une vidéo non placée

Cette mutation n'utilise pas `active_k`. Elle demande au décodeur quelles vidéos il a réellement placées dans au moins un cache (`GreedyDecoder::last_placed()`), ce qui donne la vraie frontière de l'individu.

1. Décoder l'individu et relever, pour chaque position, si la vidéo est placée ou non.
2. Tirer une vidéo non placée parmi les premières de la permutation : on ne considère qu'autant de vidéos non placées qu'il y a de vidéos placées.
3. Tirer une vidéo placée située avant elle.
4. Déplacer la vidéo non placée juste devant cette vidéo placée ; les vidéos intermédiaires se décalent d'un cran vers la droite.

Ici les vidéos placées sont notées en majuscules et les non placées en minuscules :

```
A B c D E f g h      vidéo tirée : f, cible : B
A f B c D E g h
```

`f` passe maintenant avant `B`, `D` et `E` : elle trouve de la place, et c'est une vidéo servie plus tard qui en manquera. Contrairement à `BOUNDARY_SWAP`, aucune vidéo active n'est envoyée dans la queue : les vidéos déjà placées reculent d'un seul rang. Une vidéo non placée peut aussi se trouver au milieu de la permutation (comme `c`), ce que les mutations fondées sur `active_k` ne voient pas.

Si toutes les vidéos sont placées, ou aucune, la mutation se rabat sur `BASELINE_SWAP`. Le décodage nécessaire était déjà fait pour mesurer le taux de succès : il n'y a pas d'évaluation supplémentaire.

**Complexité :** O(n) pour la modification. Le parcours qui sépare les positions placées et non placées est linéaire, la recherche des vidéos placées situées avant la vidéo tirée est une recherche dichotomique en O(log n), et le déplacement (`std::rotate`) coûte la distance parcourue, au pire O(n). Elle utilise aussi O(n) de mémoire en plus de la copie, pour les deux listes de positions.

## 3. Les crossovers (résumé)

| ID | Nom | Principe | Utilise `active_k` |
|---|---|---|---|
| 0 | `BASELINE_OX1` | Copie un segment `[a, b]` de p1 tiré sur toute la permutation, complète avec les autres vidéos dans l'ordre de p2. | non |
| 1 | `ACTIVE_PREFIX_OX1` | Comme OX1, mais le segment est tiré dans la fenêtre active. | oui |
| 2 | `POSITION_BASED_POS` | Chaque position garde la vidéo de p1 avec probabilité 0,4 ; les trous sont remplis dans l'ordre de p2. | non |
| 3 | `PARTIALLY_MAPPED_PMX` | Part de p1 et impose, par échanges, les vidéos de p2 sur un segment de la fenêtre. | oui |
| 4 | `CYCLE_CROSSOVER_CX` | Copie depuis p1 le cycle qui part de la position 0, le reste vient de p2. Déterministe. | non |
| 5 | `ACTIVE_PREFIX_PPX` | Pour chaque case de la fenêtre, tire un parent au hasard et prend sa première vidéo non utilisée ; la queue suit l'ordre de p2. | oui |
| 6 | `ACTIVE_SET_CROSSOVER` | Garde toutes les vidéos actives chez les deux parents, tire le reste parmi celles actives chez un seul ; la queue suit l'ordre de p2. | oui |

## 4. Comparaison des mutations

### Protocole

- **Instances** : 14 des 29 instances de `../instances/promo`. Les cinq instances jouets (20 vidéos ou moins) sont exclues, et le reste est choisi pour couvrir chaque famille (denses, creuses, zipf, régionales, réalistes, grandes capacités).
- **Configuration fixe** : init 2 (`NOISY_PRIORITY_ADD`), crossover 1 (`ACTIVE_PREFIX_OX1`), pas de recherche locale, décodeur 0 (`STATIC_GREEDY`). La recherche locale est coupée pour isoler l'effet de la mutation.
- **Paramètres du GA** : ceux du code (population 30, 50 générations max, `mutation_rate` 0,2, arrêt après 10 générations sans progrès).
- **Graines** : 5 par couple instance × mutation (graines 1 à 5), soit 420 runs. Pour une graine donnée, la population initiale est la même quelle que soit la mutation.
- **Mesure** : gain du meilleur score final par rapport au meilleur individu de la génération 0, en %, moyenné sur les 5 graines.

Pour relancer un run : `solver <best_known> <nom> <init> <cx> <mut> <ls> <dec> [graine]`, l'instance sur l'entrée standard. Le dernier argument est optionnel et vaut 42 par défaut.

### Synthèse sur les 14 instances

| Mutation | Gain moyen (%) | Rang moyen | Victoires | Mutations améliorantes (%) | Mutations neutres (%) |
|---|---|---|---|---|---|
| 0 `BASELINE_SWAP` | 0,808 | 3,07 | 1,8 | 10,1 | 54,1 |
| 1 `ACTIVE_PREFIX_SWAP` | 0,943 | 3,64 | 2,2 | 9,1 | 65,0 |
| 2 `ACTIVE_PREFIX_INVERSION` | **1,320** | 3,50 | **3,0** | 13,0 | 37,7 |
| 3 `ACTIVE_PREFIX_INSERTION` | 0,862 | 3,71 | 1,2 | 7,7 | 71,8 |
| 4 `BOUNDARY_SWAP` | 0,953 | 3,43 | **3,0** | 12,3 | 49,3 |
| 5 `PROMOTE_UNPLACED` | 1,069 | **2,93** | 2,8 | **20,0** | **20,6** |

- *Rang moyen* : rang de la mutation sur chaque instance (1 = meilleur gain), moyenné. Un classement au hasard donnerait 3,5.
- *Victoires* : nombre d'instances où la mutation a le meilleur gain ; une égalité est partagée entre les ex æquo.
- *Améliorantes / neutres* : part des mutations dont le résultat est strictement meilleur que / égal à l'individu avant mutation.

### Gain par instance (% au-dessus du meilleur de la génération 0)

| Instance | `active_k` | Meilleur gén. 0 | 0 `BASELINE_SWAP` | 1 `PREFIX_SWAP` | 2 `PREFIX_INVERSION` | 3 `PREFIX_INSERTION` | 4 `BOUNDARY_SWAP` | 5 `PROMOTE_UNPLACED` |
|---|---|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 965 | 1 972 980 | **0,001** | **0,001** | 0,000 | **0,001** | 0,000 | **0,001** |
| `6970_311_100000_69` | 1742 | 1 725 950 | **0,018** | 0,004 | 0,005 | 0,006 | 0,004 | **0,018** |
| `big` | 1722 | 249 794 | 0,163 | 0,199 | **1,133** | 0,203 | 0,188 | 0,540 |
| `custom_universallambda42` | 2500 | 946 459 | 0,003 | **0,003** | 0,002 | 0,002 | 0,003 | 0,002 |
| `dense` | 2499 | 52 949 | 4,854 | 7,594 | **10,859** | 6,240 | 6,523 | 7,262 |
| `instance1` | 1691 | 1 429 440 | 0,014 | 0,008 | 0,005 | 0,010 | 0,006 | **0,026** |
| `INSTANCED` | 150 | 1 163 020 | 0,461 | 0,449 | 0,467 | 0,445 | **0,495** | 0,432 |
| `medium_flat_sparse_lambda` | 750 | 16 656 | 1,358 | 0,997 | **1,750** | 1,078 | 1,087 | 1,332 |
| `medium_mixed_lambda` | 750 | 44 903 | 0,050 | **0,274** | 0,042 | 0,261 | 0,024 | 0,146 |
| `medium_skew_dense_dejavu` | 731 | 156 528 | 0,003 | 0,013 | 0,017 | 0,000 | **0,138** | 0,000 |
| `realistic_large_clustered` | 2427 | 684 882 | 0,781 | 0,823 | 0,861 | **0,867** | 0,829 | 0,781 |
| `regional_instance` | 103 | 2 240 470 | **1,324** | 0,773 | 0,794 | 1,047 | 1,227 | 1,164 |
| `supra` | 342 | 369 631 | 1,352 | 1,274 | 1,857 | 1,115 | 1,703 | **2,376** |
| `zipF_instance` | 84 | 1 761 520 | 0,929 | 0,789 | 0,685 | 0,790 | **1,112** | 0,890 |

### Meilleur score final (moyenne des 5 graines)

| Instance | 0 `BASELINE_SWAP` | 1 `PREFIX_SWAP` | 2 `PREFIX_INVERSION` | 3 `PREFIX_INSERTION` | 4 `BOUNDARY_SWAP` | 5 `PROMOTE_UNPLACED` |
|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 1 972 990 | 1 972 990 | 1 972 984 | 1 972 990 | 1 972 986 | 1 972 990 |
| `6970_311_100000_69` | 1 726 260 | 1 726 026 | 1 726 042 | 1 726 050 | 1 726 014 | 1 726 260 |
| `big` | 250 200 | 250 291 | 252 624 | 250 302 | 250 265 | 251 142 |
| `custom_universallambda42` | 946 484 | 946 485 | 946 480 | 946 482 | 946 484 | 946 482 |
| `dense` | 55 519 | 56 971 | 58 697 | 56 253 | 56 402 | 56 794 |
| `instance1` | 1 429 634 | 1 429 548 | 1 429 518 | 1 429 582 | 1 429 522 | 1 429 814 |
| `INSTANCED` | 1 168 386 | 1 168 240 | 1 168 446 | 1 168 194 | 1 168 778 | 1 168 042 |
| `medium_flat_sparse_lambda` | 16 882 | 16 822 | 16 947 | 16 836 | 16 837 | 16 878 |
| `medium_mixed_lambda` | 44 926 | 45 026 | 44 922 | 45 020 | 44 914 | 44 969 |
| `medium_skew_dense_dejavu` | 156 533 | 156 548 | 156 554 | 156 528 | 156 744 | 156 528 |
| `realistic_large_clustered` | 690 230 | 690 518 | 690 780 | 690 821 | 690 562 | 690 230 |
| `regional_instance` | 2 270 142 | 2 257 782 | 2 258 256 | 2 263 924 | 2 267 962 | 2 266 558 |
| `supra` | 374 629 | 374 340 | 376 495 | 373 753 | 375 927 | 378 412 |
| `zipF_instance` | 1 777 886 | 1 775 422 | 1 773 578 | 1 775 430 | 1 781 112 | 1 777 196 |

### Mutations améliorantes / neutres par instance (en %)

| Instance | 0 `BASELINE_SWAP` | 1 `PREFIX_SWAP` | 2 `PREFIX_INVERSION` | 3 `PREFIX_INSERTION` | 4 `BOUNDARY_SWAP` | 5 `PROMOTE_UNPLACED` |
|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 17 / 32 | 10 / 65 | 19 / 19 | 8 / 73 | 19 / 42 | 17 / 32 |
| `6970_311_100000_69` | 21 / 33 | 15 / 65 | 25 / 14 | 10 / 78 | 22 / 46 | 21 / 33 |
| `big` | 2 / 94 | 0 / 96 | 12 / 37 | 0 / 98 | 1 / 96 | 18 / 57 |
| `custom_universallambda42` | 3 / 40 | 0 / 100 | 0 / 100 | 0 / 100 | 8 / 26 | 9 / 13 |
| `dense` | 3 / 77 | 14 / 42 | 9 / 16 | 13 / 51 | 8 / 50 | 17 / 25 |
| `instance1` | 13 / 56 | 2 / 95 | 0 / 95 | 1 / 97 | 0 / 99 | 31 / 0 |
| `INSTANCED` | 13 / 17 | 30 / 9 | 24 / 9 | 27 / 18 | 34 / 0 | 20 / 1 |
| `medium_flat_sparse_lambda` | 2 / 69 | 17 / 15 | 7 / 14 | 15 / 22 | 10 / 34 | 21 / 1 |
| `medium_mixed_lambda` | 3 / 73 | 13 / 26 | 11 / 19 | 16 / 34 | 4 / 46 | 20 / 1 |
| `medium_skew_dense_dejavu` | 12 / 52 | 6 / 83 | 10 / 62 | 6 / 87 | 18 / 5 | 35 / 1 |
| `realistic_large_clustered` | 23 / 26 | 12 / 54 | 21 / 8 | 9 / 67 | 20 / 36 | 23 / 26 |
| `regional_instance` | 12 / 53 | 2 / 91 | 14 / 49 | 2 / 95 | 10 / 68 | 17 / 21 |
| `supra` | 5 / 81 | 3 / 82 | 16 / 37 | 1 / 93 | 8 / 77 | 16 / 44 |
| `zipF_instance` | 11 / 55 | 3 / 87 | 13 / 51 | 2 / 93 | 10 / 66 | 13 / 32 |

### Lecture des résultats

- **Aucune mutation ne domine.** Les rangs moyens vont de 2,93 à 3,71, tous proches de 3,5, et chaque mutation gagne sur au moins une instance.
- **`PROMOTE_UNPLACED` est la mutation la plus efficace par application** : 20 % de mutations améliorantes et 21 % de neutres, contre 8 à 13 % et 38 à 72 % pour les autres. Elle a aussi le meilleur rang moyen. Cet avantage se traduit peu dans le score final : elle est deuxième en gain moyen et ne gagne nettement que sur `supra` et `instance1`.
- **Sur trois instances, `PROMOTE_UNPLACED` donne exactement les résultats de `BASELINE_SWAP`** (`3860_246_84901_66`, `6970_311_100000_69`, `realistic_large_clustered`). Des résultats identiques graine par graine indiquent que la mutation s'est rabattue sur l'échange uniforme, ce qu'elle fait quand toutes les vidéos candidates sont placées. Sur ces instances, il n'y a donc pas de queue inactive.
- **`ACTIVE_PREFIX_INVERSION` garde le meilleur gain moyen**, mais cette avance vient surtout de deux instances, `dense` (+10,9 % contre 4,9 à 7,6 %) et `big` (+1,13 % contre 0,54 % au mieux).
- **L'échange et l'insertion dans la fenêtre sont plus souvent neutres que la mutation de référence** : 65 % et 72 %, contre 54 % pour `BASELINE_SWAP`. Déplacer une ou deux vidéos à l'intérieur de la fenêtre change rarement le résultat du décodeur.
- **La fenêtre `n/4` ne correspond pas toujours à la zone sensible.** Sur `custom_universallambda42`, les trois mutations internes à la fenêtre sont neutres à 100 %, alors que `BOUNDARY_SWAP` ne l'est qu'à 26 % et `PROMOTE_UNPLACED` à 13 %. Sur `instance1`, les quatre mutations fondées sur `active_k` sont neutres à 95 % ou plus, alors que `PROMOTE_UNPLACED` ne l'est jamais.
- **`BOUNDARY_SWAP` gagne sur trois instances** (`INSTANCED`, `zipF_instance`, `medium_skew_dense_dejavu`). Sur `medium_skew_dense_dejavu`, c'est la seule mutation qui fait progresser le GA de façon visible (+0,14 % contre moins de 0,02 %) ; `PROMOTE_UNPLACED` y produit 35 % de mutations améliorantes mais aucun gain final.
- **Sur quatre instances, le GA ne progresse presque pas** quelle que soit la mutation (`3860_246_84901_66`, `6970_311_100000_69`, `custom_universallambda42`, `instance1` : gain inférieur à 0,03 %). Le glouton de départ y est déjà très bon, et ces instances ne départagent rien.

### Limites

- Cinq graines par cellule et pas de test statistique : les écarts de quelques centièmes de pour cent ne sont pas significatifs.
- La mutation ne s'applique qu'à environ 4 enfants par génération, et les runs s'arrêtent tôt (600 à 730 évaluations en moyenne, soit une trentaine de générations). Chaque run ne contient donc qu'un peu plus d'une centaine de mutations.
- Les gains sont mesurés par rapport à la génération 0, pas par rapport à la meilleure valeur connue.
- Une seule configuration des autres blocs est testée : le classement peut changer avec un autre crossover ou avec la recherche locale activée.
- `PROMOTE_UNPLACED` lit les vidéos placées par le décodeur glouton standard. Avec un autre décodeur (IDs 1 à 4), cette information ne correspond pas exactement à la solution évaluée ; ce cas n'a pas été testé.

## 5. Comparaison des crossovers

### Protocole

Le protocole est celui de la section 4, en faisant varier le crossover au lieu de la mutation.

- **Instances** : les mêmes 14 instances de `../instances/promo`.
- **Configuration fixe** : init 2 (`NOISY_PRIORITY_ADD`), mutation 2 (`ACTIVE_PREFIX_INVERSION`, celle de la configuration par défaut de `../Utils/benchmark-Sweep1.py`), pas de recherche locale, décodeur 0 (`STATIC_GREEDY`).
- **Paramètres du GA** : ceux du code, inchangés.
- **Graines** : 5 par couple instance × crossover (graines 1 à 5), soit 490 runs. Les 70 runs du crossover 1 sont ceux de la colonne `PREFIX_INVERSION` de la section 4.
- **Mesure** : gain du meilleur score final par rapport au meilleur individu de la génération 0, en %, moyenné sur les 5 graines.

Les noms sont abrégés dans les tableaux par instance : `OX1` = `BASELINE_OX1`, `PREFIX_OX1` = `ACTIVE_PREFIX_OX1`, `POS` = `POSITION_BASED_POS`, `PMX` = `PARTIALLY_MAPPED_PMX`, `CX` = `CYCLE_CROSSOVER_CX`, `PREFIX_PPX` = `ACTIVE_PREFIX_PPX`, `ACTIVE_SET` = `ACTIVE_SET_CROSSOVER`.

### Synthèse sur les 14 instances

| Crossover | Gain moyen (%) | Rang moyen | Victoires | Crossovers améliorants (%) | Crossovers neutres (%) | Évaluations par run | Runs qui battent la gén. 0 (%) |
|---|---|---|---|---|---|---|---|
| 0 `BASELINE_OX1` | 1,022 | 2,71 | **4,0** | 5,9 | 45,6 | 652 | 76 |
| 1 `ACTIVE_PREFIX_OX1` | **1,320** | **2,43** | **4,0** | 8,8 | 44,4 | 606 | **79** |
| 2 `POSITION_BASED_POS` | 0,889 | 4,14 | 2,0 | 7,3 | 19,8 | 549 | 47 |
| 3 `PARTIALLY_MAPPED_PMX` | 1,162 | 3,00 | 2,0 | 3,8 | 45,3 | 649 | 74 |
| 4 `CYCLE_CROSSOVER_CX` | 1,146 | 4,07 | 2,0 | 1,4 | 89,2 | 514 | 64 |
| 5 `ACTIVE_PREFIX_PPX` | 0,500 | 5,93 | 0,0 | **19,7** | 22,6 | 384 | 51 |
| 6 `ACTIVE_SET_CROSSOVER` | 0,845 | 4,21 | 0,0 | 12,1 | 24,1 | 499 | 59 |

- *Rang moyen* : rang du crossover sur chaque instance (1 = meilleur gain), moyenné. Un classement au hasard donnerait 4.
- *Victoires* : nombre d'instances où le crossover a le meilleur gain.
- *Améliorants* : part des enfants strictement meilleurs que le meilleur de leurs deux parents.
- *Neutres* : part des enfants dont le score est égal à celui d'un des deux parents.
- *Évaluations par run* : un run s'arrête après 10 générations sans progrès, donc un nombre faible signifie que le GA a calé tôt.

### Gain par instance (% au-dessus du meilleur de la génération 0)

| Instance | Meilleur gén. 0 | 0 `OX1` | 1 `PREFIX_OX1` | 2 `POS` | 3 `PMX` | 4 `CX` | 5 `PREFIX_PPX` | 6 `ACTIVE_SET` |
|---|---|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 1 972 980 | **0,001** | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| `6970_311_100000_69` | 1 725 950 | 0,005 | 0,005 | 0,000 | 0,004 | **0,013** | 0,000 | 0,000 |
| `big` | 249 794 | **1,206** | 1,133 | 0,618 | 1,094 | 1,151 | 0,555 | 1,081 |
| `custom_universallambda42` | 946 459 | 0,002 | 0,002 | 0,000 | **0,007** | 0,000 | 0,002 | 0,003 |
| `dense` | 52 949 | 6,548 | 10,859 | 6,121 | 9,191 | **11,165** | 3,940 | 5,896 |
| `instance1` | 1 429 440 | 0,000 | **0,005** | 0,000 | 0,001 | 0,000 | 0,000 | 0,000 |
| `INSTANCED` | 1 163 020 | 0,428 | **0,467** | 0,399 | 0,442 | 0,327 | 0,232 | 0,386 |
| `medium_flat_sparse_lambda` | 16 656 | 0,988 | **1,750** | 0,999 | 1,216 | 0,537 | 0,000 | 0,279 |
| `medium_mixed_lambda` | 44 903 | 0,172 | 0,042 | 0,000 | **0,186** | 0,060 | 0,000 | 0,000 |
| `medium_skew_dense_dejavu` | 156 528 | **0,069** | 0,017 | 0,000 | 0,002 | 0,002 | 0,000 | 0,000 |
| `realistic_large_clustered` | 684 882 | 0,465 | **0,861** | 0,137 | 0,617 | 0,825 | 0,034 | 0,476 |
| `regional_instance` | 2 240 470 | 1,503 | 0,794 | **1,547** | 1,047 | 0,165 | 0,514 | 1,222 |
| `supra` | 369 631 | **2,165** | 1,857 | 1,775 | 1,693 | 1,714 | 1,131 | 1,706 |
| `zipF_instance` | 1 761 520 | 0,754 | 0,685 | **0,848** | 0,759 | 0,091 | 0,586 | 0,774 |

### Meilleur score final (moyenne des 5 graines)

| Instance | 0 `OX1` | 1 `PREFIX_OX1` | 2 `POS` | 3 `PMX` | 4 `CX` | 5 `PREFIX_PPX` | 6 `ACTIVE_SET` |
|---|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 1 972 990 | 1 972 984 | 1 972 982 | 1 972 982 | 1 972 980 | 1 972 980 | 1 972 984 |
| `6970_311_100000_69` | 1 726 042 | 1 726 042 | 1 725 950 | 1 726 026 | 1 726 174 | 1 725 950 | 1 725 950 |
| `big` | 252 806 | 252 624 | 251 338 | 252 528 | 252 669 | 251 180 | 252 495 |
| `custom_universallambda42` | 946 474 | 946 480 | 946 459 | 946 527 | 946 459 | 946 478 | 946 485 |
| `dense` | 56 416 | 58 697 | 56 190 | 57 814 | 58 859 | 55 035 | 56 072 |
| `instance1` | 1 429 446 | 1 429 518 | 1 429 440 | 1 429 454 | 1 429 440 | 1 429 440 | 1 429 440 |
| `INSTANCED` | 1 168 002 | 1 168 446 | 1 167 666 | 1 168 166 | 1 166 824 | 1 165 720 | 1 167 512 |
| `medium_flat_sparse_lambda` | 16 821 | 16 947 | 16 822 | 16 859 | 16 745 | 16 656 | 16 702 |
| `medium_mixed_lambda` | 44 980 | 44 922 | 44 903 | 44 986 | 44 930 | 44 903 | 44 903 |
| `medium_skew_dense_dejavu` | 156 637 | 156 554 | 156 528 | 156 532 | 156 532 | 156 528 | 156 528 |
| `realistic_large_clustered` | 688 069 | 690 780 | 685 823 | 689 109 | 690 535 | 685 116 | 688 144 |
| `regional_instance` | 2 274 146 | 2 258 256 | 2 275 136 | 2 263 934 | 2 244 160 | 2 251 984 | 2 267 842 |
| `supra` | 377 635 | 376 495 | 376 194 | 375 889 | 375 967 | 373 811 | 375 937 |
| `zipF_instance` | 1 774 806 | 1 773 578 | 1 776 458 | 1 774 890 | 1 763 122 | 1 771 840 | 1 775 156 |

### Crossovers améliorants / neutres par instance (en %)

| Instance | 0 `OX1` | 1 `PREFIX_OX1` | 2 `POS` | 3 `PMX` | 4 `CX` | 5 `PREFIX_PPX` | 6 `ACTIVE_SET` |
|---|---|---|---|---|---|---|---|
| `3860_246_84901_66` | 4 / 44 | 8 / 35 | 8 / 20 | 3 / 32 | 0 / 100 | 24 / 13 | 10 / 20 |
| `6970_311_100000_69` | 6 / 21 | 13 / 24 | 8 / 11 | 3 / 32 | 0 / 100 | 24 / 14 | 14 / 14 |
| `big` | 5 / 61 | 6 / 51 | 7 / 29 | 6 / 55 | 0 / 99 | 10 / 21 | 7 / 52 |
| `custom_universallambda42` | 6 / 42 | 14 / 41 | 9 / 13 | 7 / 50 | 9 / 33 | 18 / 39 | 14 / 35 |
| `dense` | 9 / 52 | 6 / 52 | 9 / 18 | 5 / 51 | 0 / 100 | 25 / 21 | 9 / 20 |
| `instance1` | 4 / 28 | 13 / 40 | 6 / 15 | 2 / 34 | 0 / 100 | 19 / 14 | 21 / 15 |
| `INSTANCED` | 4 / 53 | 8 / 39 | 5 / 26 | 4 / 38 | 5 / 62 | 9 / 18 | 12 / 23 |
| `medium_flat_sparse_lambda` | 7 / 52 | 3 / 68 | 5 / 23 | 2 / 51 | 5 / 67 | 34 / 14 | 8 / 20 |
| `medium_mixed_lambda` | 7 / 50 | 6 / 32 | 8 / 14 | 2 / 46 | 0 / 100 | 34 / 10 | 7 / 11 |
| `medium_skew_dense_dejavu` | 7 / 44 | 19 / 33 | 9 / 12 | 1 / 40 | 0 / 100 | 31 / 14 | 30 / 10 |
| `realistic_large_clustered` | 4 / 30 | 8 / 45 | 6 / 14 | 2 / 48 | 0 / 100 | 24 / 17 | 14 / 17 |
| `regional_instance` | 6 / 48 | 7 / 55 | 8 / 28 | 5 / 50 | 0 / 100 | 6 / 39 | 9 / 31 |
| `supra` | 6 / 56 | 7 / 50 | 9 / 29 | 6 / 53 | 1 / 87 | 9 / 38 | 8 / 40 |
| `zipF_instance` | 6 / 56 | 5 / 58 | 8 / 28 | 6 / 53 | 0 / 100 | 8 / 43 | 7 / 30 |

### Lecture des résultats

- **Contrairement aux mutations, les crossovers se départagent.** Les rangs moyens vont de 2,43 à 5,93 pour une valeur au hasard de 4.
- **`ACTIVE_PREFIX_OX1` est le meilleur choix** : meilleur rang moyen, meilleur gain moyen, et c'est avec lui que le plus de runs dépassent la génération 0 (79 %). `BASELINE_OX1` et `PARTIALLY_MAPPED_PMX` le suivent de près ; l'écart entre ces trois-là n'est pas assez grand pour être sûr de l'ordre.
- **`ACTIVE_PREFIX_PPX` est le plus mauvais** : rang moyen 5,93 sur 7, aucune victoire, strictement dernier sur six instances, et sur `medium_flat_sparse_lambda` il n'améliore jamais la génération 0. Ses runs s'arrêtent aussi le plus tôt (384 évaluations en moyenne contre 600 à 650 pour les trois meilleurs).
- **`ACTIVE_SET_CROSSOVER` est dans la moitié basse**, sans aucune victoire. Les deux crossovers ajoutés ne sont donc pas à retenir tels quels.
- **Le taux de crossovers améliorants ne prédit pas le score final.** `ACTIVE_PREFIX_PPX` a le taux le plus élevé (19,7 %) et le plus mauvais résultat ; `ACTIVE_SET_CROSSOVER` est deuxième sur ce taux et avant-dernier au rang. Ce taux compare l'enfant à ses deux parents, pas au meilleur individu de la population. Hypothèse non vérifiée : ces deux crossovers ramènent des parents bruités vers un ordre « consensus » meilleur qu'eux, mais sans jamais dépasser l'élite, et ils uniformisent la population, ce qui déclenche l'arrêt plus tôt.
- **`CYCLE_CROSSOVER_CX` ne recombine presque rien** : 89 % de ses enfants ont le score d'un parent, et 100 % sur neuf instances. Avec lui, le GA se réduit à peu près à la sélection et à la mutation. Il obtient pourtant le meilleur gain sur `dense` (+11,2 %), au niveau de `ACTIVE_PREFIX_OX1` (+10,9 %). Sur cette instance, le progrès vient donc de la mutation et non du crossover, ce qui recoupe la section 4, où `dense` est l'instance qui sépare le plus les mutations. En contrepartie, il est dernier sur `regional_instance` et `zipF_instance`.
- **`POSITION_BASED_POS` est le plus perturbateur des crossovers existants** (20 % de neutres seulement) et moins d'un run sur deux dépasse la génération 0. Il gagne quand même sur les deux petites instances `regional_instance` et `zipF_instance`.
- **Les quatre mêmes instances ne départagent rien** (`3860_246_84901_66`, `6970_311_100000_69`, `custom_universallambda42`, `instance1` : gain inférieur à 0,02 % pour tous les crossovers).

### Limites

- Mêmes réserves que pour la section 4 : cinq graines par cellule, pas de test statistique, gains mesurés par rapport à la génération 0.
- La mutation est fixée à `ACTIVE_PREFIX_INVERSION`. Le classement peut changer avec une autre mutation, en particulier avec `PROMOTE_UNPLACED`, qui n'a pas été croisée avec les crossovers.
- Le critère d'arrêt pénalise les crossovers qui font converger la population rapidement. Avec un budget d'évaluations fixe au lieu d'un arrêt sur stagnation, l'écart pourrait être différent.
