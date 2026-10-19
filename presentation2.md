Voici vos notes complétées à partir du code ([sequenceSolver.cpp](sequenceSolver.cpp), [benchmark.py](benchmark.py)). Quatre points de vos notes ne collent pas au code et sont à corriger avant de présenter :

- **POS n'utilise pas la fenêtre active.** Les crossovers à fenêtre sont PMX, ACTIVE_PPX, ACTIVE_OX1 et ACTIVE_SET ; « ceux de droite sont basés sur la fenêtre » est donc faux pour la dernière barre.
- **Le benchmark balaie 5 blocs, pas 4** : initialisation, crossover, mutation, recherche locale, décodage.
- **Graph repair ne prend pas « la pire partie »** : il prend la meilleure vidéo non placée.
- **Les 7 groupes d'instances ne sont définis nulle part dans le dépôt** (j'ai bien 33 instances dans `instances/promo`), donc je n'ai pas pu les détailler.

## Slide 1 : Crossover

### Lire le graphique

- **Axe de gauche (barres, total 100 %)** : chaque crossover produit un enfant, classé dans une seule des trois catégories.
  - **Bleu (bénéfique)** : l'enfant est strictement meilleur que le meilleur de ses deux parents.
  - **Gris (neutre)** : l'enfant a exactement le score d'un de ses parents. C'est le même score, pas forcément la même permutation.
  - **Rouge (délétère)** : tous les autres cas, donc en pratique un enfant moins bon que le meilleur parent.
- **Axe de droite (courbe)** : pour les enfants rouges uniquement, la perte moyenne de fitness par rapport au meilleur parent. Il mesure la gravité d'une détérioration, pas sa fréquence.
- Les deux axes se lisent ensemble : le rouge dit « à quelle fréquence ça casse », la courbe dit « à quel point ça casse ».

### La fenêtre active

- **Encodage** : un individu est une permutation des vidéos candidates.
- **Décodage** : le décodeur glouton lit la permutation dans l'ordre et place chaque vidéo dans le cache qui rapporte le plus, tant qu'il reste de la place.
- **Conséquence** : une fois les caches pleins, les vidéos suivantes ne sont plus placées. Seul le début de la permutation décide de la solution : c'est la fenêtre active. Modifier la queue ne change pas le score.
- **Taille** : elle est estimée, pas mesurée. `active_k` vaut entre 5 % et 40 % des candidates, selon le rapport capacité totale des caches / taille totale des vidéos (plus les caches sont grands, plus la fenêtre est large).
- **Pourquoi** : chaque évaluation coûte un décodage complet et le budget est en temps. Un opérateur qui travaille dans la queue gaspille des évaluations ; un opérateur qui y pioche au hasard fait entrer des vidéos peu utiles.

### Les crossovers

| Crossover | Principe | Fenêtre active |
|---|---|---|
| BASELINE (OX1) | Copie un segment de p1 tiré sur toute la permutation, complète dans l'ordre de p2 | non |
| CX | Copie un cycle de positions depuis p1, le reste vient de p2 | non |
| POS | Chaque position garde la vidéo de p1 avec probabilité 0,4, les trous sont remplis dans l'ordre de p2 | non |
| PMX | Part de p1 et impose les vidéos de p2 sur un segment de la fenêtre | oui |
| ACTIVE_OX1 | Comme la baseline, mais le segment est tiré dans la fenêtre | oui |
| ACTIVE_SET | Garde les vidéos actives chez les deux parents, tire le reste parmi celles actives chez un seul | oui |
| ACTIVE_PPX | Voir ci-dessous | oui |

**Active PPX (Precedence Preservative Crossover).** Pour chaque case de la fenêtre, on tire un parent à pile ou face et on prend sa première vidéo pas encore utilisée. La queue est remplie dans l'ordre de p2.
- Une vidéo en tête chez un parent reste en tête chez l'enfant : la fenêtre de l'enfant est un entrelacement des têtes des deux parents.
- Aucune vidéo ne remonte du fond de la queue.
- Coût O(V) : un passage sur la fenêtre, un passage sur p2 pour la queue.

### Ce que montre le graphique

- **Le gris domine partout (environ 52 à 75 %)** : le crossover a peu d'influence, la majorité des enfants valent un parent. CX est l'extrême, il ne recombine presque rien.
- **Le bleu est faible partout (environ 5 à 7 %)** : aucun crossover ne se démarque sur ce critère.
- **La baseline casse le plus fort** (perte moyenne d'environ 9 250, près du double d'Active PPX). Votre hypothèse est la bonne : son segment est tiré sur tout le vecteur, donc il est souvent plus long que la fenêtre entière. Quand il la touche, il en remplace une grande partie et pousse les bonnes vidéos de p2 hors de la fenêtre.
- **CX casse rarement mais fort** (environ 8 200) : lui aussi travaille sur tout le vecteur.
- **Les quatre crossovers à fenêtre ont une perte nettement plus faible** (environ 4 700 à 5 800) : le changement reste borné à la zone utile.
- **Active PPX a la perte la plus faible (environ 4 700) mais beaucoup de rouge (environ 35 %)**. L'enfant n'est presque jamais la copie d'un parent, donc peu de neutres ; mais comme il ne contient que des vidéos déjà en tête chez un parent, les détériorations sont petites.
- **POS est l'exception** : sans fenêtre, il a le plus de rouge (environ 41 %) mais une perte faible (environ 5 200). Mon interprétation : il fait beaucoup de petits changements répartis partout, et chaque vidéo reste proche de son rang chez un parent.

### Deux questions à anticiper

- **« Active PPX est-il vraiment le meilleur ? »** Ce graphique mesure la mécanique de l'opérateur, pas le score final. En multipliant fréquence par gravité (lecture approximative du graphique), ACTIVE_OX1 et PMX perdent moins par crossover qu'Active PPX. Et dans l'ancienne étude de [crossover_mutation.md](crossover_mutation.md), faite avec un autre protocole, Active PPX était dernier en gain final.
- **« La perte est en quelle unité ? »** En points de fitness bruts, moyennés sur toutes les instances. Les instances à gros scores pèsent donc plus lourd.

### Mutation : Graph Repair

1. On décode l'individu et on relève les vidéos que le décodeur n'a pas réussi à placer.
2. Parmi elles, on choisit celle qui a la plus grande priorité (temps potentiellement gagné par Mo).
3. On la retire et on la réinsère à une position aléatoire tout en haut de la fenêtre, dans le premier dixième de `active_k`.

- **Idée** : réparer le goulot d'étranglement. Une vidéo de forte valeur est restée dehors, on la force à passer en premier ; c'est une vidéo moins utile qui manquera de place.
- **« Graph-aware »** : la priorité vient du graphe requêtes / endpoints / caches, pas du hasard.
- **Repli** : si toutes les vidéos sont placées, elle fait un échange aléatoire (baseline).
- **Limite** : le choix de la vidéo est déterministe. Pour un même individu, c'est toujours la même vidéo qui remonte ; seule la position d'insertion varie.

## Slide 2 : Protocole de benchmark

### Critère d'arrêt : le temps

- Chaque run a 45 secondes. Les limites en générations et en stagnation sont réglées assez haut (10 000 et 1 000) pour que le temps soit le critère visé.
- **Pourquoi** : une génération ne coûte pas la même chose selon la stratégie (recherche locale, décodeurs plus lourds). À générations égales, les stratégies coûteuses seraient avantagées ; à temps égal, tout le monde a le même budget de calcul.

### Les blocs

- Cinq blocs dans le code : initialisation, crossover, mutation, recherche locale, décodage.
- Chaque bloc a une stratégie de base (l'encadré vert). Dans [benchmark.py:181](benchmark.py#L181), c'est : init aléatoire, crossover OX1, mutation par échange, pas de recherche locale, décodeur glouton statique.
- La base d'initialisation est `RANDOM` (id 1), pas la stratégie nommée `BASELINE` (id 0) : vérifiez que l'encadré vert de la slide dit bien la même chose.
- **Un bloc à la fois** : on teste toutes les stratégies du bloc, les autres blocs restant sur leur base. Un écart de score vient donc du bloc testé et de rien d'autre.
- **Limite à assumer** : on ne mesure pas les interactions. La meilleure mutation avec le crossover de base n'est pas forcément la meilleure avec un autre crossover.

### Côté statistique

- **5 runs par couple instance × stratégie**, avec 5 graines fixes. Le GA est aléatoire : un seul run ne prouve rien, on compare des moyennes.
- Les graines sont les mêmes pour toutes les stratégies, donc elles sont comparées dans les mêmes conditions.
- **33 instances**, regroupées en 7 groupes (taille du cache, taille générale, profil de distribution des requêtes). Le regroupement permet de voir si une stratégie gagne partout ou seulement sur un type d'instance.
- Ordre de grandeur : le bloc crossover seul, c'est 33 × 7 × 5 = 1 155 runs.

### Le JSON produit à chaque exécution

| Famille | Métriques | Ce qu'elle dit |
|---|---|---|
| Qualité de la solution | Meilleur score, score de la génération 0, paliers atteints vers la meilleure valeur connue, nombre d'évaluations, temps par évaluation | Qui gagne, et à quelle vitesse |
| Mécanique des opérateurs | Taux bénéfique / neutre et perte moyenne pour le crossover et la mutation ; taux de succès et gain moyen de la recherche locale | Pourquoi une stratégie gagne ou perd (c'est la source du graphique de la slide 1) |
| Santé structurelle | Diversité de la population, taux de remplissage des caches, place perdue par cache | Si le GA a encore de quoi chercher |

**Pourquoi c'est important.**
- Le score final seul dit qui gagne, pas pourquoi. Sans la mécanique, on ne peut pas distinguer un opérateur utile d'un opérateur qui ne fait rien pendant que la sélection travaille.
- La diversité est le carburant du crossover : il ne recombine que ce qui diffère entre deux parents. Une population uniforme converge trop tôt et reste bloquée.
- Deux diversités sont mesurées : celle des permutations (distance moyenne des positions dans la fenêtre active) et celle des solutions décodées (nombre de solutions distinctes). Deux permutations différentes peuvent donner la même solution, donc la première peut être trompeuse.
- Les deux diversités sont calculées sur la population initiale seulement. Elles servent surtout à juger le bloc initialisation, pas l'évolution au fil des générations.