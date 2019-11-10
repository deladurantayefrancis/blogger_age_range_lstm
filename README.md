# Classification des billets de blog par tranche d'âge

Le corpus est composé 512629 billets de blogs provenant de trois tranches d'âge.

10 à 19 ans : 35.1%
20 à 29 ans : 46.9%
30 ans et plus : 17.9%

## Prétraitement

Pour prétraiter les données exécutez:

```bash
python preprocess.py <nombre_lignes_a_pretraiter>
```

## Lancer l'entraîner du modèle Bayes Naïf

```bash
python src/Naive_Bayes.py
```

## Auteur

Francis de Ladurantaye
