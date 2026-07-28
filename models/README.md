# Modèle

`resnet50v2_faceguard.keras` — 97 Mo, inclus dans le dépôt.

Le serveur charge automatiquement le premier fichier `.keras` ou `.h5` trouvé
dans ce dossier, il n'y a donc rien à configurer.

## Caractéristiques

| | |
|---|---|
| Architecture | ResNet50V2 pré-entraîné ImageNet + tête binaire |
| Entrée | 224 × 224 × 3, pixels normalisés dans **[0, 1]** |
| Sortie | 1 neurone sigmoid — proche de **1 = image réelle** |
| Accuracy | 89,64 % sur 5 714 images de test |
| AUC | 96,30 % |

Le fichier a été réexporté sans l'état de l'optimiseur : 97 Mo au lieu de 277,
pour des prédictions strictement identiques. Les 180 Mo retirés étaient les
moments de l'optimiseur Adam, utiles seulement pour reprendre un entraînement.

## Utiliser un autre fichier

Pour pointer vers un modèle situé ailleurs :

```bash
# Windows
set FACEGUARD_MODEL=C:\chemin\vers\mon_modele.keras && run.bat

# macOS / Linux
FACEGUARD_MODEL=/chemin/vers/mon_modele.keras ./run.sh
```

Il doit respecter la même convention : entrée 224 × 224 × 3 normalisée dans
[0, 1], sortie d'un seul neurone sigmoid. Voir `app/model.py`.

## Note

Les formats `.h5`, `.pt`, `.pth` et `.onnx` sont exclus du versionnement. Le
`.h5` d'origine fait 277 Mo, ce qui dépasse la limite de 100 Mo par fichier
imposée par GitHub : son push serait refusé.
