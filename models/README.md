# Emplacement du modèle

Dépose ici le fichier de poids. Le serveur charge automatiquement le premier
fichier `.keras` ou `.h5` qu'il trouve dans ce dossier.

```
models/resnet50v2_faceguard.keras
```

Ce dossier est volontairement vide dans le dépôt : le modèle pèse 97 Mo, ce qui
dépasse la taille raisonnable pour un fichier versionné dans git. Il est
distribué via les **Releases** du dépôt.

## Récupérer le modèle

1. Ouvre l'onglet **Releases** du dépôt GitHub
2. Télécharge `resnet50v2_faceguard.keras`
3. Place-le dans ce dossier

## Utiliser un autre fichier

Pour pointer vers un modèle situé ailleurs :

```bash
# Windows
set FACEGUARD_MODEL=C:\chemin\vers\mon_modele.keras && run.bat

# macOS / Linux
FACEGUARD_MODEL=/chemin/vers/mon_modele.keras ./run.sh
```

Le modèle doit respecter l'architecture attendue : entrée 224×224×3, sortie
d'un seul neurone sigmoid, pixels normalisés dans [0, 1]. Voir `app/model.py`.
