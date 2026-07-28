"""
FaceGuard — serveur local de demonstration.

Sert une page unique qui envoie une image au ResNet50V2 entraine et affiche
le verdict. Aucune dependance web : uniquement la bibliotheque standard.

Lancement :
    python app/server.py            (ou run.bat / run.sh a la racine)

Variables d'environnement :
    FACEGUARD_MODEL   chemin du modele (par defaut : premier fichier trouve
                      dans models/)
    FACEGUARD_PORT    port d'ecoute (par defaut : 8000)
"""

import base64
import io
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from PIL import Image

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
PAGE = os.path.join(ICI, 'page.html')
ASSETS = os.path.join(ICI, 'assets')
DOSSIER_MODELES = os.path.join(RACINE, 'models')

sys.path.insert(0, ICI)
import model as moteur

PORT = int(os.environ.get('FACEGUARD_PORT', 8000))
TAILLE_MAX = 12 * 1024 * 1024        # garde-fou sur le corps des requetes

MODELE = None
NOM_MODELE = '—'


def trouver_modele():
    """Chemin du modele : variable d'environnement, sinon premier fichier
    de models/."""
    impose = os.environ.get('FACEGUARD_MODEL')
    if impose:
        return impose
    if os.path.isdir(DOSSIER_MODELES):
        for nom in sorted(os.listdir(DOSSIER_MODELES)):
            if nom.lower().endswith(('.keras', '.h5')):
                return os.path.join(DOSSIER_MODELES, nom)
    return None


def demarrer():
    global MODELE, NOM_MODELE

    chemin = trouver_modele()
    if not chemin:
        print("Aucun modele trouve.\n"
              f"Depose le fichier .keras ou .h5 dans {DOSSIER_MODELES}\n"
              "puis relance. Voir la section « Installation » du README.",
              file=sys.stderr)
        sys.exit(1)

    NOM_MODELE = os.path.basename(chemin)
    taille = os.path.getsize(chemin) / 1e6
    print(f"Chargement de {NOM_MODELE} ({taille:.0f} Mo)...", flush=True)
    print("La premiere fois, compter 30 a 60 secondes.", flush=True)

    MODELE, methode = moteur.load_model(chemin)
    print(f"  charge par {methode}", flush=True)

    # Une premiere prediction a vide force la construction du graphe : sans
    # cela, c'est la premiere image de l'utilisateur qui paierait l'attente.
    moteur.predict(MODELE, Image.new('RGB', (moteur.IMG_SIZE, moteur.IMG_SIZE)))
    print(f"\nPret. Ouvre http://127.0.0.1:{PORT}   (Ctrl+C pour arreter)\n", flush=True)


TYPES_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
              '.svg': 'image/svg+xml', '.webp': 'image/webp', '.ico': 'image/x-icon'}


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass                                    # pas de log par requete

    def _envoyer(self, corps, type_mime, cache='no-store'):
        self.send_response(200)
        self.send_header('Content-Type', type_mime)
        self.send_header('Cache-Control', cache)
        self.send_header('Content-Length', str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _json(self, objet, code=200):
        corps = json.dumps(objet).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self):
        route = urlparse(self.path).path

        if route == '/':
            # Relue a chaque requete : on peut retoucher page.html et
            # rafraichir sans redemarrer, donc sans recharger le modele.
            with open(PAGE, 'r', encoding='utf-8') as f:
                self._envoyer(f.read().encode('utf-8'), 'text/html; charset=utf-8')
            return

        if route.startswith('/assets/'):
            # basename() seul : empeche toute remontee de dossier via ../
            nom = os.path.basename(route)
            chemin = os.path.join(ASSETS, nom)
            if not nom or not os.path.isfile(chemin):
                self.send_error(404)
                return
            with open(chemin, 'rb') as f:
                self._envoyer(f.read(),
                              TYPES_MIME.get(os.path.splitext(nom)[1].lower(),
                                             'application/octet-stream'),
                              cache='max-age=3600')
            return

        if route == '/api/status':
            self._json({'modele': NOM_MODELE, 'taille_entree': moteur.IMG_SIZE})
            return

        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != '/api/predict':
            self.send_error(404)
            return

        try:
            taille = int(self.headers.get('Content-Length', 0))
            if taille <= 0 or taille > TAILLE_MAX:
                self._json({'erreur': 'image absente ou trop volumineuse'}, 413)
                return

            donnees = json.loads(self.rfile.read(taille).decode('utf-8'))
            brut = donnees['image'].split(',', 1)[-1]
            image = Image.open(io.BytesIO(base64.b64decode(brut)))

            label, score, confiance = moteur.predict(MODELE, image)
            self._json({'label': label,
                        'score': round(score, 4),
                        'confiance': round(confiance, 1)})
        except Exception as e:
            self._json({'erreur': str(e)}, 500)


if __name__ == '__main__':
    demarrer()
    serveur = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    try:
        webbrowser.open(f'http://127.0.0.1:{PORT}')
    except Exception:
        pass
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nArret.")
