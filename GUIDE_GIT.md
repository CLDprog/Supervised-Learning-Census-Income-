# Guide Git & GitHub de l'équipe

Guide pour travailler à 3 sur le projet **Census Income**, directement sur la branche `main`, sans connaissances préalables de Git.

**Si tu ne lis qu'une chose :** avant de travailler, `git pull`. Après avoir travaillé, `git add` → `git commit` → `git pull` → `git push`. Tout le reste du guide sert à comprendre pourquoi et à s'en sortir quand ça coince.

---

## 1. Comprendre en 2 minutes

Le projet existe à **4 endroits** : une copie sur GitHub (la référence commune) et une copie sur l'ordinateur de chacun de nous trois.

Git ne synchronise **rien automatiquement**. Ce n'est pas un Google Drive : tant que tu n'as pas envoyé ton travail, les autres ne le voient pas, et tant que tu n'as pas récupéré le leur, tu ne le vois pas.

| Mot | Ce que ça veut dire |
|---|---|
| **dépôt** (repo) | Le dossier du projet, avec tout son historique |
| **commit** | Une sauvegarde de ton travail, avec un message qui dit ce que tu as fait |
| **pull** | Récupérer sur ton ordinateur ce que les autres ont envoyé sur GitHub |
| **push** | Envoyer tes commits sur GitHub pour que les autres les aient |
| **main** | La branche principale, celle sur laquelle on travaille tous |
| **conflit** | Deux personnes ont modifié les mêmes lignes d'un même fichier, Git ne sait pas quelle version garder |

Le trajet d'une modification :

```
ton fichier modifié  --git add-->  prêt à sauvegarder  --git commit-->  sauvegardé chez toi  --git push-->  sur GitHub
```

---

## 2. Installation (à faire une seule fois)

### 2.1 Installer Git

Télécharger Git sur <https://git-scm.com/downloads> et l'installer en laissant les options par défaut. Sur Mac, taper `git --version` dans le Terminal suffit à déclencher l'installation.

Vérifier que ça marche :

```bash
git --version
```

### 2.2 Se présenter à Git

Mets ton vrai nom et **l'adresse mail de ton compte GitHub** :

```bash
git config --global user.name "Prénom Nom"
```

```bash
git config --global user.email "ton.mail@exemple.com"
```

Puis ces deux réglages, qui évitent des messages d'erreur déroutants plus tard :

```bash
git config --global pull.rebase false
```

```bash
git config --global core.editor "code --wait"
```

(Le second suppose que tu utilises VS Code. Sinon, ignore-le.)

### 2.3 Avoir accès au dépôt

Le dépôt appartient au compte `CLDprog`. Son propriétaire doit inviter les deux autres : sur GitHub, **Settings → Collaborators → Add people**. Chacun doit ensuite **accepter l'invitation** reçue par mail. Sans ça, impossible de faire `git push`.

### 2.4 Récupérer le projet sur ton ordinateur

Place-toi dans le dossier où tu veux ranger le projet, puis :

```bash
git clone https://github.com/CLDprog/Supervised-Learning-Census-Income-.git
```

```bash
cd Supervised-Learning-Census-Income-
```

Au premier `git push`, une fenêtre de navigateur s'ouvrira pour te connecter à GitHub. C'est normal, et ça n'arrive qu'une fois.

> **Évite de cloner dans un dossier OneDrive, Google Drive ou iCloud.** Ces outils synchronisent en même temps que Git et peuvent abîmer le dépôt (fichiers verrouillés, doublons). Préfère un dossier simple comme `C:\Users\toi\projets\`.

---

## 3. La routine de tous les jours

### Avant de commencer à travailler

```bash
git pull
```

Tu récupères le travail des autres. **Toujours**, même si tu penses que personne n'a rien fait.

### Pendant que tu travailles

Tu modifies tes fichiers normalement. Pour voir où tu en es :

```bash
git status
```

C'est la commande à taper dès que tu as un doute. Elle ne modifie rien, elle affiche juste les fichiers modifiés et te suggère la suite.

### Quand tu as fini quelque chose

**1. Choisir ce que tu sauvegardes**

```bash
git add nom_du_fichier
```

ou, pour tout prendre d'un coup :

```bash
git add .
```

**2. Sauvegarder avec un message**

```bash
git commit -m "Ajoute le nettoyage des valeurs manquantes"
```

**3. Récupérer ce que les autres ont envoyé entre-temps**

```bash
git pull
```

**4. Envoyer ton travail**

```bash
git push
```

Vérifie sur la page GitHub du projet que ton commit apparaît bien.

---

## 4. Les 6 règles de l'équipe

Comme on travaille tous sur `main` sans filet, ces règles sont ce qui nous évite les ennuis.

1. **`git pull` avant de commencer, `git pull` avant chaque `git push`.**
2. **On se prévient avant de toucher à un fichier.** Un message dans le groupe (« je prends le notebook de preprocessing ») suffit. Deux personnes ne travaillent jamais sur le même fichier en même temps.
3. **Des petits commits, souvent.** Un commit par étape terminée, pas un énorme commit en fin de journée. Plus tu attends, plus les conflits sont probables et pénibles.
4. **On ne pousse que du code qui tourne.** Si ton notebook plante à la cellule 3, les deux autres sont bloqués.
5. **Des messages de commit clairs.** `Ajoute la régression logistique` plutôt que `modif` ou `test`.
6. **Jamais de `git push --force`.** Cette commande peut effacer le travail des autres sans retour possible.

---

## 5. Les notebooks Jupyter : attention particulière

Un fichier `.ipynb` contient le code **et** les résultats (graphiques, tableaux, compteurs d'exécution). Il suffit de relancer une cellule pour que le fichier change, même sans toucher au code. Résultat : les conflits sur les notebooks sont fréquents et quasi illisibles.

Pour les éviter :

- **Un notebook par personne ou par étape**, avec un nom clair : `01_exploration.ipynb`, `02_preprocessing.ipynb`, `03_modeles.ipynb`…
- **Une seule personne à la fois sur un notebook donné** (règle n°2, encore plus stricte ici).
- **Le code réutilisé par plusieurs va dans des fichiers `.py`** (par exemple `src/preprocessing.py`), que les notebooks importent. Les `.py` se fusionnent très bien.
- N'ouvre pas et ne relance pas le notebook d'un autre « juste pour voir » avant un commit : ça le modifie. Si c'est arrivé, annule avec `git restore nom_du_notebook.ipynb`.

---

## 6. Ce qu'on ne met pas sur GitHub

Créer à la racine du projet un fichier nommé `.gitignore` avec ce contenu. Git ignorera alors ces fichiers :

```gitignore
# Environnements Python
.venv/
venv/
env/
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Éditeurs et système
.vscode/
.idea/
.DS_Store
Thumbs.db

# Secrets
.env
```

À retenir aussi :

- **Jamais de mot de passe ni de clé d'API** dans le dépôt. Une fois poussé, c'est dans l'historique pour toujours.
- **Pas de fichier de plus de 50 Mo** (GitHub refuse au-delà de 100 Mo). Le jeu de données Census Income est petit, il peut rester dans un dossier `data/`.
- Les dépendances se partagent via un fichier `requirements.txt`, pas en envoyant le dossier `.venv`.

---

## 7. Quand ça coince

### « Mon `git push` est refusé » (`rejected`, `fetch first`)

Quelqu'un a poussé avant toi. Rien de grave :

```bash
git pull
```

```bash
git push
```

### « `git pull` refuse à cause de mes modifications locales »

Message du type `Your local changes would be overwritten by merge`. Tu as des modifications non sauvegardées. Fais d'abord ton commit, puis recommence :

```bash
git add .
```

```bash
git commit -m "Décris ce que tu as fait"
```

```bash
git pull
```

### « Un écran bizarre s'ouvre après `git pull` »

Git te demande de valider un message de fusion.

- Si c'est VS Code : ferme simplement l'onglet.
- Si c'est un écran noir dans le terminal (l'éditeur Vim) : tape `:wq` puis Entrée.

### « J'ai un conflit » (`CONFLICT`, `Automatic merge failed`)

Deux personnes ont modifié le même endroit. Pas de panique, rien n'est perdu.

**1.** `git status` te liste les fichiers en conflit (`both modified`).

**2.** Ouvre le fichier. Git a marqué la zone comme ceci :

```
<<<<<<< HEAD
ta version à toi
=======
la version de ton coéquipier
>>>>>>> a1b2c3d
```

**3.** Garde ce qu'il faut (ta version, la sienne, ou un mélange des deux), puis **supprime les trois lignes de marqueurs** `<<<<<<<`, `=======` et `>>>>>>>`. Dans VS Code, des boutons *Accept Current / Accept Incoming / Accept Both* le font pour toi. En cas de doute, appelle la personne concernée.

**4.** Termine la fusion :

```bash
git add .
```

```bash
git commit -m "Résout le conflit sur preprocessing.py"
```

```bash
git push
```

**Si le conflit est dans un notebook**, n'essaie pas de l'éditer à la main. Choisis une version entière :

```bash
git checkout --ours nom_du_notebook.ipynb
```

garde **ta** version, tandis que

```bash
git checkout --theirs nom_du_notebook.ipynb
```

garde **celle de GitHub**. Enchaîne ensuite avec l'étape 4. La personne dont la version est écartée devra refaire ses modifications : mettez-vous d'accord avant.

**Si tu es perdu au milieu d'un conflit**, tu peux tout annuler et revenir à l'état d'avant le `git pull` :

```bash
git merge --abort
```

### « Je veux annuler ce que j'ai fait »

| Situation | Commande |
|---|---|
| Annuler les modifications d'un fichier, pas encore commitées (**définitif**) | `git restore nom_du_fichier` |
| Retirer un fichier ajouté par erreur avec `git add` | `git restore --staged nom_du_fichier` |
| Corriger le message du dernier commit, **pas encore poussé** | `git commit --amend -m "Nouveau message"` |
| Annuler un commit **déjà poussé** | `git revert <identifiant_du_commit>` puis `git push` |

L'identifiant d'un commit se trouve avec `git log --oneline`.

### « Je ne comprends plus rien »

1. Ne tape pas de commandes au hasard trouvées sur Internet, surtout pas celles contenant `--force` ou `reset --hard`.
2. Fais une copie de ton dossier de projet ailleurs, par sécurité.
3. Tape `git status`, fais une capture d'écran et envoie-la au groupe.

---

## 8. Aide-mémoire

| Commande | À quoi ça sert |
|---|---|
| `git status` | Où j'en suis ? (à utiliser sans modération) |
| `git pull` | Je récupère le travail des autres |
| `git add .` | Je prépare tous mes fichiers modifiés |
| `git commit -m "message"` | Je sauvegarde avec un message |
| `git push` | J'envoie sur GitHub |
| `git log --oneline` | Je vois l'historique des commits |
| `git diff` | Je vois précisément ce que j'ai modifié |
| `git restore fichier` | J'annule mes modifications sur un fichier |
| `git merge --abort` | J'annule une fusion en conflit |

**La séquence complète :**

```
git pull  →  je travaille  →  git add .  →  git commit -m "..."  →  git pull  →  git push
```

> Tout ce guide peut aussi se faire à la souris dans VS Code, via l'onglet **Source Control** (icône à trois points reliés, à gauche) : le `+` correspond à `git add`, le bouton **Commit** à `git commit`, et **Sync Changes** fait `git pull` puis `git push`.
