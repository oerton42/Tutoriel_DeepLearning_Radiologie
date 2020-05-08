# Tutoriel pour initiation au deep learning à l'usage des radiologues
Vous pourrez télécharger sur cette page le code pour simplifier la mise à l#étrier dans le domaine du deep learning.

Après avoir configuré votre environnement, soit via les commandes suivantes, soit via le fichier reequirements.txt, vous pourrez téléchagrcer les fichiers python et jupyter-notebook pour faire vos propres essais sur votre machine.

### Pour l'installation de l'environnement
Commençons par installer python et conda : ceci se fait en une seule action en choisissant la version sur le site d’Anaconda : https://www.anaconda.com/distribution/ .
Tensorflow nécessite une version de python en version 3.7 au maximum (alors que celui-ci possède déjà une version 3.8).

Une fois Anaconda installé, il faut ouvrir “l’invite de commandes” : 
- sur Windows : allez dans le menu Démarrer et cherchez “Anaconda Prompt”.
- sur Mac : allez dans le Dossier Applications et double cliquez sur Terminal. Il est possible également de faire une recherche via l'icône de loupe ou le raccourci clavier Cmd + Espace.




- via le fichier d'installation :
Une fois conda installé, il faut ouvrir une invite de commande anaconda et rentrer :
> conda env create -f environment.yml
 Ceci créera un environnement nommé “TutorielIA”. Assurez vous que le fichier environment.yml
se situe au même endroit que là où vous lancez cette commande. Sinon vous pouvez donner son chemin, exemple : 
> conda env create -f “C:\\Exemple\de\Chemin\environment.yml”
Pour activer l'environnement par la suite, tapez :
> conda activate TutorielIA


- via lignes de commandes :
 L’installation peut se faire via pip ou via conda (deux logiciels existants qui servent aux installations). Ce dernier a l’avantage d’une installation plus simple concernant l’usage des GPU. Cependant il existe un délai entre l’apparition d’une mise à jour d’une librairie -tensorflow par exemple- et sa disponibilité sur conda. Si la version de tensorflow peut souffrir d’un léger retard sur conda ceci ne posera pas de problème pour notre usage : TensorFlow2.1 y a été ajouté en février 2020.




### Usage
Ouvrez le fichier Usage.ipynb et suivez le guide.

Plusieurs exemples de dataset sont disponible dans l'article source, disponible à telle adresse : 
