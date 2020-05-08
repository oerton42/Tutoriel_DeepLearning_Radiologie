# Tutoriel pour initiation au deep learning à l'usage des radiologues
Vous pourrez télécharger sur cette page le code pour simplifier la mise à l#étrier dans le domaine du deep learning.

Après avoir configuré votre environnement, soit via les commandes suivantes, soit via le fichier reequirements.txt, vous pourrez téléchagrcer les fichiers python et jupyter-notebook pour faire vos propres essais sur votre machine.

### Pour l'installation de l'environnement

Commençons par installer python et conda : ceci se fait en une seule action en choisissant la version sur le site d’Anaconda : https://www.anaconda.com/distribution/ .

Tensorflow nécessite une version de python en version 3.7 au maximum (alors que celui-ci possède déjà une version 3.8).

Une fois Anaconda installé, il faut ouvrir “l’invite de commandes” : 
- sur Windows : allez dans le menu Démarrer et cherchez “Anaconda Prompt”.
![image New notebook](https://github.com/oerton42/Tutoriel_DeepLearning_Radiologie/blob/master/Annotation%202020-05-08%20095907.png?raw=true)
- sur Mac : allez dans le Dossier Applications et double cliquez sur Terminal. Il est possible également de faire une recherche via l'icône de loupe ou le raccourci clavier Cmd + Espace.
![image New notebook](https://github.com/oerton42/Tutoriel_DeepLearning_Radiologie/blob/master/Annotation%202020-05-08%200959072.png?raw=true)




**- via le fichier d'installation :**

Une fois conda installé, il faut ouvrir une invite de commande anaconda et rentrer :

'''
conda env create -f environment.yml
'''

Ceci créera un environnement nommé “TutorielIA”. Assurez vous que le fichier environment.yml
se situe au même endroit que là où vous lancez cette commande. Sinon vous pouvez donner son chemin, exemple : 

'''
conda env create -f “C:\\Exemple\de\Chemin\environment.yml”
'''

Pour activer l'environnement par la suite, tapez :

'''
conda activate TutorielIA
'''



**- via lignes de commandes :**

Si vous voulez garder la main sur le détail de l’installation, vous pouvez taper les lignes de commandes suivantes, ligne par ligne pour obtenir le même résultat et sans utiliser le fichier d’installation que nous fournissons. 

L’installation peut se faire via pip ou via conda (deux logiciels existants qui servent aux installations). Ce dernier a l’avantage d’une installation plus simple concernant l’usage des GPU. Cependant il existe un délai entre l’apparition d’une mise à jour d’une librairie -tensorflow par exemple- et sa disponibilité sur conda. Si la version de tensorflow peut souffrir d’un léger retard sur conda ceci ne posera pas de problème pour notre usage : TensorFlow2.1 y a été ajouté en février 2020.
 

'''
conda create --name <env>  python=3.7.6
conda install keras-gpu
conda install -c conda-forge pydicom
conda install scikit-image
conda install -c conda-forge opencv
conda install pandas
conda install openpyxl
conda install jupyterlab
 '''


Ceci installera toutes les API qui nous semblent nécessaires :
Pydicom  : librairie permettant de lire les fichiers DICOM
Tensorflow (2.0 ou 2.1) via la ligne contenant keras-gpu qui permet de simplifier l’installation de tensorflow, en fournissant tous les packages dont il dépend et de sa surcouche Keras. À noter que Tensorflow nécessite python au maximum en version 3.7, c’est pourquoi nous sélectionnons ce réglage lors de la création de l’environnement.
Scikit image et OpenCV permettent de modifier les images, 
pandas et openpyxl permettent de lire les tableurs (fichiers .csv et excel .xlsx) pour les convertir en dataframes (l’équivalent au sein de python). Ces deux librairies ne sont pas nécessaires pour notre tutoriel mais vous seront utiles pour la plupart des datasets publiques dont vous pourriez avoir l’usage.
jupyterlab correspond à l’interface visuelle de Jupyter Notebook que nous conseillons et qui permettra d’écrire, visualiser et tester notre code.

Voici une liste d’API non exhaustive que nous recommandons pour aller plus loin :
SimpleITK, pytables, nibabel, nipype  seront utiles en cas de passage à un 3D U-Net ou pour lire plusieurs bases de données dont les fichiers sont au format nifti. Scikit-learn est une librairie puissante pour réaliser des statistiques. Cupy permet d'accélérer les calculs en transférant les calculs lourds sur les cartes graphiques.





### Usage
Par la suite, pour ouvrir jupyter notebook il suffit de taper la commande suivante dans l’invite de commande :

'''
jupyter notebook
'''

Le logiciel s’ouvre dans le navigateur internet (sans nécessiter une connexion internet). Nous pouvons créer un “notebook” en Python 3 en cliquant sur ‘New’.
![image New notebook](https://github.com/oerton42/Tutoriel_DeepLearning_Radiologie/blob/master/Annotation%202020-05-08%20095904572.png?raw=true)
Sinon, ouvrez le fichier Usage.ipynb et suivez le guide.



### Obtention d'un dataset :

Plusieurs exemples de dataset sont disponible dans l'article source, disponible à telle adresse : XXXX

Vous pouvez également créer un dataset avec vos propres images.



Dans le cas d’un exercice sur nos propres images, le plus simple à notre niveau est de trier les images à la main en dossiers et sous-dossiers correspondants :
pour les dossiers, aux datasets d'entraînement, validation et test
pour les sous-dossiers, aux catégories (“classes”) que nous voulons départager.

La séparation entre training set et validation set est optionnelle en amont et peut être faite automatiquement par le logiciel. Il est nécessaire de séparer un jeu de test dès le début de l'entraînement dans un dossier séparé.
Dans chaque dossier de dataset, nous créons un sous-dossier de catégorie : 
dans le cas d’un réseau de labellisation cela correspond aux différents labels recherchés pour définir nos images ; 

Exemple dans le cas d’une labellisation en 3 classes :
'''
D:\User\Dataset\Train\
                       …\Train\Class1\
                       …\Train\Class1\image1.png
                       …\Train\Class1\image2.png
                       …\Train\Class2\
                       …\Train\Class2\image3.png
                       …\Train\Class2\image4.png
                       …\Train\Class3\
                       …\Train\Class3\image5.png
                       …\Train\Class3\image6.png
D:\User\Dataset\Test\
                       …\Test\Class1\
                       …\Test\Class1\image7.png
                       …\Test\Class1\image8.png
                       …\Test\Class2\
                       …\Test\Class2\image9.png
                       …\Test\Class2\image10.png
                       …\Test\Class3\
                       …\Test\Class3\image11.png
                       …\Test\Class3\image12.png
'''

Il est possible de créer autant de classes que voulues, cependant une image ne peut être que dans un seul dossier à la fois et donc n’appartenir qu’à une seule classe. Ainsi cette technique d’import ne prend pas en charge la possibilité d’associer plusieurs labels à une même image : nous parlons de labellisation “multi-class single-label”. D’autres techniques permettent de trier les fichiers afin d’obtenir une labellisation “multi-class multi-label”.



Dans le cas d’un réseau de segmentation cela correspond à un sous-dossier “images à segmenter” et un sous-dossier “résultats”.

'''
D:\User\Dataset\Train\
                       …\Train\Images\
                       …\Train\Images\image1.png
                       …\Train\Images\image2.png
                       …\Train\Segmentation\
                       …\Train\Segmentation\image3.png
                       …\Train\Segmentation\image4.png
D:\User\Dataset\Test\
                       …\Test\Images\
                       …\Test\Images\image5.png
                       …\Test\Images\image6.png
                       …\Test\SegmentationMap\
                       …\Test\Segmentation\image7.png
                       …\Test\Segmentation\image8.png
'''

