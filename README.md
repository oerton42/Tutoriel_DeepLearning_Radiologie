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

```
conda env create -f environment.yml
```

Ceci créera un environnement nommé “TutorielIA”. Assurez vous que le fichier environment.yml
se situe au même endroit que là où vous lancez cette commande. Sinon vous pouvez donner son chemin, exemple : 

```
conda env create -f “C:\\Exemple\de\Chemin\environment.yml”
```

Pour activer l'environnement par la suite, tapez :

```
conda activate TutorielIA
```



**- via lignes de commandes :**

Si vous voulez garder la main sur le détail de l’installation, vous pouvez taper les lignes de commandes suivantes, ligne par ligne pour obtenir le même résultat et sans utiliser le fichier d’installation que nous fournissons. 

L’installation peut se faire via pip ou via conda (deux logiciels existants qui servent aux installations). Ce dernier a l’avantage d’une installation plus simple concernant l’usage des GPU. Cependant il existe un délai entre l’apparition d’une mise à jour d’une librairie -tensorflow par exemple- et sa disponibilité sur conda. Si la version de tensorflow peut souffrir d’un léger retard sur conda ceci ne posera pas de problème pour notre usage : TensorFlow2.1 y a été ajouté en février 2020.
 

```
conda create --name <env>  python=3.7.6
conda install keras-gpu
conda install -c conda-forge pydicom
conda install scikit-image
conda install -c conda-forge opencv
conda install pandas
conda install openpyxl
conda install jupyterlab
```


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

```
jupyter notebook
```

Le logiciel s’ouvre dans le navigateur internet (sans nécessiter une connexion internet). Nous pouvons créer un “notebook” en Python 3 en cliquant sur ‘New’.
![image New notebook](https://github.com/oerton42/Tutoriel_DeepLearning_Radiologie/blob/master/Annotation%202020-05-08%20095904572.png?raw=true)
Sinon, ouvrez le fichier Tutoriel.ipynb et suivez le guide.



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
```
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
```

Il est possible de créer autant de classes que voulues, cependant une image ne peut être que dans un seul dossier à la fois et donc n’appartenir qu’à une seule classe. Ainsi cette technique d’import ne prend pas en charge la possibilité d’associer plusieurs labels à une même image : nous parlons de labellisation “multi-class single-label”. D’autres techniques permettent de trier les fichiers afin d’obtenir une labellisation “multi-class multi-label”.



Dans le cas d’un réseau de segmentation cela correspond à un sous-dossier “images à segmenter” et un sous-dossier “résultats”.

```
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
```



### Création d'un réseau de convolution :

Les réseaux neuronaux convolutifs mettent en œuvre des couches de neurones convolutifs décomposant l’image par fragments où chaque neurone est responsable d’une partie puis connecté à d’autres, décomposant les caractéristiques de l'image en une multitude de motifs différents (feature maps). Cette composition peut être répétée plusieurs fois de façon séquentielle au sein de l’architecture.


Nous proposons une fonction permettant de créer un tel réseau que l’on nommera “model”; il permet de labelliser des images (de hauteur HauteurImage” et largeur “LargeurImage” codées sur une échelle de gris “1”) selon 6 catégories :
```
model = build_cnn (entree =  (HauteurImage,LargeurImage,1), sortie =6)
```

Dans sa version la plus simple cette fonction ne demande quasiment aucun réglage, nous permettons en revanche de modifier la totalité de ses réglages : 
- **l’optimizer** :  il s’agit de la formule qui détermine comment le réseau doit adapter ses réglages internes. Il existe plusieurs façon de déterminer comment les poids des neurones doivent être corrigés: Certains optimizers possèdent une inertie.
- **le learning rate** : correspond à la taille du pas de modification. C’est un paramètre majeur, il est dépendant de l’optimizer choisi.
- **le Nombre de blocs de convolution** : dans sa version de base un réseau de type CNN correspond à une à deux couche(s) de neurones de convolution et d’une couche de maxpooling (qui peut être interprété comme un Maximal Intesity Projection (MIP) sur plusieurs pixels adjacents). Il est possible de répéter les blocs de convolution dont le nombre correspond à ce réglage.
- **Nombre de feature maps** : Il s’agit du nombre de filtres d’image utilisés pour interpréter l’image. Ce nombre est multiplié automatiquement par deux à chaque bloc de convolution.
- **Kernel size** : correspond à la largeur des filtres utilisés. Un filtre trop large perd le détail de l’image alors qu’un filtre trop petit ne s’intéresse qu’aux détails en oubliant l’information globale. En général ces filtres font entre 3 et 7 pixels de côté.
- **Fonction d’activation** : cette fonction indique comment est interprétée l’activation des différents neurones d’une couche avant le passage à la couche suivante. Il existe de multiples fonctions d’activation décrites dans la littérature parmi lesquelles nous conseillons “selu” et "leaky-relu". La fonction “relu” étant souvent utilisée dans la littérature, nous avons choisi celle-ci comme réglage par défaut.
- **Dropout** : Le dropout est une méthode de régularisation. Pendant l’entraînement, un certain nombre de résultats sont ignorés de manière aléatoire selon une certaine probabilité. Cela a pour effet de limiter l’overfitting. A chaque epoch les neurones ignorés sont de nouveau tirés au hasard. Nous conseillons un dropout maximal de 0,5 en fin de réseau et 0,2 en début de réseau. En effet au-delà de 0,5 de probabilité le dropout diminue les capacités d’apprentissage du réseau sans augmenter son effet sur l’overfitting.
- **Normalisation par batch** : la normalisation par batch permet d'accélérer l'entraînement du réseau. Son mécanisme est en revanche débattu. Son usage peut cependant s’avérer péjoratif sur les batchs de petite taille et à éviter avec l’usage du dropout.
- **Couche entièrement connectée** : elle correspond à la taille de l’avant-dernière couche de neurones, s’occupant d’agréger les résultats des couches précédentes.


Les réglages suivants correspondent aux réglages par défaut de la fonction :
```
model =  build_cnn(entree = <image>,
                                sortie = <nombre de catégories>,
                                optimizer = "adam",
                                Learning_rate_custom = None,
                                nombre_de_blocs = 2,
                                feature_maps = 32,
                                Kernel_size = 3,
                                activation = "relu",
                                dropout_rate = .5,
                                batch_Norm = False,
                                couche_entierement_connectée = 64)
```



### Création d'un réseau de convolution par Transfer Learning :
Si Keras permet de télécharger facilement ces réseaux, il faut tout de même quelques lignes de code pour les adapter, nous avons regroupé tout ceci pour le tutoriel dans une fonction.

```
TransferLearning(entree = <image>, 
                            sortie = <nombre de catégories>,
                            training_generator = <training_generator>, 
                            validation_generator = <validation_generator>,
                            nombre_epochs_avant_finetuning = <EPOCHS_BEFORE>,
                            nombre_epochs_apres_finetuning = <EPOCHS_AFTER>,
                            Model_dOrigine       = "InceptionV3" )
```

Parmi les paramètres disponibles, nous permettons les réglages suivants :
- **Model_dOrigine** : correspond au choix du réseau à utiliser pour le transfer learning, parmi : "Xception", "InceptionV3", "ResNet50", "VGG16", "VGG19", "MobileNetV2"
- **optimizer** : correspond aux optimizer à utiliser, en distinguant avant et après transfer learning.
- **Learning_rate_custom** = correspond au pas d’apprentissage à utiliser, en distinguant avant et après transfer learning.
- **class_weight** : permet d’ajouter une pondération aux classes selon leur prévalence dans le jeu de données.





### Création d'un réseau U-net :
Nous fournissons une fonction également pour créer un “model” U-net en une seule ligne :

```
model = U_Net (entree = (HauteurImage,LargeurImage,1))
```


