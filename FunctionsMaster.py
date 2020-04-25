"""
Created in 2020

@author: Alexandre NEROT
contact : alexandre@nerot.net

Ceci correspond aux fonctions utilisées pour traiter les données dans le cadre de ce tutoriel au deep learning à l'usage des radiologues
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
import pandas
from PIL import Image
import random
import datetime
import scipy
import openpyxl
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K


    
#___________________________________________________________________________________________
#___________________FONCTIONS POUR IMPORTER LES FICHIERS DICOM______________________________
#___________________________________________________________________________________________

    
def Dossier_DICOM_vers_ImagesPNG(DossierDICOM, #Entrer ici la localisation du dossier oú se situent les fichiers
                                 Dossier_de_sauvegarde, #indiquer le chemin où seront sauvegardées les images
                                 WINDOWCENTER = 40,
                                 WINDOWWIDTH = 400
                                ):
    """
    Prend un dossier contenant des DICOM et les sauveagrdes en images .png
    Obtient également un volume numpy correspondant au volume du scanner
    
    Parameters
    ----------
        - DossierDICOM : string, chemin vers le dossier contenant les fichiers DICOM
        - Dossier_de_sauvegarde : string, chemin vers le dossier où l'on veut enregistrer les fichiers png
        - WINDOWCENTER : int, optionnel, correspond au centre du fenetrage voulu qui sera pris par défaut si jamais le fichier dicom n'est pas lisible
        - WINDOWWIDTH : int, optionnel, correspond à la largeur du fenetrage voulu qui sera pris par défaut si jamais le fichier dicom n'est pas lisible
        
    Returns
    -------
        - volume_numpy : numpy, volume de l'ensemble de sfichiers DICOM, dont la taille est de [nb_dIamges,512,512].
    
    """
    list_files = os.listdir(DossierDICOM)
    if len(list_files) <150:
        print("   Moins de 150 coupes, le dossier n'a pas été importé")
        return

    #Nous ne récupérons pas les images dont l'épaisseur est trop épaisse :  
    echantillon1 = os.path.join(rootdir, list_files[1])
    echantillon2 = os.path.join(rootdir, list_files[100])
    if not os.path.isdir(echantillon1):
        _ds_1 = pydicom.dcmread(echantillon1,force =True, specific_tags =["ImagePositionPatient","SliceThickness","WindowCenter","BodyPartExamined", "FilterType", "SeriesDescription"])
        if (0x18, 0x50) in _ds_1:
            thickness = _ds_1["SliceThickness"].value
            if thickness >2.5: #Limitation si coupe trop épaisses : MIP...etc
                print("   Thickness is too high.")
                return

    
    #Nous ne récupérons pas les images sagittales ni coronales :        
    if not os.path.isdir(echantillon2):
        _ds_2 = pydicom.dcmread(echantillon2,force =True,specific_tags =["ImagePositionPatient","SliceThickness"])
    position1 = [5.,10.,15.] 
    position2 = [5.,10.,15.] 
    if (0x20, 0x32) in _ds_1:
        position1 = _ds_1["ImagePositionPatient"].value
    if (0x20, 0x32) in _ds_2:
        position2 = _ds_2["ImagePositionPatient"].value

    if position1[0] != position2[0]:
        print("   Sagittal plane.")
        return 
    
    if position1[1] != position2[1]:
        print("   Coronal plane.")
        return
        
    #Maintenant que l'on a arrêté la fonction précocement selon certains criteres, regardons la liste des images

    #il faut les trier les fichiers dans l'ordre de la sequence de scanner 
    #(ce qui ne correspond pas à l'ordre alphabetique du nom des fichiers)
    inter = {}
    for f in list_files:
        if not os.path.isdir(f):
            f_long = os.path.join(rootdir, f)
            _ds_   = pydicom.dcmread(f_long,specific_tags =["ImagePositionPatient","SliceThickness"])
            inter[f_long]=_ds_.ImagePositionPatient[2]
    inter_sorted=sorted(inter.items(), key=lambda x: x[1], reverse=True) 
    liste_fichiers=[x[0] for x in inter_sorted]

    #Nous avons maintenant la liste des fichiers selon leur ordre selon l'axe z du scanner
    nbcoupes = len(liste_fichiers)
    print(nbcoupes, " fichiers trouvés pour ce scanner")

    #Pour chaque fichier nous allons : l'importer, régler son contraste, puis le sauvegarder avec un nom différent
    j=0
    randomnumber = random.randint(0, 1000)
    volume_numpy=np.zeros((len(liste_fichiers),512,512))
    for k in range (0,len(liste_fichiers)):

        dicom_file = pydicom.read_file(liste_fichiers[k])
        img_orig_dcm = (dicom_file.pixel_array)

        slope=float(dicom_file[0x28,0x1053].value)
        intercept=float(dicom_file[0x28,0x1052].value)
        img_modif_dcm=(img_orig_dcm*slope) + intercept

        #Réglage du contraste
        if (0x28, 0x1050) in dicom_file:
            WindowCenter = dicom_file["WindowCenter"].value
            if not isinstance(WindowCenter, float) : WindowCenter = WINDOWCENTER
        if (0x28, 0x1051) in dicom_file:
            WindowWidth = dicom_file["WindowWidth"].value
            if not isinstance(WindowWidth, float) : WindowWidth = WINDOWWIDTH
        arraytopng = ReglageContrasteDICOM (WindowCenter,WindowWidth,img_modif_dcm) #réglages de contraste
        volume_numpy[k,:,:]=arraytopng #ecrit une ligne correspondant à l'image

        #Sauvegarde du fichier
        im = Image.fromarray(arraytopng)
        SAVING = os.path.basename(rootdir)+r"_{}_{}.png".format(randomnumber, j) 
        im.save(os.path.join(Dossier_de_sauvegarde, SAVING))    
        j+=1
            
    return volume_numpy


def fast_scandir(dir):
    """
    Prend un dossier contenant atant de sous-dossiers et sous-sous-dossiers que voulu et en crée la liste des sous dossiers.
    Utile pour généraliser une fonction adaptée à un dossier à autant de dossiers que voulus en une seule fois.
    Rq : le dossier dir n'est pas inclus dans la liste
    
    Parameters
    ----------
        - dir : string, chemin vers le dossier racine
        
    Returns
    -------
        - subfloders : liste, contient tous les sous-dossiers
    
    """
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(fast_scandir(dir))
    return subfolders


def readCSV(csv_path,name=None,indexing=None):
    """
    Fonction simple pour lire le CSV et le garder en mémoire sous la forme d'un datafile, plus facilement lisible en utilisant pandas
    si on rentre name (un des fichiers numpy disponibles), la fonction affiche cette valeur
    On peut rentrer un string pour l'arg indexing pour demander a classer selon la colonne.
    """
    df=pandas.read_csv(csv_path, delimiter=",",dtype=str)
    if indexing != None :
        df.set_index(indexing, inplace=True)
    if name:
        print(df.loc[name])
    return df


def ReglageContrasteDICOM (Global_Level,Global_Window,imageDICOM):
    """
    Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour afficher une image en échelle de gris (255 possibilités de gris sur un ordinateur classique) il faut réduire les 6000 possibilités à 255. Cette fonction est nécessaire avant d'afficher une image mais fait perdre des données (passage de 16 bits à 8 bits par pixel).
    Obligatoire pour sauvegarder une image png ou jpg mais fait perdre de l'information !
    
    On redéfinit les valeurs des pixels selon une largeur de fenêtre et un centre, les valeurs de réglages correspondent à celles ques les radiologues ont l’habitude d’utiliser sur le leur logiciel de viewer dImages.

    Parameters
    ----------
        - Global_Level : centre de la fenetre (en UH)
        - Global_Window : largeur de la fenetre (en UH)
        - imageDICOM: image ou volume numpy chargé(e) en mémoire
        
    Returns
    -------
        - image_avec_contraste : l'image ou le volume après réglage du contraste.
    
    Notes
    -----
    Ne fonctionne PAS si l'image a déjà été modifiée.
    
    """
    limite_inf   = Global_Level -  (Global_Window / 2)
    limite_sup = Global_Level + (Global_Window / 2)
    image_avec_contraste = np.clip (imageDICOM, limite_inf , limite_sup )
    image_avec_contraste = image_avec_contraste - limite_inf 
    image_avec_contraste = image_avec_contraste / (limite_sup  - limite_inf )
    image_avec_contraste *= 255
    return image_avec_contraste 


def Norm0_1 (volume_array):
    """
    les scanners ont des voxels dont la valeur est négative, ce qui sera mal interprété pour une image, il faut donc normaliser entre 0 et 1. Cela permet notamment de les afficher sous un format image apres un facteur de *255.
    
    Parameters
    ----------
        - volume_array : numpy, volume scanner [nb_de_coupes, largeur, profondeur]
        
    Returns
    -------
        - volume_array_scale : numpy, le même volume mais avec des voxels entre o et 1.
        - a : float, valeur minimale avant normalisation
        - b : float, valeur maximale avant normalisation
        - c : float, valeur moyenne avant normalisation
    
    Notes
    -----
    Ne fonctionne QUE si l'image a déjà été normalisée. 
    
    """
    a,b,c=volume_array.min(),volume_array.max(),volume_array.mean()
    volume_array_scale=(volume_array-a)/(b-a)
    return volume_array_scale,a,b,c


def WL_scaled (Global_Level,Global_Window,array,a,b):
    """
    Idem que ReglageContrasteDICOM mais corrigé par les facteurs a et b qui correpsondent au min et max, 
    >>> à utiliser à la place de ReglageContrasteDICOM si on a utilisé Norm0_1
    
    Les valeurs des voxels en DICOM sont entre -2000 et +4000, pour afficher une image en echelle de gris (255 possibilités de gris sur un ordinateur classique) il faut réduire les 6000 possibilités à 255. Cette fonction est nécessaire avant d'afficher une image mais fait perdre des données (passage de 16 bits à  8 bits par pixel).
    Obligatoire pour sauvegarder une image png ou jpg mais fait perdre de l'information !
    
    On redéfinit les valeurs des pixels selon une largeur de fenetre et un centre
    On sauvegarde les bornes initiales dans les variables a et b dans le cas où l'on veuille modifier le contraste après coup

    Parameters
    ----------
        - Global_Level : centre de la fenetre (en UH)
        - Global_Window : largeur de la fenetre (en UH)
        - array : image ou volume numpy chargé en mémoire
        - a : minimum en UH avant normalisation
        - b : maximum en UH avant normalisation
        
    Returns
    -------
        - image_ret : l'image ou le volume après réglage du contraste.
    
    Notes
    -----
    Ne fonctionne QUE si l'image a déjà été normalisée.
    A noter que le résultat reste entre les valeurs 0 et 1
    
    """
    limite_inf   = Global_Level -  (Global_Window / 2)
    limite_sup = Global_Level + (Global_Window / 2)
    limite_inf   = limite_inf/b
    limite_sup   = limite_sup/b
    image_ret=np.clip(array, limite_inf, limite_sup)
    image_ret=image_ret-limite_inf
    image_ret=image_ret/(limite_sup-limite_inf)
    return image_ret


def affichage3D(volume, k, axis=0):
    """
    affiche la coupe numéro k d'un volume, selon son axe axis

    Parameters
    ----------
        - volume : volume numpy chargé en mémoire
        - k : int, numéro de coupe
        - axis : int, 0 : axial ; 1 : coronal ; 2 : sag (dans le cas d'un volume chargé en axial)
    
    """
    f = plt.figure()
    if axis == 0:
        image1 = volume[k,:,:]
    if axis == 1:
        image1 = volume[:,k,:]
    if axis == 2:
        image1 = volume[:,:,k]
    plt.imshow(image1,cmap='gray')
    plt.show()
    return
    
    
def affichage2D(volume):
    """
    affiche un plan numpy 2D
    
    Parameters
    ----------
        - volume : plan numpy chargé en mémoire, en 2 dimensions
    
    """
    f = plt.figure()
    image1 = volume
    plt.imshow(image1,cmap='gray')
    plt.show()
    return


def AffichageMulti(volume, frequence, axis=0, FIGSIZE = 40):
    """
    affiche toutes les coupes d'un volume selon l'axe axis, avec une frequence entre les coupes définie
    
    Parameters
    ----------
        - volume : volume numpy chargé en mémoire
        - frequence : int, espace inter coupe (en voxels)
        - axis : int, 0 : axial ; 1 : coronal ; 2 : sag (dans le cas d'un volume chargé en axial)
        - FIGSIZE : taille des images pour l'affichage.
    
    """
    coupes = np.shape(volume)[axis]
    nb_images = coupes // frequence
    fig=plt.figure(figsize=(FIGSIZE, FIGSIZE))
    columns = 6
    if nb_images % columns >0 :
        rows = (nb_images // columns)+1
    else :
        rows = nb_images // columns
    for i in range(nb_images):
        i+=1
        fig.add_subplot(rows, columns, i)
        dix = frequence * i
        if axis == 0:
            plt.imshow(volume[dix,:,:], cmap='gray')
        elif axis == 1:
            plt.imshow(volume[:,dix,:], cmap='gray')
        elif axis == 2:
            plt.imshow(volume[:,:,dix], cmap='gray')
    plt.show(block=True)
    return
    

#___________________________________________________________________________________________
#___________________FONCTIONS POUR CREER UN RESEAU DE NEURONES______________________________
#___________________________________________________________________________________________



def U_Net(input_size = (256,256,1), #Correspond à la taille des images utilisées
         initial = 64 #Le nombre de features maps utilisé au départ
        ):
    """
    Crée un réseau de type U-net pour la segmentation

    Parameters
    ----------
        - input_size : la taille des images à utiliser et le nombre de channels couleur (1 pour les DICOM)
        - initial : nombre de feature map à utiliser au déaprt (allourdit le réseau)
        
    Returns
    -------
        - model : le réseau prêt à l'entrainement
    
    Notes
    -----
    Ne pas hésiter à modifier les hyperparamètres :
    - le learning rate
    - l'optimizer
    
    """
    inputs = Input(input_size)
    
    conv1 = Conv2D(initial, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(initial, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(initial * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(initial * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(initial * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(initial * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(initial * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(initial * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(initial * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(initial * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(initial * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(initial * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(initial * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(initial * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(initial * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(initial * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(initial * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(initial * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(initial * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(initial, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(initial, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(initial, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


def myCNN (INPUT, 
           sortie=10, #Le nombre de catégories prévues
           original = 64 #Le nombre de features maps utilisé au départ
          ):
    """
    Crée un réseau de type CNN pour la labellisation

    Parameters
    ----------
        - INPUT : la taille des images à utiliser et le nombre de channels couleur (1 pour les DICOM)
        - sortie : int, le nombre de class voulues en sortie, correspond au nombre de sous dossiers si vous utilisez keras.
        - original : nombre de feature map à utiliser au déaprt (allourdit le réseau)
        
    Returns
    -------
        - model : le réseau prêt à l'entrainement
    
    Notes
    -----
    Ne pas hésiter à modifier les hyperparamètres :
    - le learning rate
    - l'optimizer
    
    """
    model = tf.keras.Sequential([
        Conv2D(original, 3, activation='relu', input_shape=INPUT), Activation('relu'),
        Conv2D(original, (3, 3)), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(original *2, (3, 3), padding='same'), Activation('relu'),
        Conv2D(original *2, (3, 3)), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(original *4, (3, 3), padding='same'), Activation('relu'),
        Conv2D(original *4, (3, 3)), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(256), Activation('relu'),
        Dropout(0.5),
        Dense(sortie, activation='softmax')
    ])
    model.compile(optimizer = RMSprop(lr=0.0001),
                  loss      = "categorical_crossentropy", 
                  metrics   = ["accuracy"])
    return model

def myCNN2 (INPUT, 
           sortie=10, #Le nombre de catégories prévues
           original = 64 #Le nombre de features maps utilisé au départ
          ):
    """
    Crée un réseau de type CNN pour la labellisation

    Parameters
    ----------
        - INPUT : la taille des images à utiliser et le nombre de channels couleur (1 pour les DICOM)
        - sortie : int, le nombre de class voulues en sortie, correspond au nombre de sous dossiers si vous utilisez keras.
        - original : nombre de feature map à utiliser au déaprt (allourdit le réseau)
        
    Returns
    -------
        - model : le réseau prêt à l'entrainement
    
    Notes
    -----
    Ne pas hésiter à modifier les hyperparamètres :
    - le learning rate
    - l'optimizer
    
    """
    model = tf.keras.Sequential([
        Conv2D(original, 3, input_shape=INPUT, activation=LeakyReLU(alpha=0.3)),
        Conv2D(original, (3, 3), activation=LeakyReLU(alpha=0.3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(original *2, (3, 3), padding='same', activation=LeakyReLU(alpha=0.3)),
        Conv2D(original *2, (3, 3), activation=LeakyReLU(alpha=0.3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(original *4, (3, 3), padding='same', activation=keras.layers.LeakyReLU(alpha=0.3)),
        Conv2D(original *4, (3, 3), activation=LeakyReLU(alpha=0.3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(sortie, activation='softmax')
    ])
    model.compile(optimizer = RMSprop(lr=0.0001),
                  loss      = "categorical_crossentropy", 
                  metrics   = ["accuracy"])
    return model


def mySNN (INPUT,sortie=10):
    """
    Crée un réseau de type CNN pour la labellisation, ce type de réseau particulier réalise une normalisation autour de zero, 
    ce type de réseau porte le nom de Self Normalizing Networks (SNN)

    Parameters
    ----------
        - INPUT : tuple, la taille des images à utiliser et le nombre de channels couleur (1 pour les DICOM)
        - sortie : int, le nombre de class voulues en sortie, correspond au nombre de sous dossiers si vous utilisez keras.

    Returns
    -------
        - model : le réseau prêt à l'entrainement
    
    Notes
    -----
    Ne pas hésiter à modifier les hyperparamètres :
    - le learning rate
    - l'optimizer
    
    """
    inputs = keras.Input(shape=INPUT, dtype='float32', name='images')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=INPUT,kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.1))

    
    model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.1))
    
    model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AlphaDropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(256,kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('selu'))
    model.add(AlphaDropout(0.2))
    model.add(Dense(sortie,kernel_initializer='lecun_normal',bias_initializer='zeros'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    type_output = model(inputs)
    return model


def ComparaisonResultats(Nombre_a_afficher, model, test_gen, categories, color="gray", colonnes=2, reset=False):
    """
    Ni Keras ni Tensorflow ne propose de fonction simple pour afficher les résultats en image des prédictions faites par le réseau et
    leur comparaison par rapport à la labellisation réelle.
    Cette fonction rattrape cette lacune en permettant un affichage simple

    Parameters
    ----------
        - Nombre_a_afficher : int, le nombre d'exemples que l'on veut afficher
        - model : model tensorflow, le réseau de neurones
        - test_gen : generator, le test_generator que nous utilisons
        - categories : list, liste des classes à nommer,
        - color :  string, nom des couleurs à utiliser pour l'affichage, cf matplotlib : https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        - colonnes : combien d'images afficher sur une même ligne, ne peut dépasser la taille du batch du generator
        - reset : boolean, retourne au début du generator
  
    
    """
    if reset ==True :
        test_gen.reset()
    
    prediction=model.predict(test_gen,steps=Nombre_a_afficher/test_gen.batch_size,verbose=1)
    predicted_class_indices=np.argmax(prediction,axis=1)
    predictions = [categories[k] for k in predicted_class_indices]
    number=0
    while number < Nombre_a_afficher :
        x,y = test_gen.next()

        plt.figure(figsize=(20,20))
        lignes= int(test_gen.batch_size/colonnes)
        if colonnes > Nombre_a_afficher:
            colonnes = Nombre_a_afficher
        if test_gen.batch_size%colonnes !=0:
            lignes+=1
        for i in range(0,test_gen.batch_size):
            plt.subplot(lignes,colonnes,i+1)
            plt.imshow(x[i,:,:,0], cmap=color)
            titre = 'Vérité: '+ categories[np.argmax(y[i])] +'\n vs Predit: '+predictions[i]
            plt.title(titre)
            plt.axis('off')
        plt.show()
        number += test_gen.batch_size