# ProjetAnnuel



# Problematique

L'objectif du projet est de pouvoir déterminer à quel pays appartient une image représentant son drapeau national.
Pour ce faire nous devions créer et entrainer plusieurs PMC grace à une librairie dynamique developée en C++ avec une configuration précise jusqu'a convergence. Une fois les modèles entrainées et pret à predire nous devions developper une application web permettant à un utilisateur d'importer une image et de réaliser des prédictions dessus.
De plus, nous devions developper une application web permettant à un utilisateur de faire ses prédictions via une interface.


# Présentation des dossiers

## application :

Application réalisée en django permettant à un utilisateur de faire des prédictions sur des images qu'il aura uploadé.
Pour lancer l'application il faut lancer la commande.

```bash
python manage.py runserver
```


## dataset :

Jeu de données d'entrainement et de test sur 3 types de drapeaux nationaux (Espagne, France et Japon)


## library :

Librairie dynamique developée en C++ permettant de créer, entrainer, sauvegarder et détruire des modèles linéaires et des PMC.
La librairie est utilisée en Python à l'aide du module ctypes. Elle est d’une part utilisée sur PyCharm et sur Jupyter Notebook afin de tester les fonctions relatives aux algorithmes d’apprentissage
implémentant les modèles. 


## extern_librairies :

Librairies externes nécessaires au développement de la librairie principale . Ex: librairie JSON qui va faciliter l’enregistrement des modèles dans des fichiers JSON (ainsi que leur chargement).


## models : 

Dossier dans lequel sont enregistrés les modèles entrainés en C++ et en Keras.


## scripts_notebooks :

### Partie Keras :

| Nom | Type | Description |
| ------|-----|-----|
| create_keras_mlp_models | Notebook| Permet de créer les différents modèles PMC avec la librairie Tensorflow Keras |
| train_keras_mlp_models | Notebook | Permet d'entrainer les différents modèles PMC crées avec la librairie Tensorflow Keras |
| predict_keras_mlp_model | Script | Permet d'effectuer des prédictions sur les différents modèles PMC crées avec la librairie Tensorflow Keras |


### Partie C++ :

| Nom | Type | Description |
| ------|-----|-----|
| create_cpplibrary_mlp_models| Notebook| Permet de créer les différents modèles PMC avec la librairie codée en C++ |
| train_cpplibrary_mlp_models| Notebook | Permet d'entrainer les différents modèles PMC crées avec la librairie codée en C++ |
| predict_cpplibrary_mlp_model| Script | Permet d'effectuer des prédictions sur les différents modèles PMC crées avec la librairie codée en C++ |


## others : 

Dossier contenant :
-> Le rapport final représentant les résultats des modèles (prédictions et analyses des courbes obtenues avec Tensorboard)
-> Script permettant d'évaluer les stats du dataset avec Tensorflow Keras









