# **IA en Python**

##### **Par Tiago Da Silva**





Cette IA en Python est un élément important de mes projets personnels. C’est l’un des plus gros projets que j’ai entrepris, et j’y ai consacré énormément de temps. 



J’ai utilisé plusieurs environnements et bibliothèques afin de mener ce projet à son état actuel. 



Tout d’abord, Torch, une bibliothèque Python qui m’a fourni de nombreux outils facilitant la création d’IA. Je l’ai beaucoup utilisée au début du projet. Par la suite, j’ai préféré retirer certains éléments afin de revenir vers un code plus proche du Python classique. Cependant, lorsque j’ai donné un corpus de texte important à mon IA, le CPU de mon ordinateur a commencé à surchauffer. J’ai donc dû ajouter une seconde technologie pour résoudre ce problème. 



Il s’agit de CUDA, une technologie permettant d’exécuter les calculs sur la carte graphique NVIDIA de l’ordinateur. Grâce à cela, j’ai pu déplacer une grande partie des calculs vers le GPU, ce qui a considérablement amélioré les performances (mon PC m'en remercie).



Pour utiliser cette configuration, j’ai créé un environnement appelé "rtx\_envir". Celui-ci peut être activé via le terminal et permet d’exécuter le code en utilisant l’accélération GPU. 

# 

##### **Fonctionnement du code**



###### **Le code fonctionne en trois étapes principales :**



1. **Compilation :** le programme récupère les textes des corpus indiqués et les rassemble dans une seule chaîne de caractères.
2. **Entraînement :** l’IA lit ce texte par blocs d’une taille prédéfinie. Elle tokenize ensuite le texte et l’indexe dans sa mémoire. C’est ainsi qu’elle apprend progressivement les structures du langage, les mots et certaines règles grammaticales. 
3. **Génération :** dans cette étape, l’IA génère du texte en prédisant le caractère suivant à partir du contexte précédent. 



À l’heure actuelle, l’IA n’est pas encore capable de comprendre ou de répondre directement à une question d’un utilisateur. L’un des objectifs futurs de ce projet est donc d’orienter son développement vers des interactions plus avancées.



J'ai également l'intention de séparer le code de la génération, et celui de la compilation + entraînement. 
