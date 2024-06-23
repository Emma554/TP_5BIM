# TP_5BIM
TP 5BIM 5ème année Ecole-IT
Ce projet implémente un système de détection d'ordinateur et de souris sur des flux vidéo à l'aide de OpenCV et YOLO.

Explication des principales parties du code:

	YOLODetector:
		
	    .init
	    
	    Charge le modèle YOLO avec les fichiers de configuration (yolov3.cfg) et de poids(yolov3.weights que je n'ai malheureusement pas pu importer car il est trop lourd).
	    Récupère les noms des couches de sortie du réseau.
	    Charge les noms des classes d'objets depuis un fichier (coco.names).
	    
	    .detect:
	    
	    Prétraite l'image pour qu'elle soit compatible avec le modèle YOLO.
	    Passe l'image à travers le réseau pour obtenir les prédictions.
	    Filtre les prédictions selon la confiance et calcule les coordonnées des boîtes englobantes.
	    Applique la suppression non maximale pour éliminer les détections redondantes.
	    Dessine les boîtes englobantes et les étiquettes des objets détectés sur l'image.
	
	
	main:
	
	Initialise le détecteur YOLO avec les chemins des fichiers de configuration, de poids et des classes.
	Ouvre un flux vidéo depuis la caméra.
	Pour chaque frame, applique la détection et affiche les résultats.
	Quitte la boucle et libère les ressources si la touche 'q' est pressée.


Exécution du projet:
	Après avoir lancé le projet, il faut juste présenter les objets détectables (laptop ou souris) devant la caméra et voir le résultat de la détection par la suite. Vu que je n'ai pas pu importer le fichier yolov3.weights, vous devrez le télécharger, le mettre dans le dossier du projet au préalable avant de le lancer.
