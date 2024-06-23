import cv2
import numpy as np

class YOLODetector:
    def __init__(self, config_path, weights_path, classes_path):
        # Chargement du modèle YOLO
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # Chargement des noms des classes d'objet
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, image):
        height, width = image.shape[:2]
        # Prétraitement de l'image pour l'adapter au modèle YOLO
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        #Initialisation des listes pour stocker les boites englobantes, les confiances et les identifiants de la classe
        boxes, confidences, class_ids = [], [], []

        #Parcours des sorties du réseau
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Application de la suppression non maximale pour supprimer les détections redondantes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                # Dessiner des rectangles uniquement pour les ordinateurs (laptop) et les souris (mouse)
                if label in ['laptop', 'mouse']:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def main():
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    classes_path = 'coco.names'
    yolo = YOLODetector(config_path, weights_path, classes_path)
    
    cap = cv2.VideoCapture(0)  # Ouvre la caméra par défaut (il est possible remplacer 0 par le chemin d'un fichier vidéo) 
    while True:
        ret, frame = cap.read() #Lire une frame de la caméra
        if not ret:
            break #Si la frame n'est pas lue correctement, quitter la boucle

        #Appliquer la détection YOLO sur la frame
        frame = yolo.detect(frame)

        #Afficher la frame avec les détections
        cv2.imshow('YOLO Detection', frame)
        #Taper la touche 'q' pour quitter la boucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Libérer la capture vidéo et fermer toutes les fenetres ouvertes    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
