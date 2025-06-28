import numpy as np
import cv2
import imutils
from imutils.video import FPS
from math import sqrt
import time

# --- PARÂMETROS DE AJUSTE ---
VIDEO_SOURCE = "./video/video_transito3.mp4"
PROTOTXT_PATH = "./MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "./MobileNetSSD_deploy.caffemodel"
ITEM_IDENTIFICADO = "car"

# Ajuste estes valores para otimizar a detecção e rastreamento
CONFIDENCE_THRESHOLD = 0.5  # Confiança mínima da detecção
MAX_DISTANCE = 75           # Distância máxima (em pixels) para associar um objeto
STABILITY_THRESHOLD = 10     # Nº de frames consecutivos que um objeto deve ser visto para ser contado
DISAPPEARED_THRESHOLD = 15  # Nº de frames que um objeto pode desaparecer antes de ser esquecido

# --- INICIALIZAÇÃO ---
camera = cv2.VideoCapture(VIDEO_SOURCE)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] Carregando modelo...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
fps = FPS().start()

# --- ESTRUTURA DE DADOS DO RASTREADOR ---
tracked_objects = {}
next_object_id = 0
total_real_count = 0

# --- LOOP PRINCIPAL ---
while True:
    (grabbed, image) = camera.read()
    if not grabbed:
        print("Vídeo finalizado ou ocorreu um erro.")
        break

    image = imutils.resize(image, width=700)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    current_frame_objects = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == ITEM_IDENTIFICADO:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                current_frame_objects.append({'centroid': (cX, cY), 'box': (startX, startY, endX, endY)})

    # Se não há objetos sendo rastreados, registra todos os atuais
    if len(tracked_objects) == 0:
        for obj in current_frame_objects:
            tracked_objects[next_object_id] = {
                'centroid': obj['centroid'],
                'box': obj['box'],
                'frames_unseen': 0, 'frames_seen': 1, 'counted': False
            }
            next_object_id += 1
    else:
        object_ids = list(tracked_objects.keys())
        previous_centroids = [tracked_objects[obj_id]['centroid'] for obj_id in object_ids]
        
        used_detections = [False] * len(current_frame_objects)
        
        for i, obj_id in enumerate(object_ids):
            min_dist = MAX_DISTANCE
            best_detection_idx = -1
            
            for j, new_obj in enumerate(current_frame_objects):
                if not used_detections[j]:
                    dist = sqrt((previous_centroids[i][0] - new_obj['centroid'][0])**2 + (previous_centroids[i][1] - new_obj['centroid'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_detection_idx = j
            
            if best_detection_idx != -1:
                tracked_objects[obj_id]['centroid'] = current_frame_objects[best_detection_idx]['centroid']
                tracked_objects[obj_id]['box'] = current_frame_objects[best_detection_idx]['box'] 
                tracked_objects[obj_id]['frames_unseen'] = 0
                tracked_objects[obj_id]['frames_seen'] += 1
                used_detections[best_detection_idx] = True
            else:
                tracked_objects[obj_id]['frames_unseen'] += 1

        for i, used in enumerate(used_detections):
            if not used:
                tracked_objects[next_object_id] = {
                    'centroid': current_frame_objects[i]['centroid'],
                    'box': current_frame_objects[i]['box'],
                    'frames_unseen': 0, 'frames_seen': 1, 'counted': False
                }
                next_object_id += 1

    objects_to_delete = []
    for obj_id, data in tracked_objects.items():
        if data['frames_seen'] >= STABILITY_THRESHOLD and not data['counted']:
            total_real_count += 1
            data['counted'] = True
            print(f"CARRO {obj_id} CONFIRMADO E CONTADO! Total: {total_real_count}")

        if data['frames_unseen'] > DISAPPEARED_THRESHOLD:
            objects_to_delete.append(obj_id)

        text = f"ID {obj_id}"
        color = (0, 255, 0)
        if data['counted']:
            color = (0, 0, 255)

        (startX, startY, endX, endY) = data['box']
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        cv2.putText(image, text, (data['centroid'][0] - 10, data['centroid'][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(image, data['centroid'], 4, color, -1)
    
    for obj_id in objects_to_delete:
        del tracked_objects[obj_id]

    cv2.putText(image, f"Carros Contados: {total_real_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Frame", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break
    fps.update()

# --- FINALIZAÇÃO ---
fps.stop()
print(f"[INFO] FPS: {fps.fps():.2f}")
print(f"[INFO] Contagem final de carros unicos: {total_real_count}")
camera.release()
cv2.destroyAllWindows()