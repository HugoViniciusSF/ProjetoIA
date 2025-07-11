import numpy as np
import cv2
import imutils
from imutils.video import FPS
from math import sqrt
import time
from classificador_bayesiano import classificar_estado_bayesiano
from classificador_markov import ClassificadorMarkoviano

#Paranetros de ajustes
VIDEO_SOURCE = "./video/video_transito4.mp4"
PROTOTXT_PATH = "./MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "./MobileNetSSD_deploy.caffemodel"

#Array de objetos para identificação (carros e motos)
ITENS_IDENTIFICADOS = ["car", "motorbike"]

VEHICLE_PARAMS = {
    'car': {
        'confidence_threshold': 0.5,        #Confiança padrão para carros
        'stability_threshold': 10,          #Frames necessários para contar carros
        'disappeared_threshold': 15,        #Frames para esquecer carros
        'max_distance': 75                  #Raio de busca para carros
    },
    'motorbike': {
        'confidence_threshold': 0.2,       
        'stability_threshold': 0.5,
        'disappeared_threshold': 5,
        'max_distance': 100       
    }
}

TIME_WINDOW_SECONDS = 10  #Janela de tempo para cada análise de trânsito.

camera = cv2.VideoCapture(VIDEO_SOURCE)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"]

print("[INFO] Carregando modelo...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
fps = FPS().start()

tracked_objects = {}
next_object_id = 0

total_cars_count = 0
total_motorbikes_count = 0
total_vehicles_count = 0

classificador_markov = ClassificadorMarkoviano()

estado_bayesiano_atual = "Indeterminado"
estado_markoviano_atual = "Indeterminado"

#Variáveis para a análise baseada em janela de tempo.
cars_in_window = 0
motorbikes_in_window = 0
vehicles_in_window = 0
last_analysis_time = time.time()

while True:
    (grabbed, image) = camera.read()
    if not grabbed:
        break
    
    image = imutils.resize(image, width=700)
    (h, w) = image.shape[:2]
    
    #Detecção dos objetos
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    current_frame_objects = []
    
    #Filtra as deteções com thresholds específicos para cada tipo
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        detected_class = CLASSES[idx]
        
        #Verifica se é um veículo de interesse e aplica limite específico
        if detected_class in ITENS_IDENTIFICADOS:
            vehicle_threshold = VEHICLE_PARAMS[detected_class]['confidence_threshold']
            
            if confidence > vehicle_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                
                current_frame_objects.append({
                    'centroid': (cX, cY), 
                    'box': (startX, startY, endX, endY),
                    'vehicle_type': detected_class,
                    'confidence': confidence
                })
    
    #Lógica de associação de IDs
    if len(tracked_objects) == 0:
        for obj in current_frame_objects:
            tracked_objects[next_object_id] = {
                'centroid': obj['centroid'], 
                'box': obj['box'], 
                'vehicle_type': obj['vehicle_type'],
                'confidence': obj['confidence'],
                'frames_unseen': 0, 
                'frames_seen': 1, 
                'counted': False
            }
            next_object_id += 1
    else:
        object_ids = list(tracked_objects.keys())
        previous_centroids = [data['centroid'] for data in tracked_objects.values()]
        used_detections = [False] * len(current_frame_objects)
        
        for i, obj_id in enumerate(object_ids):
            vehicle_type = tracked_objects[obj_id]['vehicle_type']
            max_distance = VEHICLE_PARAMS[vehicle_type]['max_distance']
            
            min_dist = max_distance
            best_detection_idx = -1
            
            for j, new_obj in enumerate(current_frame_objects):
                if not used_detections[j]:
                    #Prioriza mesmo tipo de veículo, mas permite associação cruzada
                    if tracked_objects[obj_id]['vehicle_type'] == new_obj['vehicle_type']:
                        dist = sqrt((previous_centroids[i][0] - new_obj['centroid'][0])**2 + 
                                (previous_centroids[i][1] - new_obj['centroid'][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_detection_idx = j
            
            if best_detection_idx != -1:
                tracked_objects[obj_id]['centroid'] = current_frame_objects[best_detection_idx]['centroid']
                tracked_objects[obj_id]['box'] = current_frame_objects[best_detection_idx]['box']
                tracked_objects[obj_id]['confidence'] = current_frame_objects[best_detection_idx]['confidence']
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
                    'vehicle_type': current_frame_objects[i]['vehicle_type'],
                    'confidence': current_frame_objects[i]['confidence'],
                    'frames_unseen': 0, 
                    'frames_seen': 1, 
                    'counted': False
                }
                next_object_id += 1
    
    #Lógica de contagem e remoção de veiculos
    objects_to_delete = []
    
    for obj_id, data in tracked_objects.items():
        vehicle_type = data['vehicle_type']
        stability_threshold = VEHICLE_PARAMS[vehicle_type]['stability_threshold']
        disappeared_threshold = VEHICLE_PARAMS[vehicle_type]['disappeared_threshold']
        
        #Contagem com thresholds específicos
        if data['frames_seen'] >= stability_threshold and not data['counted']:
            if data['vehicle_type'] == 'car':
                total_cars_count += 1
                cars_in_window += 1
            elif data['vehicle_type'] == 'motorbike':
                total_motorbikes_count += 1
                motorbikes_in_window += 1
            
            total_vehicles_count += 1
            vehicles_in_window += 1
            data['counted'] = True
            
            #Log de consulta
            print(f"[CONTADO] {data['vehicle_type'].upper()} ID {obj_id} - Confiança: {data['confidence']:.2f} - Frames vistos: {data['frames_seen']}")
        
        #Remoção com thresholds específicos
        if data['frames_unseen'] > disappeared_threshold:
            objects_to_delete.append(obj_id)
        
        #Desenho na tela
        if data['vehicle_type'] == 'car':
            color = (0, 255, 0) if not data['counted'] else (0, 0, 255)  #Verde/vermelho
        else:  # moto
            color = (255, 255, 0) if not data['counted'] else (0, 165, 255)  # marelo/laranja
        
        (startX, startY, endX, endY) = data['box']
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        vehicle_label = "Carro" if data['vehicle_type'] == 'car' else "Moto"
        text = f"{vehicle_label} {obj_id} ({data['confidence']:.2f})"
        cv2.putText(image, text, (data['centroid'][0] - 20, data['centroid'][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    #Remove veiculos que desapareceram
    for obj_id in objects_to_delete:
        del tracked_objects[obj_id]
    
    #ANÁLISE PERIÓDICA
    current_time = time.time()
    
    if current_time - last_analysis_time >= TIME_WINDOW_SECONDS:
        hora_do_dia = 14
        
        estado_bayesiano_atual = classificar_estado_bayesiano(vehicles_in_window, hora_do_dia)
        estado_markoviano_atual = classificador_markov.classificar_estado(vehicles_in_window)
        
        print("-" * 25, f"ANÁLISE ({hora_do_dia}h)", "-" * 25)
        print(f"Carros na janela: {cars_in_window}")
        print(f"Motos na janela: {motorbikes_in_window}")
        print(f"Total de veículos na janela: {vehicles_in_window}")
        print(f"Classificação Bayesiana: {estado_bayesiano_atual}")
        print(f"Classificação Markoviana: {estado_markoviano_atual}")
        print("-" * 60)
        
        last_analysis_time = current_time
        cars_in_window = 0
        motorbikes_in_window = 0
        vehicles_in_window = 0
    
    #Informacoes
    cv2.putText(image, f"Bayes: {estado_bayesiano_atual}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Markov: {estado_markoviano_atual}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(image, f"Carros: {total_cars_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Motos: {total_motorbikes_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(image, f"Total Veiculos: {total_vehicles_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("x"):
        break
    
    fps.update()

#Fim
fps.stop()
print(f"\n[INFO] FPS: {fps.fps():.2f}")
print(f"[INFO] Contagem final de carros únicos: {total_cars_count}")
print(f"[INFO] Contagem final de motos únicas: {total_motorbikes_count}")
print(f"[INFO] Contagem final total de veículos: {total_vehicles_count}")
print(f"[INFO] Classificação Bayesiana: {estado_bayesiano_atual}")
print(f"[INFO] Classificação Markoviana: {estado_markoviano_atual}")
camera.release()
cv2.destroyAllWindows()