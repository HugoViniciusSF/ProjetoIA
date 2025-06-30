import numpy as np
import cv2
import imutils
from imutils.video import FPS
from math import sqrt
import time

# --- IMPORTAÇÃO DOS MÓDULOS DE ANÁLISE ---
from classificador_bayesiano import classificar_estado_bayesiano
from classificador_markov import ClassificadorMarkoviano

# --- PARÂMETROS DE AJUSTE ---
VIDEO_SOURCE = "./video/video_transito.mp4"
PROTOTXT_PATH = "./MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "./MobileNetSSD_deploy.caffemodel"
ITEM_IDENTIFICADO = "car"

# Parâmetros de Rastreamento (Ajustáveis)
CONFIDENCE_THRESHOLD = 0.5  # Confiança mínima para uma deteção ser considerada válida.
MAX_DISTANCE = 75           # Raio de busca (em píxeis) para associar um carro ao seu ID anterior.
STABILITY_THRESHOLD = 10    # N.º de frames que um carro precisa ser visto para ser contado (evita falsos positivos).
DISAPPEARED_THRESHOLD = 15  # N.º de frames que um carro pode desaparecer antes de ser "esquecido".
TIME_WINDOW_SECONDS = 10    # Janela de tempo (em segundos) para cada análise de trânsito.

# --- INICIALIZAÇÃO ---
camera = cv2.VideoCapture(VIDEO_SOURCE)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]

print("[INFO] Carregando modelo...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
fps = FPS().start()

# --- ESTRUTURA DE DADOS DO RASTREADOR ---
# Este dicionário é a "memória" principal. A chave é o ID do carro.
tracked_objects = {}
next_object_id = 0
total_real_count = 0

# --- INICIALIZAÇÃO DOS CLASSIFICADORES E VARIÁVEIS DE ANÁLISE ---

classificador_markov = ClassificadorMarkoviano()
# Variáveis para guardar o último estado de cada classificador para exibição.
estado_bayesiano_atual = "Indeterminado"
estado_markoviano_atual = "Indeterminado"
# Variáveis para a análise baseada em janela de tempo.
cars_in_window = 0
last_analysis_time = time.time()

# --- CICLO PRINCIPAL ---
while True:
    (grabbed, image) = camera.read()
    if not grabbed:
        break

    image = imutils.resize(image, width=700)
    (h, w) = image.shape[:2]
    
    # --- Deteção de Objetos ---
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    current_frame_objects = []
    # Filtra as deteções para manter apenas os carros com confiança suficiente.
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

    # --- Lógica de Associação de IDs ---
    # Se não há carros a serem rastreados, todos os detetados são novos.
    if len(tracked_objects) == 0:
        for obj in current_frame_objects:
            # Atribui um novo ID
            tracked_objects[next_object_id] = {'centroid': obj['centroid'], 'box': obj['box'], 'frames_unseen': 0, 'frames_seen': 1, 'counted': False}
            next_object_id += 1
    else:
        # Tenta associar os carros detetados com os IDs existentes.
        object_ids = list(tracked_objects.keys())
        previous_centroids = [data['centroid'] for data in tracked_objects.values()]
        used_detections = [False] * len(current_frame_objects)
        
        for i, obj_id in enumerate(object_ids):
            min_dist = MAX_DISTANCE
            best_detection_idx = -1
            # Procura o carro detetado mais próximo do ID que estamos a analisar.
            for j, new_obj in enumerate(current_frame_objects):
                if not used_detections[j]:
                    dist = sqrt((previous_centroids[i][0] - new_obj['centroid'][0])**2 + (previous_centroids[i][1] - new_obj['centroid'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_detection_idx = j
            
            if best_detection_idx != -1:
                # Associação bem-sucedida: atualiza os dados do ID existente.
                tracked_objects[obj_id]['centroid'] = current_frame_objects[best_detection_idx]['centroid']
                tracked_objects[obj_id]['box'] = current_frame_objects[best_detection_idx]['box']
                tracked_objects[obj_id]['frames_unseen'] = 0
                tracked_objects[obj_id]['frames_seen'] += 1
                used_detections[best_detection_idx] = True
            else:
                # Associação falhou: o carro com este ID desapareceu neste frame.
                tracked_objects[obj_id]['frames_unseen'] += 1
                
        # Adiciona os carros não associados como novos objetos com novos IDs.
        for i, used in enumerate(used_detections):
            if not used:
                tracked_objects[next_object_id] = {'centroid': current_frame_objects[i]['centroid'], 'box': current_frame_objects[i]['box'], 'frames_unseen': 0, 'frames_seen': 1, 'counted': False}
                next_object_id += 1
    
    # --- Lógica de Contagem e Remoção de Objetos ---
    objects_to_delete = []
    for obj_id, data in tracked_objects.items():
        # Se um carro atingiu a estabilidade e ainda não foi contado...
        if data['frames_seen'] >= STABILITY_THRESHOLD and not data['counted']:
            total_real_count += 1
            cars_in_window += 1 # Adiciona à contagem da janela de tempo atual.
            data['counted'] = True # Marca para nunca mais ser contado.

        # Se um carro desapareceu por tempo demais...
        if data['frames_unseen'] > DISAPPEARED_THRESHOLD:
            objects_to_delete.append(obj_id)
        
        # --- Lógica de Desenho dos Carros e IDs ---
        color = (0, 255, 0) if not data['counted'] else (0, 0, 255) # Verde se rastreando, vermelho se contado.
        (startX, startY, endX, endY) = data['box']
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        text = f"ID {obj_id}"
        cv2.putText(image, text, (data['centroid'][0] - 10, data['centroid'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Remove os objetos que desapareceram da memória.
    for obj_id in objects_to_delete:
        del tracked_objects[obj_id]

    # --- BLOCO DE ANÁLISE E COMPARAÇÃO ---
    current_time = time.time()
    # A cada tempo determinado em TIME_WINDOW_SECONDS, executamos a análise.
    if current_time - last_analysis_time >= TIME_WINDOW_SECONDS:
        # A hora é fixada para simular um horário.
        hora_do_dia = 14
        
        # --- COMPARAÇÃO DIRETA ---
        # 1. Classificação usando rede bayseana.
        estado_bayesiano_atual = classificar_estado_bayesiano(cars_in_window, hora_do_dia)
        
        # 2. Classificação usando Markov.
        estado_markoviano_atual = classificador_markov.classificar_estado(cars_in_window)
        
        # Impressões.
        print("-" * 20, f"ANÁLISE ({hora_do_dia}h)", "-" * 20)
        print(f"Carros na janela: {cars_in_window}")
        print(f"* Classificação Bayesiana: {estado_bayesiano_atual}")
        print(f"* Classificação Markoviana : {estado_markoviano_atual}")

        # Reseta a janela para a próxima análise.
        last_analysis_time = current_time
        cars_in_window = 0

    # --- Adiciona as informações de comparação na tela ---
    cv2.putText(image, f"Bayes: {estado_bayesiano_atual}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Markov: {estado_markoviano_atual}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Carros Contados: {total_real_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Frame", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break
    fps.update()

# --- FINALIZAÇÃO ---
fps.stop()
print(f"[INFO] FPS: {fps.fps():.2f}")
print(f"[INFO] Contagem final de carros únicos: {total_real_count}")
camera.release()
cv2.destroyAllWindows()
