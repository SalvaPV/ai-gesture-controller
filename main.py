import cv2
import mediapipe as mp
import math
import numpy as np

# --- CONFIGURACIÓN INICIAL ---
# Inicializamos MediaPipe Hands (modelo pre-entrenado de Google)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración de detección: confianza del 70% para evitar falsos positivos
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Captura de video
cap = cv2.VideoCapture(0)

def calcular_distancia(x1, y1, x2, y2):
    """Calcula la distancia Euclidiana entre dos puntos."""
    return math.hypot(x2 - x1, y2 - y1)

print("Iniciando sistema... Presiona 'q' para salir.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convertir color de BGR (OpenCV) a RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para detectar manos
    results = hands.process(image)

    # Volver a convertir a BGR para mostrar en pantalla
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los "esqueletos" de la mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener coordenadas de los puntos clave (Landmarks)
            # 4 = Punta del Pulgar, 8 = Punta del Índice
            h, w, c = image.shape
            x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            # Calcular el centro de la línea
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # --- DIBUJO DE INTERFAZ GRÁFICA ---
            cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Pulgar
            cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED) # Índice
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)      # Línea de conexión
            cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # Centro

            # --- MATEMÁTICA APLICADA ---
            # Calcular longitud de la línea (Distancia Euclidiana)
            length = calcular_distancia(x1, y1, x2, y2)

            # Mapeo de rangos (Interpolación lineal)
            # Rango de mano (aprox): 30 a 250 pixeles -> Rango Volumen: 0 a 100%
            vol_percentage = np.interp(length, [30, 250], [0, 100])
            
            # Feedback visual (Barra de volumen)
            cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
            vol_bar = np.interp(length, [30, 250], [400, 150]) # Mapeo para la barra gráfica
            cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            
            cv2.putText(image, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

    cv2.imshow('Ingeniería Inteligente - Gesto Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
