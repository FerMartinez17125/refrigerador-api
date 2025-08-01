import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime
import math
import json
import time
import threading
import requests
from pathlib import Path


class AdvancedCloudDetector:
    """
    DETECTOR AVANZADO CLOUD: Toda tu lógica original + Procesamiento en la nube
    - Sistema de modos (hands/movement/auto)
    - Detección de manos parciales inteligente
    - Background subtraction para movimiento
    - Sensibilidad ajustable (5 niveles)
    - Análisis cloud con Gemini
    - Inventario centralizado
    """

    def __init__(self):
        # Tu configuración original de MediaPipe con MÁS PERMISIVIDAD
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Tu configuración original
            min_tracking_confidence=0.2,  # Tu configuración original
            model_complexity=1  # Modelo más complejo para mejor detección
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Variables de estado originales
        self.detection_line = None  # Zona de ENTRADA (6 puntos)
        self.line_defined = False
        self.exit_zone = None  # Zona de SALIDA (6 puntos)
        self.exit_zone_defined = False
        self.analysis_zone = None
        self.zone_defined = False
        self.zone_mask = None
        self.last_photo_time = 0
        self.photo_cooldown = 2  # 2 segundos entre fotos

        # TU SISTEMA DE MODOS - MODO POR DEFECTO AUTO (como en tu original)
        self.detection_mode = "auto"  # "hands", "movement", o "auto" (híbrido)

        # TU SISTEMA DE DETECCIÓN DE MOVIMIENTO
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.movement_threshold = 1000  # Píxeles que deben cambiar
        self.min_contour_area = 500  # Área mínima de contorno
        self.max_contour_area = 50000  # Área máxima de contorno
        self.sensitivity_level = 2  # Nivel de sensibilidad (1-5)
        self.last_movement_time = 0

        # TU SISTEMA DE MANOS PARCIALES
        self.min_landmarks_for_detection = 5  # Mínimo de landmarks para considerar válida
        self.partial_hand_mode = True  # Habilitar detección parcial
        self.last_valid_palm_position = None  # Última posición válida de palma
        self.force_capture_mode = False  # Modo forzar captura sin detectar objeto

        # CONFIGURACIÓN CLOUD API
        self.base_url = "https://refrigerador-api-4.onrender.com"
        self.analizar_url = f"{self.base_url}/analizar"
        self.inventario_url = f"{self.base_url}/inventario"
        self.request_timeout = 30

        # Directorios
        self.photos_dir = "fotos_cloud_advanced"
        os.makedirs(self.photos_dir, exist_ok=True)

        # TU SISTEMA DE INVENTARIO (adaptado para cloud)
        self.pending_photos = []  # Lista de fotos pendientes de análisis
        self.cloud_responses = []

        # Categorías válidas para clasificación (de tu código original)
        self.valid_categories = ["carne", "verduras", "frutas", "lacteos", "granos", "otros"]

        print("✅ Detector AVANZADO CLOUD inicializado")
        print(f"🌐 API Base: {self.base_url}")
        print(f"🎯 Modo inicial: {self.detection_mode.upper()}")
        print(f"🖐️ Manos parciales: {'✅ Habilitado' if self.partial_hand_mode else '❌ Deshabilitado'}")
        print(f"🎛️ Sensibilidad: Nivel {self.sensitivity_level}/5")

    # === TU SISTEMA DE SENSIBILIDAD ORIGINAL ===

    def adjust_sensitivity(self, increase=True):
        """Ajusta la sensibilidad de detección de movimiento (TU CÓDIGO ORIGINAL)"""
        if increase and self.sensitivity_level < 5:
            self.sensitivity_level += 1
        elif not increase and self.sensitivity_level > 1:
            self.sensitivity_level -= 1

        # Ajustar parámetros según nivel de sensibilidad (TU CONFIGURACIÓN)
        sensitivity_configs = {
            1: {"threshold": 2000, "min_area": 1000, "max_area": 80000},  # Baja sensibilidad
            2: {"threshold": 1000, "min_area": 500, "max_area": 50000},  # Media-baja
            3: {"threshold": 800, "min_area": 300, "max_area": 40000},  # Media
            4: {"threshold": 500, "min_area": 200, "max_area": 30000},  # Media-alta
            5: {"threshold": 300, "min_area": 100, "max_area": 20000}  # Alta sensibilidad
        }

        config = sensitivity_configs[self.sensitivity_level]
        self.movement_threshold = config["threshold"]
        self.min_contour_area = config["min_area"]
        self.max_contour_area = config["max_area"]

        print(f"🎛️ Sensibilidad: Nivel {self.sensitivity_level}/5")
        print(f"   Umbral: {self.movement_threshold}, Área mín: {self.min_contour_area}")

    # === TU SISTEMA DE MANOS PARCIALES ORIGINAL ===

    def validate_partial_hand(self, landmarks):
        """Valida si los landmarks detectados son suficientes para una mano parcial (TU CÓDIGO)"""
        if not landmarks:
            return False, 0, None

        # Contar landmarks válidos (que tienen confianza)
        valid_landmarks = []
        for i, landmark in enumerate(landmarks):
            # Verificar que las coordenadas están en rango válido
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                valid_landmarks.append((i, landmark))

        valid_count = len(valid_landmarks)

        # Necesitamos al menos min_landmarks_for_detection puntos válidos
        if valid_count < self.min_landmarks_for_detection:
            return False, valid_count, None

        # Intentar encontrar un punto de referencia para la "posición de mano"
        estimated_center = self.estimate_hand_center(valid_landmarks, landmarks)

        return True, valid_count, estimated_center

    def estimate_hand_center(self, valid_landmarks, all_landmarks):
        """Estima el centro de la mano basándose en landmarks disponibles (TU CÓDIGO)"""
        # Prioridad de landmarks para estimar centro:
        # 1. Palma (landmark 9) si está disponible
        # 2. Muñeca (landmark 0) si está disponible
        # 3. Promedio de puntas de dedos disponibles
        # 4. Promedio de todos los landmarks válidos

        landmark_positions = {idx: lm for idx, lm in valid_landmarks}

        # Opción 1: Usar palma si está disponible
        if 9 in landmark_positions:
            palm = all_landmarks[9]
            self.last_valid_palm_position = (palm.x, palm.y)
            return (palm.x, palm.y)

        # Opción 2: Usar muñeca si está disponible
        if 0 in landmark_positions:
            wrist = all_landmarks[0]
            return (wrist.x, wrist.y)

        # Opción 3: Usar puntas de dedos (landmarks 4, 8, 12, 16, 20)
        fingertips = [4, 8, 12, 16, 20]
        available_fingertips = [(idx, all_landmarks[idx]) for idx in fingertips if idx in landmark_positions]

        if available_fingertips:
            avg_x = sum(lm.x for _, lm in available_fingertips) / len(available_fingertips)
            avg_y = sum(lm.y for _, lm in available_fingertips) / len(available_fingertips)
            return (avg_x, avg_y)

        # Opción 4: Promedio de todos los landmarks válidos
        if valid_landmarks:
            avg_x = sum(lm.x for _, lm in valid_landmarks) / len(valid_landmarks)
            avg_y = sum(lm.y for _, lm in valid_landmarks) / len(valid_landmarks)
            return (avg_x, avg_y)

        # Usar última posición válida conocida si no hay nada más
        return self.last_valid_palm_position

    def detect_object_in_partial_hand(self, landmarks):
        """Detecta si una mano parcial probablemente está sosteniendo un objeto (TU CÓDIGO)"""
        # Validar si la mano parcial es suficiente
        is_valid, landmark_count, estimated_center = self.validate_partial_hand(landmarks)

        if not is_valid:
            return False, 0, "insufficient_landmarks"

        # Para manos parciales, ser MÁS PERMISIVO en la detección
        # Si detectamos una mano parcial válida, asumir que probablemente sostiene algo

        # Contar cuántos dedos podemos ver
        fingertip_landmarks = [4, 8, 12, 16, 20]  # Puntas de dedos
        visible_fingertips = []

        for tip_idx in fingertip_landmarks:
            if tip_idx < len(landmarks):
                landmark = landmarks[tip_idx]
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    visible_fingertips.append((tip_idx, landmark))

        # LÓGICA MÁS PERMISIVA para manos parciales:
        if landmark_count >= self.min_landmarks_for_detection:
            # Si vemos muy pocos dedos = alta probabilidad de objeto
            if len(visible_fingertips) <= 2:
                return True, 0.9, "few_fingers_partial"
            # Si vemos algunos dedos = mediana-alta probabilidad
            elif len(visible_fingertips) <= 3:
                return True, 0.7, "some_fingers_partial"
            # Si vemos muchos dedos = aún podría tener objeto
            elif len(visible_fingertips) == 4:
                return True, 0.5, "many_fingers_partial"
            # Solo si vemos todos los dedos = probablemente mano abierta
            else:
                return True, 0.3, "all_fingers_partial"  # Aún darle una oportunidad

        # Si no hay suficientes landmarks, no detectar objeto
        return False, 0.1, "insufficient_landmarks"

    def check_partial_hand_zones(self, landmarks, frame_shape):
        """Verifica zonas usando detección de manos parciales (TU CÓDIGO MEJORADO)"""
        is_valid, landmark_count, estimated_center = self.validate_partial_hand(landmarks)

        if not is_valid:
            return False, False

        in_entry_zone = False
        in_exit_zone = False

        # Lista de landmarks a probar (en orden de prioridad)
        test_landmarks = [9, 0, 5, 13, 17, 1, 2, 3, 4, 8, 12, 16, 20]  # Palma, muñeca, bases, puntas

        # Recopilar todos los puntos válidos para verificar
        valid_test_points = []

        # Método 1: Usar centro estimado como primer punto
        if estimated_center:
            center_x = int(estimated_center[0] * frame_shape[1])
            center_y = int(estimated_center[1] * frame_shape[0])
            valid_test_points.append((center_x, center_y, "centro"))

        # Método 2: Agregar múltiples landmarks disponibles
        for landmark_idx in test_landmarks:
            if landmark_idx < len(landmarks):
                landmark = landmarks[landmark_idx]
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    test_x = int(landmark.x * frame_shape[1])
                    test_y = int(landmark.y * frame_shape[0])
                    valid_test_points.append((test_x, test_y, f"landmark_{landmark_idx}"))

        # Verificar cada punto contra ambas zonas
        entry_votes = 0
        exit_votes = 0

        for test_x, test_y, point_type in valid_test_points:
            # Verificar zona de ENTRADA
            if self.line_defined:
                area_points = np.array(self.detection_line, dtype=np.int32)
                result = cv2.pointPolygonTest(area_points, (test_x, test_y), False)
                if result >= 0:
                    entry_votes += 1

            # Verificar zona de SALIDA
            if self.exit_zone_defined:
                area_points = np.array(self.exit_zone, dtype=np.int32)
                result = cv2.pointPolygonTest(area_points, (test_x, test_y), False)
                if result >= 0:
                    exit_votes += 1

        # Determinar si está en zona basado en votos (más robusto)
        min_votes = max(1, len(valid_test_points) // 3)

        in_entry_zone = entry_votes >= min_votes
        in_exit_zone = exit_votes >= min_votes

        return in_entry_zone, in_exit_zone

    def draw_partial_hand_info(self, frame, landmarks, estimated_center, landmark_count):
        """Dibuja información específica para manos parciales (TU CÓDIGO)"""
        if estimated_center:
            center_x = int(estimated_center[0] * frame.shape[1])
            center_y = int(estimated_center[1] * frame.shape[0])

            # Dibujar centro estimado con color diferente
            cv2.circle(frame, (center_x, center_y), 10, (255, 165, 0), -1)  # Naranja
            cv2.putText(frame, "MANO PARCIAL", (center_x - 50, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            cv2.putText(frame, f"Landmarks: {landmark_count}", (center_x - 50, center_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    # === TU SISTEMA DE DETECCIÓN DE MOVIMIENTO ORIGINAL ===

    def detect_movement_in_zones(self, frame):
        """Detecta movimiento en las zonas de entrada y salida (TU CÓDIGO)"""
        # Aplicar background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por área
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                valid_contours.append(contour)

        # Verificar si hay movimiento en zonas específicas
        movement_in_entry = False
        movement_in_exit = False

        for contour in valid_contours:
            # Obtener centro del contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Verificar zona de ENTRADA
                if self.line_defined:
                    area_points = np.array(self.detection_line, dtype=np.int32)
                    result = cv2.pointPolygonTest(area_points, (cx, cy), False)
                    if result >= 0:
                        movement_in_entry = True

                # Verificar zona de SALIDA
                if self.exit_zone_defined:
                    area_points = np.array(self.exit_zone, dtype=np.int32)
                    result = cv2.pointPolygonTest(area_points, (cx, cy), False)
                    if result >= 0:
                        movement_in_exit = True

        return movement_in_entry, movement_in_exit, valid_contours, fg_mask

    def draw_movement_info(self, frame, contours):
        """Dibuja información de movimiento en el frame (TU CÓDIGO)"""
        # Dibujar contornos detectados
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        # Dibujar centros de contornos
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

                # Mostrar área del contorno
                area = cv2.contourArea(contour)
                cv2.putText(frame, f"A:{int(area)}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # === FUNCIONES CLOUD NUEVAS ===

    def send_photo_to_cloud_with_inventory(self, image_path, zone_type="unknown"):
        """Envía foto a la nube con actualización automática de inventario"""
        try:
            print(f"📤 Enviando {os.path.basename(image_path)} a la nube...")
            print(f"🎯 Zona: {zone_type.upper()}")

            with open(image_path, 'rb') as img_file:
                files = {'imagen': img_file}
                data = {'zone_type': zone_type}

                print("⏳ Analizando con Gemini + actualizando inventario...")
                response = requests.post(
                    self.analizar_url,
                    files=files,
                    data=data,
                    timeout=self.request_timeout
                )

            print(f"📊 Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                alimentos = result.get('alimentos_detectados', [])
                inventory_updated = result.get('inventory_updated', False)
                inventory_changes = result.get('inventory_changes', [])
                inventory_action = result.get('inventory_action', 'none')

                if alimentos:
                    print(f"✅ {len(alimentos)} alimentos detectados:")
                    for alimento in alimentos:
                        print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")

                    if inventory_updated:
                        action_text = "AGREGADOS AL" if inventory_action == "added" else "REMOVIDOS DEL"
                        print(f"📊 Inventario actualizado - {action_text} inventario cloud:")
                        for change in inventory_changes:
                            print(f"   {change}")
                    else:
                        print("⚠️ Inventario cloud no se pudo actualizar")

                else:
                    print("🤷 Nube: No se detectaron alimentos")

                cloud_response = {
                    'timestamp': datetime.now().isoformat(),
                    'zone_type': zone_type,
                    'image_path': image_path,
                    'alimentos': alimentos,
                    'inventory_updated': inventory_updated,
                    'inventory_changes': inventory_changes
                }
                self.cloud_responses.append(cloud_response)

                return result
            else:
                print(f"❌ Error del servidor: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("⏰ Timeout - La nube tardó demasiado")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_cloud_inventory(self):
        """Obtiene el inventario completo desde la nube"""
        try:
            print("📊 Obteniendo inventario desde la nube...")
            response = requests.get(self.inventario_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('inventario'), data.get('estadisticas')
            else:
                print(f"❌ Error obteniendo inventario: {response.status_code}")
                return None, None

        except Exception as e:
            print(f"❌ Error conectando con inventario cloud: {e}")
            return None, None

    def display_cloud_inventory(self):
        """Muestra el inventario desde la nube"""
        print("\n" + "=" * 60)
        print("☁️ INVENTARIO DEL REFRIGERADOR (CLOUD)")
        print("=" * 60)

        inventory_data, stats = self.get_cloud_inventory()

        if not inventory_data:
            print("❌ No se pudo obtener inventario desde la nube")
            print("=" * 60 + "\n")
            return

        alimentos = inventory_data.get('alimentos', [])

        if not alimentos:
            print("📭 Inventario cloud vacío")
        else:
            print(f"📊 Total: {stats['total_types']} tipos, {stats['total_items']} unidades")
            print(f"🕒 Actualizado: {inventory_data.get('actualizado', 'N/A')[:19].replace('T', ' ')}")
            print("")

            categories = {}
            for food in alimentos:
                categoria = food['categoria']
                if categoria not in categories:
                    categories[categoria] = []
                categories[categoria].append(food)

            for categoria in sorted(categories.keys()):
                foods = categories[categoria]
                categoria_items = sum(food['cantidad'] for food in foods)

                print(f"📂 {categoria.upper()} ({categoria_items} unidades):")
                for food in sorted(foods, key=lambda x: x['nombre']):
                    fecha = food.get('fecha_ingreso', 'N/A')
                    print(f"   • {food['cantidad']}x {food['nombre']} (desde {fecha})")
                print("")

        print("=" * 60 + "\n")

    def save_photo_and_send_to_cloud(self, frame, zone_type="unknown"):
        """Guarda foto localmente y envía a la nube para análisis"""
        current_time = datetime.now()

        # Verificar cooldown
        if (current_time.timestamp() - self.last_photo_time) < self.photo_cooldown:
            return False

        # Guardar imagen localmente
        filename = f"advanced_{zone_type}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.photos_dir, filename)
        cv2.imwrite(filepath, frame)

        print(f"📸 Foto guardada: {filename}")

        # Enviar a la nube con inventario automático
        result = self.send_photo_to_cloud_with_inventory(filepath, zone_type)

        self.last_photo_time = current_time.timestamp()
        return result is not None

    def show_cloud_responses(self):
        """Muestra historial de respuestas cloud"""
        print("\n" + "=" * 60)
        print("☁️ HISTORIAL DE ANÁLISIS AVANZADO + INVENTARIO CLOUD")
        print("=" * 60)

        if not self.cloud_responses:
            print("📭 No hay análisis registrados")
            return

        for i, response in enumerate(self.cloud_responses, 1):
            print(f"\n🔍 Análisis #{i} ({response['zone_type'].upper()})")
            print(f"⏰ {response['timestamp'][:19].replace('T', ' ')}")
            print(f"📄 {os.path.basename(response['image_path'])}")

            if response['alimentos']:
                for alimento in response['alimentos']:
                    print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")

                if response['inventory_updated'] and response['inventory_changes']:
                    print("   📊 Cambios en inventario cloud:")
                    for change in response['inventory_changes']:
                        print(f"      {change}")
            else:
                print("   🤷 Sin alimentos detectados")

        print("\n" + "=" * 60)

    # === TUS FUNCIONES DE DEFINICIÓN DE ZONAS ORIGINALES ===

    def define_detection_line(self, frame):
        """Define zona de ENTRADA con 6 puntos (TU CÓDIGO ORIGINAL)"""
        print("Haz clic en 6 puntos para definir la ZONA DE ENTRADA")
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                print(f"Punto ENTRADA {len(points)}: ({x}, {y})")

        window_name = 'Definir Zona de ENTRADA (6 puntos)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        temp_frame = frame.copy()
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(1)
        time.sleep(0.1)
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            while len(points) < 6:
                display_frame = temp_frame.copy()

                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"E{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)

                if len(points) == 5:
                    cv2.line(display_frame, points[4], points[0], (0, 255, 0), 1)

                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA ENTRADA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27: break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            time.sleep(0.1)

        if len(points) == 6:
            self.detection_line = points
            self.line_defined = True
            print("✅ Zona de ENTRADA definida")

    def define_exit_zone(self, frame):
        """Define zona de SALIDA con 6 puntos (TU CÓDIGO ORIGINAL)"""
        print("Haz clic en 6 puntos para definir la ZONA DE SALIDA")
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                print(f"Punto SALIDA {len(points)}: ({x}, {y})")

        window_name = 'Definir Zona de SALIDA (6 puntos)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        temp_frame = frame.copy()
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(1)
        time.sleep(0.1)
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            while len(points) < 6:
                display_frame = temp_frame.copy()

                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"S{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 0, 255), 2)

                if len(points) == 5:
                    cv2.line(display_frame, points[4], points[0], (0, 0, 255), 1)

                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA SALIDA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27: break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            time.sleep(0.1)

        if len(points) == 6:
            self.exit_zone = points
            self.exit_zone_defined = True
            print("✅ Zona de SALIDA definida")

    def draw_detection_areas(self, frame):
        """Dibuja las zonas de entrada y salida (TU CÓDIGO ORIGINAL)"""
        # Dibujar zona de ENTRADA (verde) - 6 puntos
        if self.line_defined and len(self.detection_line) == 6:
            pts = np.array(self.detection_line, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            for i, point in enumerate(self.detection_line):
                cv2.circle(frame, point, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"E{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            center_x = int(np.mean([p[0] for p in self.detection_line]))
            center_y = int(np.mean([p[1] for p in self.detection_line]))
            cv2.putText(frame, "AREA DE ENTRADA", (center_x - 70, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar zona de SALIDA (roja) - 6 puntos
        if self.exit_zone_defined and len(self.exit_zone) == 6:
            pts = np.array(self.exit_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

            for i, point in enumerate(self.exit_zone):
                cv2.circle(frame, point, 6, (0, 0, 255), -1)
                cv2.putText(frame, f"S{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            center_x = int(np.mean([p[0] for p in self.exit_zone]))
            center_y = int(np.mean([p[1] for p in self.exit_zone]))
            cv2.putText(frame, "AREA DE SALIDA", (center_x - 70, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def run(self):
        """FUNCIÓN PRINCIPAL AVANZADA - Tu lógica completa + Cloud"""
        print("📹 Abriendo cámara...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return

        print("✅ Cámara abierta!")

        # Configurar cámara para alto rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 120)

        print("\n=== REFRIGERADOR INTELIGENTE - AVANZADO CLOUD ===")
        print("'l' = Definir zona de ENTRADA")
        print("'s' = Definir zona de SALIDA")
        print("'h' = Modo DETECCIÓN DE MANOS")
        print("'n' = Modo DETECCIÓN DE MOVIMIENTO")
        print("'u' = Modo AUTOMÁTICO HÍBRIDO (recomendado)")
        print("'+/-' = Aumentar/Disminuir sensibilidad")
        print("'i' = Ver inventario cloud")
        print("'r' = Ver historial de análisis")
        print("'p' = Alternar modo FORZAR CAPTURA")
        print("'ESPACIO' = Captura manual")
        print("'q' = Salir")
        print("🌐 SISTEMA AVANZADO: Tu lógica completa + Procesamiento cloud")

        # Variables para FPS
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 30

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                clean_frame = frame.copy()

                # Calcular FPS
                fps_counter += 1
                if fps_counter % 20 == 0:
                    elapsed = time.time() - fps_start_time
                    if elapsed > 0:
                        current_fps = 20 / elapsed
                    fps_start_time = time.time()

                # Variables para detección
                should_analyze_cloud = False
                zone_type = "unknown"

                # TU LÓGICA HÍBRIDA ORIGINAL COMPLETA
                if self.detection_mode == "auto":
                    # PASO 1: Intentar detectar manos PRIMERO
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)

                    hands_found = False

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = hand_landmarks.landmark
                            is_valid_partial, landmark_count, estimated_center = self.validate_partial_hand(landmarks)

                            if is_valid_partial:
                                hands_found = True

                                # Dibujar landmarks
                                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                                # TU DETECCIÓN DE OBJETOS EN MANOS PARCIALES
                                is_holding, confidence, detection_type = self.detect_object_in_partial_hand(landmarks)

                                # TU VERIFICACIÓN DE ZONAS CON MÚLTIPLES PUNTOS
                                in_entry_zone, in_exit_zone = self.check_partial_hand_zones(landmarks, frame.shape)

                                # Dibujar info de mano parcial
                                self.draw_partial_hand_info(frame, landmarks, estimated_center, landmark_count)

                                # Lógica de captura
                                if (is_holding and confidence > 0.2) or self.force_capture_mode:
                                    if in_entry_zone and not in_exit_zone:
                                        cv2.putText(frame, "CLOUD: ENTRADA (MANOS)!", (30, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                                        should_analyze_cloud = True
                                        zone_type = "entrada"
                                    elif in_exit_zone and not in_entry_zone:
                                        cv2.putText(frame, "CLOUD: SALIDA (MANOS)!", (30, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                        should_analyze_cloud = True
                                        zone_type = "salida"

                                cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 180),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                break

                    # PASO 2: Si NO hay manos, usar TU DETECCIÓN DE MOVIMIENTO
                    if not hands_found:
                        cv2.putText(frame, "MODO AUTO: USANDO MOVIMIENTO", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        movement_entry, movement_exit, contours, fg_mask = self.detect_movement_in_zones(frame)
                        self.draw_movement_info(frame, contours)

                        if movement_entry and not movement_exit:
                            cv2.putText(frame, "CLOUD: ENTRADA (MOVIMIENTO)!", (30, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            should_analyze_cloud = True
                            zone_type = "entrada"
                        elif movement_exit and not movement_entry:
                            cv2.putText(frame, "CLOUD: SALIDA (MOVIMIENTO)!", (30, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                            should_analyze_cloud = True
                            zone_type = "salida"

                        cv2.putText(frame, f"Contornos: {len(contours)}", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, "MODO AUTO: MANOS DETECTADAS", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # MODO SOLO MANOS
                elif self.detection_mode == "hands":
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = hand_landmarks.landmark
                            is_valid_partial, landmark_count, estimated_center = self.validate_partial_hand(landmarks)

                            if is_valid_partial:
                                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                                is_holding, confidence, detection_type = self.detect_object_in_partial_hand(landmarks)
                                in_entry_zone, in_exit_zone = self.check_partial_hand_zones(landmarks, frame.shape)
                                self.draw_partial_hand_info(frame, landmarks, estimated_center, landmark_count)

                                if (is_holding and confidence > 0.2) or self.force_capture_mode:
                                    if in_entry_zone and not in_exit_zone:
                                        cv2.putText(frame, "CLOUD: ENTRADA!", (50, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                        should_analyze_cloud = True
                                        zone_type = "entrada"
                                    elif in_exit_zone and not in_entry_zone:
                                        cv2.putText(frame, "CLOUD: SALIDA!", (50, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                        should_analyze_cloud = True
                                        zone_type = "salida"

                                cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 180),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # MODO SOLO MOVIMIENTO
                elif self.detection_mode == "movement":
                    movement_entry, movement_exit, contours, fg_mask = self.detect_movement_in_zones(frame)
                    self.draw_movement_info(frame, contours)

                    if movement_entry and not movement_exit:
                        cv2.putText(frame, "CLOUD: ENTRADA!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        should_analyze_cloud = True
                        zone_type = "entrada"
                    elif movement_exit and not movement_entry:
                        cv2.putText(frame, "CLOUD: SALIDA!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        should_analyze_cloud = True
                        zone_type = "salida"

                    cv2.putText(frame, f"Contornos: {len(contours)}", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # ENVÍO AUTOMÁTICO A CLOUD
                if should_analyze_cloud:
                    self.save_photo_and_send_to_cloud(clean_frame, zone_type)

                # Dibujar zonas
                self.draw_detection_areas(frame)

                # FPS e info en pantalla
                fps_color = (0, 255, 0) if current_fps > 100 else (0, 255, 255) if current_fps > 60 else (0, 165, 255)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

                # Mostrar modo de detección
                if self.detection_mode == "hands":
                    detection_text = "Deteccion: MANOS AVANZADA"
                    detection_color = (255, 255, 0)
                elif self.detection_mode == "movement":
                    detection_text = "Deteccion: MOVIMIENTO AVANZADO"
                    detection_color = (255, 0, 255)
                else:
                    detection_text = "Deteccion: HIBRIDO AVANZADO"
                    detection_color = (0, 255, 255)

                cv2.putText(frame, detection_text, (10, frame.shape[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)

                # Mostrar sensibilidad en modo movimiento o auto
                if self.detection_mode in ["movement", "auto"]:
                    sensitivity_text = f"Sensibilidad: {self.sensitivity_level}/5"
                    cv2.putText(frame, sensitivity_text, (250, frame.shape[0] - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Mostrar modo forzar captura si está activado
                if self.force_capture_mode:
                    cv2.putText(frame, "FORZAR CAPTURA ACTIVADO", (250, frame.shape[0] - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Análisis cloud count
                cv2.putText(frame, f"Analisis cloud: {len(self.cloud_responses)}", (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('DETECTOR AVANZADO CLOUD - Refrigerador Inteligente', frame)

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    self.define_detection_line(frame)
                elif key == ord('s'):
                    self.define_exit_zone(frame)
                elif key == ord('h'):
                    self.detection_mode = "hands"
                    print("🎯 Modo: DETECCIÓN DE MANOS AVANZADA")
                elif key == ord('n'):
                    self.detection_mode = "movement"
                    print("🎯 Modo: DETECCIÓN DE MOVIMIENTO AVANZADA")
                elif key == ord('u'):
                    self.detection_mode = "auto"
                    print("🎯 Modo: AUTOMÁTICO HÍBRIDO AVANZADO")
                elif key == ord('+') or key == ord('='):
                    if self.detection_mode in ["movement", "auto"]:
                        self.adjust_sensitivity(increase=True)
                elif key == ord('-') or key == ord('_'):
                    if self.detection_mode in ["movement", "auto"]:
                        self.adjust_sensitivity(increase=False)
                elif key == ord('i'):
                    self.display_cloud_inventory()
                elif key == ord('r'):
                    self.show_cloud_responses()
                elif key == ord('p'):
                    self.force_capture_mode = not self.force_capture_mode
                    mode_text = "ACTIVADO" if self.force_capture_mode else "DESACTIVADO"
                    print(f"⚡ Modo FORZAR CAPTURA: {mode_text}")
                elif key == 32:  # ESPACIO
                    print("📸 Captura manual avanzada...")
                    self.save_photo_and_send_to_cloud(clean_frame, "manual")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            print(f"\n🎯 Sesión avanzada terminada:")
            print(f"   ☁️ Análisis cloud: {len(self.cloud_responses)}")
            print(f"   🎛️ Modo final: {self.detection_mode.upper()}")
            print(f"   📊 Sensibilidad final: {self.sensitivity_level}/5")


if __name__ == "__main__":
    print("🚀 DETECTOR AVANZADO CLOUD - REFRIGERADOR INTELIGENTE")
    print("=" * 60)
    print("🧠 Lógica: Tu código original completo")
    print("☁️ Procesamiento: Render.com + Gemini AI")
    print("🎯 Modos: Manos/Movimiento/Híbrido avanzados")
    print("🖐️ Manos parciales: Detección inteligente")
    print("📊 Inventario: Centralizado en la nube")
    print("=" * 60)

    detector = AdvancedCloudDetector()
    detector.run()