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


class FullCloudDetector:
    """
    DETECTOR COMPLETAMENTE EN LA NUBE:
    - Análisis de imágenes: Render.com + Gemini
    - Inventario: Centralizado en Render.com
    - Hardware: Solo cámara + detección de manos
    """

    def __init__(self):
        # Configuración MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Variables de estado
        self.detection_line = None
        self.line_defined = False
        self.exit_zone = None
        self.exit_zone_defined = False
        self.last_photo_time = 0
        self.photo_cooldown = 2

        # CONFIGURACIÓN API CLOUD COMPLETA
        self.base_url = "https://refrigerador-api-4.onrender.com"
        self.analizar_url = f"{self.base_url}/analizar"
        self.inventario_url = f"{self.base_url}/inventario"
        self.request_timeout = 30

        # Directorio local solo para fotos (backup)
        self.photos_dir = "fotos_cloud_backup"
        os.makedirs(self.photos_dir, exist_ok=True)

        # Variables de respuesta
        self.cloud_responses = []

        print("✅ Detector FULL CLOUD inicializado")
        print(f"🌐 API Base: {self.base_url}")
        print("📊 Inventario: Completamente en la nube")

    def send_photo_to_cloud_with_inventory(self, image_path, zone_type="unknown"):
        """
        ENVÍA FOTO A LA NUBE CON ACTUALIZACIÓN AUTOMÁTICA DE INVENTARIO
        """
        try:
            print(f"📤 Enviando {os.path.basename(image_path)} a la nube...")
            print(f"🎯 Zona: {zone_type.upper()}")

            # Preparar datos
            with open(image_path, 'rb') as img_file:
                files = {'imagen': img_file}
                data = {'zone_type': zone_type}  # ← IMPORTANTE: Indica zona para inventario

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

                # Extraer datos de la respuesta
                alimentos = result.get('alimentos_detectados', [])
                inventory_updated = result.get('inventory_updated', False)
                inventory_changes = result.get('inventory_changes', [])
                inventory_action = result.get('inventory_action', 'none')

                if alimentos:
                    print(f"✅ {len(alimentos)} alimentos detectados:")
                    for alimento in alimentos:
                        print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")

                    # Mostrar cambios de inventario
                    if inventory_updated:
                        action_text = "AGREGADOS AL" if inventory_action == "added" else "REMOVIDOS DEL"
                        print(f"📊 Inventario actualizado - {action_text} inventario cloud:")
                        for change in inventory_changes:
                            print(f"   {change}")
                    else:
                        print("⚠️ Inventario cloud no se pudo actualizar")

                else:
                    print("🤷 Nube: No se detectaron alimentos")

                # Guardar respuesta
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
                try:
                    error_data = response.json()
                    print(f"   Detalle: {error_data}")
                except:
                    print(f"   Respuesta: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print("⏰ Timeout - La nube tardó demasiado")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de red: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_cloud_inventory(self):
        """
        OBTIENE EL INVENTARIO COMPLETO DESDE LA NUBE
        """
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
        """
        MUESTRA EL INVENTARIO DESDE LA NUBE
        """
        print("\n" + "=" * 60)
        print("☁️ INVENTARIO DEL REFRIGERADOR (CLOUD)")
        print("=" * 60)

        inventory_data, stats = self.get_cloud_inventory()

        if not inventory_data:
            print("❌ No se pudo obtener inventario desde la nube")
            print("   Verifica tu conexión a internet")
            print("=" * 60 + "\n")
            return

        alimentos = inventory_data.get('alimentos', [])

        if not alimentos:
            print("📭 Inventario cloud vacío")
        else:
            # Mostrar estadísticas
            print(f"📊 Total: {stats['total_types']} tipos, {stats['total_items']} unidades")
            print(f"🕒 Actualizado: {inventory_data.get('actualizado', 'N/A')[:19].replace('T', ' ')}")
            print(f"🌐 Almacenado en: Render.com")
            print("")

            # Agrupar por categorías
            categories = {}
            for food in alimentos:
                categoria = food['categoria']
                if categoria not in categories:
                    categories[categoria] = []
                categories[categoria].append(food)

            # Mostrar por categorías
            for categoria in sorted(categories.keys()):
                foods = categories[categoria]
                categoria_items = sum(food['cantidad'] for food in foods)

                print(f"📂 {categoria.upper()} ({categoria_items} unidades):")
                for food in sorted(foods, key=lambda x: x['nombre']):
                    fecha = food.get('fecha_ingreso', 'N/A')
                    print(f"   • {food['cantidad']}x {food['nombre']} (desde {fecha})")
                print("")

        print("=" * 60 + "\n")

    def clear_cloud_inventory(self):
        """
        LIMPIA TODO EL INVENTARIO EN LA NUBE
        """
        try:
            print("🗑️ Limpiando inventario cloud...")
            response = requests.post(f"{self.base_url}/inventario/limpiar", timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Inventario cloud limpiado completamente")
                else:
                    print(f"❌ Error: {result.get('message')}")
            else:
                print(f"❌ Error limpiando inventario: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def save_photo_and_send_to_cloud(self, frame, zone_type="unknown"):
        """
        GUARDA FOTO Y ENVÍA A LA NUBE CON INVENTARIO AUTOMÁTICO
        """
        current_time = datetime.now()

        # Cooldown
        if (current_time.timestamp() - self.last_photo_time) < self.photo_cooldown:
            return False

        # Guardar localmente (backup)
        filename = f"fullcloud_{zone_type}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
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
        print("☁️ HISTORIAL DE ANÁLISIS + INVENTARIO CLOUD")
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

                # Mostrar cambios de inventario
                if response['inventory_updated'] and response['inventory_changes']:
                    print("   📊 Cambios en inventario cloud:")
                    for change in response['inventory_changes']:
                        print(f"      {change}")
            else:
                print("   🤷 Sin alimentos detectados")

        print("\n" + "=" * 60)

    # === FUNCIONES DE DETECCIÓN ORIGINALES (sin cambios) ===

    def define_detection_line(self, frame):
        """Define zona de entrada"""
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

                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA ENTRADA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27: break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if len(points) == 6:
            self.detection_line = points
            self.line_defined = True
            print("✅ Zona de ENTRADA definida")

    def define_exit_zone(self, frame):
        """Define zona de salida"""
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

                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA SALIDA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27: break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if len(points) == 6:
            self.exit_zone = points
            self.exit_zone_defined = True
            print("✅ Zona de SALIDA definida")

    def check_hand_zones(self, landmarks, frame_shape):
        """Verifica zonas"""
        palm_center = landmarks[9]
        palm_x = int(palm_center.x * frame_shape[1])
        palm_y = int(palm_center.y * frame_shape[0])

        in_entry_zone = False
        in_exit_zone = False

        if self.line_defined:
            area_points = np.array(self.detection_line, dtype=np.int32)
            result = cv2.pointPolygonTest(area_points, (palm_x, palm_y), False)
            in_entry_zone = result >= 0

        if self.exit_zone_defined:
            area_points = np.array(self.exit_zone, dtype=np.int32)
            result = cv2.pointPolygonTest(area_points, (palm_x, palm_y), False)
            in_exit_zone = result >= 0

        return in_entry_zone, in_exit_zone

    def detect_object_in_hand(self, landmarks):
        """Detecta objeto en mano"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        distances = []
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            distances.append(dist)

        avg_distance = sum(distances) / len(distances)
        is_holding = avg_distance < 0.12
        confidence = 0.8 if is_holding else 0.2

        return is_holding, confidence

    def draw_detection_areas(self, frame):
        """Dibuja áreas de detección"""
        if self.line_defined and len(self.detection_line) == 6:
            pts = np.array(self.detection_line, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            center_x = int(np.mean([p[0] for p in self.detection_line]))
            center_y = int(np.mean([p[1] for p in self.detection_line]))
            cv2.putText(frame, "AREA ENTRADA", (center_x - 70, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.exit_zone_defined and len(self.exit_zone) == 6:
            pts = np.array(self.exit_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
            center_x = int(np.mean([p[0] for p in self.exit_zone]))
            center_y = int(np.mean([p[1] for p in self.exit_zone]))
            cv2.putText(frame, "AREA SALIDA", (center_x - 70, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def run(self):
        """FUNCIÓN PRINCIPAL - FULL CLOUD"""
        print("📹 Abriendo cámara...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return

        print("✅ Cámara abierta!")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== REFRIGERADOR INTELIGENTE - FULL CLOUD ===")
        print("'l' = Definir zona de ENTRADA")
        print("'s' = Definir zona de SALIDA")
        print("'i' = Ver inventario cloud")
        print("'h' = Ver historial de análisis")
        print("'x' = Limpiar inventario cloud")
        print("'ESPACIO' = Captura manual")
        print("'q' = Salir")
        print("🌐 TODO EN LA NUBE: Análisis + Inventario")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                clean_frame = frame.copy()

                # Detección de manos
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                should_analyze_cloud = False
                zone_type = "unknown"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        landmarks = hand_landmarks.landmark
                        is_holding, confidence = self.detect_object_in_hand(landmarks)
                        in_entry, in_exit = self.check_hand_zones(landmarks, frame.shape)

                        # Análisis automático FULL CLOUD
                        if is_holding and confidence > 0.5:
                            if in_entry and not in_exit:
                                cv2.putText(frame, "CLOUD: ENTRADA + INVENTARIO!", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                                should_analyze_cloud = True
                                zone_type = "entrada"
                            elif in_exit and not in_entry:
                                cv2.putText(frame, "CLOUD: SALIDA + INVENTARIO!", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                should_analyze_cloud = True
                                zone_type = "salida"

                        cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # ENVÍO AUTOMÁTICO A CLOUD COMPLETO
                if should_analyze_cloud:
                    self.save_photo_and_send_to_cloud(clean_frame, zone_type)

                # Interfaz
                self.draw_detection_areas(frame)

                # Info en pantalla
                info_lines = [
                    f"☁️ Full Cloud: Análisis + Inventario",
                    f"📊 Análisis: {len(self.cloud_responses)}",
                    f"🌐 API: {self.base_url}",
                    "'l'=Entrada, 's'=Salida, 'i'=Inventario, 'h'=Historial, 'x'=Limpiar"
                ]

                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 150 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('FULL CLOUD - Refrigerador Inteligente', frame)

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    self.define_detection_line(frame)
                elif key == ord('s'):
                    self.define_exit_zone(frame)
                elif key == ord('i'):
                    self.display_cloud_inventory()
                elif key == ord('h'):
                    self.show_cloud_responses()
                elif key == ord('x'):
                    print("⚠️ ¿Seguro que quieres limpiar todo el inventario cloud? (s/N)")
                    # En una implementación real, podrías agregar confirmación
                    self.clear_cloud_inventory()
                elif key == 32:  # ESPACIO
                    print("📸 Captura manual - enviando a cloud completo...")
                    self.save_photo_and_send_to_cloud(clean_frame, "manual")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            print(f"\n🎯 Sesión terminada:")
            print(f"   ☁️ Análisis cloud: {len(self.cloud_responses)}")
            print(f"   🌐 Inventario: Centralizado en {self.base_url}")


if __name__ == "__main__":
    print("🚀 REFRIGERADOR INTELIGENTE - FULL CLOUD")
    print("=" * 60)
    print("📱 Hardware: Solo cámara + detección")
    print("☁️ Procesamiento: Render.com completo")
    print("🔍 Análisis: Gemini AI")
    print("📊 Inventario: Centralizado y sincronizado")
    print("🌐 API: https://refrigerador-api-4.onrender.com")
    print("=" * 60)

    detector = FullCloudDetector()
    detector.run()