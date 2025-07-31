# INTEGRACIÓN: Tu código original + nuestra función optimizada de Gemini
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime
import time
from PIL import Image

# IMPORTAR NUESTRA FUNCIÓN OPTIMIZADA
from detector import detectar_alimentos


class IntegratedHandDetector:
    """
    Tu detector original MEJORADO con nuestra función optimizada de Gemini
    """

    def __init__(self):
        # Tu configuración original de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Variables de estado (de tu código original)
        self.detection_line = None
        self.line_defined = False
        self.exit_zone = None
        self.exit_zone_defined = False
        self.last_photo_time = 0
        self.photo_cooldown = 2

        # Directorios para fotos
        self.photos_dir = "fotos_detectadas"
        os.makedirs(self.photos_dir, exist_ok=True)

        # Variables para almacenar detecciones
        self.pending_photos = []
        self.detected_foods = []

        print("✅ Detector integrado inicializado")
        print("🤖 Función optimizada de Gemini cargada")

    def define_detection_line(self, frame):
        """Tu función original para definir zona de ENTRADA"""
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

                # Dibujar puntos ya seleccionados
                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"E{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Instrucciones
                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA ENTRADA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    print("Definición cancelada")
                    break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if len(points) == 6:
            self.detection_line = points
            self.line_defined = True
            print("✅ Zona de ENTRADA definida")

    def define_exit_zone(self, frame):
        """Tu función original para definir zona de SALIDA"""
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

                # Dibujar puntos ya seleccionados
                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"S{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Instrucciones
                if len(points) < 6:
                    cv2.putText(display_frame, f"ZONA SALIDA - Punto {len(points) + 1}/6",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    print("Definición cancelada")
                    break

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if len(points) == 6:
            self.exit_zone = points
            self.exit_zone_defined = True
            print("✅ Zona de SALIDA definida")

    def check_hand_zones(self, landmarks, frame_shape):
        """Verifica si la mano está en zona de entrada o salida"""
        palm_center = landmarks[9]
        palm_x = int(palm_center.x * frame_shape[1])
        palm_y = int(palm_center.y * frame_shape[0])

        in_entry_zone = False
        in_exit_zone = False

        # Verificar zona de ENTRADA
        if self.line_defined:
            area_points = np.array(self.detection_line, dtype=np.int32)
            result = cv2.pointPolygonTest(area_points, (palm_x, palm_y), False)
            in_entry_zone = result >= 0

        # Verificar zona de SALIDA
        if self.exit_zone_defined:
            area_points = np.array(self.exit_zone, dtype=np.int32)
            result = cv2.pointPolygonTest(area_points, (palm_x, palm_y), False)
            in_exit_zone = result >= 0

        return in_entry_zone, in_exit_zone

    def detect_object_in_hand(self, landmarks):
        """Detección simple de objeto en mano (tu lógica original simplificada)"""
        # Calcular apertura de la mano
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        # Calcular distancias promedio
        distances = []
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
            distances.append(dist)

        avg_distance = sum(distances) / len(distances)

        # Si la mano está relativamente cerrada, probablemente sostiene algo
        is_holding = avg_distance < 0.12
        confidence = 0.8 if is_holding else 0.2

        return is_holding, confidence

    def save_photo_and_analyze(self, frame, zone_type="unknown"):
        """
        NUEVA FUNCIÓN INTEGRADA:
        Guarda foto Y la analiza inmediatamente con nuestra función optimizada
        """
        current_time = datetime.now()

        # Verificar cooldown
        if (current_time.timestamp() - self.last_photo_time) < self.photo_cooldown:
            return False

        # Guardar imagen
        filename = f"hand_object_{zone_type}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.photos_dir, filename)
        cv2.imwrite(filepath, frame)

        print(f"📸 Foto guardada: {filename}")

        # ANALIZAR INMEDIATAMENTE con nuestra función optimizada
        try:
            # Convertir imagen para PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            print("🤖 Analizando con Gemini...")
            alimentos_detectados = detectar_alimentos(pil_image)

            if alimentos_detectados:
                print(f"✅ {len(alimentos_detectados)} alimentos detectados:")
                for alimento in alimentos_detectados:
                    print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")

                # Agregar a la lista de detecciones
                detection_info = {
                    'timestamp': current_time.isoformat(),
                    'zone_type': zone_type,
                    'filename': filename,
                    'alimentos': alimentos_detectados
                }
                self.detected_foods.append(detection_info)
            else:
                print("🤷 No se detectaron alimentos")

        except Exception as e:
            print(f"❌ Error en análisis: {e}")

        self.last_photo_time = current_time.timestamp()
        return True

    def draw_detection_areas(self, frame):
        """Dibuja las zonas de entrada y salida"""
        # Zona de ENTRADA (verde)
        if self.line_defined and len(self.detection_line) == 6:
            pts = np.array(self.detection_line, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            center_x = int(np.mean([p[0] for p in self.detection_line]))
            center_y = int(np.mean([p[1] for p in self.detection_line]))
            cv2.putText(frame, "AREA ENTRADA", (center_x - 70, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Zona de SALIDA (roja)
        if self.exit_zone_defined and len(self.exit_zone) == 6:
            pts = np.array(self.exit_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

            center_x = int(np.mean([p[0] for p in self.exit_zone]))
            center_y = int(np.mean([p[1] for p in self.exit_zone]))
            cv2.putText(frame, "AREA SALIDA", (center_x - 70, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def show_detection_summary(self):
        """Muestra resumen de todas las detecciones"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE DETECCIONES")
        print("=" * 60)

        if not self.detected_foods:
            print("📭 No hay detecciones registradas")
            return

        for i, detection in enumerate(self.detected_foods, 1):
            print(f"\n🔍 Detección #{i} ({detection['zone_type'].upper()})")
            print(f"⏰ {detection['timestamp'][:19].replace('T', ' ')}")
            print(f"📄 {detection['filename']}")

            if detection['alimentos']:
                for alimento in detection['alimentos']:
                    print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")
            else:
                print("   🤷 Sin alimentos detectados")

        print("\n" + "=" * 60)

    def run(self):
        """Función principal - Tu código original mejorado"""
        print("📹 Abriendo cámara...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return

        print("✅ Cámara abierta exitosamente!")

        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== CONTROLES ===")
        print("'l' = Definir zona de ENTRADA")
        print("'s' = Definir zona de SALIDA")
        print("'r' = Ver resumen de detecciones")
        print("'ESPACIO' = Captura manual + análisis")
        print("'q' = Salir")
        print("🤖 Gemini activado - análisis automático")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error leyendo frame")
                    break

                frame = cv2.flip(frame, 1)  # Espejo
                clean_frame = frame.copy()  # Frame sin anotaciones para análisis

                # Procesar con MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Variables para detección
                should_analyze = False
                zone_type = "unknown"

                # Detectar manos
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Dibujar landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Verificar si sostiene objeto
                        landmarks = hand_landmarks.landmark
                        is_holding, confidence = self.detect_object_in_hand(landmarks)

                        # Verificar zonas
                        in_entry, in_exit = self.check_hand_zones(landmarks, frame.shape)

                        # Lógica de detección
                        if is_holding and confidence > 0.5:
                            if in_entry and not in_exit:
                                cv2.putText(frame, "OBJETO DETECTADO - ENTRADA!", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                should_analyze = True
                                zone_type = "entrada"
                            elif in_exit and not in_entry:
                                cv2.putText(frame, "OBJETO DETECTADO - SALIDA!", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                should_analyze = True
                                zone_type = "salida"

                        # Mostrar info
                        cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Análisis automático
                if should_analyze:
                    self.save_photo_and_analyze(clean_frame, zone_type)

                # Dibujar zonas
                self.draw_detection_areas(frame)

                # Info en pantalla
                info_lines = [
                    "'l' = Zona entrada",
                    "'s' = Zona salida",
                    "'r' = Ver resumen",
                    "'SPACE' = Captura manual",
                    "'q' = Salir"
                ]

                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 150 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Mostrar estado
                status_text = "LISTO" if (self.line_defined and self.exit_zone_defined) else "CONFIGURANDO..."
                status_color = (0, 255, 0) if (self.line_defined and self.exit_zone_defined) else (0, 255, 255)
                cv2.putText(frame, status_text, (10, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Mostrar número de detecciones
                cv2.putText(frame, f"Detecciones: {len(self.detected_foods)}", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('Detector Integrado - Refrigerador Inteligente', frame)

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    self.define_detection_line(frame)
                elif key == ord('s'):
                    self.define_exit_zone(frame)
                elif key == ord('r'):
                    self.show_detection_summary()
                elif key == 32:  # ESPACIO
                    print("📸 Captura manual + análisis...")
                    self.save_photo_and_analyze(clean_frame, "manual")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Mostrar resumen final
            self.show_detection_summary()
            print(f"🎯 Sesión terminada. Total detecciones: {len(self.detected_foods)}")


if __name__ == "__main__":
    print("🚀 DETECTOR INTEGRADO - REFRIGERADOR INTELIGENTE")
    print("=" * 60)
    print("✅ Cámara local + MediaPipe + Gemini optimizado")
    print("🤖 Análisis automático activado")
    print("=" * 60)

    detector = IntegratedHandDetector()
    detector.run()