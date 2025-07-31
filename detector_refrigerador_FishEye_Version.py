import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime
import math
import google.generativeai as genai
import json
from PIL import Image
import time
import threading  # Para controlar duración de sonidos
import simpleaudio as sa  # Reemplazar winsound con simpleaudio
from pathlib import Path


class HandObjectDetector:
    def __init__(self):
        # Inicializar MediaPipe con configuración MÁS PERMISIVA
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Reducido de 0.7 a 0.3
            min_tracking_confidence=0.2,  # Reducido de 0.5 a 0.2
            model_complexity=1  # Modelo más complejo para mejor detección
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Variables de estado
        self.detection_line = None  # Zona de ENTRADA (6 puntos)
        self.line_defined = False
        self.exit_zone = None  # Zona de SALIDA (6 puntos)
        self.exit_zone_defined = False
        self.analysis_zone = None
        self.zone_defined = False
        self.zone_mask = None
        self.last_photo_time = 0
        self.photo_cooldown = 2  # 2 segundos entre fotos

        # NUEVO: Sistema de modos - MODO POR DEFECTO CAMBIADO A AUTO
        self.detection_mode = "auto"  # "hands", "movement", o "auto" (híbrido)

        # NUEVO: Variables para detección de movimiento
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.movement_threshold = 1000  # Píxeles que deben cambiar
        self.min_contour_area = 500  # Área mínima de contorno
        self.max_contour_area = 50000  # Área máxima de contorno
        self.sensitivity_level = 2  # Nivel de sensibilidad (1-5)
        self.last_movement_time = 0

        # NUEVO: Variables para detección de manos parciales
        self.min_landmarks_for_detection = 5  # Mínimo de landmarks para considerar válida
        self.partial_hand_mode = True  # Habilitar detección parcial
        self.last_valid_palm_position = None  # Última posición válida de palma
        self.force_capture_mode = False  # NUEVO: Modo forzar captura sin detectar objeto

        # Configuración de Gemini
        self.gemini_enabled = False
        self.gemini_api_key = None
        self.pending_photos = []  # Lista de fotos pendientes de análisis

        # Inventario con nuevo formato JSON
        self.inventory_data = {
            "actualizado": datetime.now().isoformat(),
            "alimentos": []
        }

        # Categorías válidas para clasificación
        self.valid_categories = ["carne", "verduras", "frutas", "lacteos", "granos", "otros"]

        # INTEGRACIÓN: Usar rutas del sistema SmartChef - RUTAS PERSONALIZADAS
        # Rutas específicas para C:\BASE_SEGURA
        self.photos_dir = r"C:\BASE_SEGURA\smartchef_data\photos"
        self.analysis_dir = r"C:\BASE_SEGURA\smartchef_data\json_analysis"

        # Crear carpetas si no existen
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # INTEGRACIÓN: Archivo JSON único para inventario (compatible con orquestador)
        self.inventory_file = "database/inventory.json"

        # NUEVO: Configuración de sonidos para Raspberry Pi
        self.sound_entry_path = "/home/marco/Descargas/SmartChef/BASE_SEGURA_RASP/smartchef_sounds/GRAVE_ok.wav"
        self.sound_exit_path = "/home/marco/Descargas/SmartChef/BASE_SEGURA_RASP/smartchef_sounds/AGUDO_ok.wav"

        # Pre-cargar archivos de sonido si existen
        self.sound_entry_wave = None
        self.sound_exit_wave = None
        self.setup_sounds()

        # RASPBERRY PI: Variable para controlar soporte de callback (ya no se usa)
        # self.mouse_callback_works = False

        # Configurar Gemini si hay API key
        self.setup_gemini()

        # Cargar inventario previo al iniciar
        self.load_inventory_from_json()

        # RASPBERRY PI FIX: Inicializar OpenCV con configuración específica
        self.setup_opencv_raspberry()

    def setup_opencv_raspberry(self):
        """Configuración específica de OpenCV para Raspberry Pi - SIMPLIFICADA"""
        try:
            # Inicializar el thread de OpenCV para mejor manejo de ventanas
            cv2.startWindowThread()
            print("✅ OpenCV configurado para Raspberry Pi")
        except Exception as e:
            print(f"⚠️ Advertencia configurando OpenCV: {e}")

    def setup_sounds(self):
        """Configura los archivos de sonido para Raspberry Pi"""
        try:
            # Verificar que los archivos de sonido existen
            if os.path.exists(self.sound_entry_path):
                self.sound_entry_wave = sa.WaveObject.from_wave_file(self.sound_entry_path)
                print(f"✅ Sonido ENTRADA cargado: {self.sound_entry_path}")
            else:
                print(f"⚠️ Archivo de sonido ENTRADA no encontrado: {self.sound_entry_path}")

            if os.path.exists(self.sound_exit_path):
                self.sound_exit_wave = sa.WaveObject.from_wave_file(self.sound_exit_path)
                print(f"✅ Sonido SALIDA cargado: {self.sound_exit_path}")
            else:
                print(f"⚠️ Archivo de sonido SALIDA no encontrado: {self.sound_exit_path}")

            # Variables para controlar reproducción
            self.current_entry_play = None
            self.current_exit_play = None

        except Exception as e:
            print(f"❌ Error configurando sonidos: {e}")
            print("   Los sonidos no funcionarán pero el programa continuará")

    def play_entry_sound(self):
        """Reproduce sonido grave para indicar ENTRADA detectada - LIMITADO A 0.5 SEGUNDOS"""
        try:
            if self.sound_entry_wave:
                # Detener sonido anterior si está reproduciéndose
                if self.current_entry_play and self.current_entry_play.is_playing():
                    self.current_entry_play.stop()

                # Reproducir nuevo sonido
                self.current_entry_play = self.sound_entry_wave.play()
                print("🔊 ♪ Sonido ENTRADA reproducido (0.5s)")

                # Detener después de 0.5 segundos usando threading
                def stop_after_delay():
                    time.sleep(0.5)
                    if self.current_entry_play and self.current_entry_play.is_playing():
                        self.current_entry_play.stop()

                threading.Thread(target=stop_after_delay, daemon=True).start()

            else:
                print("🔊 ♪ Sonido ENTRADA (archivo no disponible)")
        except Exception as e:
            print(f"⚠️ Error reproduciendo sonido de entrada: {e}")

    def play_exit_sound(self):
        """Reproduce sonido agudo para indicar SALIDA detectada - LIMITADO A 0.5 SEGUNDOS"""
        try:
            if self.sound_exit_wave:
                # Detener sonido anterior si está reproduciéndose
                if self.current_exit_play and self.current_exit_play.is_playing():
                    self.current_exit_play.stop()

                # Reproducir nuevo sonido
                self.current_exit_play = self.sound_exit_wave.play()
                print("🔊 ♫ Sonido SALIDA reproducido (0.5s)")

                # Detener después de 0.5 segundos usando threading
                def stop_after_delay():
                    time.sleep(0.5)
                    if self.current_exit_play and self.current_exit_play.is_playing():
                        self.current_exit_play.stop()

                threading.Thread(target=stop_after_delay, daemon=True).start()

            else:
                print("🔊 ♫ Sonido SALIDA (archivo no disponible)")
        except Exception as e:
            print(f"⚠️ Error reproduciendo sonido de salida: {e}")

    def validate_partial_hand(self, landmarks):
        """Valida si los landmarks detectados son suficientes para una mano parcial"""
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
        """Estima el centro de la mano basándose en landmarks disponibles"""
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
        """Detecta si una mano parcial probablemente está sosteniendo un objeto"""
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
        # Si tenemos suficientes landmarks, probablemente sostiene algo
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
        """Verifica zonas usando detección de manos parciales - MEJORADA PARA ENTRADA Y SALIDA"""
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
        # Si al menos 1/3 de los puntos están en la zona, considerarla válida
        min_votes = max(1, len(valid_test_points) // 3)

        in_entry_zone = entry_votes >= min_votes
        in_exit_zone = exit_votes >= min_votes

        return in_entry_zone, in_exit_zone

    def draw_partial_hand_info(self, frame, landmarks, estimated_center, landmark_count):
        """Dibuja información específica para manos parciales"""
        if estimated_center:
            center_x = int(estimated_center[0] * frame.shape[1])
            center_y = int(estimated_center[1] * frame.shape[0])

            # Dibujar centro estimado con color diferente
            cv2.circle(frame, (center_x, center_y), 10, (255, 165, 0), -1)  # Naranja
            cv2.putText(frame, "MANO PARCIAL", (center_x - 50, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            cv2.putText(frame, f"Landmarks: {landmark_count}", (center_x - 50, center_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    def setup_gemini(self):
        """Configura la API de Gemini usando la ruta específica - RUTA PERSONALIZADA"""
        # INTEGRACIÓN: Usar ruta específica personalizada
        config_file = "gemini_config.txt.txt"

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.gemini_api_key = f.read().strip()

                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_enabled = True
                print("✅ [Visión] Gemini API configurada correctamente (desde archivo)")

            except Exception as e:
                print(f"❌ [Visión] Error configurando Gemini: {e}")
                self.gemini_enabled = False
        else:
            # FALLBACK: Intentar con variable de entorno como en el orquestador
            try:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                    self.gemini_enabled = True
                    print("✅ [Visión] Gemini API configurada desde variable de entorno")
                else:
                    print(
                        "⚠️ [Visión] Archivo gemini_config.txt no encontrado y variable GOOGLE_API_KEY no configurada")
                    print("   Crea el archivo con tu API key para habilitar Gemini")
                    self.gemini_enabled = False
            except Exception as e:
                print(f"❌ [Visión] Error configurando Gemini con variable de entorno: {e}")
                self.gemini_enabled = False

    def find_food_by_name(self, name):
        """Busca un alimento por nombre en el inventario"""
        name_lower = name.lower().strip()
        for i, food in enumerate(self.inventory_data["alimentos"]):
            if food["nombre"].lower() == name_lower:
                return i, food
        return None, None

    def validate_category(self, category):
        """Valida y normaliza una categoría"""
        if not category:
            return "otros"

        category_lower = category.lower().strip()

        # Buscar coincidencia exacta
        for valid_cat in self.valid_categories:
            if category_lower == valid_cat:
                return valid_cat

        # Buscar coincidencias parciales
        category_mappings = {
            "carne": ["pollo", "res", "cerdo", "pescado", "mariscos"],
            "verduras": ["vegetales", "hortalizas"],
            "frutas": ["fruta"],
            "lacteos": ["leche", "queso", "yogurt", "mantequilla"],
            "granos": ["cereales", "arroz", "pasta", "pan"]
        }

        for valid_cat, synonyms in category_mappings.items():
            if category_lower in synonyms or any(syn in category_lower for syn in synonyms):
                return valid_cat

        return "otros"

    def analyze_image_with_gemini(self, image_path):
        """Envía imagen a Gemini para análisis con nuevo formato"""
        if not self.gemini_enabled:
            return {"error": "Gemini no está configurado"}

        try:
            # Abrir imagen
            img = Image.open(image_path)

            # Nuevo prompt para detectar nombre, categoría y cantidad - OPTIMIZADO PARA IMÁGENES LIMPIAS
            prompt = f"""
            Eres un refrigerador inteligente con capacidad de reconocimiento de alimentos. Analiza esta imagen en la que una persona sostiene alimentos en su mano dominante (Puede que la persona introduzca cosas con sus dos manos).

            IMPORTANTE:
            - La imagen fue tomada con una cámara *fish-eye*, por lo que puede tener una ligera distorsión visual.

            Tu tarea es identificar únicamente los alimentos que la persona sostiene en su **mano dominante**.

            Por cada alimento detectado, proporciona:
            1. **nombre** específico del alimento (por ejemplo: "catsup", "coca cola", "manzana")
            2. **categoría**, que debe ser una de estas: {', '.join(self.valid_categories)}
            3. **cantidad exacta**, es decir, cuántos objetos iguales están presentes en esa mano
            Responde ÚNICAMENTE con este formato JSON:
            {{
                "alimentos_detectados": [
                    {{
                        "nombre": "nombre_especifico_del_alimento",
                        "categoria": "categoria_valida",
                        "cantidad": numero_entero
                    }}
                ]
            }}

            ANÁLISIS INTELIGENTE REQUERIDO:
            1. ELIMINA DUPLICADOS: Si ves el mismo alimento en múltiples imágenes, cuenta UNA SOLA VEZ, algunos usuarios pueden ser lentos y puede que se tome multiples imagenes.
            2. DIFERENCIA ENTRADA vs SALIDA: Cuenta por separado los alimentos que entran vs los que salen

REGLAS ESPECÍFICAS PARA SECUENCIAS:
✅ Prioriza la imagen con mayor claridad para contar
✅ Si hay duda entre imágenes, usa el conteo más conservador
✅ CUENTA BIEN CUANTOS ALIMENTOS HAY EN LA MANO DOMINANTE DE LAS IMAGENES CUIDANDO QUE SI DOS IMAGENES SON IGUALES NO CUENTES DOBLE

❌ No sumas automáticamente cantidades entre imágenes, a menos que sea claro que representan acciones diferentes (por ejemplo, introdujo 2 tomates y luego otros 2).
❌ No cuentes el mismo objeto que aparece en varios frames mientras se mueve.
❌  NO incluyas alimentos que no esten en la mano dominante, puede que el usuario este sosteniendo alimentos en otra mano y que no planea entrar al refrigerador, entonces no tendría sentido que cuentes esos.
⚠️ ADVERTENCIA IMPORTANTE:
No todos los alimentos visibles deben contarse. Algunos pueden estar en la otra mano o simplemente estar siendo sostenidos sin intención de meterlos al refrigerador. 
✔️ Solo cuenta aquellos que están en la mano dominante **y** que claramente están siendo introducidos o retirados.
✖️ Ignora objetos que simplemente aparecen en escena pero no están siendo colocados o sacados.

            """

            # Enviar a Gemini
            response = self.model.generate_content([prompt, img])

            # Procesar respuesta
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "gemini_response": response.text,
                "analysis_successful": True
            }

            # Intentar parsear JSON de la respuesta
            try:
                # Extraer JSON de la respuesta
                response_text = response.text
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text

                parsed_data = json.loads(json_text)
                analysis_result["parsed_data"] = parsed_data

                # Mostrar resultado
                alimentos = parsed_data.get('alimentos_detectados', [])
                if alimentos:
                    for alimento in alimentos:
                        print(f"🔍 Detectado: {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")

            except json.JSONDecodeError:
                analysis_result["parsed_data"] = None
                analysis_result["raw_response"] = response.text
                print("⚠️ Error parseando respuesta de Gemini")

            return analysis_result

        except Exception as e:
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "error": str(e),
                "analysis_successful": False
            }

            print(f"❌ Error en análisis: {e}")
            return error_result

    def analyze_multiple_images_with_gemini(self, image_paths):
        """Envía múltiples imágenes a Gemini para análisis conjunto con nuevo formato"""
        if not self.gemini_enabled:
            return {"error": "Gemini no está configurado"}

        if not image_paths:
            return {"error": "No hay imágenes para analizar"}

        # Extraer solo las rutas si recibimos objetos con información de zona
        clean_paths = []
        for item in image_paths:
            if isinstance(item, dict) and 'path' in item:
                clean_paths.append(item['path'])
            else:
                clean_paths.append(item)

        try:
            print(f"🤖 Analizando {len(clean_paths)} imágenes con Gemini...")

            # Si hay muchas imágenes, analizar en lotes más pequeños
            if len(clean_paths) > 4:
                print(f"⚠️  Demasiadas imágenes ({len(clean_paths)}). Analizando en lotes...")
                return self.analyze_in_batches(clean_paths)

            # Preparar las imágenes
            images = []
            for i, path in enumerate(clean_paths):
                try:
                    img = Image.open(path)
                    # Redimensionar si es muy grande para reducir carga
                    if img.width > 1024 or img.height > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    images.append(img)
                    print(f"✅ Imagen {i + 1} cargada: {os.path.basename(path)}")
                except Exception as e:
                    print(f"⚠️  Error cargando imagen {path}: {e}")

            if not images:
                return {"error": "No se pudieron cargar las imágenes"}

            # Nuevo prompt para múltiples imágenes - OPTIMIZADO PARA IMÁGENES LIMPIAS
            prompt = f"""
Eres un refrigerador inteligente que reconoce los alimentos que entran, analiza estas imagenes {len(images)} , de una persona sosteniendo alimento(s) en su mano(s) dominantes (Puede que introduzca alimentos con las 2 manos a la vez).

Toma en cuenta que la imagen fue tomada con una camara fish eye, así que puede tener una ligera distorsión.

IDENTIFICA TODOS los alimentos que ves en TODAS las imágenes y proporciona:
1. NOMBRE específico del alimento (ej: "catsup", "coca cola", "manzana")
2. CATEGORÍA (debe ser una de estas: {', '.join(self.valid_categories)})
3. CANTIDAD exacta (cuenta cuántos del mismo alimento ves en la mano dominante de la persona)
Responde ÚNICAMENTE con este formato JSON:

{{
    "alimentos_detectados": [
        {{
            "nombre": "nombre_especifico_del_alimento",
            "categoria": "categoria_valida", 
            "cantidad": numero_entero_total
        }}
    ]
}}

ANÁLISIS INTELIGENTE REQUERIDO:
1. ELIMINA DUPLICADOS: Si ves el mismo alimento en múltiples imágenes, cuenta UNA SOLA VEZ, algunos usuarios pueden ser lentos y se puede tomar multiples imagenes.
2. DIFERENCIA ENTRADA vs SALIDA: Cuenta por separado los alimentos que entran vs los que salen

REGLAS ESPECÍFICAS PARA SECUENCIAS:
✅ Prioriza la imagen con mayor claridad para contar
✅ Si hay duda entre imágenes, usa el conteo más conservador
✅ CUENTA BIEN CUANTOS HAY EN LAS IMAGENES CUIDANDO QUE SI DOS IMAGENES SON IGUALES NO CUENTES DOBLE

❌ NO sumes automáticamente cantidades entre imágenes, solo suma cantidades que esten en la misma imagen.
❌ NO cuentes el mismo objeto viajando a través de múltiples frames
❌  NO incluyas alimentos que no esten en la mano dominante, puede que el usuario este sosteniendo alimentos en otra mano y que no planea entrar al refrigerador, entonces no tendría sentido que cuentes esos.
            """

            # Preparar contenido para Gemini (prompt + todas las imágenes)
            content = [prompt] + images

            # Enviar a Gemini con configuración más permisiva
            try:
                print("📤 Enviando solicitud a Gemini...")
                response = self.model.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Más determinístico
                        top_p=0.8,
                        top_k=20,
                        max_output_tokens=1024
                    )
                )
                response_text = response.text

                print(f"📝 Respuesta recibida ({len(response_text)} caracteres)")

            except Exception as gemini_error:
                print(f"❌ Error en la API de Gemini: {gemini_error}")
                return {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": "multi_image_batch",
                    "total_images": len(clean_paths),
                    "image_paths": clean_paths,
                    "error": f"Error API Gemini: {str(gemini_error)}",
                    "analysis_successful": False
                }

            # Procesar respuesta
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "multi_image_batch",
                "total_images": len(clean_paths),
                "image_paths": clean_paths,
                "gemini_response": response_text,
                "analysis_successful": True
            }

            # Parsing más robusto del JSON
            parsed_data = None
            try:
                # Limpiar la respuesta
                clean_response = response_text.strip()

                # Buscar JSON en diferentes formatos
                json_text = ""
                if "```json" in clean_response:
                    json_start = clean_response.find("```json") + 7
                    json_end = clean_response.find("```", json_start)
                    json_text = clean_response[json_start:json_end].strip()
                elif "{" in clean_response and "}" in clean_response:
                    # Extraer solo la parte JSON
                    json_start = clean_response.find("{")
                    json_end = clean_response.rfind("}") + 1
                    json_text = clean_response[json_start:json_end]
                else:
                    # Si no hay JSON claro, crear uno básico
                    json_text = '{"alimentos_detectados": []}'

                print(f"🔧 JSON extraído: {json_text[:100]}...")

                # Intentar parsear
                parsed_data = json.loads(json_text)
                analysis_result["parsed_data"] = parsed_data

                print("✅ JSON parseado exitosamente")

                # Mostrar resumen SIMPLE en consola
                self.display_simple_summary(parsed_data, len(clean_paths))

            except json.JSONDecodeError as json_error:
                print(f"⚠️  Error parseando JSON: {json_error}")

                # Análisis manual de texto si falla el JSON
                manual_analysis = self.simple_text_analysis(response_text)
                analysis_result["parsed_data"] = manual_analysis

                print("\n🔧 Análisis manual activado:")
                self.display_simple_summary(manual_analysis, len(clean_paths))

            return analysis_result

        except Exception as e:
            print(f"❌ Error general en análisis conjunto: {e}")

            error_result = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "multi_image_batch",
                "total_images": len(clean_paths),
                "image_paths": clean_paths,
                "error": str(e),
                "analysis_successful": False
            }

            return error_result

    def display_simple_summary(self, parsed_data, num_images):
        """Muestra los alimentos detectados con nuevo formato"""
        print("\n" + "=" * 50)
        print("🔍 ALIMENTOS DETECTADOS")
        print("=" * 50)

        # Verificar si parsed_data es válido
        if not parsed_data or not isinstance(parsed_data, dict):
            print("⚠️  No se pudieron detectar alimentos")
            print("=" * 50 + "\n")
            return

        # Obtener alimentos detectados
        alimentos = parsed_data.get('alimentos_detectados', [])

        # Mostrar los alimentos de manera simple
        if alimentos and len(alimentos) > 0:
            for alimento in alimentos:
                nombre = alimento.get('nombre', 'Desconocido')
                categoria = alimento.get('categoria', 'otros')
                cantidad = alimento.get('cantidad', 1)

                # Validar categoría
                categoria_validada = self.validate_category(categoria)

                print(f"📦 {cantidad}x {nombre} ({categoria_validada})")
        else:
            print("🤷 No se detectaron alimentos en las imágenes")

        print("=" * 50 + "\n")

    def simple_text_analysis(self, text):
        """Análisis manual simplificado cuando falla el parsing de JSON"""
        print("🔍 Realizando análisis manual de texto...")

        # Palabras clave para buscar alimentos
        food_keywords = {
            "manzana": "frutas", "banana": "frutas", "naranja": "frutas", "limón": "frutas",
            "leche": "lacteos", "queso": "lacteos", "yogurt": "lacteos", "mantequilla": "lacteos",
            "pollo": "carne", "carne": "carne", "pescado": "carne", "jamón": "carne",
            "lechuga": "verduras", "tomate": "verduras", "zanahoria": "verduras", "papa": "verduras",
            "cebolla": "verduras", "brócoli": "verduras", "apio": "verduras", "pimiento": "verduras",
            "pan": "granos", "arroz": "granos", "pasta": "granos", "cereal": "granos"
        }

        found_foods = []
        text_lower = text.lower()

        for food, category in food_keywords.items():
            if food in text_lower:
                found_foods.append({
                    "nombre": food,
                    "categoria": category,
                    "cantidad": 1
                })

        return {
            "alimentos_detectados": found_foods if found_foods else []
        }

    def analyze_in_batches(self, image_paths):
        """Analiza imágenes en lotes más pequeños con nuevo formato"""
        batch_size = 3
        all_foods = []

        print(f"📦 Dividiendo {len(image_paths)} imágenes en lotes de {batch_size}")

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            print(f"🔄 Procesando lote {i // batch_size + 1}: {len(batch)} imágenes")

            result = self.analyze_multiple_images_with_gemini(batch)
            if result.get("analysis_successful") and result.get("parsed_data"):
                data = result["parsed_data"]
                if "alimentos_detectados" in data:
                    all_foods.extend(data["alimentos_detectados"])

        # Combinar alimentos duplicados
        combined_foods = self.combine_duplicate_foods(all_foods)

        combined_result = {
            "alimentos_detectados": combined_foods
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "multi_batch",
            "total_images": len(image_paths),
            "parsed_data": combined_result,
            "analysis_successful": True
        }

    def combine_duplicate_foods(self, food_list):
        """Combina alimentos duplicados sumando sus cantidades"""
        combined = {}

        for food in food_list:
            nombre = food.get('nombre', '').lower()
            categoria = self.validate_category(food.get('categoria', 'otros'))
            cantidad = food.get('cantidad', 1)

            if nombre:
                if nombre in combined:
                    combined[nombre]['cantidad'] += cantidad
                else:
                    combined[nombre] = {
                        'nombre': nombre,
                        'categoria': categoria,
                        'cantidad': cantidad
                    }

        return list(combined.values())

    def save_inventory_to_json(self):
        """Guarda el inventario con nuevo formato JSON"""
        try:
            # Actualizar timestamp
            self.inventory_data["actualizado"] = datetime.now().isoformat()

            # INTEGRACIÓN: Asegurar que el directorio existe
            os.makedirs(os.path.dirname(self.inventory_file), exist_ok=True)

            with open(self.inventory_file, 'w', encoding='utf-8') as f:
                json.dump(self.inventory_data, f, indent=2, ensure_ascii=False)

            total_items = sum(food['cantidad'] for food in self.inventory_data['alimentos'])
            unique_foods = len(self.inventory_data['alimentos'])

            print(f"💾 Inventario actualizado: {unique_foods} tipos de alimentos, {total_items} unidades totales")
            print(f"📄 Guardado en: {self.inventory_file}")

        except Exception as e:
            print(f"❌ Error guardando inventario: {e}")

    def load_inventory_from_json(self):
        """Carga el inventario desde el JSON con nuevo formato"""
        try:
            if os.path.exists(self.inventory_file):
                with open(self.inventory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Cargar inventario con nuevo formato
                if "alimentos" in data:
                    self.inventory_data = data
                    total_items = sum(food['cantidad'] for food in self.inventory_data['alimentos'])
                    unique_foods = len(self.inventory_data['alimentos'])
                    print(f"📦 Inventario cargado: {unique_foods} tipos, {total_items} unidades")
                else:
                    print("⚠️ Archivo de inventario sin formato válido")
                    # Crear estructura nueva
                    self.inventory_data = {
                        "actualizado": datetime.now().isoformat(),
                        "alimentos": []
                    }
            else:
                print("📭 No existe inventario previo, empezando desde cero")

        except Exception as e:
            print(f"❌ Error cargando inventario: {e}")
            print("📭 Empezando con inventario vacío")
            self.inventory_data = {
                "actualizado": datetime.now().isoformat(),
                "alimentos": []
            }

    def display_inventory(self):
        """Muestra el inventario con nuevo formato"""
        print("\n" + "=" * 60)
        print("🏪 INVENTARIO DEL REFRIGERADOR")
        print("=" * 60)

        if not self.inventory_data['alimentos']:
            print("📭 No hay alimentos en el inventario")
            print("   Analiza algunas fotos primero (presiona 'a')")
        else:
            # Agrupar por categoría
            categories = {}
            total_items = 0

            for food in self.inventory_data['alimentos']:
                categoria = food['categoria']
                if categoria not in categories:
                    categories[categoria] = []
                categories[categoria].append(food)
                total_items += food['cantidad']

            print(f"📊 Total: {len(self.inventory_data['alimentos'])} tipos de alimentos, {total_items} unidades")
            print(f"🕒 Actualizado: {self.inventory_data['actualizado'][:19].replace('T', ' ')}")
            print("")

            # Mostrar por categorías
            for categoria in sorted(categories.keys()):
                foods = categories[categoria]
                categoria_items = sum(food['cantidad'] for food in foods)

                print(f"📂 {categoria.upper()} ({categoria_items} unidades):")
                for food in sorted(foods, key=lambda x: x['nombre']):
                    fecha_ingreso = food.get('fecha_ingreso', 'N/A')
                    print(f"   • {food['cantidad']}x {food['nombre']} (desde {fecha_ingreso})")
                print("")

        print("=" * 60 + "\n")

    def add_to_inventory(self, detected_foods):
        """Agrega alimentos al inventario con nuevo formato"""
        if not detected_foods:
            return

        added_count = 0
        current_date = datetime.now().strftime('%Y-%m-%d')

        for food_data in detected_foods:
            if not food_data or not isinstance(food_data, dict):
                continue

            nombre = food_data.get('nombre', '').strip().lower()
            categoria = self.validate_category(food_data.get('categoria', 'otros'))
            cantidad = max(1, food_data.get('cantidad', 1))  # Mínimo 1

            if not nombre:
                continue

            # Buscar si ya existe el alimento
            index, existing_food = self.find_food_by_name(nombre)

            if existing_food:
                # Incrementar cantidad del alimento existente
                old_quantity = existing_food['cantidad']
                self.inventory_data['alimentos'][index]['cantidad'] += cantidad
                new_quantity = self.inventory_data['alimentos'][index]['cantidad']
                print(f"➕ {nombre}: {old_quantity} → {new_quantity} unidades")
            else:
                # Agregar nuevo alimento
                new_food = {
                    "nombre": nombre,
                    "categoria": categoria,
                    "cantidad": cantidad,
                    "fecha_ingreso": current_date
                }
                self.inventory_data['alimentos'].append(new_food)
                print(f"🆕 Nuevo alimento: {cantidad}x {nombre} ({categoria})")

            added_count += cantidad

        if added_count > 0:
            total_items = sum(food['cantidad'] for food in self.inventory_data['alimentos'])
            print(f"📊 Total en inventario: {len(self.inventory_data['alimentos'])} tipos, {total_items} unidades")
            # Guardar inventario actualizado
            self.save_inventory_to_json()

    def remove_from_inventory(self, detected_foods):
        """Remueve alimentos del inventario con nuevo formato"""
        if not detected_foods:
            return

        removed_count = 0

        for food_data in detected_foods:
            if not food_data or not isinstance(food_data, dict):
                continue

            nombre = food_data.get('nombre', '').strip().lower()
            cantidad_to_remove = max(1, food_data.get('cantidad', 1))

            if not nombre:
                continue

            # Buscar el alimento en el inventario
            index, existing_food = self.find_food_by_name(nombre)

            if existing_food:
                old_quantity = existing_food['cantidad']

                if old_quantity >= cantidad_to_remove:
                    # Reducir cantidad
                    new_quantity = old_quantity - cantidad_to_remove

                    if new_quantity == 0:
                        # Eliminar alimento del inventario
                        self.inventory_data['alimentos'].pop(index)
                        print(f"🗑️  {nombre}: eliminado completamente del inventario")
                    else:
                        # Actualizar cantidad
                        self.inventory_data['alimentos'][index]['cantidad'] = new_quantity
                        print(f"➖ {nombre}: {old_quantity} → {new_quantity} unidades")

                    removed_count += cantidad_to_remove
                else:
                    print(f"⚠️  {nombre}: solo hay {old_quantity} unidades, no se pueden remover {cantidad_to_remove}")
            else:
                print(f"⚠️  {nombre}: no está en el inventario")

        if removed_count > 0:
            total_items = sum(food['cantidad'] for food in self.inventory_data['alimentos'])
            print(f"📊 Total en inventario: {len(self.inventory_data['alimentos'])} tipos, {total_items} unidades")
            # Guardar inventario actualizado
            self.save_inventory_to_json()

    def adjust_sensitivity(self, increase=True):
        """Ajusta la sensibilidad de detección de movimiento"""
        if increase and self.sensitivity_level < 5:
            self.sensitivity_level += 1
        elif not increase and self.sensitivity_level > 1:
            self.sensitivity_level -= 1

        # Ajustar parámetros según nivel de sensibilidad
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

    def detect_movement_in_zones(self, frame):
        """Detecta movimiento en las zonas de entrada y salida"""
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
        """Dibuja información de movimiento en el frame"""
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

    def process_pending_photos(self):
        """Procesa todas las fotos pendientes con Gemini usando nuevo formato"""
        if not self.pending_photos:
            print("📭 No hay fotos pendientes para analizar")
            return

        if not self.gemini_enabled:
            print("❌ Gemini no está configurado")
            return

        print(f"🔄 Iniciando análisis conjunto de {len(self.pending_photos)} fotos...")

        # Separar fotos por tipo de zona
        entry_photos = []
        exit_photos = []

        for photo_info in self.pending_photos:
            if isinstance(photo_info, dict):
                if photo_info['zone_type'] == 'entrada':
                    entry_photos.append(photo_info['path'])
                elif photo_info['zone_type'] == 'salida':
                    exit_photos.append(photo_info['path'])
            else:
                # Compatibilidad con formato anterior (solo path)
                entry_photos.append(photo_info)

        # Procesar fotos de ENTRADA
        if entry_photos:
            print(f"📥 Analizando {len(entry_photos)} fotos de ENTRADA...")
            result = self.analyze_multiple_images_with_gemini(entry_photos)

            if result.get("analysis_successful") and result.get("parsed_data"):
                foods_detected = result["parsed_data"].get("alimentos_detectados", [])

                if foods_detected:
                    print("➕ Agregando alimentos al inventario...")
                    self.add_to_inventory(foods_detected)

        # Procesar fotos de SALIDA
        if exit_photos:
            print(f"📤 Analizando {len(exit_photos)} fotos de SALIDA...")
            result = self.analyze_multiple_images_with_gemini(exit_photos)

            if result.get("analysis_successful") and result.get("parsed_data"):
                foods_detected = result["parsed_data"].get("alimentos_detectados", [])

                if foods_detected:
                    print("➖ Removiendo alimentos del inventario...")
                    self.remove_from_inventory(foods_detected)

        # Limpiar lista de fotos pendientes
        self.pending_photos.clear()
        print("✅ Análisis conjunto completado exitosamente")
        print("🗑️  Lista de fotos pendientes limpiada")

    def define_exit_zone(self, frame):
        """Permite al usuario definir la zona de SALIDA con 6 puntos - MEJORADO PARA RASPBERRY PI"""
        print("Haz clic en 6 puntos para definir la ZONA DE SALIDA")
        print("Los puntos deben formar un polígono")
        print("Presiona 'ESC' para cancelar")
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                print(f"Punto SALIDA {len(points)}: ({x}, {y})")

        # Crear ventana con nombre único
        window_name = 'Definir Zona de SALIDA (6 puntos)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # RASPBERRY PI FIX: Asegurar que la ventana se muestre antes del callback
        temp_frame = frame.copy()
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(1)  # Pequeña pausa para que se renderice la ventana
        time.sleep(0.1)  # Tiempo adicional para estabilizar

        # Ahora asignar el callback después de que la ventana esté activa
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            while len(points) < 6:
                display_frame = temp_frame.copy()

                # Dibujar puntos ya seleccionados en ROJO
                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"S{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Dibujar líneas entre puntos consecutivos en ROJO
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 0, 255), 2)

                # Si hay 5 puntos, dibujar línea temporal al primer punto
                if len(points) == 5:
                    cv2.line(display_frame, points[4], points[0], (0, 0, 255), 1)

                # Instrucciones dinámicas
                instructions = [
                    "ZONA SALIDA - Haz clic en el primer punto",
                    "ZONA SALIDA - Haz clic en el segundo punto",
                    "ZONA SALIDA - Haz clic en el tercer punto",
                    "ZONA SALIDA - Haz clic en el cuarto punto",
                    "ZONA SALIDA - Haz clic en el quinto punto",
                    "ZONA SALIDA - Haz clic en el sexto punto"
                ]

                if len(points) < 6:
                    cv2.putText(display_frame, instructions[len(points)],
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(display_frame, "ESC para cancelar",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    print("Definición de zona de SALIDA cancelada")
                    break
                elif key == ord('q'):
                    break

        except Exception as e:
            print(f"Error en definición de zona de SALIDA: {e}")

        finally:
            # Asegurar que la ventana se cierre correctamente
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Permitir que OpenCV procese la destrucción de ventana
            time.sleep(0.1)  # Tiempo adicional para limpieza

        if len(points) == 6:
            self.exit_zone = points  # Contiene 6 puntos de SALIDA
            self.exit_zone_defined = True
            print(f"Zona de SALIDA definida con 6 puntos:")
            for i, point in enumerate(points):
                print(f"  Punto S{i + 1}: {point}")
        else:
            print("Zona de SALIDA no definida - se necesitan 6 puntos")

    def define_detection_line(self, frame):
        """Permite al usuario definir la zona de ENTRADA con 6 puntos - MEJORADO PARA RASPBERRY PI"""
        print("Haz clic en 6 puntos para definir la ZONA DE ENTRADA")
        print("Los puntos deben formar un polígono")
        print("Presiona 'ESC' para cancelar")
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                print(f"Punto ENTRADA {len(points)}: ({x}, {y})")

        # Crear ventana con nombre único
        window_name = 'Definir Zona de ENTRADA (6 puntos)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # RASPBERRY PI FIX: Asegurar que la ventana se muestre antes del callback
        temp_frame = frame.copy()
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(1)  # Pequeña pausa para que se renderice la ventana
        time.sleep(0.1)  # Tiempo adicional para estabilizar

        # Ahora asignar el callback después de que la ventana esté activa
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            while len(points) < 6:
                display_frame = temp_frame.copy()

                # Dibujar puntos ya seleccionados en VERDE
                for i, point in enumerate(points):
                    cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"E{i + 1}", (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Dibujar líneas entre puntos consecutivos en VERDE
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)

                # Si hay 5 puntos, dibujar línea temporal al primer punto
                if len(points) == 5:
                    cv2.line(display_frame, points[4], points[0], (0, 255, 0), 1)

                # Instrucciones dinámicas
                instructions = [
                    "ZONA ENTRADA - Haz clic en el primer punto",
                    "ZONA ENTRADA - Haz clic en el segundo punto",
                    "ZONA ENTRADA - Haz clic en el tercer punto",
                    "ZONA ENTRADA - Haz clic en el cuarto punto",
                    "ZONA ENTRADA - Haz clic en el quinto punto",
                    "ZONA ENTRADA - Haz clic en el sexto punto"
                ]

                if len(points) < 6:
                    cv2.putText(display_frame, instructions[len(points)],
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(display_frame, "ESC para cancelar",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    print("Definición de zona de ENTRADA cancelada")
                    break
                elif key == ord('q'):
                    break

        except Exception as e:
            print(f"Error en definición de zona de ENTRADA: {e}")

        finally:
            # Asegurar que la ventana se cierre correctamente
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Permitir que OpenCV procese la destrucción de ventana
            time.sleep(0.1)  # Tiempo adicional para limpieza

        if len(points) == 6:
            self.detection_line = points  # Contiene 6 puntos de ENTRADA
            self.line_defined = True
            print(f"Zona de ENTRADA definida con 6 puntos:")
            for i, point in enumerate(points):
                print(f"  Punto E{i + 1}: {point}")
        else:
            print("Zona de ENTRADA no definida - se necesitan 6 puntos")

    def check_hand_zones(self, landmarks, frame_shape):
        """Verifica si la mano está en zona de entrada, salida o ambas"""
        # Usar el centro de la palma (landmark 9)
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

    def hand_crosses_line(self, landmarks, frame_shape):
        """Verifica si la mano está en alguna zona (mantener para compatibilidad)"""
        in_entry, in_exit = self.check_hand_zones(landmarks, frame_shape)
        return in_entry or in_exit

    def draw_detection_areas(self, frame):
        """Dibuja las zonas de entrada y salida en el frame"""
        # Dibujar zona de ENTRADA (verde) - 6 puntos
        if self.line_defined and len(self.detection_line) == 6:
            # Dibujar el polígono de entrada
            pts = np.array(self.detection_line, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            # Dibujar puntos numerados
            for i, point in enumerate(self.detection_line):
                cv2.circle(frame, point, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"E{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            # Etiqueta del área de entrada
            center_x = int(np.mean([p[0] for p in self.detection_line]))
            center_y = int(np.mean([p[1] for p in self.detection_line]))
            cv2.putText(frame, "AREA DE ENTRADA", (center_x - 70, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar zona de SALIDA (roja) - 6 puntos
        if self.exit_zone_defined and len(self.exit_zone) == 6:
            # Dibujar el polígono de salida
            pts = np.array(self.exit_zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

            # Dibujar puntos numerados
            for i, point in enumerate(self.exit_zone):
                cv2.circle(frame, point, 6, (0, 0, 255), -1)
                cv2.putText(frame, f"S{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            # Etiqueta del área de salida
            center_x = int(np.mean([p[0] for p in self.exit_zone]))
            center_y = int(np.mean([p[1] for p in self.exit_zone]))
            cv2.putText(frame, "AREA DE SALIDA", (center_x - 70, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def define_analysis_zone_alternative(self, frame):
        """Método alternativo para definir zona cuando callback del mouse no funciona"""
        print("\n" + "=" * 60)
        print("🔧 MÉTODO ALTERNATIVO - DEFINIR ZONA DE ANÁLISIS")
        print("=" * 60)
        print("El callback del mouse no funciona en este sistema.")
        print("Usaremos un método alternativo con teclado.")
        print("")
        print("OPCIONES:")
        print("1. Usar coordenadas predefinidas (rápido)")
        print("2. Ingresar coordenadas manualmente")
        print("3. Definir zona con teclado interactivo")
        print("4. Cancelar")

        window_name = 'Zona de Análisis - Método Alternativo'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            # Mostrar frame con instrucciones
            display_frame = frame.copy()

            # Instrucciones en la imagen
            instructions = [
                "METODO ALTERNATIVO - SIN MOUSE",
                "Presiona '1' = Zona predefinida (centro)",
                "Presiona '2' = Coordenadas manuales",
                "Presiona '3' = Teclado interactivo",
                "Presiona 'ESC' = Cancelar"
            ]

            for i, instruction in enumerate(instructions):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                thickness = 2 if i == 0 else 1
                cv2.putText(display_frame, instruction, (10, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

            cv2.imshow(window_name, display_frame)

            while True:
                key = cv2.waitKey(30) & 0xFF

                if key == ord('1'):
                    # Opción 1: Zona predefinida (centro de la imagen)
                    h, w = frame.shape[:2]
                    margin_x, margin_y = w // 4, h // 4
                    self.zone_points = [
                        (margin_x, margin_y),  # Esquina superior izquierda
                        (w - margin_x, h - margin_y)  # Esquina inferior derecha
                    ]
                    self.zone_type = "rectangle"
                    print(f"✅ Zona predefinida aplicada: {self.zone_points}")
                    break

                elif key == ord('2'):
                    # Opción 2: Coordenadas manuales
                    self.zone_points = self.get_manual_coordinates(frame)
                    if self.zone_points:
                        self.zone_type = "rectangle"
                        print(f"✅ Coordenadas manuales aplicadas: {self.zone_points}")
                        break

                elif key == ord('3'):
                    # Opción 3: Teclado interactivo
                    self.zone_points = self.define_zone_with_keyboard(frame, window_name)
                    if self.zone_points:
                        self.zone_type = "rectangle"
                        print(f"✅ Zona con teclado aplicada: {self.zone_points}")
                        break

                elif key == 27:  # ESC
                    print("❌ Definición de zona cancelada")
                    break

                elif key == ord('q'):
                    print("❌ Salida solicitada")
                    break

        except Exception as e:
            print(f"❌ Error en método alternativo: {e}")

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        # Aplicar zona si se definió
        if hasattr(self, 'zone_points') and self.zone_points and len(self.zone_points) == 2:
            self.analysis_zone = {
                'type': self.zone_type,
                'points': self.zone_points
            }
            self.zone_defined = True
            self.create_zone_mask(frame.shape)
            print(f"✅ Zona de análisis definida exitosamente: {self.zone_type}")
            return True
        else:
            print("❌ No se definió zona de análisis")
            return False

    def get_manual_coordinates(self, frame):
        """Permite al usuario ingresar coordenadas manualmente"""
        print("\n🔢 INGRESO MANUAL DE COORDENADAS")
        print("Ingresa las coordenadas de la zona rectangular:")
        h, w = frame.shape[:2]
        print(f"Tamaño de la imagen: {w}x{h}")
        print("Formato: x1 y1 x2 y2 (separados por espacios)")
        print("Ejemplo: 100 50 500 400")

        try:
            coords_input = input("Coordenadas (x1 y1 x2 y2): ").strip()
            coords = list(map(int, coords_input.split()))

            if len(coords) != 4:
                print("❌ Error: Se necesitan exactamente 4 números")
                return None

            x1, y1, x2, y2 = coords

            # Validar coordenadas
            if not (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h):
                print(f"❌ Error: Coordenadas fuera de rango (0-{w}, 0-{h})")
                return None

            if x1 >= x2 or y1 >= y2:
                print("❌ Error: x1 debe ser < x2 y y1 debe ser < y2")
                return None

            return [(x1, y1), (x2, y2)]

        except (ValueError, EOFError):
            print("❌ Error: Formato inválido")
            return None

    def define_zone_with_keyboard(self, frame, window_name):
        """Define zona usando teclado para mover un rectángulo"""
        print("\n⌨️  DEFINICIÓN CON TECLADO")
        print("Usa las teclas para ajustar el rectángulo:")
        print("WASD = mover rectángulo")
        print("Flechas = cambiar tamaño")
        print("ENTER = confirmar")
        print("ESC = cancelar")

        # Inicializar rectángulo en el centro
        h, w = frame.shape[:2]
        rect_w, rect_h = w // 3, h // 3  # Tamaño inicial
        rect_x = (w - rect_w) // 2  # Posición inicial centrada
        rect_y = (h - rect_h) // 2

        step_move = 10  # Píxeles de movimiento por tecla
        step_size = 10  # Píxeles de cambio de tamaño

        try:
            while True:
                display_frame = frame.copy()

                # Dibujar rectángulo actual
                cv2.rectangle(display_frame, (rect_x, rect_y),
                              (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 3)

                # Instrucciones en pantalla
                instructions = [
                    "CONTROL CON TECLADO:",
                    "W/A/S/D = Mover rectangulo",
                    "Flechas = Cambiar tamano",
                    "ENTER = Confirmar zona",
                    "ESC = Cancelar"
                ]

                for i, instruction in enumerate(instructions):
                    color = (255, 255, 0) if i == 0 else (255, 255, 255)
                    cv2.putText(display_frame, instruction, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                # Mostrar coordenadas actuales
                coords_text = f"Zona: ({rect_x},{rect_y}) - ({rect_x + rect_w},{rect_y + rect_h})"
                cv2.putText(display_frame, coords_text, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF

                # Movimiento
                if key == ord('w') and rect_y > step_move:
                    rect_y -= step_move
                elif key == ord('s') and rect_y + rect_h < h - step_move:
                    rect_y += step_move
                elif key == ord('a') and rect_x > step_move:
                    rect_x -= step_move
                elif key == ord('d') and rect_x + rect_w < w - step_move:
                    rect_x += step_move

                # Cambio de tamaño
                elif key == 82:  # Flecha arriba - aumentar altura
                    if rect_y > step_size:
                        rect_y -= step_size
                        rect_h += step_size
                elif key == 84:  # Flecha abajo - disminuir altura
                    if rect_h > step_size * 2:
                        rect_h -= step_size
                elif key == 81:  # Flecha izquierda - disminuir anchura
                    if rect_w > step_size * 2:
                        rect_w -= step_size
                elif key == 83:  # Flecha derecha - aumentar anchura
                    if rect_x + rect_w < w - step_size:
                        rect_w += step_size

                # Confirmar o cancelar
                elif key == 13:  # ENTER
                    return [(rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h)]
                elif key == 27:  # ESC
                    return None
                elif key == ord('q'):
                    return None

        except Exception as e:
            print(f"❌ Error en definición con teclado: {e}")
            return None

    def define_analysis_zone(self, frame):
        """Permite al usuario definir la zona de análisis con clic y arrastre - ROBUSTO PARA RASPBERRY PI"""
        print("Haz clic y arrastra para definir la ZONA DE ANÁLISIS")
        print("Presiona 'ESC' para cancelar")
        points = []
        drawing = False
        # Usamos una lista para que sea mutable dentro del callback
        start_point_ref = [None]

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point_ref
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point_ref[0] = (x, y)
                drawing = True
                points.clear()  # Limpiar puntos anteriores al empezar un nuevo arrastre
                print(f"Punto ANÁLISIS iniciado: ({x}, {y})")

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    # Este bloque es solo para visualización mientras se arrastra
                    temp_frame = frame.copy()
                    cv2.rectangle(temp_frame, start_point_ref[0], (x, y), (0, 255, 0), 2)
                    cv2.imshow('Definir Zona de ANÁLISIS (rectángulo)', temp_frame)

            elif event == cv2.EVENT_LBUTTONUP:
                if drawing:
                    end_point = (x, y)
                    # Asegurarse de que el punto inicial es la esquina superior izquierda
                    x1 = min(start_point_ref[0][0], end_point[0])
                    y1 = min(start_point_ref[0][1], end_point[1])
                    x2 = max(start_point_ref[0][0], end_point[0])
                    y2 = max(start_point_ref[0][1], end_point[1])
                    points.extend([(x1, y1), (x2, y2)])
                    drawing = False
                    print(f"Zona de ANÁLISIS definida: {(x1, y1)} -> {(x2, y2)}")

        # Crear la ventana con un nombre único
        window_name = 'Definir Zona de ANÁLISIS (rectángulo)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # RASPBERRY PI FIX: Mostrar la ventana y esperar ANTES de asignar el callback
        temp_frame = frame.copy()
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(1)
        time.sleep(0.1)

        # Ahora que la ventana está garantizada a existir, asignamos el callback
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            # Bucle principal para la definición de la zona
            while len(points) < 2:
                display_frame = frame.copy()

                # Instrucciones
                cv2.putText(display_frame, "ZONA ANALISIS - Haz clic y arrastra",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Presiona 'ESC' para salir",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # Tecla ESC
                    print("Definición de zona de ANÁLISIS cancelada")
                    break
                elif key == ord('q'):
                    break

                # Si el usuario suelta el mouse, `points` tendrá 2 elementos y el bucle terminará.

        except Exception as e:
            print(f"Error en definición de zona de ANÁLISIS: {e}")

        finally:
            # Asegurarse de que la ventana se cierre correctamente
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            time.sleep(0.1)

        if len(points) == 2:
            self.analysis_zone = {
                'type': 'rectangle',
                'points': points
            }
            self.zone_defined = True
            self.create_zone_mask(frame.shape)
            print("✅ Zona de ANÁLISIS definida con rectángulo:")
            print(f"   Esquina 1: {points[0]}")
            print(f"   Esquina 2: {points[1]}")
        else:
            print("❌ Zona de ANÁLISIS no definida.")

    def create_zone_mask(self, frame_shape):
        """Crea una máscara basada en la zona definida"""
        self.zone_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        if self.analysis_zone['type'] == "rectangle":
            pt1, pt2 = self.analysis_zone['points']
            cv2.rectangle(self.zone_mask, pt1, pt2, 255, -1)

        elif self.analysis_zone['type'] == "polygon":
            pts = np.array(self.analysis_zone['points'], np.int32)
            cv2.fillPoly(self.zone_mask, [pts], 255)

    def apply_zone_mask(self, frame):
        """Aplica la máscara de zona al frame"""
        if not self.zone_defined:
            return frame

        # Crear frame enmascarado
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.zone_mask)
        return masked_frame

    def calculate_hand_openness(self, landmarks):
        """Calcula qué tan abierta está la mano basándose en las distancias entre dedos"""
        # Obtener puntos clave de los dedos
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        # Calcular distancias desde la muñeca a cada punta de dedo
        distances = []
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            distances.append(dist)

        # Promedio de distancias (mano más cerrada = distancias menores)
        avg_distance = sum(distances) / len(distances)

        # Calcular "apertura" relativa (esto es una heurística)
        # Mano abierta típicamente tiene distancias mayores
        return avg_distance

    def detect_object_in_hand(self, landmarks, frame_shape):
        """Detecta si la mano probablemente está sosteniendo un objeto"""
        openness = self.calculate_hand_openness(landmarks)

        # Umbral para considerar que hay un objeto
        # Este valor puede necesitar ajuste según tu uso
        openness_threshold = 0.12

        # Si la mano está relativamente cerrada, probablemente sostiene algo
        is_holding = openness < openness_threshold

        return is_holding, openness

    def save_photo(self, frame, clean_frame, zone_type="unknown"):
        """Guarda una foto LIMPIA (sin textos) y la agrega a la lista de análisis pendiente"""
        current_time = datetime.now()

        # Verificar cooldown
        if (current_time.timestamp() - self.last_photo_time) < self.photo_cooldown:
            return False

        filename = f"hand_object_{zone_type}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.photos_dir, filename)

        # GUARDAR LA IMAGEN LIMPIA (sin textos de debugging)
        cv2.imwrite(filepath, clean_frame)

        # REPRODUCIR SONIDO según el tipo de zona
        if zone_type == "entrada":
            print("🔊 ♪ Sonido ENTRADA (grave)")
            self.play_entry_sound()
        elif zone_type == "salida":
            print("🔊 ♫ Sonido SALIDA (agudo)")
            self.play_exit_sound()

        # Modo acumulación: agregar a lista pendiente con tipo de zona
        photo_info = {
            'path': filepath,
            'zone_type': zone_type,
            'timestamp': current_time.isoformat()
        }
        self.pending_photos.append(photo_info)
        print(f"📸 Foto LIMPIA {len(self.pending_photos)} guardada ({zone_type}): {filename}")
        print(f"📋 Total fotos pendientes: {len(self.pending_photos)} (Presiona 'a' para analizar)")

        self.last_photo_time = current_time.timestamp()
        return True

    def test_camera_capabilities(self, cap):
        """Prueba diferentes configuraciones para encontrar el mejor rendimiento"""
        print("\n🔍 DIAGNÓSTICO DE CÁMARA:")
        print("=" * 50)

        # Configuraciones a probar (resolución, fps objetivo)
        test_configs = [
            (640, 480, 120),  # Probar 120fps primero
            (640, 480, 90),  # 90fps
            (640, 480, 60),  # 60fps
            (320, 240, 120),  # Muy baja resolución a 120fps
            (480, 360, 120),  # Baja resolución a 120fps
            (800, 600, 60),  # Mayor resolución, menos FPS
        ]

        best_config = None
        best_real_fps = 0
        all_results = []

        for width, height, target_fps in test_configs:
            print(f"\n📊 Probando {width}x{height} @ {target_fps}fps...")

            try:
                # Configurar con manejo de errores
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)

                # Dar tiempo a la cámara para aplicar cambios
                time.sleep(0.3)

                # Verificar configuración actual
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)

                # Medir FPS real tomando algunas muestras
                real_fps = self.measure_real_fps(cap)

                print(f"   Configurado: {actual_width}x{actual_height} @ {actual_fps}fps")
                print(f"   Real medido: {real_fps:.1f}fps")

                # Guardar resultado
                all_results.append({
                    'width': actual_width,
                    'height': actual_height,
                    'configured_fps': actual_fps,
                    'real_fps': real_fps,
                    'target_fps': target_fps
                })

                # Guardar la mejor configuración (priorizar FPS real más alto)
                if real_fps > best_real_fps:
                    best_real_fps = real_fps
                    best_config = (actual_width, actual_height, target_fps, real_fps)

            except Exception as e:
                print(f"   ❌ Error en configuración: {e}")
                continue

        print(f"\n📋 RESUMEN DE TODAS LAS CONFIGURACIONES:")
        for i, result in enumerate(all_results):
            fps_indicator = "🚀" if result['real_fps'] > 100 else "✅" if result['real_fps'] > 60 else "⚠️"
            print(
                f"   {fps_indicator} {result['width']}x{result['height']} @ {result['configured_fps']}fps → {result['real_fps']:.1f}fps reales")

        print(f"\n🏆 CONFIGURACIÓN DE MÁXIMO RENDIMIENTO:")
        if best_config:
            w, h, target_fps, real_fps = best_config
            print(f"   Resolución: {w}x{h}")
            print(f"   FPS objetivo: {target_fps}")
            print(f"   FPS real esperado: {real_fps:.1f}")

            # Aplicar la configuración de máximo rendimiento CORRECTAMENTE
            try:
                # APLICAR LA CONFIGURACIÓN ÓPTIMA
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, target_fps)  # Usar el target_fps que dio mejor resultado
                time.sleep(0.8)  # Más tiempo para que se aplique

                # Verificar configuración final aplicada
                final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                final_fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"   ✅ Aplicado: {final_width}x{final_height} @ {final_fps}fps")

                # Medir FPS real final para confirmar
                final_real_fps = self.measure_real_fps(cap, sample_frames=25)
                print(f"   🎯 FPS real final: {final_real_fps:.1f}")

                # RETORNAR LA CONFIGURACIÓN APLICADA CORRECTAMENTE
                return final_width, final_height, target_fps, final_real_fps  # target_fps, NO final_fps
            except Exception as e:
                print(f"   ⚠️ Error aplicando mejor configuración: {e}")

        else:
            print("   ⚠️ No se encontró configuración válida, usando valores por defecto")
            return 640, 480, 30, 30

    def measure_real_fps(self, cap, sample_frames=15):
        """Mide el FPS real de la cámara tomando muestras"""
        print("   Midiendo FPS real...", end="", flush=True)

        # Dar tiempo a la cámara para estabilizarse después de cambiar configuración
        time.sleep(0.5)

        # Descartar algunos frames iniciales que pueden ser problemáticos
        for _ in range(5):
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
            except:
                continue

        start_time = time.time()
        frames_captured = 0
        successful_reads = 0

        for i in range(sample_frames):
            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames_captured += 1
                    successful_reads += 1
                else:
                    frames_captured += 1  # Contar intento incluso si falló

            except Exception as e:
                # Si hay error en la lectura, continuar con el siguiente frame
                frames_captured += 1
                continue

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f" ✓")

        if elapsed_time > 0 and successful_reads > 0:
            measured_fps = successful_reads / elapsed_time
            return measured_fps
        else:
            print(f"   ⚠️ No se pudieron leer frames válidos")
            return 0

    def force_max_fps_config(self, cap):
        """Fuerza la configuración para máximo FPS"""
        print("\n🚀 FORZANDO CONFIGURACIÓN DE MÁXIMO FPS...")

        # Configuraciones agresivas para máximo FPS
        high_fps_configs = [
            (640, 480, 120),
            (640, 480, 144),
            (640, 480, 165),
            (480, 360, 120),
            (320, 240, 120),
        ]

        best_real_fps = 0
        best_config = None

        for width, height, target_fps in high_fps_configs:
            print(f"🔥 Probando {width}x{height} @ {target_fps}fps...")

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                time.sleep(0.5)

                real_fps = self.measure_real_fps(cap, sample_frames=20)
                print(f"   → FPS real: {real_fps:.1f}")

                if real_fps > best_real_fps:
                    best_real_fps = real_fps
                    best_config = (width, height, target_fps, real_fps)

            except Exception as e:
                print(f"   ❌ Error: {e}")

        if best_config:
            w, h, target_fps, real_fps = best_config
            print(f"\n🎯 APLICANDO CONFIGURACIÓN DE MÁXIMO RENDIMIENTO:")
            print(f"   {w}x{h} @ {target_fps}fps → {real_fps:.1f}fps reales")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            time.sleep(0.5)

            # Verificar la aplicación y medir FPS final
            final_real_fps = self.measure_real_fps(cap, sample_frames=20)
            print(f"   🎯 FPS real aplicado: {final_real_fps:.1f}")

            return w, h, target_fps, final_real_fps
        else:
            print("❌ No se pudo optimizar para máximo FPS")
            return 640, 480, 30, 30

    def force_fps_update(self, cap):
        """Fuerza una actualización inmediata del FPS para sincronizar la visualización"""
        print("🔄 Forzando actualización de FPS...")

        # Descarta frames iniciales
        for _ in range(3):
            cap.read()

        # Medición rápida pero precisa
        start_time = time.time()
        frame_count = 0
        target_frames = 20

        for i in range(target_frames):
            ret, frame = cap.read()
            if ret:
                frame_count += 1

        elapsed = time.time() - start_time
        if elapsed > 0 and frame_count > 0:
            measured_fps = frame_count / elapsed
            print(f"   📊 FPS actualizado: {measured_fps:.1f}")
            return measured_fps
        return 0

    def measure_current_fps(self, cap, sample_frames=15):
        """Mide el FPS actual en tiempo real de manera más precisa"""
        frames_processed = 0
        start_time = time.time()

        for i in range(sample_frames):
            ret, frame = cap.read()
            if ret:
                frames_processed += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time > 0 and frames_processed > 0:
            return frames_processed / elapsed_time
        return 0

    # INTEGRACIÓN: Agregar parámetro stop_event para compatibilidad con orquestador
    def run(self, stop_event=None):
        """Función principal del detector - INTEGRADA CON ORQUESTADOR"""
        print("📹 Intentando abrir directamente cámara índice 0...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("\n" + "=" * 60)
            print("❌ No se pudo abrir la cámara en índice 0")
            print("   Esto confirma que hay un problema con el acceso a la cámara")
            print("   Posibles causas:")
            print("   • La cámara está siendo usada por otro programa")
            print("   • Permisos insuficientes")
            print("   • Driver de cámara no disponible")
            print("   • Cámara físicamente desconectada")
            print("=" * 60)
            sys.exit(1)

        print("✅ Cámara índice 0 abierta exitosamente!")
        print(f"    Backend utilizado: {cap.getBackendName()}")

        # Test inmediato de lectura de frame
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"✅ Frame de prueba leído correctamente: {test_frame.shape}")
        else:
            print("❌ ERROR: La cámara se abrió pero no puede leer frames")
            cap.release()
            sys.exit(1)

        # Configurar cámara para 120 fps MEJORADO
        print("🔧 Configurando cámara fisheye...")

        # Establecer resolución (menor resolución = mejor para altos FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Configurar FPS a 120
        cap.set(cv2.CAP_PROP_FPS, 120)

        # Dar MÁS tiempo para que se aplique la configuración
        time.sleep(1.0)

        # Verificar configuración actual
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✅ Cámara configurada:")
        print(f"   Resolución: {actual_width}x{actual_height}")
        print(f"   FPS objetivo: 120")
        print(f"   FPS actual: {actual_fps}")

        # Medir FPS real inicial
        print("📊 Midiendo FPS real inicial...")
        initial_real_fps = self.force_fps_update(cap)

        if actual_fps < 120:
            print(f"⚠️  La cámara no pudo configurarse a 120fps")
            print(f"   Funcionando a {actual_fps}fps configurado")
        print(f"🚀 FPS real inicial: {initial_real_fps:.1f}fps")

        print("\n=== CONFIGURACIÓN INICIAL ===")
        print("Presiona 'z' para definir la zona de análisis")
        print("Presiona 'l' para definir el área de ENTRADA (6 puntos)")
        print("Presiona 's' para definir el área de SALIDA (6 puntos)")
        print("Presiona 'h' para modo DETECCIÓN DE MANOS")
        print("Presiona 'n' para modo DETECCIÓN DE MOVIMIENTO")
        print("Presiona 'u' para modo AUTOMÁTICO HÍBRIDO (recomendado)")
        print("Presiona '+' para aumentar sensibilidad")
        print("Presiona '-' para disminuir sensibilidad")
        print(f"🤖 Gemini API: {'✅ Habilitado' if self.gemini_enabled else '❌ Deshabilitado'}")
        print(
            f"🎯 Detección: {'👋 Manos' if self.detection_mode == 'hands' else '🏃 Movimiento' if self.detection_mode == 'movement' else '🤖 Automático (Híbrido)'}")
        print(f"🖐️ Manos parciales: {'✅ Habilitado' if self.partial_hand_mode else '❌ Deshabilitado'}")
        print(f"🎛️ Sensibilidad: Nivel {self.sensitivity_level}/5")
        print(f"⚡ Forzar captura: {'✅ Activado' if self.force_capture_mode else '❌ Desactivado'}")
        print("Presiona 'a' para analizar fotos pendientes")
        print("Presiona 'c' para limpiar fotos pendientes")
        print("Presiona 'p' para alternar modo FORZAR CAPTURA")
        print("Presiona 'ESPACIO' para captura manual de emergencia")
        print("** NUEVO: Las fotos se guardan SIN textos para mejor análisis de Gemini **")
        print("** SONIDOS: Sonido grave=ENTRADA, Sonido agudo=SALIDA **")
        print("** COOLDOWN: 2 segundos entre fotos **")
        print("Presiona 'i' para mostrar inventario de alimentos")
        print("Presiona 'f' para forzar configuración de máximo FPS")
        print("Presiona 'd' para diagnóstico completo de cámara")
        print("Presiona 'q' para salir")

        # Variables para mostrar FPS real - MEJORADAS Y MÁS RESPONSIVAS
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = initial_real_fps  # Inicializar con el FPS real medido
        fps_update_frequency = 20  # Actualizar cada 20 frames para mayor responsividad
        fps_history = [initial_real_fps]  # Inicializar con medición real
        last_fps_update = 0  # Para controlar actualizaciones

        # INTEGRACIÓN: Bucle principal compatible con stop_event del orquestador
        while not (stop_event and stop_event.is_set()):
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer frame de la cámara")
                break

            # Calcular FPS real de manera más precisa y responsiva
            fps_counter += 1
            current_time = time.time()

            # Actualizar FPS cada cierto número de frames O cada segundo
            if fps_counter % fps_update_frequency == 0 or (current_time - last_fps_update) > 1.0:
                elapsed_time = current_time - fps_start_time
                if elapsed_time > 0:
                    instant_fps = fps_update_frequency / elapsed_time
                    fps_history.append(instant_fps)

                    # Mantener solo las últimas 3 mediciones para mayor responsividad
                    if len(fps_history) > 3:
                        fps_history.pop(0)

                    # Usar promedio de las últimas mediciones
                    current_fps = sum(fps_history) / len(fps_history)

                fps_start_time = current_time
                last_fps_update = current_time

            frame = cv2.flip(frame, 1)  # Espejo horizontal

            # GUARDAR FRAME LIMPIO ANTES DE AGREGAR TEXTOS DE DEBUGGING
            clean_frame = frame.copy()

            # Variables para detección
            in_entry_zone = False
            in_exit_zone = False
            should_take_photo = False
            photo_reason = ""

            # MODO AUTOMÁTICO HÍBRIDO - LÓGICA SIMPLIFICADA
            if self.detection_mode == "auto":
                # PASO 1: Intentar detectar manos PRIMERO
                analysis_frame = self.apply_zone_mask(frame) if self.zone_defined else frame
                rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                hands_found = False

                # Verificar si hay manos válidas
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        is_valid_partial, landmark_count, estimated_center = self.validate_partial_hand(landmarks)

                        if is_valid_partial:
                            hands_found = True

                            # Dibujar landmarks
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                            # Detectar objetos
                            is_holding, confidence, detection_type = self.detect_object_in_partial_hand(landmarks)

                            # Verificar zonas
                            in_entry_zone, in_exit_zone = self.check_partial_hand_zones(landmarks, frame.shape)

                            # Dibujar info de mano
                            self.draw_partial_hand_info(frame, landmarks, estimated_center, landmark_count)

                            # Lógica de captura
                            if (is_holding and confidence > 0.2) or self.force_capture_mode:
                                if in_entry_zone and not in_exit_zone:
                                    cv2.putText(frame, "FOTO ENTRADA (MANOS)!", (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    should_take_photo = True
                                    photo_reason = "entrada"
                                elif in_exit_zone and not in_entry_zone:
                                    cv2.putText(frame, "FOTO SALIDA (MANOS)!", (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    should_take_photo = True
                                    photo_reason = "salida"

                            # Mostrar info de debug
                            cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            break  # Solo procesar la primera mano válida

                # PASO 2: Si NO hay manos, usar detección de movimiento
                if not hands_found:
                    cv2.putText(frame, "MODO AUTO: USANDO MOVIMIENTO", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    # Detectar movimiento
                    movement_entry, movement_exit, contours, fg_mask = self.detect_movement_in_zones(frame)

                    # Dibujar contornos
                    self.draw_movement_info(frame, contours)

                    # Lógica de captura por movimiento
                    if movement_entry and not movement_exit:
                        cv2.putText(frame, "MOVIMIENTO ENTRADA!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        should_take_photo = True
                        photo_reason = "entrada"
                    elif movement_exit and not movement_entry:
                        cv2.putText(frame, "MOVIMIENTO SALIDA!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        should_take_photo = True
                        photo_reason = "salida"

                    cv2.putText(frame, f"Contornos: {len(contours)}", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "MODO AUTO: MANOS DETECTADAS", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # MODO SOLO MANOS
            elif self.detection_mode == "hands":
                analysis_frame = self.apply_zone_mask(frame) if self.zone_defined else frame
                rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
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
                                    cv2.putText(frame, "FOTO ENTRADA!", (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    should_take_photo = True
                                    photo_reason = "entrada"
                                elif in_exit_zone and not in_entry_zone:
                                    cv2.putText(frame, "FOTO SALIDA!", (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                    should_take_photo = True
                                    photo_reason = "salida"

                            cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # MODO SOLO MOVIMIENTO
            elif self.detection_mode == "movement":
                movement_entry, movement_exit, contours, fg_mask = self.detect_movement_in_zones(frame)
                self.draw_movement_info(frame, contours)

                if movement_entry and not movement_exit:
                    cv2.putText(frame, "MOVIMIENTO ENTRADA!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    should_take_photo = True
                    photo_reason = "entrada"
                elif movement_exit and not movement_entry:
                    cv2.putText(frame, "MOVIMIENTO SALIDA!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    should_take_photo = True
                    photo_reason = "salida"

                cv2.putText(frame, f"Contornos: {len(contours)}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Tomar foto si es necesario
            if should_take_photo:
                self.save_photo(frame, clean_frame, photo_reason)

            # Dibujar zona de análisis en el frame original
            if self.zone_defined:
                if self.analysis_zone['type'] == "rectangle":
                    pt1, pt2 = self.analysis_zone['points']
                    cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
                    cv2.putText(frame, "ZONA DE ANALISIS", (pt1[0], pt1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif self.analysis_zone['type'] == "polygon":
                    pts = np.array(self.analysis_zone['points'], np.int32)
                    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                    centroid_x = int(np.mean([p[0] for p in self.analysis_zone['points']]))
                    centroid_y = int(np.mean([p[1] for p in self.analysis_zone['points']]))
                    cv2.putText(frame, "ZONA DE ANALISIS", (centroid_x - 50, centroid_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Dibujar zonas de entrada y salida si están definidas
            self.draw_detection_areas(frame)

            # Mostrar FPS real en pantalla con mejor indicador visual
            fps_color = (0, 255, 0) if current_fps > 100 else (0, 255, 255) if current_fps > 60 else (0, 165, 255)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

            # Instrucciones en pantalla ACTUALIZADAS
            instructions = []
            if not self.zone_defined:
                instructions.append("'z' = Definir zona")
            if not self.line_defined:
                instructions.append("'l' = Zona ENTRADA (6pts)")
            if not self.exit_zone_defined:
                instructions.append("'s' = Zona SALIDA (6pts)")

            # Controles de modo
            instructions.append(f"'h' = Modo manos {'✓' if self.detection_mode == 'hands' else ''}")
            instructions.append(f"'n' = Modo movimiento {'✓' if self.detection_mode == 'movement' else ''}")
            instructions.append(f"'u' = Modo automatico {'✓' if self.detection_mode == 'auto' else ''}")

            # Controles de sensibilidad (solo en modo movimiento o auto)
            if self.detection_mode in ["movement", "auto"]:
                instructions.append(f"'+/-' = Sensibilidad ({self.sensitivity_level}/5)")

            if self.pending_photos:
                instructions.append(f"'a' = Analizar {len(self.pending_photos)} fotos")
            instructions.append("'c' = Limpiar pendientes")
            instructions.append("'p' = Forzar captura")
            instructions.append("'SPACE' = Captura manual")
            instructions.append("'i' = Ver inventario")
            instructions.append("'d' = Diagnostico")
            instructions.append("'f' = Max FPS")
            instructions.append("'q' = Salir")

            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 30 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Mostrar estado del sistema
            status_color = (0, 255, 0) if (self.zone_defined and self.line_defined) else (0, 255, 255)
            status_text = "SISTEMA LISTO" if (self.zone_defined and self.line_defined) else "CONFIGURANDO..."
            cv2.putText(frame, status_text, (10, frame.shape[0] - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Mostrar modo de detección
            if self.detection_mode == "hands":
                detection_text = "Deteccion: MANOS"
                detection_color = (255, 255, 0)
            elif self.detection_mode == "movement":
                detection_text = "Deteccion: MOVIMIENTO"
                detection_color = (255, 0, 255)
            else:  # auto
                detection_text = "Deteccion: AUTOMATICO"
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

            if self.pending_photos:
                # Contar fotos por tipo
                entry_count = sum(
                    1 for p in self.pending_photos if isinstance(p, dict) and p.get('zone_type') == 'entrada')
                exit_count = sum(
                    1 for p in self.pending_photos if isinstance(p, dict) and p.get('zone_type') == 'salida')
                other_count = len(self.pending_photos) - entry_count - exit_count

                if entry_count + exit_count + other_count > 0:
                    pending_parts = []
                    if entry_count > 0:
                        pending_parts.append(f"{entry_count}E")
                    if exit_count > 0:
                        pending_parts.append(f"{exit_count}S")
                    if other_count > 0:
                        pending_parts.append(f"{other_count}?")

                    pending_text = f"Pendientes: {'/'.join(pending_parts)} fotos"
                    cv2.putText(frame, pending_text, (10, frame.shape[0] - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Mostrar resolución de la cámara e inventario
            resolution_text = f"Resolucion: {actual_width}x{actual_height}"
            cv2.putText(frame, resolution_text, (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            total_food_items = sum(food['cantidad'] for food in self.inventory_data['alimentos'])
            unique_foods = len(self.inventory_data['alimentos'])
            inventory_text = f"Inventario: {unique_foods} tipos, {total_food_items} unidades"
            cv2.putText(frame, inventory_text, (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Indicador de que las fotos se guardan LIMPIAS
            cv2.putText(frame, "FOTOS SIN DEBUG PARA GEMINI", (frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Detector de Manos con Objetos - Fisheye', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.define_detection_line(frame)
            elif key == ord('s'):
                self.define_exit_zone(frame)
            elif key == ord('z'):
                self.define_analysis_zone(frame)
            elif key == ord('h'):
                # Cambiar a modo detección de manos
                self.detection_mode = "hands"
                print("🎯 Modo cambiado a: DETECCIÓN DE MANOS")
                print("   Detectará objetos en manos usando MediaPipe")
                print(f"   🖐️ Detección parcial: {'✅ Habilitada' if self.partial_hand_mode else '❌ Deshabilitada'}")
                print(f"   📊 Mínimo landmarks: {self.min_landmarks_for_detection}")
            elif key == ord('n'):
                # Cambiar a modo detección de movimiento
                self.detection_mode = "movement"
                print("🎯 Modo cambiado a: DETECCIÓN DE MOVIMIENTO")
                print(f"   Sensibilidad: Nivel {self.sensitivity_level}/5")
                print("   Detectará cualquier movimiento en las zonas")
            elif key == ord('u'):
                # Cambiar a modo automático híbrido
                self.detection_mode = "auto"
                print("🎯 Modo cambiado a: AUTOMÁTICO HÍBRIDO")
                print("   🤖 Prioridad 1: Detectar manos con objetos")
                print("   🏃 Prioridad 2: Si no hay manos, usar detección de movimiento")
                print("   ✨ Máxima cobertura de detección automática")
            elif key == ord('+') or key == ord('='):
                # Aumentar sensibilidad (solo en modo movimiento o auto)
                if self.detection_mode in ["movement", "auto"]:
                    self.adjust_sensitivity(increase=True)
                else:
                    print("⚠️ Ajuste de sensibilidad solo disponible en modo movimiento o automático")
            elif key == ord('-') or key == ord('_'):
                # Disminuir sensibilidad (solo en modo movimiento o auto)
                if self.detection_mode in ["movement", "auto"]:
                    self.adjust_sensitivity(increase=False)
                else:
                    print("⚠️ Ajuste de sensibilidad solo disponible en modo movimiento o automático")
            elif key == ord('a'):
                # Analizar fotos pendientes
                if self.pending_photos:
                    self.process_pending_photos()
                else:
                    print("📭 No hay fotos pendientes para analizar")
            elif key == ord('c'):
                # Limpiar fotos pendientes
                if self.pending_photos:
                    count = len(self.pending_photos)
                    self.pending_photos.clear()
                    print(f"🗑️  {count} fotos pendientes eliminadas de la lista")
                else:
                    print("📭 No hay fotos pendientes que limpiar")
            elif key == ord('p'):
                # Alternar modo forzar captura
                self.force_capture_mode = not self.force_capture_mode
                mode_text = "ACTIVADO" if self.force_capture_mode else "DESACTIVADO"
                print(f"⚡ Modo FORZAR CAPTURA: {mode_text}")
                if self.force_capture_mode:
                    print("   ⚠️  Capturará fotos al detectar cualquier mano en las zonas")
                else:
                    print("   🎯 Capturará fotos solo al detectar objetos en manos")
            elif key == 32:  # ESPACIO key
                # Captura manual de emergencia
                print("📸 CAPTURA MANUAL DE EMERGENCIA")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"manual_emergency_{timestamp}.jpg"
                filepath = os.path.join(self.photos_dir, filename)

                # GUARDAR IMAGEN LIMPIA (sin textos de debugging)
                cv2.imwrite(filepath, clean_frame)

                # Agregar a fotos pendientes como entrada por defecto
                photo_info = {
                    'path': filepath,
                    'zone_type': 'entrada',  # Por defecto entrada, el usuario puede cambiar
                    'timestamp': datetime.now().isoformat()
                }
                self.pending_photos.append(photo_info)
                print(f"📸 Foto LIMPIA manual guardada: {filename}")
                print(f"📋 Agregada como ENTRADA - Cambia manualmente si es SALIDA")
            elif key == ord('i'):
                # Mostrar inventario con nuevo formato
                self.display_inventory()
            elif key == ord('d'):
                # Ejecutar diagnóstico completo y APLICAR la mejor configuración
                print("\n🔧 Ejecutando diagnóstico completo...")
                new_width, new_height, new_fps, new_real_fps = self.test_camera_capabilities(cap)
                # Actualizar variables para mostrar en pantalla
                actual_width, actual_height, actual_fps = new_width, new_height, new_fps
                # REINICIAR COMPLETAMENTE la medición de FPS
                fps_counter = 0
                fps_start_time = time.time()
                fps_history.clear()
                current_fps = new_real_fps  # Forzar actualización inmediata
                print("✅ Diagnóstico completado y configuración aplicada")
                print(f"🎯 Nuevos valores: {actual_width}x{actual_height} @ {actual_fps}fps (real: {new_real_fps:.1f})")
                # Forzar una medición inmediata para verificar
                immediate_fps = self.measure_current_fps(cap, 10)
                print(f"🔍 Verificación inmediata: {immediate_fps:.1f}fps")
            elif key == ord('f'):
                # Forzar configuración de máximo FPS
                print("\n🚀 Forzando configuración de máximo FPS...")
                new_width, new_height, new_fps, new_real_fps = self.force_max_fps_config(cap)
                # Actualizar variables para mostrar en pantalla
                actual_width, actual_height, actual_fps = new_width, new_height, new_fps
                # REINICIAR COMPLETAMENTE la medición de FPS
                fps_counter = 0
                fps_start_time = time.time()
                fps_history.clear()
                current_fps = new_real_fps  # Forzar actualización inmediata
                last_fps_update = time.time()
                print(f"✅ Configuración aplicada: {new_real_fps:.1f}fps reales")
                # Verificación inmediata
                immediate_fps = self.force_fps_update(cap)
                print(f"🔍 FPS verificado: {immediate_fps:.1f}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"🎯 Cámara cerrada. FPS promedio final: {current_fps:.1f}")


if __name__ == "__main__":
    detector = HandObjectDetector()
    detector.run()