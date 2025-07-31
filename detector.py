import os
import json
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any


class FoodDetector:
    """
    Detector de alimentos usando Google Gemini API
    Diseñado para funcionar en la nube sin dependencias de cámara
    """

    def __init__(self):
        """Inicializa el detector y configura Gemini API"""
        self.gemini_enabled = False
        self.model = None

        # Categorías válidas para clasificación (extraídas del código original)
        self.valid_categories = ["carne", "verduras", "frutas", "lacteos", "granos", "otros"]

        # Configurar Gemini API
        self._setup_gemini()

    def _setup_gemini(self):
        """Configura la API de Gemini usando variable de entorno"""
        try:
            # Obtener API key desde variable de entorno
            api_key = os.getenv("AIzaSyAKcJSkpXKwBpH0zUxu21DYIQYzJZywlhA")

            if not api_key:
                raise ValueError("Variable de entorno GEMINI_API_KEY no configurada")

            # Configurar Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_enabled = True

            print("✅ Gemini API configurada correctamente")

        except Exception as e:
            print(f"❌ Error configurando Gemini API: {e}")
            self.gemini_enabled = False
            raise

    def validate_category(self, category: str) -> str:
        """
        Valida y normaliza una categoría de alimento
        (Lógica extraída del código original)
        """
        if not category:
            return "otros"

        category_lower = category.lower().strip()

        # Buscar coincidencia exacta
        for valid_cat in self.valid_categories:
            if category_lower == valid_cat:
                return valid_cat

        # Buscar coincidencias parciales con sinónimos
        category_mappings = {
            "carne": ["pollo", "res", "cerdo", "pescado", "mariscos", "jamón", "salchicha"],
            "verduras": ["vegetales", "hortalizas", "lechuga", "tomate", "zanahoria", "papa", "cebolla", "brócoli",
                         "apio", "pimiento"],
            "frutas": ["fruta", "manzana", "banana", "naranja", "limón", "fresa", "uva"],
            "lacteos": ["leche", "queso", "yogurt", "mantequilla", "crema"],
            "granos": ["cereales", "arroz", "pasta", "pan", "avena", "quinoa", "trigo"]
        }

        for valid_cat, synonyms in category_mappings.items():
            if category_lower in synonyms or any(syn in category_lower for syn in synonyms):
                return valid_cat

        return "otros"

    def _parse_gemini_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parsea la respuesta de Gemini y extrae los alimentos detectados
        (Lógica extraída y simplificada del código original)
        """
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

            # Intentar parsear JSON
            parsed_data = json.loads(json_text)

            # Extraer alimentos detectados
            alimentos = parsed_data.get('alimentos_detectados', [])

            # Validar y limpiar cada alimento
            alimentos_limpios = []
            for alimento in alimentos:
                if isinstance(alimento, dict):
                    nombre = alimento.get('nombre', '').strip().lower()
                    categoria = self.validate_category(alimento.get('categoria', 'otros'))
                    cantidad = max(1, int(alimento.get('cantidad', 1)))  # Mínimo 1

                    if nombre:  # Solo agregar si tiene nombre válido
                        alimentos_limpios.append({
                            "nombre": nombre,
                            "categoria": categoria,
                            "cantidad": cantidad
                        })

            return alimentos_limpios

        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando JSON de Gemini: {e}")
            # Intentar análisis manual como fallback
            return self._manual_text_analysis(response_text)
        except Exception as e:
            print(f"❌ Error procesando respuesta de Gemini: {e}")
            return []

    def _manual_text_analysis(self, text: str) -> List[Dict[str, Any]]:
        """
        Análisis manual simplificado cuando falla el parsing de JSON
        (Extraído del código original)
        """
        print("🔍 Realizando análisis manual de texto...")

        # Palabras clave para buscar alimentos
        food_keywords = {
            "manzana": "frutas", "banana": "frutas", "naranja": "frutas", "limón": "frutas",
            "fresa": "frutas", "uva": "frutas", "pera": "frutas", "durazno": "frutas",
            "leche": "lacteos", "queso": "lacteos", "yogurt": "lacteos", "mantequilla": "lacteos",
            "crema": "lacteos", "yogur": "lacteos",
            "pollo": "carne", "carne": "carne", "pescado": "carne", "jamón": "carne",
            "res": "carne", "cerdo": "carne", "salchicha": "carne",
            "lechuga": "verduras", "tomate": "verduras", "zanahoria": "verduras", "papa": "verduras",
            "cebolla": "verduras", "brócoli": "verduras", "apio": "verduras", "pimiento": "verduras",
            "pan": "granos", "arroz": "granos", "pasta": "granos", "cereal": "granos",
            "avena": "granos", "quinoa": "granos"
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

        return found_foods

    def detectar_alimentos(self, imagen: Image.Image) -> List[Dict[str, Any]]:
        """
        Función principal: detecta alimentos en una imagen usando Gemini

        Args:
            imagen: Imagen PIL ya cargada

        Returns:
            Lista de diccionarios con formato:
            [{"nombre": str, "categoria": str, "cantidad": int}, ...]
        """
        if not self.gemini_enabled:
            raise Exception("Gemini API no está configurada correctamente")

        if not isinstance(imagen, Image.Image):
            raise ValueError("La imagen debe ser un objeto PIL.Image.Image")

        try:
            # Redimensionar imagen si es muy grande (optimización para Gemini)
            if imagen.width > 1024 or imagen.height > 1024:
                imagen.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                print(f"🔧 Imagen redimensionada a {imagen.width}x{imagen.height}")

            # Prompt optimizado (extraído del código original)
            prompt = f"""
Eres un refrigerador inteligente con capacidad de reconocimiento de alimentos. Analiza esta imagen donde una persona sostiene alimentos en sus manos.

IMPORTANTE:
- La imagen puede haber sido tomada con una cámara fish-eye, por lo que puede tener ligera distorsión visual.
- Identifica únicamente los alimentos que la persona sostiene en sus manos.

Por cada alimento detectado, proporciona:
1. **nombre** específico del alimento (por ejemplo: "catsup", "coca cola", "manzana")
2. **categoría**, que debe ser una de estas: {', '.join(self.valid_categories)}
3. **cantidad exacta**, es decir, cuántos objetos iguales están presentes

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

REGLAS ESPECÍFICAS:
✅ Solo cuenta alimentos que estén claramente en las manos
✅ Usa nombres específicos (no genéricos)
✅ Las categorías deben ser exactamente una de: {', '.join(self.valid_categories)}
✅ Cuenta la cantidad exacta de cada tipo de alimento

❌ NO incluyas alimentos que no estén siendo sostenidos
❌ NO inventes alimentos que no veas claramente
❌ NO uses categorías que no estén en la lista válida
"""

            # Enviar a Gemini
            print("📤 Enviando imagen a Gemini para análisis...")
            response = self.model.generate_content([prompt, imagen])

            if not response.text:
                print("⚠️ Gemini no devolvió respuesta")
                return []

            print("📝 Respuesta recibida de Gemini")

            # Procesar respuesta
            alimentos_detectados = self._parse_gemini_response(response.text)

            # Log de resultados
            if alimentos_detectados:
                print(f"✅ {len(alimentos_detectados)} tipos de alimentos detectados:")
                for alimento in alimentos_detectados:
                    print(f"   📦 {alimento['cantidad']}x {alimento['nombre']} ({alimento['categoria']})")
            else:
                print("🤷 No se detectaron alimentos en la imagen")

            return alimentos_detectados

        except Exception as e:
            print(f"❌ Error en análisis con Gemini: {e}")
            raise


# Función de conveniencia para usar directamente
def detectar_alimentos(imagen: Image.Image) -> List[Dict[str, Any]]:
    """
    Función simple para detectar alimentos en una imagen

    Args:
        imagen: Imagen PIL ya cargada

    Returns:
        Lista de alimentos detectados: [{"nombre": str, "categoria": str, "cantidad": int}]
    """
    detector = FoodDetector()
    return detector.detectar_alimentos(imagen)


# Ejemplo de uso para testing local
if __name__ == "__main__":
    # Este bloque solo se ejecuta si ejecutas el archivo directamente
    # No afecta cuando lo importas en tu Flask app

    print("🧪 Modo de testing - detector.py")

    # Verificar que la variable de entorno esté configurada
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Configura la variable de entorno GEMINI_API_KEY para testing")
        exit(1)

    # Ejemplo de uso (puedes comentar esto)
    try:
        # Crear una imagen de prueba (1x1 pixel blanco)
        test_image = Image.new('RGB', (100, 100), color='white')

        print("🔬 Probando detector con imagen de prueba...")
        result = detectar_alimentos(test_image)
        print(f"📊 Resultado: {result}")

    except Exception as e:
        print(f"❌ Error en test: {e}")