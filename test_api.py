import requests
import json
from PIL import Image
import os


class RefrigeradorAPIClient:
    """Cliente para probar la API del refrigerador inteligente en Render"""

    def __init__(self, base_url="https://refrigerador-api-4.onrender.com"):
        self.base_url = base_url
        self.analizar_endpoint = f"{base_url}/analizar"

    def test_connection(self):
        """Prueba la conexión básica a la API"""
        print("🔍 Probando conexión a la API...")
        try:
            # Intentar acceder a la raíz (esperamos 404, pero confirma que el servidor responde)
            response = requests.get(self.base_url, timeout=10)
            print(f"✅ Servidor responde - Status: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return False

    def analizar_imagen(self, ruta_imagen):
        """Envía una imagen a la API para análisis"""

        if not os.path.exists(ruta_imagen):
            print(f"❌ Error: No se encuentra la imagen en {ruta_imagen}")
            return None

        print(f"📤 Enviando imagen: {ruta_imagen}")
        print(f"📍 URL: {self.analizar_endpoint}")

        try:
            # Abrir la imagen para verificar que es válida
            with Image.open(ruta_imagen) as img:
                print(f"🖼️  Imagen válida: {img.size} ({img.format})")

            # Enviar a la API
            with open(ruta_imagen, 'rb') as img_file:
                files = {'imagen': img_file}

                print("⏳ Enviando solicitud a Render...")
                response = requests.post(
                    self.analizar_endpoint,
                    files=files,
                    timeout=30  # 30 segundos de timeout para Gemini
                )

            print(f"📊 Status Code: {response.status_code}")

            if response.status_code == 200:
                resultado = response.json()
                print("✅ ¡Análisis exitoso!")
                return resultado
            else:
                print(f"❌ Error del servidor:")
                try:
                    error_data = response.json()
                    print(f"   Mensaje: {error_data}")
                except:
                    print(f"   Respuesta: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print("⏰ Timeout - La API tardó demasiado en responder")
            print("   (Esto puede ser normal la primera vez que se usa Gemini)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de red: {e}")
            return None
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return None

    def mostrar_resultados(self, resultado):
        """Muestra los resultados de manera ordenada"""
        if not resultado:
            print("😔 No hay resultados para mostrar")
            return

        print("\n" + "=" * 60)
        print("🔍 RESULTADOS DEL ANÁLISIS")
        print("=" * 60)

        alimentos = resultado.get('alimentos_detectados', [])

        if not alimentos:
            print("🤷 No se detectaron alimentos en la imagen")
        else:
            print(f"📦 {len(alimentos)} tipos de alimentos detectados:")
            print()

            for i, alimento in enumerate(alimentos, 1):
                nombre = alimento.get('nombre', 'N/A')
                categoria = alimento.get('categoria', 'N/A')
                cantidad = alimento.get('cantidad', 0)

                print(f"   {i}. {cantidad}x {nombre.title()}")
                print(f"      Categoría: {categoria.capitalize()}")
                print()

        print("=" * 60)


def main():
    """Función principal de prueba"""
    print("🧪 PROBADOR DE API - REFRIGERADOR INTELIGENTE")
    print("=" * 60)

    # Crear cliente
    client = RefrigeradorAPIClient()

    # Paso 1: Probar conexión
    if not client.test_connection():
        print("❌ No se pudo conectar a la API. Verifica la URL.")
        return

    # Paso 2: Solicitar ruta de imagen
    print("\n📸 ANÁLISIS DE IMAGEN")
    print("-" * 30)

    # Puedes cambiar esta ruta por la de tu imagen
    ruta_imagen = input("Ingresa la ruta de tu imagen (o presiona Enter para usar 'test_image.jpg'): ").strip()

    if not ruta_imagen:
        ruta_imagen = "test_image.jpg"

    # Paso 3: Analizar imagen
    resultado = client.analizar_imagen(ruta_imagen)

    # Paso 4: Mostrar resultados
    client.mostrar_resultados(resultado)

    # Paso 5: Mostrar JSON completo
    if resultado:
        print("\n📋 RESPUESTA JSON COMPLETA:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))


def test_rapido():
    """Función para test rápido con imagen específica"""
    print("⚡ TEST RÁPIDO")

    # Tu imagen de prueba
    ruta_imagen = r"C:\Users\fermt\Documents\refrigerador_api\foto_prueba.jpg"

    client = RefrigeradorAPIClient()
    resultado = client.analizar_imagen(ruta_imagen)
    client.mostrar_resultados(resultado)


if __name__ == "__main__":
    print("Opciones:")
    print("1. Test completo interactivo")
    print("2. Test rápido (edita la ruta en el código)")

    opcion = input("Elige una opción (1 o 2): ").strip()

    if opcion == "2":
        test_rapido()
    else:
        main()