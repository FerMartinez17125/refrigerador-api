import requests

with open("foto_prueba.jpg", "rb") as img:
    r = requests.post(url="https://refrigerador-api-3.onrender.com/analizar", files={"imagen": img})

    print("Código de estado:", r.status_code)
    print("Texto de respuesta:")
    print(r.text)
