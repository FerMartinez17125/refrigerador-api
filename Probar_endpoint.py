import requests

with open("foto_prueba.jpg", "rb") as img:
    r = requests.post("https://refrigerador-api.onrender.com/analizar", files={"imagen": img})
    print(r.json())
