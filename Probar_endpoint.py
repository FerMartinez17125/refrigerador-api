import requests

# Asegúrate de tener una imagen en el mismo directorio

with open("foto_prueba.jpg", "rb") as img:
    res = requests.post(

        "https://refrigerador-api.onrender.com/analizar",

        files={"imagen": img}

    )

    print(res.json())

