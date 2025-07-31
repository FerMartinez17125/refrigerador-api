from flask import Flask, request, jsonify

from datetime import datetime

import os

app = Flask(__name__)


@app.route('/analizar', methods=['POST'])
def analizar():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró imagen"}), 400

    imagen = request.files['imagen']

    filename = f"uploads/{datetime.now().isoformat()}.jpg"

    os.makedirs("uploads", exist_ok=True)

    imagen.save(filename)

    # Simulación de resultado IA

    resultado = {

        "alimentos_detectados": [

            {

                "nombre": "manzana",

                "categoria": "frutas",

                "cantidad": 2

            }

        ]

    }

    return jsonify(resultado)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

