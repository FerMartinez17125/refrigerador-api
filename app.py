from flask import Flask, request, jsonify
from PIL import Image
from detector import detectar_alimentos

app = Flask(__name__)

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se recibió una imagen'}), 400

        imagen = Image.open(request.files['imagen'])
        resultado = detectar_alimentos(imagen)

        return jsonify({'alimentos_detectados': resultado})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
