from flask import Flask, request, jsonify
from PIL import Image
from detector import detectar_alimentos
import json
import os
from datetime import datetime

app = Flask(__name__)

# Archivo de inventario en la nube (persistente en Render)
INVENTORY_FILE = "inventory_cloud.json"


def load_inventory():
    """Carga el inventario desde archivo en la nube"""
    try:
        if os.path.exists(INVENTORY_FILE):
            with open(INVENTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Inventario inicial vacío
            return {
                "actualizado": datetime.now().isoformat(),
                "alimentos": []
            }
    except Exception as e:
        print(f"Error cargando inventario: {e}")
        return {
            "actualizado": datetime.now().isoformat(),
            "alimentos": []
        }


def save_inventory(inventory_data):
    """Guarda el inventario en archivo en la nube"""
    try:
        inventory_data["actualizado"] = datetime.now().isoformat()
        with open(INVENTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(inventory_data, f, indent=2, ensure_ascii=False)

        total_items = sum(food['cantidad'] for food in inventory_data['alimentos'])
        print(f"💾 Inventario cloud actualizado: {len(inventory_data['alimentos'])} tipos, {total_items} unidades")
        return True
    except Exception as e:
        print(f"Error guardando inventario: {e}")
        return False


def find_food_by_name(inventory_data, name):
    """Busca un alimento por nombre en el inventario"""
    name_lower = name.lower().strip()
    for i, food in enumerate(inventory_data["alimentos"]):
        if food["nombre"].lower() == name_lower:
            return i, food
    return None, None


def add_to_cloud_inventory(detected_foods):
    """Agrega alimentos al inventario en la nube"""
    if not detected_foods:
        return {"success": False, "message": "No hay alimentos para agregar"}

    inventory_data = load_inventory()
    added_count = 0
    current_date = datetime.now().strftime('%Y-%m-%d')
    changes = []

    for food_data in detected_foods:
        if not food_data or not isinstance(food_data, dict):
            continue

        nombre = food_data.get('nombre', '').strip().lower()
        categoria = food_data.get('categoria', 'otros')
        cantidad = max(1, food_data.get('cantidad', 1))

        if not nombre:
            continue

        # Buscar si ya existe
        index, existing_food = find_food_by_name(inventory_data, nombre)

        if existing_food:
            old_quantity = existing_food['cantidad']
            inventory_data['alimentos'][index]['cantidad'] += cantidad
            new_quantity = inventory_data['alimentos'][index]['cantidad']
            changes.append(f"➕ {nombre}: {old_quantity} → {new_quantity} unidades")
        else:
            new_food = {
                "nombre": nombre,
                "categoria": categoria,
                "cantidad": cantidad,
                "fecha_ingreso": current_date
            }
            inventory_data['alimentos'].append(new_food)
            changes.append(f"🆕 Nuevo: {cantidad}x {nombre} ({categoria})")

        added_count += cantidad

    if added_count > 0:
        if save_inventory(inventory_data):
            return {
                "success": True,
                "message": f"Se agregaron {added_count} alimentos",
                "changes": changes,
                "total_types": len(inventory_data['alimentos']),
                "total_items": sum(food['cantidad'] for food in inventory_data['alimentos'])
            }

    return {"success": False, "message": "No se pudieron agregar alimentos"}


def remove_from_cloud_inventory(detected_foods):
    """Remueve alimentos del inventario en la nube"""
    if not detected_foods:
        return {"success": False, "message": "No hay alimentos para remover"}

    inventory_data = load_inventory()
    removed_count = 0
    changes = []

    for food_data in detected_foods:
        if not food_data or not isinstance(food_data, dict):
            continue

        nombre = food_data.get('nombre', '').strip().lower()
        cantidad_to_remove = max(1, food_data.get('cantidad', 1))

        if not nombre:
            continue

        index, existing_food = find_food_by_name(inventory_data, nombre)

        if existing_food:
            old_quantity = existing_food['cantidad']

            if old_quantity >= cantidad_to_remove:
                new_quantity = old_quantity - cantidad_to_remove

                if new_quantity == 0:
                    inventory_data['alimentos'].pop(index)
                    changes.append(f"🗑️ {nombre}: eliminado completamente")
                else:
                    inventory_data['alimentos'][index]['cantidad'] = new_quantity
                    changes.append(f"➖ {nombre}: {old_quantity} → {new_quantity} unidades")

                removed_count += cantidad_to_remove
            else:
                changes.append(f"⚠️ {nombre}: solo hay {old_quantity}, no se pueden remover {cantidad_to_remove}")
        else:
            changes.append(f"⚠️ {nombre}: no está en el inventario")

    if removed_count > 0:
        if save_inventory(inventory_data):
            return {
                "success": True,
                "message": f"Se removieron {removed_count} alimentos",
                "changes": changes,
                "total_types": len(inventory_data['alimentos']),
                "total_items": sum(food['cantidad'] for food in inventory_data['alimentos'])
            }

    return {"success": False, "message": "No se pudieron remover alimentos"}


@app.route('/')
def home():
    return jsonify({
        'mensaje': 'API Refrigerador Inteligente - CON INVENTARIO CLOUD',
        'status': 'funcionando',
        'endpoints': {
            '/analizar (POST)': 'Analiza imagen con Gemini',
            '/inventario (GET)': 'Obtiene inventario completo',
            '/inventario/agregar (POST)': 'Agrega alimentos al inventario',
            '/inventario/remover (POST)': 'Remueve alimentos del inventario',
            '/inventario/limpiar (POST)': 'Limpia todo el inventario'
        }
    })


@app.route('/analizar', methods=['POST'])
def analizar():
    """Analiza imagen Y maneja inventario automáticamente"""
    try:
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se recibió una imagen'}), 400

        # Obtener tipo de zona (entrada o salida)
        zone_type = request.form.get('zone_type', 'unknown')

        # Analizar imagen con Gemini
        imagen = Image.open(request.files['imagen'])
        alimentos_detectados = detectar_alimentos(imagen)

        response_data = {
            'alimentos_detectados': alimentos_detectados,
            'zone_type': zone_type,
            'inventory_updated': False,
            'inventory_changes': []
        }

        # Si hay alimentos detectados, actualizar inventario automáticamente
        if alimentos_detectados:
            if zone_type == 'entrada':
                inventory_result = add_to_cloud_inventory(alimentos_detectados)
                response_data['inventory_updated'] = inventory_result['success']
                response_data['inventory_changes'] = inventory_result.get('changes', [])
                response_data['inventory_action'] = 'added'

            elif zone_type == 'salida':
                inventory_result = remove_from_cloud_inventory(alimentos_detectados)
                response_data['inventory_updated'] = inventory_result['success']
                response_data['inventory_changes'] = inventory_result.get('changes', [])
                response_data['inventory_action'] = 'removed'

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inventario', methods=['GET'])
def get_inventario():
    """Obtiene el inventario completo"""
    try:
        inventory_data = load_inventory()

        # Calcular estadísticas
        total_items = sum(food['cantidad'] for food in inventory_data['alimentos'])
        categories = {}

        for food in inventory_data['alimentos']:
            categoria = food['categoria']
            if categoria not in categories:
                categories[categoria] = 0
            categories[categoria] += food['cantidad']

        return jsonify({
            'inventario': inventory_data,
            'estadisticas': {
                'total_types': len(inventory_data['alimentos']),
                'total_items': total_items,
                'categories': categories,
                'last_updated': inventory_data.get('actualizado')
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inventario/agregar', methods=['POST'])
def agregar_inventario():
    """Agrega alimentos al inventario manualmente"""
    try:
        data = request.get_json()
        alimentos = data.get('alimentos', [])

        if not alimentos:
            return jsonify({'error': 'No se proporcionaron alimentos'}), 400

        result = add_to_cloud_inventory(alimentos)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inventario/remover', methods=['POST'])
def remover_inventario():
    """Remueve alimentos del inventario manualmente"""
    try:
        data = request.get_json()
        alimentos = data.get('alimentos', [])

        if not alimentos:
            return jsonify({'error': 'No se proporcionaron alimentos'}), 400

        result = remove_from_cloud_inventory(alimentos)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inventario/limpiar', methods=['POST'])
def limpiar_inventario():
    """Limpia todo el inventario"""
    try:
        inventory_data = {
            "actualizado": datetime.now().isoformat(),
            "alimentos": []
        }

        if save_inventory(inventory_data):
            return jsonify({
                'success': True,
                'message': 'Inventario limpiado completamente'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Error limpiando inventario'
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 API REFRIGERADOR INTELIGENTE - CON INVENTARIO CLOUD")
    print("=" * 60)
    print("🔍 Análisis: Gemini AI")
    print("📊 Inventario: Centralizado en la nube")
    print("🌐 Endpoints disponibles:")
    print("   POST /analizar - Analiza imagen + actualiza inventario")
    print("   GET  /inventario - Obtiene inventario completo")
    print("   POST /inventario/agregar - Agrega alimentos")
    print("   POST /inventario/remover - Remueve alimentos")
    print("   POST /inventario/limpiar - Limpia inventario")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000)