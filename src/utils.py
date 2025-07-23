#!/usr/bin/env python3
"""
Utilidades y constantes para el Sistema Experto de Rutas en Bogotá
"""

import time
import math
from typing import Dict, Tuple, Optional

# Información del sistema
SISTEMA_INFO = {
    'nombre': 'Sistema Experto de Rutas Bogotá',
    'version': '1.0.0',
    'autor': 'Estudiante UNIMINUTO',
    'algoritmos': ['dijkstra', 'a_estrella'],
    'criterios': ['distancia', 'tiempo']
}

# Emojis para la interfaz
EMOJIS = {
    'inicio': '🚀',
    'ruta': '📍',
    'exito': '✅',
    'error': '❌',
    'info': 'ℹ️',
    'advertencia': '⚠️',
    'exploracion': '🔍',
    'reporte': '📊',
    'mapa': '🗺️',
    'ubicacion': '📍',
    'tiempo': '⏱️',
    'distancia': '📏',
    'velocidad': '🏃',
    'trafico': '🚦'
}

# Coordenadas de los principales puntos de Bogotá (ESTRUCTURA CORREGIDA)
COORDENADAS_BOGOTA = {
    'CENTRO_BOGOTA': {'latitud': 4.5981, 'longitud': -74.0758, 'descripcion': 'Centro de Bogotá', 'tipo': 'centro'},
    'ZONA_ROSA': {'latitud': 4.6631, 'longitud': -74.0606, 'descripcion': 'Zona Rosa', 'tipo': 'comercial'},
    'UNIMINUTO_CALLE_80': {'latitud': 4.7050, 'longitud': -74.0900, 'descripcion': 'UNIMINUTO Calle 80', 'tipo': 'universidad'},
    'UNIMINUTO_PERDOMO': {'latitud': 4.5820, 'longitud': -74.1390, 'descripcion': 'UNIMINUTO Perdomo', 'tipo': 'universidad'},
    'CIUDAD_UNIVERSITARIA': {'latitud': 4.6356, 'longitud': -74.0817, 'descripcion': 'Ciudad Universitaria (Universidad Nacional)', 'tipo': 'universidad'},
    'RESTREPO': {'latitud': 4.6013, 'longitud': -74.0936, 'descripcion': 'Restrepo', 'tipo': 'residencial'},
    'PERDOMO': {'latitud': 4.5820, 'longitud': -74.1390, 'descripcion': 'Perdomo', 'tipo': 'residencial'},
    'PLAZA_LOURDES': {'latitud': 4.5970, 'longitud': -74.0800, 'descripcion': 'Plaza Lourdes', 'tipo': 'comercial'},
    'GRAN_ESTACION': {'latitud': 4.6280, 'longitud': -74.1050, 'descripcion': 'Gran Estación', 'tipo': 'comercial'},
    'USAQUEN': {'latitud': 4.6951, 'longitud': -74.0308, 'descripcion': 'Usaquén', 'tipo': 'residencial'},
    'SUBA': {'latitud': 4.7570, 'longitud': -74.0820, 'descripcion': 'Suba', 'tipo': 'residencial'},
    'ENGATIVA': {'latitud': 4.7180, 'longitud': -74.1070, 'descripcion': 'Engativá', 'tipo': 'residencial'},
    'KENNEDY': {'latitud': 4.6268, 'longitud': -74.1370, 'descripcion': 'Kennedy', 'tipo': 'residencial'},
    'BOSA': {'latitud': 4.6138, 'longitud': -74.1791, 'descripcion': 'Bosa', 'tipo': 'residencial'},
    'TINTAL': {'latitud': 4.6450, 'longitud': -74.1580, 'descripcion': 'Tintal', 'tipo': 'residencial'}
}

# Conexiones entre nodos (origen, destino, distancia_km, tiempo_min)
CONEXIONES = [
    ("UNIMINUTO_CALLE_80", "GRAN_ESTACION", 5.0, 12),
    ("GRAN_ESTACION", "CIUDAD_UNIVERSITARIA", 3.2, 9),
    ("CIUDAD_UNIVERSITARIA", "PLAZA_LOURDES", 2.0, 6),
    ("PLAZA_LOURDES", "ZONA_ROSA", 1.8, 5),
    ("ZONA_ROSA", "USAQUEN", 6.5, 15),
    ("USAQUEN", "SUBA", 4.5, 10),
    ("SUBA", "ENGATIVA", 3.0, 8),
    ("ENGATIVA", "UNIMINUTO_CALLE_80", 2.5, 7),
    ("UNIMINUTO_PERDOMO", "PERDOMO", 0.8, 3),
    ("PERDOMO", "KENNEDY", 5.2, 13),
    ("KENNEDY", "TINTAL", 3.1, 9),
    ("TINTAL", "BOSA", 2.9, 8),
    ("BOSA", "RESTREPO", 4.7, 12),
    ("RESTREPO", "CIUDAD_UNIVERSITARIA", 2.4, 7),
    ("CENTRO_BOGOTA", "PLAZA_LOURDES", 1.3, 4),
    ("CENTRO_BOGOTA", "RESTREPO", 2.2, 6)
]

# Colores para la interfaz
COLORES = {
    'RESET': '\033[0m',
    'BOLD': '\033[1m',
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'PURPLE': '\033[95m',
    'CYAN': '\033[96m',
    'WHITE': '\033[97m'
}

# ----------------------------- Funciones -----------------------------

def imprimir_encabezado(titulo: str):
    """Imprime un encabezado formateado"""
    print(f"\n{COLORES['CYAN']}{'=' * 80}{COLORES['RESET']}")
    print(f"{COLORES['BOLD']}{COLORES['BLUE']}{titulo.center(80)}{COLORES['RESET']}")
    print(f"{COLORES['CYAN']}{'=' * 80}{COLORES['RESET']}")

def imprimir_mensaje(tipo: str, mensaje: str):
    """Imprime un mensaje con formato y color"""
    colores_tipo = {
        'error': COLORES['RED'],
        'exito': COLORES['GREEN'],
        'advertencia': COLORES['YELLOW'],
        'info': COLORES['BLUE']
    }
    emoji = EMOJIS.get(tipo, '')
    color = colores_tipo.get(tipo, COLORES['WHITE'])
    print(f"{color}{emoji} {mensaje}{COLORES['RESET']}")

def formatear_tiempo(segundos: float) -> str:
    """Formatea tiempo en segundos a formato legible"""
    if segundos < 60:
        return f"{segundos:.1f}s"
    elif segundos < 3600:
        return f"{segundos/60:.1f}m"
    else:
        return f"{segundos/3600:.1f}h"

def formatear_distancia(km: float) -> str:
    """Formatea distancia en kilómetros a formato legible"""
    return f"{km:.2f}km" if km >= 1 else f"{km*1000:.0f}m"

def obtener_descripcion_nodo(codigo: str) -> str:
    """Obtiene la descripción de un nodo"""
    return COORDENADAS_BOGOTA.get(codigo, {}).get('descripcion', codigo)

def obtener_coordenadas_nodo(codigo: str) -> Tuple[Optional[float], Optional[float]]:
    """Obtiene las coordenadas (latitud, longitud) de un nodo"""
    nodo = COORDENADAS_BOGOTA.get(codigo, {})
    return (nodo.get('latitud'), nodo.get('longitud'))

def obtener_tipo_nodo(codigo: str) -> str:
    """Obtiene el tipo de un nodo"""
    return COORDENADAS_BOGOTA.get(codigo, {}).get('tipo', 'desconocido')

def validar_nodo(codigo: str) -> bool:
    """Valida si un nodo existe"""
    return codigo in COORDENADAS_BOGOTA

def listar_nodos_disponibles() -> Dict[str, str]:
    """Lista todos los nodos disponibles con sus descripciones"""
    return {codigo: info['descripcion'] for codigo, info in COORDENADAS_BOGOTA.items()}

def listar_nodos_por_tipo(tipo: str) -> Dict[str, str]:
    """Lista nodos de un tipo específico"""
    return {
        codigo: info['descripcion'] 
        for codigo, info in COORDENADAS_BOGOTA.items() 
        if info.get('tipo') == tipo
    }

def calcular_velocidad_promedio(distancia_km: float, tiempo_min: float) -> float:
    """Calcula velocidad promedio en km/h"""
    return (distancia_km / tiempo_min) * 60 if tiempo_min > 0 else 0

def calcular_distancia_euclidiana(nodo1: str, nodo2: str) -> float:
    """Calcula distancia euclidiana entre dos nodos"""
    coords1 = obtener_coordenadas_nodo(nodo1)
    coords2 = obtener_coordenadas_nodo(nodo2)
    
    if None in coords1 or None in coords2:
        return float('inf')
    
    lat1, lng1 = coords1
    lat2, lng2 = coords2
    
    return math.sqrt((lat2 - lat1)**2 + (lng2 - lng1)**2)

def calcular_distancia_haversine(nodo1: str, nodo2: str) -> float:
    """Calcula distancia real usando fórmula de Haversine (más precisa)"""
    coords1 = obtener_coordenadas_nodo(nodo1)
    coords2 = obtener_coordenadas_nodo(nodo2)
    
    if None in coords1 or None in coords2:
        return float('inf')
    
    lat1, lng1 = coords1
    lat2, lng2 = coords2
    
    # Convertir a radianes
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # Fórmula de Haversine
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radio de la Tierra en km
    
    return c * r

def obtener_timestamp() -> str:
    """Obtiene timestamp actual"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def limpiar_pantalla():
    """Limpia la pantalla de la consola"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar():
    """Pausa la ejecución hasta que el usuario presione Enter"""
    input(f"\n{EMOJIS['info']} Presiona Enter para continuar...")

def confirmar_accion(mensaje: str) -> bool:
    """Solicita confirmación del usuario"""
    respuesta = input(f"\n{EMOJIS['info']} {mensaje} (s/n): ").strip().lower()
    return respuesta in ['s', 'si', 'sí', 'y', 'yes']

def mostrar_progreso(iteracion: int, total: int, prefijo: str = "Progreso"):
    """Muestra barra de progreso"""
    porcentaje = (iteracion / total) * 100
    barra = '█' * int(porcentaje // 2) + '░' * (50 - int(porcentaje // 2))
    print(f"\r{prefijo}: |{barra}| {porcentaje:.1f}% ({iteracion}/{total})", end='')
    if iteracion == total:
        print()

def generar_nombre_archivo(prefijo: str, extension: str = "txt") -> str:
    """Genera nombre de archivo con timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefijo}_{timestamp}.{extension}"

def guardar_texto(contenido: str, nombre_archivo: str) -> bool:
    """Guarda texto en un archivo"""
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(contenido)
        return True
    except Exception as e:
        print(f"Error al guardar archivo: {e}")
        return False

def cargar_texto(nombre_archivo: str) -> str:
    """Carga texto desde un archivo"""
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error al cargar archivo: {e}")
        return ""

def validar_entrada_usuario(entrada: str, opciones_validas: list) -> bool:
    """Valida entrada del usuario contra opciones válidas"""
    return entrada.strip().upper() in [opcion.upper() for opcion in opciones_validas]

def menu_seleccion(opciones: Dict[str, str], titulo: str = "Selecciona una opción") -> str:
    """Muestra menú de selección y retorna opción elegida"""
    print(f"\n{EMOJIS['info']} {titulo}:")
    for clave, descripcion in opciones.items():
        print(f"  {clave}: {descripcion}")
    
    while True:
        seleccion = input("\n>>> ").strip().upper()
        if seleccion in opciones:
            return seleccion
        else:
            print(f"{EMOJIS['error']} Opción inválida. Intenta de nuevo.")

def formatear_ruta(ruta: list) -> str:
    """Formatea una ruta como string con emojis"""
    if not ruta:
        return "Sin ruta"
    
    ruta_formateada = []
    for i, nodo in enumerate(ruta):
        descripcion = obtener_descripcion_nodo(nodo)
        if i == 0:
            ruta_formateada.append(f"🏁 {descripcion}")
        elif i == len(ruta) - 1:
            ruta_formateada.append(f"🎯 {descripcion}")
        else:
            ruta_formateada.append(f"📍 {descripcion}")
    
    return "\n".join(ruta_formateada)

def mostrar_estadisticas_ruta(resultado):
    """Muestra estadísticas de una ruta encontrada"""
    if not resultado.exito:
        print(f"{EMOJIS['error']} {resultado.mensaje}")
        return
    
    print(f"\n{EMOJIS['exito']} Ruta encontrada:")
    print(f"  {EMOJIS['distancia']} Distancia: {formatear_distancia(resultado.distancia_total)}")
    print(f"  {EMOJIS['tiempo']} Tiempo estimado: {formatear_tiempo(resultado.tiempo_total * 60)}")
    print(f"  {EMOJIS['exploracion']} Nodos explorados: {resultado.nodos_explorados}")
    print(f"  {EMOJIS['ruta']} Pasos en la ruta: {len(resultado.ruta)}")
    
    if hasattr(resultado, 'detalles') and resultado.detalles:
        print(f"  {EMOJIS['info']} Algoritmo: {resultado.detalles.get('algoritmo', 'N/A')}")
        print(f"  {EMOJIS['info']} Criterio: {resultado.detalles.get('criterio', 'N/A')}")

def mostrar_bienvenida():
    """Muestra mensaje de bienvenida del sistema"""
    imprimir_encabezado("SISTEMA EXPERTO DE RUTAS EN BOGOTÁ")
    print(f"{EMOJIS['inicio']} ¡Bienvenido al Sistema Experto de Rutas!")
    print(f"{EMOJIS['info']} Universidad UNIMINUTO")
    print(f"{EMOJIS['info']} Versión {SISTEMA_INFO['version']}")
    print(f"{EMOJIS['mapa']} Encuentra la mejor ruta en Bogotá")

def validar_coordenadas_completas() -> bool:
    """Valida que todos los nodos tengan coordenadas completas"""
    nodos_invalidos = []
    
    for codigo, datos in COORDENADAS_BOGOTA.items():
        if 'latitud' not in datos or 'longitud' not in datos:
            nodos_invalidos.append(codigo)
        elif datos['latitud'] is None or datos['longitud'] is None:
            nodos_invalidos.append(codigo)
    
    if nodos_invalidos:
        print(f"{EMOJIS['error']} Nodos con coordenadas incompletas: {', '.join(nodos_invalidos)}")
        return False
    
    return True

def generar_resumen_nodos() -> str:
    """Genera resumen de todos los nodos del sistema"""
    tipos = {}
    for codigo, info in COORDENADAS_BOGOTA.items():
        tipo = info.get('tipo', 'desconocido')
        if tipo not in tipos:
            tipos[tipo] = []
        tipos[tipo].append(f"{codigo}: {info['descripcion']}")
    
    resumen = f"=== RESUMEN DE NODOS ({len(COORDENADAS_BOGOTA)} total) ===\n\n"
    
    for tipo, nodos in tipos.items():
        resumen += f"{tipo.upper()} ({len(nodos)}):\n"
        for nodo in nodos:
            resumen += f"  - {nodo}\n"
        resumen += "\n"
    
    return resumen