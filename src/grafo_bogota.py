import json
import math
from typing import Dict, List, Optional, Tuple
import networkx as nx
import logging
from utils import COORDENADAS_BOGOTA, CONEXIONES, EMOJIS, validar_nodo

# Configurar logging
logger = logging.getLogger(__name__)

class Nodo:
    def __init__(self, nombre: str, latitud: float, longitud: float, descripcion: Optional[str] = None):
        self.nombre = nombre
        self.latitud = latitud
        self.longitud = longitud
        self.descripcion = descripcion

    def __str__(self):
        return f"{self.nombre} ({self.latitud}, {self.longitud})"

    def __repr__(self):
        return self.__str__()

class Arista:
    def __init__(self, destino: str, distancia: float, tiempo: float):
        self.destino = destino
        self.distancia = distancia
        self.tiempo = tiempo

    def __str__(self):
        return f"{self.destino} (Distancia: {self.distancia:.1f} km, Tiempo: {self.tiempo:.1f} min)"

class GrafoBogota:
    def __init__(self):
        self.nodos: Dict[str, Nodo] = {}
        self.grafo: Dict[str, List[Arista]] = {}
        self.nx_grafo = nx.Graph()

    def agregar_nodo(self, nodo: Nodo):
        if not validar_nodo(nodo.nombre):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido ignorado: {nodo.nombre}")
            return
        if nodo.nombre not in self.nodos:
            self.nodos[nodo.nombre] = nodo
            self.grafo[nodo.nombre] = []
            self.nx_grafo.add_node(nodo.nombre, pos=(nodo.latitud, nodo.longitud))
            logger.info(f"{EMOJIS['exito']} Nodo agregado: {nodo}")
        else:
            logger.warning(f"{EMOJIS['advertencia']} Nodo duplicado ignorado: {nodo.nombre}")

    def agregar_nodo_por_datos(self, nombre: str, latitud: float, longitud: float, descripcion: Optional[str] = None):
        if not validar_nodo(nombre):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido ignorado: {nombre}")
            return
        self.agregar_nodo(Nodo(nombre, latitud, longitud, descripcion))

    def agregar_arista(self, origen: str, destino: str, distancia: float, tiempo: float):
        if not (validar_nodo(origen) and validar_nodo(destino)):
            logger.error(f"{EMOJIS['error']} No se pudo agregar arista: {origen} -> {destino} (nodo inválido)")
            return
        if origen not in self.grafo:
            self.grafo[origen] = []
        if destino not in self.grafo:
            self.grafo[destino] = []
        if origen in self.nodos and destino in self.nodos:
            arista = Arista(destino, distancia, tiempo)
            self.grafo[origen].append(arista)
            self.nx_grafo.add_edge(origen, destino, weight=distancia, time=tiempo)
            logger.info(f"{EMOJIS['exito']} Arista agregada: {origen} -> {arista}")
        else:
            logger.error(f"{EMOJIS['error']} No se pudo agregar arista: {origen} -> {destino} (nodo no encontrado)")

    def agregar_arista_bidireccional(self, nodo1: str, nodo2: str, distancia: float, tiempo: float):
        self.agregar_arista(nodo1, nodo2, distancia, tiempo)
        self.agregar_arista(nodo2, nodo1, distancia, tiempo)

    def obtener_vecinos(self, nodo: str) -> List[Arista]:
        if not validar_nodo(nodo):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido: {nodo}")
            return []
        return self.grafo.get(nodo, [])

    def calcular_distancia_euclidiana(self, nodo1: str, nodo2: str) -> float:
        if not (validar_nodo(nodo1) and validar_nodo(nodo2)):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido: {nodo1} o {nodo2}")
            return float('inf')
        if nodo1 not in self.nodos or nodo2 not in self.nodos:
            logger.warning(f"{EMOJIS['advertencia']} Nodo no encontrado: {nodo1} o {nodo2}")
            return float('inf')
        n1 = self.nodos[nodo1]
        n2 = self.nodos[nodo2]
        lat_diff = n2.latitud - n1.latitud
        lon_diff = n2.longitud - n1.longitud
        # Factor de conversión ajustado para Bogotá (aproximadamente a 2600 m de altitud)
        km_por_grado = 111.139  # Más preciso para latitudes cercanas al ecuador
        return math.sqrt(lat_diff ** 2 + lon_diff ** 2) * km_por_grado

    def obtener_costo_arista(self, origen: str, destino: str, criterio: str = 'distancia') -> float:
        if not (validar_nodo(origen) and validar_nodo(destino)):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido: {origen} o {destino}")
            raise ValueError(f"Nodo inválido: {origen} o {destino}")
        for arista in self.obtener_vecinos(origen):
            if arista.destino == destino:
                return arista.distancia if criterio == 'distancia' else arista.tiempo
        raise ValueError(f"No existe arista de {origen} a {destino}")

    def adyacencias(self, nodo: str) -> List[str]:
        if not validar_nodo(nodo):
            logger.warning(f"{EMOJIS['advertencia']} Nodo inválido: {nodo}")
            return []
        return [arista.destino for arista in self.obtener_vecinos(nodo)]

    def obtener_peso(self, origen: str, destino: str) -> float:
        return self.obtener_costo_arista(origen, destino, criterio='distancia')

    def mostrar_grafo(self):
        logger.info(f"\n{EMOJIS['mapa']} GRAFO DE BOGOTÁ")
        logger.info(f"Nodos totales: {len(self.nodos)}")
        logger.info(f"Aristas totales: {self.nx_grafo.number_of_edges()}")
        for nodo in self.nodos.values():
            logger.info(f"\n{nodo}")
            vecinos = self.obtener_vecinos(nodo.nombre)
            if vecinos:
                for arista in vecinos:
                    logger.info(f"  {arista}")
            else:
                logger.info("  (Sin conexiones)")

def cargar_coordenadas(archivo_json: str = "src/coordenadas_bogota.json") -> Dict[str, Dict[str, float]]:
    """Carga las coordenadas de los nodos desde un archivo JSON, alineado con COORDENADAS_BOGOTA."""
    try:
        with open(archivo_json, 'r', encoding='utf-8') as file:
            coordenadas = json.load(file)
        resultado = {}
        for nombre, coords in coordenadas.items():
            if not validar_nodo(nombre):
                logger.warning(f"{EMOJIS['advertencia']} Nodo inválido en {archivo_json}: {nombre}")
                continue
            if isinstance(coords, dict) and 'latitud' in coords and 'longitud' in coords:
                lat, lon = coords['latitud'], coords['longitud']
            elif isinstance(coords, (list, tuple)) and len(coords) == 2:
                lat, lon = coords
            else:
                logger.warning(f"{EMOJIS['advertencia']} Formato inválido para el nodo {nombre} en {archivo_json}: {coords}")
                lat, lon = COORDENADAS_BOGOTA.get(nombre, {}).get('latitud'), COORDENADAS_BOGOTA.get(nombre, {}).get('longitud')
            if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                logger.warning(f"{EMOJIS['advertencia']} Coordenadas inválidas para {nombre}: {coords}. Usando respaldo.")
                lat, lon = COORDENADAS_BOGOTA.get(nombre, {}).get('latitud'), COORDENADAS_BOGOTA.get(nombre, {}).get('longitud')
            if lat is not None and lon is not None:
                resultado[nombre] = {'latitud': float(lat), 'longitud': float(lon)}
        # Asegurar que todos los nodos de COORDENADAS_BOGOTA estén presentes
        for nombre, coords in COORDENADAS_BOGOTA.items():
            if nombre not in resultado and validar_nodo(nombre):
                resultado[nombre] = {'latitud': float(coords['latitud']), 'longitud': float(coords['longitud'])}
        return resultado
    except FileNotFoundError:
        logger.error(f"{EMOJIS['error']} El archivo {archivo_json} no se encuentra. Usando COORDENADAS_BOGOTA.")
        return {k: {'latitud': float(v['latitud']), 'longitud': float(v['longitud'])} for k, v in COORDENADAS_BOGOTA.items() if validar_nodo(k)}
    except json.JSONDecodeError:
        logger.error(f"{EMOJIS['error']} El archivo {archivo_json} no tiene un formato JSON válido. Usando COORDENADAS_BOGOTA.")
        return {k: {'latitud': float(v['latitud']), 'longitud': float(v['longitud'])} for k, v in COORDENADAS_BOGOTA.items() if validar_nodo(k)}
    except Exception as e:
        logger.error(f"{EMOJIS['error']} Error al cargar coordenadas: {e}. Usando COORDENADAS_BOGOTA.")
        return {k: {'latitud': float(v['latitud']), 'longitud': float(v['longitud'])} for k, v in COORDENADAS_BOGOTA.items() if validar_nodo(k)}

def crear_mapa_bogota() -> GrafoBogota:
    grafo = GrafoBogota()
    coordenadas = cargar_coordenadas()

    if not coordenadas:
        logger.warning(f"{EMOJIS['advertencia']} No se cargaron coordenadas. Usando COORDENADAS_BOGOTA.")
        coordenadas = {k: {'latitud': float(v['latitud']), 'longitud': float(v['longitud'])} for k, v in COORDENADAS_BOGOTA.items() if validar_nodo(k)}

    logger.info(f"\n{EMOJIS['mapa']} AGREGANDO NODOS AL GRAFO...")
    for nombre, coords in coordenadas.items():
        grafo.agregar_nodo_por_datos(nombre, coords['latitud'], coords['longitud'])

    logger.info(f"\n{EMOJIS['mapa']} CREANDO CONEXIONES ENTRE PUNTOS...")
    for origen, destino, dist, tiempo in CONEXIONES:
        if validar_nodo(origen) and validar_nodo(destino):
            grafo.agregar_arista_bidireccional(origen, destino, dist, tiempo)
        else:
            logger.warning(f"{EMOJIS['advertencia']} Conexión inválida ignorada: {origen} -> {destino}")

    return grafo

if __name__ == "__main__":
    grafo = crear_mapa_bogota()
    grafo.mostrar_grafo()
