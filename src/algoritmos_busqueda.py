import heapq
import math
from typing import Dict, List, Optional
import logging
from utils import EMOJIS, obtener_coordenadas_nodo, validar_nodo, formatear_ruta

# Configuración del logger
logger = logging.getLogger(__name__)

class ResultadoBusqueda:
    def __init__(self, exito: bool, ruta: Optional[List[str]] = None, distancia_total: float = 0.0, 
                 tiempo_total: float = 0.0, nodos_explorados: int = 0, mensaje: str = '', detalles: Optional[Dict] = None):
        self.exito = exito
        self.ruta = ruta or []
        self.distancia_total = distancia_total
        self.tiempo_total = tiempo_total
        self.nodos_explorados = nodos_explorados
        self.mensaje = mensaje
        self.detalles = detalles or {}

def reconstruir_camino(padres: Dict[str, str], inicio: str, fin: str) -> List[str]:
    """Reconstruye el camino desde el destino hasta el origen."""
    camino = [fin]
    while camino[-1] != inicio and camino[-1] in padres:
        camino.append(padres[camino[-1]])
    camino.reverse()
    if camino[0] != inicio:
        logger.warning(f"{EMOJIS['advertencia']} No se pudo reconstruir el camino desde {inicio} a {fin}")
        return []
    logger.info(f"{EMOJIS['exito']} Camino reconstruido: {formatear_ruta(camino)}")
    return camino

def calcular_heuristica(coord_actual: Dict[str, float], coord_destino: Dict[str, float]) -> float:
    """Calcula la distancia euclidiana entre dos coordenadas en kilómetros."""
    try:
        lat1, lon1 = coord_actual['latitud'], coord_actual['longitud']
        lat2, lon2 = coord_destino['latitud'], coord_destino['longitud']
        km_por_grado = 111.139  # Alineado con grafo_bogota.py
        distancia = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * km_por_grado
        logger.debug(f"{EMOJIS['info']} Heurística calculada: {distancia:.2f} km entre {lat1},{lon1} y {lat2},{lon2}")
        return distancia
    except (TypeError, ValueError, KeyError) as e:
        logger.error(f"{EMOJIS['error']} Error calculando heurística: {e}")
        return float('inf')

def _dijkstra(grafo: Dict, coordenadas: Dict, inicio: str, fin: str, criterio: str = "distancia") -> ResultadoBusqueda:
    """Implementación interna del algoritmo Dijkstra."""
    if criterio not in ["distancia", "tiempo"]:
        logger.error(f"{EMOJIS['error']} Criterio inválido: {criterio}. Use 'distancia' o 'tiempo'")
        return ResultadoBusqueda(False, mensaje=f"Criterio '{criterio}' no válido. Use 'distancia' o 'tiempo'.")

    if not (validar_nodo(inicio) and validar_nodo(fin)):
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no válido.")

    if inicio not in grafo or fin not in grafo:
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no encontrado en el grafo.")

    cola = [(0, inicio)]
    visitados = set()
    costos = {nodo: float('inf') for nodo in grafo}
    costos[inicio] = 0
    padres = {}
    nodos_explorados = 0

    while cola:
        costo_actual, nodo_actual = heapq.heappop(cola)
        if nodo_actual in visitados:
            continue
        visitados.add(nodo_actual)
        nodos_explorados += 1

        if nodo_actual == fin:
            ruta = reconstruir_camino(padres, inicio, fin)
            if not ruta:
                return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")
            distancia_total = sum(
                next((d for (v, d, t) in grafo[u] if v == vtx), 0.0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            tiempo_total = sum(
                next((t for (v, d, t) in grafo[u] if v == vtx), 0.0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            logger.info(f"{EMOJIS['ruta']} Ruta calculada por Dijkstra: {formatear_ruta(ruta)}, "
                        f"Distancia: {distancia_total:.2f} km, Tiempo: {tiempo_total:.2f} min")
            return ResultadoBusqueda(
                True, ruta, distancia_total, tiempo_total, nodos_explorados,
                "Ruta encontrada exitosamente.",
                {"algoritmo": "Dijkstra", "criterio": criterio}
            )

        for vecino, distancia, tiempo in grafo.get(nodo_actual, []):
            nuevo_costo = costo_actual + (distancia if criterio == "distancia" else tiempo)
            if nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = nodo_actual
                heapq.heappush(cola, (nuevo_costo, vecino))

    return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")

def _a_estrella(grafo: Dict, coordenadas: Dict, inicio: str, fin: str, criterio: str = "distancia") -> ResultadoBusqueda:
    """Implementación interna del algoritmo A*."""
    if criterio not in ["distancia", "tiempo"]:
        logger.error(f"{EMOJIS['error']} Criterio inválido: {criterio}. Use 'distancia' o 'tiempo'")
        return ResultadoBusqueda(False, mensaje=f"Criterio '{criterio}' no válido. Use 'distancia' o 'tiempo'.")

    if not (validar_nodo(inicio) and validar_nodo(fin)):
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no válido.")

    if inicio not in grafo or fin not in grafo:
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no encontrado en el grafo.")

    coord_inicio = obtener_coordenadas_nodo(inicio)
    coord_fin = obtener_coordenadas_nodo(fin)
    if not coord_inicio or not coord_fin:
        logger.warning(f"{EMOJIS['advertencia']} Coordenadas no encontradas para {inicio} o {fin}, ejecutando Dijkstra como fallback")
        return _dijkstra(grafo, coordenadas, inicio, fin, criterio)

    cola = [(0, inicio)]
    visitados = set()
    costos = {nodo: float('inf') for nodo in grafo}
    costos[inicio] = 0
    padres = {}
    nodos_explorados = 0

    while cola:
        costo_estimado, nodo_actual = heapq.heappop(cola)
        if nodo_actual in visitados:
            continue
        visitados.add(nodo_actual)
        nodos_explorados += 1

        if nodo_actual == fin:
            ruta = reconstruir_camino(padres, inicio, fin)
            if not ruta:
                return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")
            distancia_total = sum(
                next((d for (v, d, t) in grafo[u] if v == vtx), 0.0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            tiempo_total = sum(
                next((t for (v, d, t) in grafo[u] if v == vtx), 0.0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            logger.info(f"{EMOJIS['ruta']} Ruta calculada por A*: {formatear_ruta(ruta)}, "
                        f"Distancia: {distancia_total:.2f} km, Tiempo: {tiempo_total:.2f} min")
            return ResultadoBusqueda(
                True, ruta, distancia_total, tiempo_total, nodos_explorados,
                "Ruta encontrada exitosamente.",
                {"algoritmo": "A*", "criterio": criterio}
            )

        for vecino, distancia, tiempo in grafo.get(nodo_actual, []):
            nuevo_costo = costos[nodo_actual] + (distancia if criterio == "distancia" else tiempo)
            if nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = nodo_actual
                coord_vecino = obtener_coordenadas_nodo(vecino)
                if not coord_vecino:
                    logger.warning(f"{EMOJIS['advertencia']} Sin coordenadas para {vecino}, usando costo sin heurística")
                    costo_estimado = nuevo_costo
                else:
                    heuristica = calcular_heuristica(coord_vecino, coord_fin)
                    if criterio == "tiempo":
                        heuristica *= 2.5 / 111.139  # Aproximación: km a minutos (ajustado para tiempos realistas)
                    costo_estimado = nuevo_costo + heuristica
                heapq.heappush(cola, (costo_estimado, vecino))

    return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")

class BuscadorRutas:
    """Clase principal para buscar rutas usando diferentes algoritmos."""
    
    def __init__(self, grafo, coordenadas: Dict):
        """Inicializa el buscador con un grafo y coordenadas pre-cargadas."""
        self.grafo = grafo
        self.coordenadas = coordenadas
        self.grafo_dict = self._convertir_grafo_a_dict()

    def _convertir_grafo_a_dict(self) -> Dict:
        """Convierte el grafo de Bogotá al formato requerido por los algoritmos."""
        try:
            grafo_dict = {}
            for nodo, aristas in self.grafo.grafo.items():
                if not validar_nodo(nodo):
                    logger.warning(f"{EMOJIS['advertencia']} Nodo inválido en grafo: {nodo}")
                    continue
                grafo_dict[nodo] = [
                    (arista.destino, arista.distancia, arista.tiempo)
                    for arista in aristas if validar_nodo(arista.destino)
                ]
            logger.info(f"{EMOJIS['exito']} Grafo convertido con {len(grafo_dict)} nodos")
            return grafo_dict
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error convirtiendo grafo: {e}")
            return {}

    def dijkstra(self, origen: str, destino: str, criterio: str = 'distancia') -> ResultadoBusqueda:
        """Implementa el algoritmo de Dijkstra."""
        if not (validar_nodo(origen) and validar_nodo(destino)):
            return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' o destino '{destino}' no válido.")
        try:
            resultado = _dijkstra(self.grafo_dict, self.coordenadas, origen, destino, criterio)
            if resultado.exito:
                logger.info(f"{EMOJIS['exito']} Ruta Dijkstra: {formatear_ruta(resultado.ruta)}, "
                            f"Distancia: {resultado.distancia_total:.2f} km, Tiempo: {resultado.tiempo_total:.2f} min")
            return resultado
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error en Dijkstra: {e}")
            return ResultadoBusqueda(False, mensaje=f"Error en Dijkstra: {e}")

    def a_estrella(self, origen: str, destino: str, criterio: str = 'distancia') -> ResultadoBusqueda:
        """Implementa el algoritmo A*."""
        if not (validar_nodo(origen) and validar_nodo(destino)):
            return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' o destino '{destino}' no válido.")
        try:
            if not self.coordenadas:
                logger.warning(f"{EMOJIS['advertencia']} No hay coordenadas disponibles, ejecutando Dijkstra como fallback")
                return self.dijkstra(origen, destino, criterio)
            resultado = _a_estrella(self.grafo_dict, self.coordenadas, origen, destino, criterio)
            if resultado.exito:
                logger.info(f"{EMOJIS['exito']} Ruta A*: {formatear_ruta(resultado.ruta)}, "
                            f"Distancia: {resultado.distancia_total:.2f} km, Tiempo: {resultado.tiempo_total:.2f} min")
            return resultado
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error en A*: {e}")
            return ResultadoBusqueda(False, mensaje=f"Error en A*: {e}")

    def buscar(self, inicio: str, fin: str, algoritmo: str = "dijkstra", criterio: str = "distancia", coordenadas: Optional[Dict] = None) -> ResultadoBusqueda:
        """Método de búsqueda genérico."""
        if coordenadas is not None:
            self.coordenadas = coordenadas
            self.grafo_dict = self._convertir_grafo_a_dict()
        
        algoritmo = algoritmo.lower()
        if algoritmo == "dijkstra":
            return self.dijkstra(inicio, fin, criterio)
        elif algoritmo in ["a*", "a_estrella"]:
            return self.a_estrella(inicio, fin, criterio)
        logger.error(f"{EMOJIS['error']} Algoritmo '{algoritmo}' no reconocido")
        return ResultadoBusqueda(False, mensaje=f"Algoritmo '{algoritmo}' no reconocido.")

    def comparar_algoritmos(self, origen: str, destino: str, criterio: str = 'distancia') -> Dict:
        """Compara el rendimiento de ambos algoritmos."""
        import time
        
        inicio = time.time()
        resultado_dijkstra = self.dijkstra(origen, destino, criterio)
        tiempo_dijkstra = time.time() - inicio
        
        inicio = time.time()
        resultado_a_estrella = self.a_estrella(origen, destino, criterio)
        tiempo_a_estrella = time.time() - inicio
        
        logger.info(f"{EMOJIS['info']} Comparación: Dijkstra ({tiempo_dijkstra*1000:.2f} ms), A* ({tiempo_a_estrella*1000:.2f} ms)")
        return {
            'dijkstra': {
                'resultado': resultado_dijkstra,
                'tiempo_ejecucion': tiempo_dijkstra * 1000
            },
            'a_estrella': {
                'resultado': resultado_a_estrella,
                'tiempo_ejecucion': tiempo_a_estrella * 1000
            }
        }

    def obtener_estadisticas(self) -> Dict:
        """Obtiene estadísticas del sistema de búsqueda."""
        estadisticas = {
            'nodos_grafo': len(self.grafo_dict),
            'nodos_con_coordenadas': len(self.coordenadas),
            'coordenadas_disponibles': list(self.coordenadas.keys())[:10],
            'grafo_disponible': bool(self.grafo_dict)
        }
        logger.info(f"{EMOJIS['info']} Estadísticas: {estadisticas}")
        return estadisticas
