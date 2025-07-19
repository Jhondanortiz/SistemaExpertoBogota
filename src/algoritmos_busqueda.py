import heapq
import math
import json
import os
from typing import Dict, Tuple, List, Optional
from utils import EMOJIS


class ResultadoBusqueda:
    def __init__(self, exito, ruta=None, distancia_total=0, tiempo_total=0, nodos_explorados=0, mensaje='', detalles=None):
        self.exito = exito
        self.ruta = ruta or []
        self.distancia_total = distancia_total
        self.tiempo_total = tiempo_total
        self.nodos_explorados = nodos_explorados
        self.mensaje = mensaje
        self.detalles = detalles or {}


def cargar_coordenadas():
    """Carga las coordenadas desde el archivo JSON"""
    rutas_posibles = [
        "src/coordenadas_bogota.json",
        "coordenadas_bogota.json", 
        "data/coordenadas_bogota.json",
        os.path.join(os.path.dirname(__file__), "coordenadas_bogota.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "coordenadas_bogota.json")
    ]
    
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    coordenadas = json.load(f)
                print(f"✅ Coordenadas cargadas desde: {ruta}")
                return coordenadas
            except Exception as e:
                print(f"❌ Error cargando {ruta}: {e}")
                continue
    
    print("❌ No se pudo cargar el archivo de coordenadas")
    return {}


def reconstruir_camino(padres: Dict[str, str], inicio: str, fin: str) -> List[str]:
    """Reconstruye el camino desde el destino hasta el origen"""
    camino = [fin]
    while camino[-1] != inicio:
        if camino[-1] not in padres:
            break
        camino.append(padres[camino[-1]])
    camino.reverse()
    return camino


def calcular_heuristica(coord_actual: Tuple[float, float], coord_destino: Tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos coordenadas"""
    try:
        lat1, lon1 = coord_actual
        lat2, lon2 = coord_destino
        
        # Distancia euclidiana simple (para heurística)
        return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    except Exception as e:
        print(f"Error calculando heurística: {e}")
        return 0.0


def obtener_coordenadas_nodo(coordenadas: Dict, nodo: str) -> Tuple[float, float]:
    """Obtiene las coordenadas de un nodo desde el diccionario de coordenadas"""
    if nodo not in coordenadas:
        raise ValueError(f"Nodo '{nodo}' no encontrado en coordenadas")
    
    coord_data = coordenadas[nodo]
    
    # Manejar diferentes formatos de coordenadas
    if isinstance(coord_data, dict):
        if "latitud" in coord_data and "longitud" in coord_data:
            return float(coord_data["latitud"]), float(coord_data["longitud"])
        elif "lat" in coord_data and "lon" in coord_data:
            return float(coord_data["lat"]), float(coord_data["lon"])
        elif "latitude" in coord_data and "longitude" in coord_data:
            return float(coord_data["latitude"]), float(coord_data["longitude"])
    elif isinstance(coord_data, (list, tuple)) and len(coord_data) >= 2:
        return float(coord_data[0]), float(coord_data[1])
    
    raise ValueError(f"Formato de coordenadas inválido para {nodo}: {coord_data}")


def _dijkstra(grafo, coordenadas, inicio, fin, criterio="distancia") -> ResultadoBusqueda:
    """Implementación interna del algoritmo Dijkstra"""
    if inicio not in grafo or fin not in grafo:
        return ResultadoBusqueda(False, mensaje="Nodo de inicio o fin no válido.")

    cola = [(0, inicio)]
    visitados = set()
    costos = {inicio: 0}
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
            distancia_total = sum(
                next((d for (v, d, t) in grafo[u] if v == vtx), 0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            tiempo_total = sum(
                next((t for (v, d, t) in grafo[u] if v == vtx), 0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            return ResultadoBusqueda(
                True, ruta, distancia_total, tiempo_total, nodos_explorados, 
                "Ruta encontrada exitosamente.", 
                {"algoritmo": "Dijkstra", "criterio": criterio}
            )

        for vecino, distancia, tiempo in grafo.get(nodo_actual, []):
            nuevo_costo = costo_actual + (distancia if criterio == "distancia" else tiempo)
            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = nodo_actual
                heapq.heappush(cola, (nuevo_costo, vecino))

    return ResultadoBusqueda(False, mensaje="No se encontró una ruta al destino.")


def _a_estrella(grafo, coordenadas, inicio, fin, criterio="distancia") -> ResultadoBusqueda:
    """Implementación interna del algoritmo A* - COMPLETAMENTE CORREGIDA"""
    if inicio not in grafo or fin not in grafo:
        return ResultadoBusqueda(False, mensaje="Nodo de inicio o fin no válido.")
    
    # Verificar y obtener coordenadas
    try:
        coord_inicio = obtener_coordenadas_nodo(coordenadas, inicio)
        coord_fin = obtener_coordenadas_nodo(coordenadas, fin)
    except Exception as e:
        print(f"Error en a_estrella: {e}")
        return ResultadoBusqueda(False, mensaje=f"Error obteniendo coordenadas: {e}")

    cola = [(0, inicio)]
    visitados = set()
    costos = {inicio: 0}
    padres = {}
    nodos_explorados = 0

    while cola:
        _, nodo_actual = heapq.heappop(cola)
        if nodo_actual in visitados:
            continue
        visitados.add(nodo_actual)
        nodos_explorados += 1

        if nodo_actual == fin:
            ruta = reconstruir_camino(padres, inicio, fin)
            distancia_total = sum(
                next((d for (v, d, t) in grafo[u] if v == vtx), 0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            tiempo_total = sum(
                next((t for (v, d, t) in grafo[u] if v == vtx), 0)
                for u, vtx in zip(ruta[:-1], ruta[1:])
            )
            return ResultadoBusqueda(
                True, ruta, distancia_total, tiempo_total, nodos_explorados, 
                "Ruta encontrada exitosamente.", 
                {"algoritmo": "A*", "criterio": criterio}
            )

        for vecino, distancia, tiempo in grafo.get(nodo_actual, []):
            nuevo_costo = costos[nodo_actual] + (distancia if criterio == "distancia" else tiempo)
            
            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = nodo_actual
                
                try:
                    # Obtener coordenadas del vecino para calcular heurística
                    coord_vecino = obtener_coordenadas_nodo(coordenadas, vecino)
                    heuristica = calcular_heuristica(coord_vecino, coord_fin)
                    
                    # Ajustar heurística según el criterio
                    if criterio == "tiempo":
                        # Convertir distancia euclidiana a estimación de tiempo
                        # Factor aproximado: 1 unidad de distancia ≈ 2.5 minutos
                        heuristica *= 2.5
                    elif criterio == "distancia":
                        # Convertir a kilómetros aproximados (factor de escala)
                        heuristica *= 100  # Ajustar según la escala de tu sistema
                    
                    costo_estimado = nuevo_costo + heuristica
                    heapq.heappush(cola, (costo_estimado, vecino))
                    
                except Exception as e:
                    # Si hay error con heurística, usar como Dijkstra
                    print(f"Warning: Error calculando heurística para {vecino}: {e}")
                    heapq.heappush(cola, (nuevo_costo, vecino))

    return ResultadoBusqueda(False, mensaje="No se encontró una ruta al destino.")


class BuscadorRutas:
    """
    Clase principal para buscar rutas usando diferentes algoritmos
    Compatible con el sistema de validación
    """
    
    def __init__(self, grafo):
        """
        Inicializa el buscador con un grafo
        """
        self.grafo = grafo
        # Cargar coordenadas desde archivo JSON
        self.coordenadas = cargar_coordenadas()
        
        # Si no se pudieron cargar desde archivo, intentar desde el grafo
        if not self.coordenadas:
            self.coordenadas = self._obtener_coordenadas_desde_grafo()

    def _obtener_coordenadas_desde_grafo(self):
        """Extrae las coordenadas del grafo de Bogotá como fallback"""
        coordenadas = {}
        try:
            if hasattr(self.grafo, 'nodos'):
                for nombre, nodo in self.grafo.nodos.items():
                    if hasattr(nodo, 'latitud') and hasattr(nodo, 'longitud'):
                        coordenadas[nombre] = {
                            "latitud": nodo.latitud,
                            "longitud": nodo.longitud
                        }
            print(f"✅ Coordenadas extraídas del grafo: {len(coordenadas)} nodos")
        except Exception as e:
            print(f"❌ Error extrayendo coordenadas del grafo: {e}")
        
        return coordenadas

    def _convertir_grafo_a_dict(self):
        """
        Convierte el grafo de Bogotá al formato requerido por los algoritmos
        """
        try:
            if hasattr(self.grafo, 'grafo') and self.grafo.grafo:
                # Formato: Dict[str, List[Tuple[destino, distancia, tiempo]]]
                grafo_dict = {}
                for nodo, aristas in self.grafo.grafo.items():
                    grafo_dict[nodo] = []
                    for arista in aristas:
                        if hasattr(arista, 'destino') and hasattr(arista, 'distancia') and hasattr(arista, 'tiempo'):
                            grafo_dict[nodo].append((arista.destino, arista.distancia, arista.tiempo))
                return grafo_dict
            elif hasattr(self.grafo, 'adyacencia') and self.grafo.adyacencia:
                # Convertir formato alternativo del grafo
                grafo_dict = {}
                for nodo, vecinos in self.grafo.adyacencia.items():
                    grafo_dict[nodo] = []
                    for vecino, datos in vecinos.items():
                        distancia = datos.get('distancia', 0)
                        tiempo = datos.get('tiempo', 0)
                        grafo_dict[nodo].append((vecino, distancia, tiempo))
                return grafo_dict
            else:
                print("❌ Formato de grafo no reconocido")
                return {}
        except Exception as e:
            print(f"❌ Error convirtiendo grafo: {e}")
            return {}

    def dijkstra(self, origen: str, destino: str, criterio: str = 'distancia') -> 'ResultadoBusqueda':
        """
        Implementa el algoritmo de Dijkstra - MÉTODO REQUERIDO POR VALIDACIÓN
        """
        try:
            # Convertir el grafo al formato requerido
            grafo_dict = self._convertir_grafo_a_dict()
            
            if not grafo_dict:
                return ResultadoBusqueda(False, mensaje="Formato de grafo no compatible.")
            
            # Verificar que los nodos existen
            if origen not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' no encontrado.")
            if destino not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo destino '{destino}' no encontrado.")
            
            resultado = _dijkstra(grafo_dict, self.coordenadas, origen, destino, criterio)
            return resultado
                
        except Exception as e:
            print(f"Error en dijkstra: {e}")
            return ResultadoBusqueda(False, mensaje=f"Error en dijkstra: {e}")

    def a_estrella(self, origen: str, destino: str, criterio: str = 'distancia') -> 'ResultadoBusqueda':
        """
        Implementa el algoritmo A* - MÉTODO REQUERIDO POR VALIDACIÓN - COMPLETAMENTE CORREGIDO
        """
        try:
            # Convertir el grafo al formato requerido
            grafo_dict = self._convertir_grafo_a_dict()
            
            if not grafo_dict:
                return ResultadoBusqueda(False, mensaje="Formato de grafo no compatible.")
            
            # Verificar que los nodos existen
            if origen not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' no encontrado.")
            if destino not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo destino '{destino}' no encontrado.")
            
            # Verificar que las coordenadas están disponibles
            if not self.coordenadas:
                return ResultadoBusqueda(False, mensaje="No hay coordenadas disponibles para A*.")
            
            if origen not in self.coordenadas:
                return ResultadoBusqueda(False, mensaje=f"Coordenadas no encontradas para '{origen}'.")
            if destino not in self.coordenadas:
                return ResultadoBusqueda(False, mensaje=f"Coordenadas no encontradas para '{destino}'.")
            
            resultado = _a_estrella(grafo_dict, self.coordenadas, origen, destino, criterio)
            return resultado
                
        except Exception as e:
            print(f"Error en a_estrella: {e}")
            return ResultadoBusqueda(False, mensaje=f"Error en a_estrella: {e}")

    def buscar(self, inicio, fin, algoritmo="dijkstra", criterio="distancia", coordenadas=None):
        """
        Método de búsqueda genérico (mantiene compatibilidad con código existente)
        """
        if coordenadas:
            self.coordenadas = coordenadas
            
        # Usar el método de conversión centralizado
        grafo_dict = self._convertir_grafo_a_dict()
        
        if not grafo_dict:
            return ResultadoBusqueda(False, mensaje="Formato de grafo no compatible.")
        
        if algoritmo == "dijkstra":
            return _dijkstra(grafo_dict, self.coordenadas, inicio, fin, criterio)
        elif algoritmo == "a*" or algoritmo == "a_estrella":
            return _a_estrella(grafo_dict, self.coordenadas, inicio, fin, criterio)
        else:
            return ResultadoBusqueda(False, mensaje=f"Algoritmo '{algoritmo}' no reconocido.")

    def comparar_algoritmos(self, origen: str, destino: str, criterio: str = 'distancia') -> Dict:
        """
        Compara el rendimiento de ambos algoritmos
        """
        import time
        
        # Ejecutar Dijkstra
        inicio = time.time()
        resultado_dijkstra = self.dijkstra(origen, destino, criterio)
        tiempo_dijkstra = time.time() - inicio
        
        # Ejecutar A*
        inicio = time.time()
        resultado_a_estrella = self.a_estrella(origen, destino, criterio)
        tiempo_a_estrella = time.time() - inicio
        
        return {
            'dijkstra': {
                'resultado': resultado_dijkstra,
                'tiempo_ejecucion': tiempo_dijkstra * 1000  # milisegundos
            },
            'a_estrella': {
                'resultado': resultado_a_estrella,
                'tiempo_ejecucion': tiempo_a_estrella * 1000  # milisegundos
            }
        }

    def obtener_estadisticas(self):
        """Obtiene estadísticas del sistema de búsqueda"""
        grafo_dict = self._convertir_grafo_a_dict()
        
        return {
            'nodos_grafo': len(grafo_dict),
            'nodos_con_coordenadas': len(self.coordenadas),
            'coordenadas_disponibles': list(self.coordenadas.keys())[:10],  # Primeros 10
            'grafo_disponible': bool(grafo_dict)
        }