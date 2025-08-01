import heapq
import math
import json
import os
from typing import Dict, Tuple, List, Optional
from unidecode import unidecode

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
    """Carga las coordenadas desde el archivo JSON, normalizando nombres"""
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
                    data = json.load(f)
                coordenadas = {unidecode(key).upper(): value for key, value in data.items()}
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
    """Obtiene las coordenadas de un nodo desde el diccionario de coordenadas, normalizando el nombre"""
    nodo_normalizado = unidecode(nodo).upper()
    if nodo_normalizado not in coordenadas:
        raise ValueError(f"Nodo '{nodo}' (normalizado: '{nodo_normalizado}') no encontrado en coordenadas")
    
    coord_data = coordenadas[nodo_normalizado]
    
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
    inicio_normalizado = unidecode(inicio).upper()
    fin_normalizado = unidecode(fin).upper()
    
    if inicio_normalizado not in grafo or fin_normalizado not in grafo:
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no válido.")

    cola = [(0, inicio_normalizado)]
    visitados = set()
    costos = {inicio_normalizado: 0}
    padres = {}
    nodos_explorados = 0

    while cola:
        costo_actual, nodo_actual = heapq.heappop(cola)
        if nodo_actual in visitados:
            continue
        visitados.add(nodo_actual)
        nodos_explorados += 1

        if nodo_actual == fin_normalizado:
            ruta = reconstruir_camino(padres, inicio_normalizado, fin_normalizado)
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

    return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")

def _a_estrella(grafo, coordenadas, inicio, fin, criterio="distancia") -> ResultadoBusqueda:
    """Implementación interna del algoritmo A*, con normalización de nombres"""
    inicio_normalizado = unidecode(inicio).upper()
    fin_normalizado = unidecode(fin).upper()
    
    if inicio_normalizado not in grafo or fin_normalizado not in grafo:
        return ResultadoBusqueda(False, mensaje=f"Nodo de inicio '{inicio}' o fin '{fin}' no válido.")
    
    # Verificar y obtener coordenadas
    try:
        coord_inicio = obtener_coordenadas_nodo(coordenadas, inicio_normalizado)
        coord_fin = obtener_coordenadas_nodo(coordenadas, fin_normalizado)
    except Exception as e:
        print(f"Error en a_estrella: {e}")
        return ResultadoBusqueda(False, mensaje=f"Error obteniendo coordenadas: {e}")

    cola = [(0, inicio_normalizado)]
    visitados = set()
    costos = {inicio_normalizado: 0}
    padres = {}
    nodos_explorados = 0

    while cola:
        _, nodo_actual = heapq.heappop(cola)
        if nodo_actual in visitados:
            continue
        visitados.add(nodo_actual)
        nodos_explorados += 1

        if nodo_actual == fin_normalizado:
            ruta = reconstruir_camino(padres, inicio_normalizado, fin_normalizado)
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
                        heuristica *= 2.5  # Convertir distancia a tiempo aproximado
                    elif criterio == "distancia":
                        heuristica *= 100  # Convertir a kilómetros aproximados
                    
                    costo_estimado = nuevo_costo + heuristica
                    heapq.heappush(cola, (costo_estimado, vecino))
                    
                except Exception as e:
                    # Si hay error con heurística, usar como Dijkstra
                    print(f"Warning: Error calculando heurística para {vecino}: {e}")
                    heapq.heappush(cola, (nuevo_costo, vecino))

    return ResultadoBusqueda(False, mensaje=f"No se encontró una ruta de {inicio} a {fin}.")

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
        """Extrae las coordenadas del grafo de Bogotá como fallback, normalizando nombres"""
        coordenadas = {}
        try:
            if hasattr(self.grafo, 'nodos'):
                for nombre, nodo in self.grafo.nodos.items():
                    nombre_normalizado = unidecode(nombre).upper()
                    if hasattr(nodo, 'latitud') and hasattr(nodo, 'longitud'):
                        coordenadas[nombre_normalizado] = {
                            "latitud": nodo.latitud,
                            "longitud": nodo.longitud
                        }
            print(f"✅ Coordenadas extraídas del grafo: {len(coordenadas)} nodos")
        except Exception as e:
            print(f"❌ Error extrayendo coordenadas del grafo: {e}")
        
        return coordenadas

    def _convertir_grafo_a_dict(self):
        """
        Convierte el grafo de Bogotá al formato requerido por los algoritmos, normalizando nombres
        """
        try:
            if hasattr(self.grafo, 'grafo') and self.grafo.grafo:
                # Formato: Dict[str, List[Tuple[destino, distancia, tiempo]]]
                grafo_dict = {}
                for nodo, aristas in self.grafo.grafo.items():
                    nodo_normalizado = unidecode(nodo).upper()
                    grafo_dict[nodo_normalizado] = []
                    for arista in aristas:
                        if hasattr(arista, 'destino') and hasattr(arista, 'distancia') and hasattr(arista, 'tiempo'):
                            destino_normalizado = unidecode(arista.destino).upper()
                            grafo_dict[nodo_normalizado].append((destino_normalizado, arista.distancia, arista.tiempo))
                return grafo_dict
            elif hasattr(self.grafo, 'adyacencia') and self.grafo.adyacencia:
                # Convertir formato alternativo del grafo
                grafo_dict = {}
                for nodo, vecinos in self.grafo.adyacencia.items():
                    nodo_normalizado = unidecode(nodo).upper()
                    grafo_dict[nodo_normalizado] = []
                    for vecino, datos in vecinos.items():
                        vecino_normalizado = unidecode(vecino).upper()
                        distancia = datos.get('distancia', 0)
                        tiempo = datos.get('tiempo', 0)
                        grafo_dict[nodo_normalizado].append((vecino_normalizado, distancia, tiempo))
                return grafo_dict
            else:
                print("❌ Formato de grafo no reconocido")
                return {}
        except Exception as e:
            print(f"❌ Error convirtiendo grafo: {e}")
            return {}

    def dijkstra(self, origen: str, destino: str, criterio: str = 'distancia') -> ResultadoBusqueda:
        """
        Implementa el algoritmo de Dijkstra - MÉTODO REQUERIDO POR VALIDACIÓN
        """
        try:
            # Convertir el grafo al formato requerido
            grafo_dict = self._convertir_grafo_a_dict()
            
            if not grafo_dict:
                return ResultadoBusqueda(False, mensaje="Formato de grafo no compatible.")
            
            # Normalizar nombres de nodos
            origen_normalizado = unidecode(origen).upper()
            destino_normalizado = unidecode(destino).upper()
            
            # Verificar que los nodos existen
            if origen_normalizado not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' no encontrado.")
            if destino_normalizado not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo destino '{destino}' no encontrado.")
            
            resultado = _dijkstra(grafo_dict, self.coordenadas, origen_normalizado, destino_normalizado, criterio)
            return resultado
                
        except Exception as e:
            print(f"Error en dijkstra: {e}")
            return ResultadoBusqueda(False, mensaje=f"Error en dijkstra: {e}")

    def a_estrella(self, origen: str, destino: str, criterio: str = 'distancia') -> ResultadoBusqueda:
        """
        Implementa el algoritmo A* - MÉTODO REQUERIDO POR VALIDACIÓN
        """
        try:
            # Convertir el grafo al formato requerido
            grafo_dict = self._convertir_grafo_a_dict()
            
            if not grafo_dict:
                return ResultadoBusqueda(False, mensaje="Formato de grafo no compatible.")
            
            # Normalizar nombres de nodos
            origen_normalizado = unidecode(origen).upper()
            destino_normalizado = unidecode(destino).upper()
            
            print("Nodos en coordenadas:", list(self.coordenadas.keys()))
            print("Nodos en grafo:", list(grafo_dict.keys()))
            
            # Verificar que los nodos existen
            if origen_normalizado not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo origen '{origen}' no encontrado.")
            if destino_normalizado not in grafo_dict:
                return ResultadoBusqueda(False, mensaje=f"Nodo destino '{destino}' no encontrado.")
            
            # Verificar que las coordenadas están disponibles
            if not self.coordenadas:
                return ResultadoBusqueda(False, mensaje="No hay coordenadas disponibles para A*.")
            
            if origen_normalizado not in self.coordenadas:
                return ResultadoBusqueda(False, mensaje=f"Coordenadas no encontradas para '{origen}'.")
            if destino_normalizado not in self.coordenadas:
                return ResultadoBusqueda(False, mensaje=f"Coordenadas no encontradas para '{destino}'.")
            
            resultado = _a_estrella(grafo_dict, self.coordenadas, origen_normalizado, destino_normalizado, criterio)
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
        
        # Normalizar nombres
        inicio_normalizado = unidecode(inicio).upper()
        fin_normalizado = unidecode(fin).upper()
        
        if algoritmo == "dijkstra":
            return _dijkstra(grafo_dict, self.coordenadas, inicio_normalizado, fin_normalizado, criterio)
        elif algoritmo == "a*" or algoritmo == "a_estrella":
            return _a_estrella(grafo_dict, self.coordenadas, inicio_normalizado, fin_normalizado, criterio)
        else:
            return ResultadoBusqueda(False, mensaje=f"Algoritmo '{algoritmo}' no reconocido.")

    def comparar_algoritmos(self, origen: str, destino: str, criterio: str = 'distancia') -> Dict:
        """
        Compara el rendimiento de ambos algoritmos
        """
        import time
        
        # Normalizar nombres
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        # Ejecutar Dijkstra
        inicio = time.time()
        resultado_dijkstra = self.dijkstra(origen_normalizado, destino_normalizado, criterio)
        tiempo_dijkstra = time.time() - inicio
        
        # Ejecutar A*
        inicio = time.time()
        resultado_a_estrella = self.a_estrella(origen_normalizado, destino_normalizado, criterio)
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