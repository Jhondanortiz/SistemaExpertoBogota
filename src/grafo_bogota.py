import math
from typing import Dict, List, Optional


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
        return f"{self.destino} (Distancia: {self.distancia} km, Tiempo: {self.tiempo} min)"


class GrafoBogota:
    def __init__(self):
        self.nodos: Dict[str, Nodo] = {}
        self.grafo: Dict[str, List[Arista]] = {}

    def agregar_nodo(self, nodo: Nodo):
        if nodo.nombre not in self.nodos:
            self.nodos[nodo.nombre] = nodo
            self.grafo[nodo.nombre] = []
            print(f"Nodo agregado: {nodo}")
        else:
            print(f"⚠️ Nodo duplicado ignorado: {nodo.nombre}")

    def agregar_nodo_por_datos(self, nombre: str, latitud: float, longitud: float, descripcion: Optional[str] = None):
        self.agregar_nodo(Nodo(nombre, latitud, longitud, descripcion))

    def agregar_arista(self, origen: str, destino: str, distancia: float, tiempo: float):
        if origen not in self.grafo:
            self.grafo[origen] = []
        if destino not in self.grafo:
            self.grafo[destino] = []

        if origen in self.nodos and destino in self.nodos:
            arista = Arista(destino, distancia, tiempo)
            self.grafo[origen].append(arista)
            print(f"Arista agregada: {origen} -> {arista}")
        else:
            print(f"❌ No se pudo agregar arista: {origen} -> {destino} (nodo no encontrado)")

    def agregar_arista_bidireccional(self, nodo1: str, nodo2: str, distancia: float, tiempo: float):
        self.agregar_arista(nodo1, nodo2, distancia, tiempo)
        self.agregar_arista(nodo2, nodo1, distancia, tiempo)

    def obtener_vecinos(self, nodo: str) -> List[Arista]:
        return self.grafo.get(nodo, [])

    def calcular_distancia_euclidiana(self, nodo1: str, nodo2: str) -> float:
        if nodo1 not in self.nodos or nodo2 not in self.nodos:
            return float('inf')
        n1 = self.nodos[nodo1]
        n2 = self.nodos[nodo2]
        lat_diff = n2.latitud - n1.latitud
        lon_diff = n2.longitud - n1.longitud
        km_por_grado = 111.32
        return math.sqrt(lat_diff ** 2 + lon_diff ** 2) * km_por_grado

    def obtener_costo_arista(self, origen: str, destino: str, criterio: str = 'distancia') -> float:
        for arista in self.obtener_vecinos(origen):
            if arista.destino == destino:
                return arista.distancia if criterio == 'distancia' else arista.tiempo
        raise ValueError(f"No existe arista de {origen} a {destino}")

    # ✅ MÉTODOS REQUERIDOS POR ALGORTIMOS DE BÚSQUEDA
    def adyacencias(self, nodo: str) -> List[str]:
        return [arista.destino for arista in self.obtener_vecinos(nodo)]

    def obtener_peso(self, origen: str, destino: str) -> float:
        return self.obtener_costo_arista(origen, destino, criterio='distancia')

    def mostrar_grafo(self):
        print("\n=== GRAFO DE BOGOTÁ ===")
        for nodo in self.nodos.values():
            print(f"\n{nodo}")
            for arista in self.obtener_vecinos(nodo.nombre):
                print(f"  {arista}")


def crear_mapa_bogota() -> GrafoBogota:
    grafo = GrafoBogota()

    puntos = {
        "UNIMINUTO_CALLE_80": (4.6757, -74.1028),
        "PLAZA_LOURDES": (4.6589, -74.0892),
        "CIUDAD_UNIVERSITARIA": (4.6367, -74.0834),
        "ZONA_ROSA": (4.6643, -74.0632),
        "CENTRO_BOGOTA": (4.5981, -74.0758),
        "RESTREPO": (4.5678, -74.0892),
        "PERDOMO": (4.5489, -74.1456),
        "UNIMINUTO_PERDOMO": (4.5501, -74.1441),
        "SUBA": (4.7391, -74.0834),
        "CHAPINERO": (4.6307, -74.0678),
        "USAQUEN": (4.6955, -74.0303),
        "KENNEDY": (4.6289, -74.1420),
        "BOSA": (4.6289, -74.1777),
        "FONTIBÓN": (4.6656, -74.1464),
        "ENGATIVÁ": (4.7556, -74.1164),
        "GRAN_ESTACION": (4.628, -74.105),
    }

    print("\n" + "=" * 50)
    print("AGREGANDO NODOS AL GRAFO...")
    print("=" * 50)
    for nombre, (lat, lon) in puntos.items():
        grafo.agregar_nodo_por_datos(nombre, lat, lon)

    print("\n" + "=" * 50)
    print("CREANDO CONEXIONES ENTRE PUNTOS...")
    print("=" * 50)
    conexiones = [
        ("UNIMINUTO_CALLE_80", "PLAZA_LOURDES", 3.2, 15),
        ("PLAZA_LOURDES", "CIUDAD_UNIVERSITARIA", 2.8, 12),
        ("CIUDAD_UNIVERSITARIA", "ZONA_ROSA", 4.1, 18),
        ("ZONA_ROSA", "CENTRO_BOGOTA", 5.6, 25),
        ("CENTRO_BOGOTA", "RESTREPO", 4.3, 20),
        ("RESTREPO", "PERDOMO", 6.8, 30),
        ("PERDOMO", "UNIMINUTO_PERDOMO", 0.5, 3),
        ("UNIMINUTO_CALLE_80", "CENTRO_BOGOTA", 9.2, 35),
        ("PLAZA_LOURDES", "RESTREPO", 7.5, 28),
        ("CIUDAD_UNIVERSITARIA", "PERDOMO", 8.9, 32),
        ("ZONA_ROSA", "CHAPINERO", 2.1, 10),
        ("CHAPINERO", "USAQUEN", 3.4, 16),
        ("UNIMINUTO_CALLE_80", "SUBA", 4.8, 22),
        ("SUBA", "ENGATIVÁ", 3.2, 18),
        ("ENGATIVÁ", "FONTIBÓN", 4.1, 25),
        ("FONTIBÓN", "KENNEDY", 3.7, 20),
        ("KENNEDY", "BOSA", 2.9, 15),
        ("BOSA", "PERDOMO", 4.2, 25),
        ("CENTRO_BOGOTA", "CHAPINERO", 3.8, 18),
        ("CHAPINERO", "CIUDAD_UNIVERSITARIA", 3.2, 15),
        ("USAQUEN", "SUBA", 5.2, 28),
        ("FONTIBÓN", "UNIMINUTO_CALLE_80", 6.3, 30),
        ("KENNEDY", "RESTREPO", 5.1, 25),
        ("BOSA", "KENNEDY", 2.9, 15),
        ("UNIMINUTO_CALLE_80", "ENGATIVÁ", 7.2, 35),
        ("CENTRO_BOGOTA", "KENNEDY", 8.4, 40),
        ("UNIMINUTO_CALLE_80", "GRAN_ESTACION", 5.0, 12),
        ("GRAN_ESTACION", "CIUDAD_UNIVERSITARIA", 3.2, 9),
    ]

    for origen, destino, dist, tiempo in conexiones:
        grafo.agregar_arista_bidireccional(origen, destino, dist, tiempo)

    return grafo


# Singleton
_grafo_instancia = None

def obtener_grafo_bogota():
    global _grafo_instancia
    if _grafo_instancia is None:
        _grafo_instancia = crear_mapa_bogota()
    return _grafo_instancia

def inicializar_grafo():
    return obtener_grafo_bogota()


if __name__ == "__main__":
    grafo = crear_mapa_bogota()
    grafo.mostrar_grafo()
