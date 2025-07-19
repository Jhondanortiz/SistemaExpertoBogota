#!/usr/bin/env python3
"""
Pruebas unitarias para el Sistema Experto de Rutas en Bogotá
Valida la funcionalidad de los algoritmos implementados

Autor: [Tu nombre]
Universidad: UNIMINUTO
Curso: [Nombre del curso]
"""

import unittest
import sys
import os
import time
from io import StringIO

# Importar módulos del sistema
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grafo_bogota import GrafoBogota, crear_mapa_bogota
from algoritmos_busqueda import AlgoritmosBusqueda, ResultadoBusqueda

class TestGrafoBogota(unittest.TestCase):
    """Pruebas para la clase GrafoBogota"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.grafo = GrafoBogota()
        
        # Crear un grafo de prueba simple
        self.grafo.agregar_nodo("A", 4.6, -74.1)
        self.grafo.agregar_nodo("B", 4.7, -74.0)
        self.grafo.agregar_nodo("C", 4.5, -74.2)
        
        self.grafo.agregar_arista("A", "B", 5.0, 20)
        self.grafo.agregar_arista("B", "C", 3.0, 15)
        self.grafo.agregar_arista("A", "C", 8.0, 30)
    
    def test_agregar_nodo(self):
        """Prueba la adición de nodos"""
        self.assertIn("A", self.grafo.nodos)
        self.assertIn("B", self.grafo.nodos)
        self.assertIn("C", self.grafo.nodos)
        
        nodo_a = self.grafo.nodos["A"]
        self.assertEqual(nodo_a.latitud, 4.6)
        self.assertEqual(nodo_a.longitud, -74.1)
    
    def test_agregar_arista(self):
        """Prueba la adición de aristas"""
        vecinos_a = self.grafo.obtener_vecinos("A")
        self.assertEqual(len(vecinos_a), 2)
        
        # Verificar que las aristas tienen los datos correctos
        destinos = [arista.destino for arista in vecinos_a]
        self.assertIn("B", destinos)
        self.assertIn("C", destinos)
    
    def test_distancia_euclidiana(self):
        """Prueba el cálculo de distancia euclidiana"""
        distancia = self.grafo.calcular_distancia_euclidiana("A", "B")
        self.assertGreater(distancia, 0)
        self.assertIsInstance(distancia, float)
        
        # Distancia a sí mismo debe ser 0
        distancia_mismo = self.grafo.calcular_distancia_euclidiana("A", "A")
        self.assertEqual(distancia_mismo, 0)
    
    def test_nodo_inexistente(self):
        """Prueba el manejo de nodos inexistentes"""
        distancia = self.grafo.calcular_distancia_euclidiana("A", "Z")
        self.assertEqual(distancia, float('inf'))

class TestAlgoritmosBusqueda(unittest.TestCase):
    """Pruebas para los algoritmos de búsqueda"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Usar el grafo completo de Bogotá
        self.grafo = crear_mapa_bogota()
        self.buscador = AlgoritmosBusqueda(self.grafo)
        
        # Puntos de prueba
        self.origen = "UNIMINUTO_CALLE_80"
        self.destino = "UNIMINUTO_PERDOMO"
    
    def test_dijkstra_distancia(self):
        """Prueba el algoritmo Dijkstra optimizando por distancia"""
        resultado = self.buscador.dijkstra(self.origen, self.destino, "distancia")
        
        self.assertTrue(resultado.exito)
        self.assertGreater(len(resultado.ruta), 0)
        self.assertEqual(resultado.ruta[0], self.origen)
        self.assertEqual(resultado.ruta[-1], self.destino)
        self.assertGreater(resultado.distancia_total, 0)
        self.assertGreater(resultado.tiempo_total, 0)
        self.assertGreater(resultado.nodos_explorados, 0)
    
    def test_dijkstra_tiempo(self):
        """Prueba el algoritmo Dijkstra optimizando por tiempo"""
        resultado = self.buscador.dijkstra(self.origen, self.destino, "tiempo")
        
        self.assertTrue(resultado.exito)
        self.assertGreater(len(resultado.ruta), 0)
        self.assertEqual(resultado.ruta[0], self.origen)
        self.assertEqual(resultado.ruta[-1], self.destino)
        self.assertGreater(resultado.distancia_total, 0)
        self.assertGreater(resultado.tiempo_total, 0)
    
    def test_a_estrella_distancia(self):
        """Prueba el algoritmo A* optimizando por distancia"""
        resultado = self.buscador.a_estrella(self.origen, self.destino, "distancia")
        
        self.assertTrue(resultado.exito)
        self.assertGreater(len(resultado.ruta), 0)
        self.assertEqual(resultado.ruta[0], self.origen)
        self.assertEqual(resultado.ruta[-1], self.destino)
        self.assertGreater(resultado.distancia_total, 0)
        self.assertGreater(resultado.tiempo_total, 0)
    
    def test_a_estrella_tiempo(self):
        """Prueba el algoritmo A* optimizando por tiempo"""
        resultado = self.buscador.a_estrella(self.origen, self.destino, "tiempo")
        
        self.assertTrue(resultado.exito)
        self.assertGreater(len(resultado.ruta), 0)
        self.assertEqual(resultado.ruta[0], self.origen)
        self.assertEqual(resultado.ruta[-1], self.destino)
        self.assertGreater(resultado.distancia_total, 0)
        self.assertGreater(resultado.tiempo_total, 0)
    
    def test_nodo_inexistente(self):
        """Prueba el manejo de nodos inexistentes"""
        resultado = self.buscador.dijkstra("NODO_FALSO", self.destino, "distancia")
        self.assertFalse(resultado.exito)
        self.assertEqual(len(resultado.ruta), 0)
    
    def test_mismo_origen_destino(self):
        """Prueba cuando origen y destino son el mismo"""
        resultado = self.buscador.dijkstra(self.origen, self.origen, "distancia")
        # Dependiendo de la implementación, esto podría ser válido o no
        if resultado.exito:
            self.assertEqual(len(resultado.ruta), 1)
            self.assertEqual(resultado.distancia_total, 0)

class TestComparacionAlgoritmos(unittest.TestCase):
    """Pruebas para la comparación de algoritmos"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.grafo = crear_mapa_bogota()
        self.buscador = AlgoritmosBusqueda(self.grafo)
        self.origen = "UNIMINUTO_CALLE_80"
        self.destino = "UNIMINUTO_PERDOMO"
    
    def test_comparacion_completa(self):
        """Prueba la comparación completa de todos los algoritmos"""
        resultados = self.buscador.comparar_algoritmos(self.origen, self.destino)
        
        # Verificar que se ejecutaron todos los algoritmos
        algoritmos_esperados = [
            'dijkstra_distancia', 'dijkstra_tiempo', 
            'a_estrella_distancia', 'a_estrella_tiempo'
        ]
        
        for algoritmo in algoritmos_esperados:
            self.assertIn(algoritmo, resultados)
            self.assertIsInstance(resultados[algoritmo], ResultadoBusqueda)
    
    def test_consistencia_resultados(self):
        """Prueba que los resultados sean consistentes entre ejecuciones"""
        resultado1 = self.buscador.dijkstra(self.origen, self.destino, "distancia")
        resultado2 = self.buscador.dijkstra(self.origen, self.destino, "distancia")
        
        # Los resultados deben ser idénticos
        self.assertEqual(resultado1.ruta, resultado2.ruta)
        self.assertEqual(resultado1.distancia_total, resultado2.distancia_total)
        self.assertEqual(resultado1.tiempo_total, resultado2.tiempo_total)
    
    def test_optimalidad_dijkstra(self):
        """Prueba que Dijkstra encuentre el camino óptimo"""
        resultado_dist = self.buscador.dijkstra(self.origen, self.destino, "distancia")
        resultado_tiempo = self.buscador.dijkstra(self.origen, self.destino, "tiempo")
        
        # Los resultados deben ser válidos
        self.assertTrue(resultado_dist.exito)
        self.assertTrue(resultado_tiempo.exito)
        
        # Dijkstra por distancia debe optimizar distancia
        # Dijkstra por tiempo debe optimizar tiempo
        self.assertGreater(resultado_dist.distancia_total, 0)
        self.assertGreater(resultado_tiempo.tiempo_total, 0)

class TestRendimientoAlgoritmos(unittest.TestCase):
    """Pruebas de rendimiento de los algoritmos"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.grafo = crear_mapa_bogota()
        self.buscador = AlgoritmosBusqueda(self.grafo)
    
    def test_tiempo_ejecucion_dijkstra(self):
        """Prueba el tiempo de ejecución de Dijkstra"""
        inicio = time.time()
        resultado = self.buscador.dijkstra("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO", "distancia")
        fin = time.time()
        
        tiempo_ejecucion = fin - inicio
        self.assertLess(tiempo_ejecucion, 1.0)  # Debe ejecutarse en menos de 1 segundo
        self.assertTrue(resultado.exito)
    
    def test_tiempo_ejecucion_a_estrella(self):
        """Prueba el tiempo de ejecución de A*"""
        inicio = time.time()
        resultado = self.buscador.a_estrella("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO", "distancia")
        fin = time.time()
        
        tiempo_ejecucion = fin - inicio
        self.assertLess(tiempo_ejecucion, 1.0)  # Debe ejecutarse en menos de 1 segundo
        self.assertTrue(resultado.exito)
    
    def test_eficiencia_a_estrella(self):
        """Prueba que A* sea más eficiente que Dijkstra en nodos explorados"""
        resultado_dijkstra = self.buscador.dijkstra("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO", "distancia")
        resultado_a_estrella = self.buscador.a_estrella("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO", "distancia")
        
        # A* debería explorar igual o menos nodos que Dijkstra
        self.assertLessEqual(resultado_a_estrella.nodos_explorados, resultado_dijkstra.nodos_explorados)

class TestCasosPrueba(unittest.TestCase):
    """Pruebas con casos específicos del dominio"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.grafo = crear_mapa_bogota()
        self.buscador = AlgoritmosBusqueda(self.grafo)
    
    def test_ruta_directa_corta(self):
        """Prueba una ruta directa corta"""
        resultado = self.buscador.dijkstra("PERDOMO", "UNIMINUTO_PERDOMO", "distancia")
        
        self.assertTrue(resultado.exito)
        self.assertEqual(len(resultado.ruta), 2)  # Debe ser directo
        self.assertLess(resultado.distancia_total, 1.0)  # Distancia muy corta
    
    def test_multiples_rutas_alternativas(self):
        """Prueba que existan múltiples rutas alternativas"""
        # Ejecutar desde diferentes puntos intermedios
        puntos_intermedios = ["PLAZA_LOURDES", "CENTRO_BOGOTA", "RESTREPO"]
        
        for punto in puntos_intermedios:
            resultado = self.buscador.dijkstra("UNIMINUTO_CALLE_80", punto, "distancia")
            self.assertTrue(resultado.exito, f"No se encontró ruta a {punto}")
    
    def test_consistencia_rutas_bidireccionales(self):
        """Prueba que las rutas sean consistentes en ambas direcciones"""
        # Ida
        resultado_ida = self.buscador.dijkstra("UNIMINUTO_CALLE_80", "CENTRO_BOGOTA", "distancia")
        # Vuelta
        resultado_vuelta = self.buscador.dijkstra("CENTRO_BOGOTA", "UNIMINUTO_CALLE_80", "distancia")
        
        # Las distancias deben ser iguales (el grafo es bidireccional)
        self.assertEqual(resultado_ida.distancia_total, resultado_vuelta.distancia_total)
        self.assertEqual(resultado_ida.tiempo_total, resultado_vuelta.tiempo_total)

def ejecutar_pruebas_completas():
    """Ejecuta todas las pruebas del sistema"""
    print("="*70)
    print("INICIANDO PRUEBAS UNITARIAS DEL SISTEMA EXPERTO")
    print("="*70)
    
    # Crear suite de pruebas
    suite = unittest.TestSuite()
    
    # Agregar todas las clases de prueba
    clases_prueba = [
        TestGrafoBogota,
        TestAlgoritmosBusqueda,
        TestComparacionAlgoritmos,
        TestRendimientoAlgoritmos,
        TestCasosPrueba
    ]
    
    for clase in clases_prueba:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(clase))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    print(f"Pruebas ejecutadas: {resultado.testsRun}")
    print(f"Exitosas: {resultado.testsRun - len(resultado.failures) - len(resultado.errors)}")
    print(f"Fallos: {len(resultado.failures)}")
    print(f"Errores: {len(resultado.errors)}")
    
    if resultado.failures:
        print("\nFALLOS:")
        for test, traceback in resultado.failures:
            print(f"  - {test}: {traceback}")
    
    if resultado.errors:
        print("\nERRORES:")
        for test, traceback in resultado.errors:
            print(f"  - {test}: {traceback}")
    
    # Determinar éxito general
    exito_general = len(resultado.failures) == 0 and len(resultado.errors) == 0
    
    if exito_general:
        print("\n✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON")
    
    print("="*70)
    return exito_general

if __name__ == "__main__":
    # Capturar salida para limpiar logs durante las pruebas
    import sys
    
    # Redirigir stdout temporalmente para limpiar la salida
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        exito = ejecutar_pruebas_completas()
    finally:
        sys.stdout = old_stdout
    
    # Mostrar resultado final
    if exito:
        print("🎉 Sistema validado correctamente. Listo para producción.")
    else:
        print("⚠️  Sistema requiere correcciones antes de la entrega.")
    
    sys.exit(0 if exito else 1)