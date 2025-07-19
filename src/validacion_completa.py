#!/usr/bin/env python3
"""
Validación Completa del Sistema Experto de Rutas en Bogotá
Ejecuta todas las pruebas y genera un reporte completo para la entrega

Autor: [Tu nombre]
Universidad: UNIMINUTO
Curso: [Nombre del curso]
"""

import sys
import time
from datetime import datetime

# Importar los módulos del sistema
try:
    from grafo_bogota import GrafoBogota, Nodo
    from algoritmos_busqueda import BuscadorRutas
    from utils import COORDENADAS_BOGOTA, CONEXIONES, EMOJIS, imprimir_encabezado
except ImportError as e:
    print(f"❌ Error al importar módulos: {e}")
    print("Asegúrate de que todos los archivos estén en el mismo directorio y correctamente nombrados.")
    sys.exit(1)


class ValidadorSistema:
    """Clase para validar el funcionamiento completo del sistema"""

    def __init__(self):
        self.grafo = GrafoBogota()
        self.buscador = BuscadorRutas(self.grafo)
        self.resultados_validacion = []

    def poblar_grafo(self):
        """Agrega nodos y conexiones al grafo desde las constantes"""
        for codigo, datos in COORDENADAS_BOGOTA.items():
            try:
                nodo = Nodo(
                    nombre=codigo,
                    latitud=datos['latitud'],
                    longitud=datos['longitud'],
                    descripcion=datos.get('descripcion', '')
                )
                self.grafo.agregar_nodo(nodo)
            except KeyError as e:
                print(f"❌ Error al agregar nodo {codigo}: falta clave {e}")
                raise

        for nodo_a, nodo_b, distancia, tiempo in CONEXIONES:
            self.grafo.agregar_arista_bidireccional(nodo_a, nodo_b, distancia, tiempo)

    def validar_estructura_grafo(self):
        """Valida que el grafo esté correctamente construido"""
        print(f"\n{EMOJIS['exploracion']} Validando estructura del grafo...")

        nodos_esperados = len(COORDENADAS_BOGOTA)
        nodos_actuales = len(self.grafo.nodos)

        if nodos_actuales == nodos_esperados:
            print(f"✅ Nodos: {nodos_actuales}/{nodos_esperados} - CORRECTO")
            self.resultados_validacion.append(("Estructura de nodos", "CORRECTO", f"{nodos_actuales} nodos"))
        else:
            print(f"❌ Nodos: {nodos_actuales}/{nodos_esperados} - ERROR")
            self.resultados_validacion.append(
                ("Estructura de nodos", "ERROR", f"Faltan {nodos_esperados - nodos_actuales} nodos")
            )

        conexiones_validas = sum(1 for nodo_id in self.grafo.nodos if self.grafo.grafo.get(nodo_id))

        if conexiones_validas == len(self.grafo.nodos):
            print(f"✅ Conectividad: Todos los nodos están conectados")
            self.resultados_validacion.append(("Conectividad", "CORRECTO", "Todos los nodos conectados"))
        else:
            nodos_desconectados = len(self.grafo.nodos) - conexiones_validas
            print(f"❌ Conectividad: {conexiones_validas}/{len(self.grafo.nodos)} nodos conectados")
            self.resultados_validacion.append(
                ("Conectividad", "ERROR", f"Nodos desconectados: {nodos_desconectados}")
            )

    def validar_algoritmos(self):
        """Valida que los algoritmos funcionen correctamente"""
        print(f"\n{EMOJIS['exploracion']} Validando algoritmos...")

        # Usar nodos específicos que sabemos que existen
        casos_prueba = [
            ('CENTRO_BOGOTA', 'ZONA_ROSA'),
            ('UNIMINUTO_CALLE_80', 'UNIMINUTO_PERDOMO'),
            ('USAQUEN', 'BOSA')
        ]
        
        # Filtrar solo los casos donde ambos nodos existen
        casos_validos = []
        for origen, destino in casos_prueba:
            if origen in self.grafo.nodos and destino in self.grafo.nodos:
                casos_validos.append((origen, destino))
        
        # Si no hay casos válidos, usar los primeros dos nodos disponibles
        if not casos_validos:
            nodos_disponibles = list(self.grafo.nodos.keys())
            if len(nodos_disponibles) >= 2:
                casos_validos = [(nodos_disponibles[0], nodos_disponibles[1])]
            else:
                print("❌ No hay suficientes nodos para realizar pruebas.")
                return

        algoritmos = ['dijkstra', 'a_estrella']
        criterios = ['distancia', 'tiempo']

        for origen, destino in casos_validos[:1]:  # Solo probar con el primer caso válido
            for algoritmo in algoritmos:
                for criterio in criterios:
                    print(f"\n   Caso: {algoritmo.upper()} - {criterio} ({origen} → {destino})")
                    try:
                        if algoritmo == 'dijkstra':
                            resultado = self.buscador.dijkstra(origen, destino, criterio)
                        else:
                            resultado = self.buscador.a_estrella(origen, destino, criterio)

                        # Validar que el resultado sea del tipo correcto
                        if resultado is None:
                            print(f"   ❌ Algoritmo retornó None")
                            self.resultados_validacion.append(
                                (f"{algoritmo}-{criterio}", "ERROR", "Resultado None")
                            )
                            continue

                        # Verificar si tiene el atributo ruta
                        if hasattr(resultado, 'ruta'):
                            if resultado.ruta and len(resultado.ruta) > 0:
                                # Calcular costo basado en el criterio
                                if criterio == 'distancia':
                                    costo = getattr(resultado, 'distancia_total', 0)
                                else:
                                    costo = getattr(resultado, 'tiempo_total', 0)
                                
                                print(f"   ✅ Ruta encontrada ({len(resultado.ruta)} pasos, costo: {costo:.2f})")
                                self.resultados_validacion.append(
                                    (f"{algoritmo}-{criterio}", "CORRECTO", f"{len(resultado.ruta)} pasos, costo: {costo:.2f}")
                                )
                            else:
                                print(f"   ❌ Ruta vacía")
                                self.resultados_validacion.append(
                                    (f"{algoritmo}-{criterio}", "ERROR", "Ruta vacía")
                                )
                        else:
                            print(f"   ❌ Resultado no tiene atributo 'ruta': {type(resultado)}")
                            self.resultados_validacion.append(
                                (f"{algoritmo}-{criterio}", "ERROR", f"Tipo incorrecto: {type(resultado)}")
                            )

                    except AttributeError as e:
                        print(f"   ❌ Error de atributo: {e}")
                        self.resultados_validacion.append(
                            (f"{algoritmo}-{criterio}", "ERROR", f"AttributeError: {e}")
                        )
                    except Exception as e:
                        print(f"   ❌ Excepción: {e}")
                        self.resultados_validacion.append(
                            (f"{algoritmo}-{criterio}", "ERROR", f"Excepción: {e}")
                        )

    def ejecutar_pruebas_rendimiento(self):
        """Simulación simple de rendimiento"""
        print(f"\n{EMOJIS['exploracion']} Ejecutando pruebas de rendimiento...")

        # Obtener nodos válidos para las pruebas
        nodos_disponibles = list(self.grafo.nodos.keys())
        if len(nodos_disponibles) < 2:
            print("❌ No hay suficientes nodos para pruebas de rendimiento")
            self.resultados_validacion.append(
                ("Prueba rendimiento", "ERROR", "Nodos insuficientes")
            )
            return

        # Usar el primer y último nodo disponible
        origen = nodos_disponibles[0]
        destino = nodos_disponibles[-1]

        try:
            inicio = time.time()
            resultado = self.buscador.dijkstra(origen, destino)
            duracion = time.time() - inicio

            if resultado and hasattr(resultado, 'ruta') and resultado.ruta:
                print(f"   ✅ {origen} → {destino} en {duracion:.4f}s con {len(resultado.ruta)} pasos")
                self.resultados_validacion.append(
                    ("Prueba rendimiento", "CORRECTO", f"{duracion:.4f}s, {len(resultado.ruta)} pasos")
                )
            else:
                print(f"   ❌ No se encontró ruta")
                self.resultados_validacion.append(
                    ("Prueba rendimiento", "ERROR", "Ruta no encontrada")
                )

        except Exception as e:
            print(f"   ❌ Error en prueba de rendimiento: {e}")
            self.resultados_validacion.append(
                ("Prueba rendimiento", "ERROR", f"Excepción: {e}")
            )

    def generar_reporte_validacion(self):
        """Genera reporte final"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo = f"reporte_validacion_{timestamp}.txt"
        
        try:
            with open(archivo, "w", encoding="utf-8") as f:
                f.write("=== REPORTE DE VALIDACIÓN ===\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Sistema: Sistema Experto de Rutas en Bogotá\n")
                f.write(f"Universidad: UNIMINUTO\n\n")
                
                # Estadísticas
                total_pruebas = len(self.resultados_validacion)
                exitosas = sum(1 for _, estado, _ in self.resultados_validacion if estado == "CORRECTO")
                fallidas = total_pruebas - exitosas
                
                f.write(f"RESUMEN:\n")
                f.write(f"- Total de pruebas: {total_pruebas}\n")
                f.write(f"- Exitosas: {exitosas}\n")
                f.write(f"- Fallidas: {fallidas}\n")
                f.write(f"- Porcentaje de éxito: {(exitosas/total_pruebas*100):.1f}%\n\n")
                
                f.write("DETALLES:\n")
                for nombre, estado, detalle in self.resultados_validacion:
                    f.write(f"• {nombre}: {estado} - {detalle}\n")
                    
            print(f"\n{EMOJIS['reporte']} Reporte guardado en: {archivo}")
            
        except Exception as e:
            print(f"❌ Error al generar reporte: {e}")
            archivo = "reporte_no_generado"
        
        reporte_texto = "\n".join([f"{n}: {e} - {d}" for n, e, d in self.resultados_validacion])
        return reporte_texto, archivo

    def mostrar_resumen_consola(self):
        """Muestra un resumen en la consola"""
        total_pruebas = len(self.resultados_validacion)
        if total_pruebas == 0:
            print(f"\n{EMOJIS['error']} No se ejecutaron pruebas")
            return
            
        exitosas = sum(1 for _, estado, _ in self.resultados_validacion if estado == "CORRECTO")
        fallidas = total_pruebas - exitosas
        porcentaje = (exitosas/total_pruebas*100) if total_pruebas > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"                    RESUMEN DE VALIDACIÓN")
        print(f"{'='*60}")
        print(f"Total de pruebas: {total_pruebas}")
        print(f"Exitosas: {exitosas} ✅")
        print(f"Fallidas: {fallidas} ❌")
        print(f"Porcentaje de éxito: {porcentaje:.1f}%")
        
        if fallidas > 0:
            print(f"\n{EMOJIS['error']} PRUEBAS FALLIDAS:")
            for nombre, estado, detalle in self.resultados_validacion:
                if estado == "ERROR":
                    print(f"  • {nombre}: {detalle}")

    def ejecutar_validacion_completa(self):
        """Ejecuta toda la validación del sistema"""
        print(f"\n{EMOJIS['info']} Iniciando validación completa del sistema...")

        inicio = time.time()
        
        try:
            self.poblar_grafo()
            self.validar_estructura_grafo()
            self.validar_algoritmos()
            self.ejecutar_pruebas_rendimiento()
            
        except Exception as e:
            print(f"\n{EMOJIS['error']} Error crítico durante la validación: {e}")
            self.resultados_validacion.append(("Error crítico", "ERROR", str(e)))
        
        fin = time.time()
        duracion = fin - inicio

        print(f"\n{EMOJIS['exito']} Validación completada en {duracion:.2f} segundos")
        
        self.mostrar_resumen_consola()
        reporte, archivo = self.generar_reporte_validacion()
        
        return reporte, archivo


def main():
    imprimir_encabezado("VALIDACIÓN COMPLETA DEL SISTEMA")
    print(f"{EMOJIS['info']} Sistema Experto de Rutas en Bogotá")
    print(f"{EMOJIS['info']} Validación para Primera Entrega")
    print(f"{EMOJIS['info']} Universidad UNIMINUTO")

    validador = ValidadorSistema()

    try:
        reporte, archivo = validador.ejecutar_validacion_completa()
        print(f"\n{EMOJIS['exito']} Proceso de validación finalizado")
        
        if archivo != "reporte_no_generado":
            print(f"{EMOJIS['reporte']} Consulta el archivo {archivo} para detalles completos")
        
    except Exception as e:
        print(f"\n{EMOJIS['error']} Error durante la validación: {e}")
        print(f"{EMOJIS['info']} Verifica que todos los archivos estén presentes")
        print(f"{EMOJIS['info']} Archivos requeridos: grafo_bogota.py, algoritmos_busqueda.py, utils.py")


if __name__ == "__main__":
    main()