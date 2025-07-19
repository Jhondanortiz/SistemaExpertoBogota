#!/usr/bin/env python3 
"""
Sistema Experto para Búsqueda de Rutas en Bogotá
"""

import sys
from datetime import datetime
from grafo_bogota import crear_mapa_bogota
from algoritmos_busqueda import BuscadorRutas, ResultadoBusqueda


def mostrar_resultado_detallado(resultado: ResultadoBusqueda, nombre_algoritmo):
    print("\n" + "=" * 60)
    print("RESULTADO DE LA BÚSQUEDA")
    print("=" * 60)

    if not resultado.exito:
        print("❌ No se encontró una ruta válida")
        return

    print(f"✅ Algoritmo: {nombre_algoritmo}")
    print(f"📍 Ruta encontrada: {' → '.join(resultado.ruta)}")
    print(f"📏 Distancia total: {resultado.distancia_total:.2f} km")
    print(f"⏱️  Tiempo total: {resultado.tiempo_total:.2f} minutos")
    print(f"🔍 Nodos explorados: {getattr(resultado, 'nodos_explorados', 'Desconocido')}")
    print(f"📊 Pasos en la ruta: {len(resultado.ruta)}")

    print(f"\n📋 DETALLES DE LA RUTA:")
    for i, nodo in enumerate(resultado.ruta):
        if i == 0:
            print(f"  {i+1}. 🚩 INICIO: {nodo}")
        elif i == len(resultado.ruta) - 1:
            print(f"  {i+1}. 🏁 DESTINO: {nodo}")
        else:
            print(f"  {i+1}. ➡️  {nodo}")
    print("=" * 60)


def ejecutar_pruebas_rendimiento(buscador: BuscadorRutas):
    print("\n" + "=" * 60)
    print("EJECUTANDO PRUEBAS DE RENDIMIENTO")
    print("=" * 60)

    casos_prueba = [
        ("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO"),
        ("UNIMINUTO_CALLE_80", "CENTRO_BOGOTA"),
        ("PLAZA_LOURDES", "PERDOMO"),
        ("ZONA_ROSA", "RESTREPO")
    ]

    for origen, destino in casos_prueba:
        print(f"\n🧪 Caso de prueba: {origen} → {destino}")
        print("-" * 50)

        resultados = buscador.comparar_algoritmos(origen, destino)

        print(f"\n📊 RESUMEN DE EFICIENCIA:")
        for nombre, resultado in resultados.items():
            if getattr(resultado, "exito", False):
                eficiencia = resultado.distancia_total / max(1, getattr(resultado, 'nodos_explorados', 1))
                print(f"  {nombre:30} | Eficiencia: {eficiencia:6.2f} km/nodo")


def generar_reporte_detallado(buscador: BuscadorRutas, grafo):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_archivo = f"reporte_sistema_{timestamp}.txt"

    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write("REPORTE DETALLADO DEL SISTEMA EXPERTO DE RUTAS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha: {datetime.now()}\n")
            f.write(f"Número de nodos: {len(grafo.nodos)}\n")
            f.write(f"Número de aristas: {sum(len(v) for v in grafo.grafo.values())}\n\n")

            f.write("ANÁLISIS DE RUTAS\n")
            ruta_principal = ("UNIMINUTO_CALLE_80", "UNIMINUTO_PERDOMO")
            resultados = buscador.comparar_algoritmos(*ruta_principal)

            for nombre, resultado in resultados.items():
                if not getattr(resultado, "exito", False):
                    continue
                f.write(f"\n{nombre.upper()}:\n")
                f.write(f"  Ruta: {' → '.join(resultado.ruta)}\n")
                f.write(f"  Distancia: {resultado.distancia_total:.2f} km\n")
                f.write(f"  Tiempo: {resultado.tiempo_total:.2f} min\n")
                f.write(f"  Nodos explorados: {getattr(resultado, 'nodos_explorados', 'Desconocido')}\n")

            f.write("\nReporte generado automáticamente.\n")

        print(f"✅ Reporte guardado como: {nombre_archivo}")

    except Exception as e:
        print(f"❌ Error al generar reporte: {e}")


def main():
    print("\n🚀 INICIANDO SISTEMA EXPERTO DE RUTAS EN BOGOTÁ 🚀")

    try:
        grafo = crear_mapa_bogota()
        buscador = BuscadorRutas(grafo)
        print("✅ Grafo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el grafo: {e}")
        sys.exit(1)

    while True:
        print("\n=== MENÚ ===")
        print("1. Ejecutar prueba específica")
        print("2. Comparar algoritmos")
        print("3. Pruebas de rendimiento")
        print("4. Generar reporte")
        print("5. Salir")

        opcion = input("Selecciona una opción (1-5): ").strip()

        if opcion == "1":
            origen = input("Nombre del nodo origen: ").strip().upper()
            destino = input("Nombre del nodo destino: ").strip().upper()
            criterio = input("Criterio (distancia/tiempo): ").strip().lower()

            resultado = buscador.buscar(origen, destino, algoritmo="a*", criterio=criterio)
            mostrar_resultado_detallado(resultado, "a*")

        elif opcion == "2":
            origen = input("Nombre del nodo origen: ").strip().upper()
            destino = input("Nombre del nodo destino: ").strip().upper()
            resultados = buscador.comparar_algoritmos(origen, destino)
            for nombre, value in resultados.items():
                print(f"\n▶️  {nombre}")
                print(f"\nTiempo {value['tiempo_ejecucion']}")
                mostrar_resultado_detallado(value['resultado'], nombre)

        elif opcion == "3":
            ejecutar_pruebas_rendimiento(buscador)

        elif opcion == "4":
            generar_reporte_detallado(buscador, grafo)

        elif opcion == "5":
            print("👋 Saliendo del sistema. ¡Hasta pronto!")
            break

        else:
            print("❌ Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
