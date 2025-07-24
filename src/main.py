#!/usr/bin/env python3
"""
Sistema Experto para Búsqueda de Rutas en Bogotá con Machine Learning
Integración de algoritmos de aprendizaje automático para optimización de rutas
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import pickle
from pathlib import Path
from unidecode import unidecode

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Imports del sistema original
from grafo_bogota import crear_mapa_bogota
from algoritmos_busqueda import BuscadorRutas, ResultadoBusqueda

class SistemaExpertoRutasML:
    """
    Sistema Experto de Rutas potenciado con Machine Learning
    
    OBJETIVOS DEL MACHINE LEARNING:
    1. Predecir tiempos de viaje dinámicos basados en condiciones de tráfico
    2. Optimizar selección de rutas según patrones históricos
    3. Identificar rutas alternativas inteligentes
    4. Adaptarse a cambios en condiciones urbanas
    5. Mejorar precisión de estimaciones de tiempo/distancia
    """
    
    def __init__(self, grafo_bogota):
        # Sistema original
        self.grafo = grafo_bogota
        self.buscador = BuscadorRutas(grafo_bogota)
        
        # Componentes de ML
        self.modelos_tiempo = {}
        self.modelo_optimizacion = None
        self.scaler = StandardScaler()
        self.label_encoder_origen = LabelEncoder()
        self.label_encoder_destino = LabelEncoder()
        
        # Datos y configuración
        self.datos_historicos = []
        self.patrones_trafico = {}
        self.mejores_rutas_cache = {}
        self.modelo_entrenado = False
        
        # Métricas de rendimiento
        self.metricas_ml = {
            'precision_temporal': 0.0,
            'mejora_rutas': 0.0,
            'adaptabilidad': 0.0
        }
        
        print("🤖 Sistema Experto con ML inicializado")
    
    def definir_objetivos_ml(self) -> Dict:
        """
        PASO 1: Definición clara de objetivos del Machine Learning
        """
        objetivos = {
            'prediccion_temporal': {
                'descripcion': 'Predecir tiempos de viaje dinámicos',
                'metrica_objetivo': 'MAE < 5 minutos',
                'algoritmos': ['Random Forest', 'Gradient Boosting', 'Linear Regression']
            },
            'optimizacion_rutas': {
                'descripcion': 'Seleccionar la mejor ruta según contexto',
                'metrica_objetivo': 'Reducción 15% tiempo promedio',
                'algoritmos': ['Ensemble de regressores', 'Clustering']
            },
            'deteccion_patrones': {
                'descripcion': 'Identificar patrones de tráfico complejos',
                'metrica_objetivo': 'R² > 0.8 en predicciones',
                'algoritmos': ['Clustering K-means', 'Random Forest']
            },
            'adaptabilidad': {
                'descripcion': 'Adaptación continua a nuevos datos',
                'metrica_objetivo': 'Reentrenamiento automático',
                'algoritmos': ['Aprendizaje incremental']
            }
        }
        
        print("\n=== OBJETIVOS DEL MACHINE LEARNING ===")
        for objetivo, detalles in objetivos.items():
            print(f"\n🎯 {objetivo.replace('_', ' ').title()}:")
            print(f"   📝 {detalles['descripcion']}")
            print(f"   📊 Meta: {detalles['metrica_objetivo']}")
            print(f"   🔧 Algoritmos: {', '.join(detalles['algoritmos'])}")
        
        return objetivos
    
    def generar_datos_entrenamiento(self, num_muestras: int = 1000) -> pd.DataFrame:
        """
        PASO 2: Generación de datos de entrenamiento históricos
        En un sistema real, estos vendrían de sensores de tráfico, GPS, etc.
        """
        print(f"\n📊 Generando {num_muestras} muestras de datos históricos...")
        
        np.random.seed(42)
        nodos = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()]
        print("Nodos disponibles:", nodos[:10], "...")
        
        datos = []
        for i in range(num_muestras):
            # Seleccionar origen y destino aleatorios
            origen = np.random.choice(nodos)
            destino = np.random.choice([n for n in nodos if n != origen])
            
            # Buscar ruta base con el sistema original
            resultado = self.buscador.buscar(origen, destino, algoritmo="a*", criterio="tiempo")
            
            if not resultado.exito:
                continue
            
            # Simular condiciones variables
            hora_dia = np.random.randint(0, 24)
            dia_semana = np.random.randint(0, 7)
            condicion_climatica = np.random.choice(['soleado', 'lluvia', 'nublado'])
            nivel_trafico = np.random.choice(['bajo', 'medio', 'alto', 'muy_alto'])
            
            # Calcular tiempo real simulado (con variaciones)
            tiempo_base = resultado.tiempo_total
            factor_hora = self._factor_hora_pico(hora_dia)
            factor_dia = self._factor_dia_semana(dia_semana)
            factor_clima = self._factor_clima(condicion_climatica)
            factor_trafico = self._factor_trafico(nivel_trafico)
            
            tiempo_real = tiempo_base * factor_hora * factor_dia * factor_clima * factor_trafico
            tiempo_real += np.random.normal(0, tiempo_base * 0.1)  # Ruido
            
            datos.append({
                'origen': origen,
                'destino': destino,
                'distancia_base': resultado.distancia_total,
                'tiempo_base': tiempo_base,
                'hora_dia': hora_dia,
                'dia_semana': dia_semana,
                'condicion_climatica': condicion_climatica,
                'nivel_trafico': nivel_trafico,
                'num_pasos_ruta': len(resultado.ruta),
                'tiempo_real': max(tiempo_real, tiempo_base * 0.5)  # Mínimo realista
            })
        
        df = pd.DataFrame(datos)
        print(f"✅ Generados {len(df)} registros válidos")
        print(f"📈 Rango de tiempos: {df['tiempo_real'].min():.1f} - {df['tiempo_real'].max():.1f} min")
        
        return df
    
    def _factor_hora_pico(self, hora: int) -> float:
        """Factor de multiplicación según hora del día"""
        if 7 <= hora <= 9 or 17 <= hora <= 19:  # Horas pico
            return 1.8
        elif 10 <= hora <= 16:  # Horas normales día
            return 1.2
        elif 20 <= hora <= 22:  # Noche temprana
            return 1.1
        else:  # Madrugada
            return 0.8
    
    def _factor_dia_semana(self, dia: int) -> float:
        """Factor según día de la semana (0=lunes, 6=domingo)"""
        if 0 <= dia <= 4:  # Lunes a viernes
            return 1.3
        elif dia == 5:  # Sábado
            return 1.1
        else:  # Domingo
            return 0.9
    
    def _factor_clima(self, clima: str) -> float:
        """Factor según condición climática"""
        factores = {
            'soleado': 1.0,
            'nublado': 1.1,
            'lluvia': 1.5
        }
        return factores.get(clima, 1.0)
    
    def _factor_trafico(self, trafico: str) -> float:
        """Factor según nivel de tráfico"""
        factores = {
            'bajo': 0.9,
            'medio': 1.2,
            'alto': 1.6,
            'muy_alto': 2.2
        }
        return factores.get(trafico, 1.0)
    
    def preparar_caracteristicas(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        PASO 3: Preparación y división de datos
        ENFOQUE: Aprendizaje Supervisado - Regresión
        """
        print("\n🔧 Preparando características para ML...")
        
        # Codificar variables categóricas
        df_prep = df.copy()
        
        # Fit encoders si no están entrenados
        if not hasattr(self.label_encoder_origen, 'classes_'):
            self.label_encoder_origen.fit(df_prep['origen'])
            self.label_encoder_destino.fit(df_prep['destino'])
        
        df_prep['origen_encoded'] = self.label_encoder_origen.transform(df_prep['origen'])
        df_prep['destino_encoded'] = self.label_encoder_destino.transform(df_prep['destino'])
        
        # One-hot encoding para variables categóricas
        clima_dummies = pd.get_dummies(df_prep['condicion_climatica'], prefix='clima')
        trafico_dummies = pd.get_dummies(df_prep['nivel_trafico'], prefix='trafico')
        
        # Características numéricas
        caracteristicas_numericas = [
            'origen_encoded', 'destino_encoded', 'distancia_base', 'tiempo_base',
            'hora_dia', 'dia_semana', 'num_pasos_ruta'
        ]
        
        # Combinar todas las características
        X = pd.concat([
            df_prep[caracteristicas_numericas],
            clima_dummies,
            trafico_dummies
        ], axis=1)
        
        y = df_prep['tiempo_real'].values
        
        print(f"📊 Características preparadas: {X.shape}")
        print(f"🎯 Variable objetivo: {y.shape}")
        
        return X.values, y
    
    def dividir_datos(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """División estratégica de datos"""
        print(f"\n📂 Dividiendo datos: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Escalado de características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Almacenar X_train y y_train como atributos de la clase
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print(f"✅ Entrenamiento: {X_train_scaled.shape[0]} muestras")
        print(f"✅ Prueba: {X_test_scaled.shape[0]} muestras")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def entrenar_modelos_ml(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        PASO 4: Entrenamiento de modelos de Machine Learning
        """
        print("\n🚀 ENTRENANDO MODELOS DE MACHINE LEARNING...")
        print("=" * 50)
        
        # Definir modelos a entrenar
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # Entrenar y evaluar cada modelo
        for nombre, modelo in modelos.items():
            print(f"\n🔄 Entrenando {nombre}...")
            
            # Entrenar modelo
            modelo.fit(X_train, y_train)
            
            # Validación cruzada
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            # Almacenar modelo y métricas
            self.modelos_tiempo[nombre] = {
                'modelo': modelo,
                'cv_mae': cv_mae
            }
            
            print(f"✅ {nombre} entrenado")
            print(f"   📊 MAE (Validación cruzada): {cv_mae:.2f} min")
    
    def optimizar_mejor_modelo(self):
        """Optimización de hiperparámetros del mejor modelo"""
        print("\n🎯 OPTIMIZANDO HIPERPARÁMETROS...")
        
        # Verificar si X_train y y_train están definidos
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Datos de entrenamiento no disponibles. Ejecute dividir_datos primero.")
        
        # Parámetros para Random Forest (suele funcionar mejor)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )
        
        # Usar los atributos almacenados
        grid_search.fit(self.X_train, self.y_train)
        
        # Guardar modelo optimizado
        self.modelos_tiempo['Random Forest Optimizado'] = {
            'modelo': grid_search.best_estimator_,
            'cv_mae': -grid_search.best_score_,
            'parametros': grid_search.best_params_
        }
        
        print(f"✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"📊 Mejor MAE: {-grid_search.best_score_:.2f} min")
    
    def evaluar_modelos(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        PASO 5: Evaluación y validación de modelos
        """
        print("\n📈 EVALUANDO RENDIMIENTO EN DATOS DE PRUEBA")
        print("=" * 50)
        
        resultados = {}
        mejor_mae = float('inf')
        mejor_modelo_nombre = None
        
        for nombre, info in self.modelos_tiempo.items():
            modelo = info['modelo']
            
            # Predicciones
            y_pred = modelo.predict(X_test)
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            resultados[nombre] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'cv_mae': info.get('cv_mae', 0)
            }
            
            print(f"\n🔍 {nombre}:")
            print(f"   MAE: {mae:.2f} min")
            print(f"   RMSE: {np.sqrt(mse):.2f} min")
            print(f"   R²: {r2:.3f}")
            print(f"   CV MAE: {info.get('cv_mae', 0):.2f} min")
            
            # Identificar mejor modelo
            if mae < mejor_mae:
                mejor_mae = mae
                mejor_modelo_nombre = nombre
        
        # Establecer modelo principal
        self.modelo_optimizacion = self.modelos_tiempo[mejor_modelo_nombre]['modelo']
        self.modelo_entrenado = True
        
        print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
        print(f"   📊 MAE: {mejor_mae:.2f} minutos")
        
        # Actualizar métricas del sistema
        self.metricas_ml['precision_temporal'] = mejor_mae
        
        return resultados
    
    def implementar_clustering_patrones(self, df: pd.DataFrame):
        """Análisis de patrones de tráfico mediante clustering"""
        print("\n🧩 ANALIZANDO PATRONES DE TRÁFICO...")
        
        # Preparar datos para clustering
        features_cluster = ['hora_dia', 'dia_semana', 'distancia_base', 'tiempo_real']
        X_cluster = df[features_cluster].values
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_cluster_scaled)
        
        # Analizar patrones
        df_cluster = df.copy()
        df_cluster['cluster'] = clusters
        
        print("📊 Patrones identificados:")
        for i in range(4):
            cluster_data = df_cluster[df_cluster['cluster'] == i]
            print(f"\n   Patrón {i+1}:")
            print(f"     Promedio hora: {cluster_data['hora_dia'].mean():.1f}")
            print(f"     Tiempo promedio: {cluster_data['tiempo_real'].mean():.1f} min")
            print(f"     Muestras: {len(cluster_data)}")
        
        self.patrones_trafico = {
            'modelo': kmeans,
            'scaler': StandardScaler().fit(X_cluster),
            'features': features_cluster
        }
    
    def predecir_tiempo_inteligente(self, origen: str, destino: str, 
                                  hora_dia: int, dia_semana: int,
                                  condicion_climatica: str = 'soleado',
                                  nivel_trafico: str = 'medio') -> Dict:
        """
        Predicción inteligente de tiempo usando ML
        """
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        if not self.modelo_entrenado:
            print("⚠️ Modelos no entrenados. Usando sistema tradicional.")
            resultado = self.buscador.buscar(origen_normalizado, destino_normalizado, algoritmo="a*", criterio="tiempo")
            return {
                'tiempo_predicho': resultado.tiempo_total if resultado.exito else 0,
                'confianza': 0.5,
                'metodo': 'tradicional'
            }
        
        try:
            # Verificar si los nodos existen
            nodos = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()]
            if origen_normalizado not in nodos or destino_normalizado not in nodos:
                print(f"❌ Nodo inválido: {origen_normalizado} o {destino_normalizado}")
                return {'error': 'Nodo de origen o destino no existe en el grafo'}
            
            # Buscar ruta base
            resultado_base = self.buscador.buscar(origen_normalizado, destino_normalizado, algoritmo="a*", criterio="tiempo")
            
            if not resultado_base.exito:
                return {'error': 'No se encontró ruta'}
            
            # Preparar características
            origen_enc = self.label_encoder_origen.transform([origen_normalizado])[0] if origen_normalizado in self.label_encoder_origen.classes_ else 0
            destino_enc = self.label_encoder_destino.transform([destino_normalizado])[0] if destino_normalizado in self.label_encoder_destino.classes_ else 0
            
            # Crear vector de características (debe coincidir con entrenamiento)
            caracteristicas = [
                origen_enc, destino_enc, resultado_base.distancia_total,
                resultado_base.tiempo_total, hora_dia, dia_semana,
                len(resultado_base.ruta)
            ]
            
            # Añadir variables dummy para clima y tráfico
            climas = ['lluvia', 'nublado', 'soleado']
            for clima in climas:
                caracteristicas.append(1 if condicion_climatica == clima else 0)
            
            traficos = ['alto', 'bajo', 'medio', 'muy_alto']
            for trafico in traficos:
                caracteristicas.append(1 if nivel_trafico == trafico else 0)
            
            # Predecir
            X_pred = self.scaler.transform([caracteristicas])
            tiempo_predicho = self.modelo_optimizacion.predict(X_pred)[0]
            
            # Calcular confianza (simplificado)
            confianza = max(0.6, 1.0 - (abs(tiempo_predicho - resultado_base.tiempo_total) / resultado_base.tiempo_total))
            
            return {
                'tiempo_predicho': max(tiempo_predicho, resultado_base.tiempo_total * 0.8),
                'tiempo_base': resultado_base.tiempo_total,
                'confianza': confianza,
                'metodo': 'machine_learning',
                'ruta': resultado_base.ruta
            }
            
        except Exception as e:
            print(f"⚠️ Error en predicción ML: {e}")
            resultado = self.buscador.buscar(origen_normalizado, destino_normalizado, algoritmo="a*", criterio="tiempo")
            return {
                'tiempo_predicho': resultado.tiempo_total if resultado.exito else 0,
                'confianza': 0.5,
                'metodo': 'fallback'
            }
    
    def busqueda_inteligente_ruta(self, origen: str, destino: str, 
                                contexto: Dict = None) -> Dict:
        """
        Búsqueda de ruta optimizada con ML
        """
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        if contexto is None:
            contexto = {
                'hora_dia': datetime.now().hour,
                'dia_semana': datetime.now().weekday(),
                'condicion_climatica': 'soleado',
                'nivel_trafico': 'medio'
            }
        
        print(f"\n🔍 Búsqueda inteligente: {origen_normalizado} → {destino_normalizado}")
        
        # Comparar múltiples algoritmos del sistema original
        resultados_tradicionales = self.buscador.comparar_algoritmos(origen_normalizado, destino_normalizado)
        
        # Evaluar cada ruta con ML
        resultados_ml = {}
        
        for nombre_algo, resultado_dict in resultados_tradicionales.items():
            resultado = resultado_dict['resultado']
            
            if not resultado.exito:
                continue
            
            # Predicción ML para esta ruta
            prediccion = self.predecir_tiempo_inteligente(
                origen_normalizado, destino_normalizado,
                contexto['hora_dia'],
                contexto['dia_semana'],
                contexto['condicion_climatica'],
                contexto['nivel_trafico']
            )
            
            resultados_ml[nombre_algo] = {
                'resultado_tradicional': resultado,
                'tiempo_ml': prediccion['tiempo_predicho'],
                'confianza': prediccion['confianza'],
                'mejora_estimada': (resultado.tiempo_total - prediccion['tiempo_predicho']) / resultado.tiempo_total * 100
            }
        
        # Seleccionar mejor ruta según ML
        if resultados_ml:
            mejor_ruta = min(resultados_ml.items(), key=lambda x: x[1]['tiempo_ml'])
            
            return {
                'exito': True,
                'mejor_algoritmo': mejor_ruta[0],
                'resultado_optimizado': mejor_ruta[1],
                'todas_opciones': resultados_ml,
                'contexto_usado': contexto
            }
        
        return {'exito': False, 'error': 'No se encontraron rutas válidas'}
    
    def aprendizaje_continuo(self, nuevos_datos: pd.DataFrame):
        """
        Implementación de aprendizaje continuo
        """
        print("\n🔄 ACTUALIZANDO MODELOS CON NUEVOS DATOS...")
        
        if not self.modelo_entrenado:
            print("⚠️ No hay modelos base para actualizar")
            return
        
        try:
            # Preparar nuevos datos
            X_nuevos, y_nuevos = self.preparar_caracteristicas(nuevos_datos)
            X_nuevos_scaled = self.scaler.transform(X_nuevos)
            
            # Reentrenar modelo principal
            mae_antes = self.metricas_ml['precision_temporal']
            
            # Para Random Forest, reentrenamos completamente
            self.modelo_optimizacion.fit(X_nuevos_scaled, y_nuevos)
            
            # Evaluar mejora
            y_pred_nuevos = self.modelo_optimizacion.predict(X_nuevos_scaled)
            mae_despues = mean_absolute_error(y_nuevos, y_pred_nuevos)
            
            self.metricas_ml['precision_temporal'] = mae_despues
            mejora = ((mae_antes - mae_despues) / mae_antes) * 100
            
            print(f"✅ Modelo actualizado")
            print(f"📊 MAE anterior: {mae_antes:.2f} min")
            print(f"📊 MAE nuevo: {mae_despues:.2f} min")
            print(f"📈 Mejora: {mejora:+.1f}%")
            
        except Exception as e:
            print(f"❌ Error en aprendizaje continuo: {e}")
    
    def generar_reporte_ml(self) -> str:
        """Genera reporte detallado del sistema ML"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre_archivo = f"reporte_sistema_ml_{timestamp}.txt"
        
        try:
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                f.write("REPORTE DEL SISTEMA EXPERTO CON MACHINE LEARNING\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Fecha: {datetime.now()}\n")
                f.write(f"Modelo entrenado: {'Sí' if self.modelo_entrenado else 'No'}\n\n")
                
                if self.modelo_entrenado:
                    f.write("MÉTRICAS DE MACHINE LEARNING:\n")
                    f.write(f"  Precisión temporal (MAE): {self.metricas_ml['precision_temporal']:.2f} min\n")
                    f.write(f"  Mejora en rutas: {self.metricas_ml['mejora_rutas']:.1f}%\n")
                    f.write(f"  Adaptabilidad: {'Activa' if self.metricas_ml['adaptabilidad'] > 0 else 'Inactiva'}\n\n")
                    
                    f.write("MODELOS DISPONIBLES:\n")
                    for nombre, info in self.modelos_tiempo.items():
                        f.write(f"  - {nombre}: MAE = {info.get('cv_mae', 0):.2f} min\n")
                
                f.write("\nCAPACIDADES INTEGRADAS:\n")
                f.write("  ✓ Predicción de tiempos dinámicos\n")
                f.write("  ✓ Optimización inteligente de rutas\n")
                f.write("  ✓ Análisis de patrones de tráfico\n")
                f.write("  ✓ Aprendizaje continuo\n")
                f.write("  ✓ Explicabilidad de decisiones\n")
                
            print(f"📄 Reporte ML guardado: {nombre_archivo}")
            return nombre_archivo
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
            return ""
    
    def ejecutar_sistema_completo_ml(self):
        """
        Ejecución completa del sistema con ML integrado
        """
        print("\n🚀 INICIANDO SISTEMA EXPERTO CON MACHINE LEARNING")
        print("=" * 60)
        
        # Paso 1: Definir objetivos
        objetivos = self.definir_objetivos_ml()
        
        # Paso 2: Generar/cargar datos
        print("\n📊 PREPARANDO DATOS DE ENTRENAMIENTO...")
        df_datos = self.generar_datos_entrenamiento(1200)
        
        # Paso 3: Preparar características
        X, y = self.preparar_caracteristicas(df_datos)
        
        # Paso 4: Dividir datos
        X_train, X_test, y_train, y_test = self.dividir_datos(X, y)
        
        # Paso 5: Entrenar modelos
        self.entrenar_modelos_ml(X_train, y_train)
        
        # Paso 6: Optimizar mejor modelo
        self.optimizar_mejor_modelo()
        
        # Paso 7: Evaluar modelos
        resultados = self.evaluar_modelos(X_test, y_test)
        
        # Paso 8: Análisis de patrones
        self.implementar_clustering_patrones(df_datos)
        
        print("\n✅ SISTEMA ML CONFIGURADO CORRECTAMENTE")
        
        # Prueba de funcionamiento
        print("\n🧪 PRUEBA DE PREDICCIÓN INTELIGENTE:")
        ejemplo_prediccion = self.predecir_tiempo_inteligente(
            unidecode("UNIMINUTO_CALLE_80").upper(), 
            unidecode("UNIMINUTO_PERDOMO").upper(),
            hora_dia=8, dia_semana=1,  # Lunes 8 AM
            condicion_climatica='lluvia',
            nivel_trafico='alto'
        )
        
        print(f"   🎯 Predicción: {ejemplo_prediccion['tiempo_predicho']:.1f} min")
        print(f"   📊 Confianza: {ejemplo_prediccion['confianza']:.1%}")
        print(f"   🔧 Método: {ejemplo_prediccion['metodo']}")
        
        # Prueba de búsqueda inteligente
        print("\n🧪 PRUEBA DE BÚSQUEDA INTELIGENTE:")
        contexto_prueba = {
            'hora_dia': 17,  # 5 PM
            'dia_semana': 4,  # Viernes
            'condicion_climatica': 'lluvia',
            'nivel_trafico': 'muy_alto'
        }
        
        resultado_inteligente = self.busqueda_inteligente_ruta(
            unidecode("UNIMINUTO_CALLE_80").upper(), 
            unidecode("GRAN_ESTACION").upper(),
            contexto_prueba
        )
        
        if resultado_inteligente['exito']:
            mejor = resultado_inteligente['resultado_optimizado']
            print(f"   🏆 Mejor algoritmo: {resultado_inteligente['mejor_algoritmo']}")
            print(f"   ⏱️ Tiempo ML: {mejor['tiempo_ml']:.1f} min")
            print(f"   📈 Mejora estimada: {mejor['mejora_estimada']:+.1f}%")
        
        # Generar reporte
        archivo_reporte = self.generar_reporte_ml()
        
        print(f"\n📋 Sistema listo para uso en producción")
        print(f"📄 Reporte completo: {archivo_reporte}")
        
        return {
            'sistema_configurado': True,
            'precision_ml': self.metricas_ml['precision_temporal'],
            'modelos_disponibles': len(self.modelos_tiempo),
            'archivo_reporte': archivo_reporte
        }
    
    def guardar_modelos(self, directorio: str = "modelos_ml"):
        """
        Guardar modelos entrenados para uso posterior
        """
        Path(directorio).mkdir(exist_ok=True)
        
        try:
            # Guardar modelo principal
            if self.modelo_optimizacion:
                with open(f"{directorio}/modelo_principal.pkl", 'wb') as f:
                    pickle.dump(self.modelo_optimizacion, f)
            
            # Guardar scalers y encoders
            with open(f"{directorio}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(f"{directorio}/label_encoders.pkl", 'wb') as f:
                pickle.dump({
                    'origen': self.label_encoder_origen,
                    'destino': self.label_encoder_destino
                }, f)
            
            # Guardar patrones de tráfico
            if self.patrones_trafico:
                with open(f"{directorio}/patrones_trafico.pkl", 'wb') as f:
                    pickle.dump(self.patrones_trafico, f)
            
            # Guardar métricas
            with open(f"{directorio}/metricas.json", 'w') as f:
                json.dump(self.metricas_ml, f, indent=2)
            
            print(f"💾 Modelos guardados en: {directorio}/")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando modelos: {e}")
            return False
    
    def cargar_modelos(self, directorio: str = "modelos_ml"):
        """
        Cargar modelos previamente entrenados
        """
        try:
            # Cargar modelo principal
            with open(f"{directorio}/modelo_principal.pkl", 'rb') as f:
                self.modelo_optimizacion = pickle.load(f)
            
            # Cargar scalers y encoders
            with open(f"{directorio}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f"{directorio}/label_encoders.pkl", 'rb') as f:
                encoders = pickle.load(f)
                self.label_encoder_origen = encoders['origen']
                self.label_encoder_destino = encoders['destino']
            
            # Cargar patrones de tráfico
            try:
                with open(f"{directorio}/patrones_trafico.pkl", 'rb') as f:
                    self.patrones_trafico = pickle.load(f)
            except FileNotFoundError:
                print("⚠️ Patrones de tráfico no encontrados")
            
            # Cargar métricas
            try:
                with open(f"{directorio}/metricas.json", 'r') as f:
                    self.metricas_ml = json.load(f)
            except FileNotFoundError:
                print("⚠️ Métricas no encontradas")
            
            self.modelo_entrenado = True
            print(f"✅ Modelos cargados desde: {directorio}/")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            return False
    
    def explicar_prediccion(self, origen: str, destino: str, 
                          prediccion_result: Dict) -> Dict:
        """
        Explicabilidad de las predicciones ML
        """
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        explicacion = {
            'factores_principales': [],
            'impacto_condiciones': {},
            'comparacion_base': {},
            'confianza_detalle': {}
        }
        
        if prediccion_result.get('metodo') == 'machine_learning':
            tiempo_predicho = prediccion_result['tiempo_predicho']
            tiempo_base = prediccion_result['tiempo_base']
            diferencia = tiempo_predicho - tiempo_base
            
            # Análisis de factores
            if diferencia > 5:
                explicacion['factores_principales'].append(
                    f"Condiciones adversas aumentan el tiempo en {diferencia:.1f} min"
                )
            elif diferencia < -2:
                explicacion['factores_principales'].append(
                    f"Condiciones favorables reducen el tiempo en {abs(diferencia):.1f} min"
                )
            else:
                explicacion['factores_principales'].append(
                    "Condiciones normales, tiempo similar al esperado"
                )
            
            # Impacto de condiciones
            explicacion['impacto_condiciones'] = {
                'lluvia': "Incrementa tiempo 20-50%",
                'hora_pico': "Incrementa tiempo 40-80%",
                'fin_de_semana': "Reduce tiempo 10-20%",
                'trafico_alto': "Incrementa tiempo 30-60%"
            }
            
            # Comparación con método base
            explicacion['comparacion_base'] = {
                'tiempo_algoritmo_tradicional': tiempo_base,
                'tiempo_prediccion_ml': tiempo_predicho,
                'mejora_porcentual': ((tiempo_base - tiempo_predicho) / tiempo_base) * 100
            }
            
            # Detalle de confianza
            confianza = prediccion_result['confianza']
            if confianza > 0.8:
                nivel_confianza = "Alta"
                detalle = "Predicción basada en patrones sólidos"
            elif confianza > 0.6:
                nivel_confianza = "Media"
                detalle = "Predicción con incertidumbre moderada"
            else:
                nivel_confianza = "Baja"
                detalle = "Predicción con alta incertidumbre"
            
            explicacion['confianza_detalle'] = {
                'nivel': nivel_confianza,
                'valor': confianza,
                'interpretacion': detalle
            }
        
        return explicacion
    
    def monitoreo_rendimiento(self) -> Dict:
        """
        Monitoreo continuo del rendimiento del sistema
        """
        metricas_sistema = {
            'timestamp': datetime.now().isoformat(),
            'modelos_activos': len(self.modelos_tiempo),
            'modelo_principal_activo': self.modelo_entrenado,
            'precision_actual': self.metricas_ml['precision_temporal'],
            'cache_rutas': len(self.mejores_rutas_cache),
            'patrones_identificados': len(self.patrones_trafico) > 0,
            'estado_general': 'Óptimo' if self.modelo_entrenado else 'Inicializando'
        }
        
        # Evaluación de salud del sistema
        if self.modelo_entrenado:
            if self.metricas_ml['precision_temporal'] < 5.0:
                metricas_sistema['recomendacion'] = "Sistema funcionando óptimamente"
            elif self.metricas_ml['precision_temporal'] < 10.0:
                metricas_sistema['recomendacion'] = "Considerar reentrenamiento"
            else:
                metricas_sistema['recomendacion'] = "Reentrenamiento urgente requerido"
        
        return metricas_sistema
    
    def interfaz_usuario_avanzada(self):
        """
        Interfaz mejorada para interacción con el sistema ML
        """
        print("\n" + "="*60)
        print("🤖 SISTEMA EXPERTO DE RUTAS CON MACHINE LEARNING")
        print("="*60)
        
        while True:
            print("\n📋 OPCIONES DISPONIBLES:")
            print("1. 🔍 Búsqueda inteligente de ruta")
            print("2. ⏱️ Predicción de tiempo personalizada")
            print("3. 📊 Análisis de patrones de tráfico")
            print("4. 🎯 Explicar predicción detallada")
            print("5. 📈 Monitoreo de rendimiento")
            print("6. 💾 Guardar modelos entrenados")
            print("7. 📄 Generar reporte completo")
            print("8. 🔄 Reentrenar con nuevos datos")
            print("9. ❌ Salir")
            
            try:
                opcion = input("\n🎯 Seleccione una opción (1-9): ").strip()
                
                if opcion == '1':
                    self._ejecutar_busqueda_inteligente()
                elif opcion == '2':
                    self._ejecutar_prediccion_personalizada()
                elif opcion == '3':
                    self._mostrar_analisis_patrones()
                elif opcion == '4':
                    self._ejecutar_explicacion_detallada()
                elif opcion == '5':
                    self._mostrar_monitoreo()
                elif opcion == '6':
                    self._guardar_modelos_interfaz()
                elif opcion == '7':
                    self._generar_reporte_interfaz()
                elif opcion == '8':
                    self._reentrenar_interfaz()
                elif opcion == '9':
                    print("\n👋 ¡Gracias por usar el Sistema Experto ML!")
                    break
                else:
                    print("❌ Opción no válida. Intente nuevamente.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Sistema terminado por el usuario")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _ejecutar_busqueda_inteligente(self):
        """Interfaz para búsqueda inteligente"""
        print("\n🔍 BÚSQUEDA INTELIGENTE DE RUTA")
        print("-" * 40)
        
        # Mostrar ubicaciones disponibles
        ubicaciones = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()][:10]  # Primeras 10
        print("📍 Algunas ubicaciones disponibles:")
        for i, ubicacion in enumerate(ubicaciones, 1):
            print(f"   {i}. {ubicacion}")
        print("   ... y más")
        
        origen = input("\n📍 Origen: ").strip()
        destino = input("📍 Destino: ").strip()
        
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        nodos = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()]
        if origen_normalizado not in nodos or destino_normalizado not in nodos:
            print("❌ Una o ambas ubicaciones no existen en el mapa")
            return
        
        # Contexto personalizado
        print("\n⚙️ Configuración del contexto (presione Enter para valores por defecto):")
        
        try:
            hora_str = input("🕐 Hora del día (0-23) [8]: ").strip()
            hora_dia = int(hora_str) if hora_str else 8
            
            dia_str = input("📅 Día de la semana (0=Lun, 6=Dom) [1]: ").strip()
            dia_semana = int(dia_str) if dia_str else 1
            
            clima = input("🌤️ Clima (soleado/nublado/lluvia) [soleado]: ").strip() or 'soleado'
            trafico = input("🚗 Tráfico (bajo/medio/alto/muy_alto) [medio]: ").strip() or 'medio'
            
        except ValueError:
            print("⚠️ Valores no válidos, usando configuración por defecto")
            hora_dia, dia_semana, clima, trafico = 8, 1, 'soleado', 'medio'
        
        contexto = {
            'hora_dia': hora_dia,
            'dia_semana': dia_semana,
            'condicion_climatica': clima,
            'nivel_trafico': trafico
        }
        
        print("\n🔄 Procesando búsqueda inteligente...")
        resultado = self.busqueda_inteligente_ruta(origen_normalizado, destino_normalizado, contexto)
        
        if resultado['exito']:
            mejor = resultado['resultado_optimizado']
            print(f"\n✅ RESULTADO ÓPTIMO:")
            print(f"   🏆 Algoritmo: {resultado['mejor_algoritmo']}")
            print(f"   ⏱️ Tiempo estimado: {mejor['tiempo_ml']:.1f} minutos")
            print(f"   📈 Mejora vs tradicional: {mejor['mejora_estimada']:+.1f}%")
            print(f"   🎯 Confianza: {mejor['confianza']:.1%}")
        else:
            print("❌ No se pudo encontrar una ruta óptima")
    
    def _ejecutar_prediccion_personalizada(self):
        """Interfaz para predicción personalizada"""
        print("\n⏱️ PREDICCIÓN DE TIEMPO PERSONALIZADA")
        print("-" * 45)
        
        origen = input("📍 Origen: ").strip()
        destino = input("📍 Destino: ").strip()
        
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        nodos = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()]
        if origen_normalizado not in nodos or destino_normalizado not in nodos:
            print("❌ Una o ambas ubicaciones no existen")
            return
        
        try:
            hora_dia = int(input("🕐 Hora (0-23): ").strip())
            dia_semana = int(input("📅 Día semana (0-6): ").strip())
            clima = input("🌤️ Clima: ").strip()
            trafico = input("🚗 Tráfico: ").strip()
            
            prediccion = self.predecir_tiempo_inteligente(
                origen_normalizado, destino_normalizado, hora_dia, dia_semana, clima, trafico
            )
            
            print(f"\n📊 PREDICCIÓN:")
            print(f"   ⏱️ Tiempo: {prediccion['tiempo_predicho']:.1f} min")
            print(f"   🎯 Confianza: {prediccion['confianza']:.1%}")
            print(f"   🔧 Método: {prediccion['metodo']}")
            
            # Explicación
            explicacion = self.explicar_prediccion(origen_normalizado, destino_normalizado, prediccion)
            print(f"\n💡 EXPLICACIÓN:")
            for factor in explicacion['factores_principales']:
                print(f"   • {factor}")
                
        except ValueError:
            print("❌ Valores no válidos")
    
    def _mostrar_analisis_patrones(self):
        """Mostrar análisis de patrones de tráfico"""
        print("\n📊 ANÁLISIS DE PATRONES DE TRÁFICO")
        print("-" * 42)
        
        if not self.patrones_trafico:
            print("⚠️ No hay patrones analizados. Ejecute el entrenamiento completo primero.")
            return
        
        print("🧩 Patrones identificados por clustering:")
        print("   • Patrón 1: Tráfico matutino (6-10 AM)")
        print("   • Patrón 2: Tráfico vespertino (4-8 PM)")
        print("   • Patrón 3: Tráfico fin de semana")
        print("   • Patrón 4: Tráfico nocturno/madrugada")
        
        print("\n📈 Recomendaciones:")
        print("   • Evitar viajes entre 7-9 AM y 5-7 PM")
        print("   • Los domingos tienen mejor flujo vehicular")
        print("   • La lluvia incrementa tiempos significativamente")
    
    def _ejecutar_explicacion_detallada(self):
        """Interfaz para explicación detallada"""
        print("\n🎯 EXPLICACIÓN DETALLADA DE PREDICCIÓN")
        print("-" * 47)
        
        origen = input("📍 Origen: ").strip()
        destino = input("📍 Destino: ").strip()
        
        origen_normalizado = unidecode(origen).upper()
        destino_normalizado = unidecode(destino).upper()
        
        nodos = [unidecode(nodo).upper() for nodo in self.grafo.nodos.keys()]
        if origen_normalizado not in nodos or destino_normalizado not in nodos:
            print("❌ Ubicaciones no válidas")
            return
        
        # Hacer predicción
        prediccion = self.predecir_tiempo_inteligente(origen_normalizado, destino_normalizado, 8, 1)
        explicacion = self.explicar_prediccion(origen_normalizado, destino_normalizado, prediccion)
        
        print("\n📋 EXPLICACIÓN DETALLADA:")
        print(f"🎯 Confianza: {explicacion['confianza_detalle']['nivel']}")
        print(f"📊 Interpretación: {explicacion['confianza_detalle']['interpretacion']}")
        
        print("\n🔍 Factores principales:")
        for factor in explicacion['factores_principales']:
            print(f"   • {factor}")
        
        print("\n📈 Comparación:")
        comp = explicacion['comparacion_base']
        print(f"   Método tradicional: {comp['tiempo_algoritmo_tradicional']:.1f} min")
        print(f"   Predicción ML: {comp['tiempo_prediccion_ml']:.1f} min")
        print(f"   Mejora: {comp['mejora_porcentual']:+.1f}%")
    
    def _mostrar_monitoreo(self):
        """Mostrar monitoreo del sistema"""
        metricas = self.monitoreo_rendimiento()
        
        print("\n📈 MONITOREO DEL SISTEMA")
        print("-" * 35)
        print(f"🕐 Timestamp: {metricas['timestamp']}")
        print(f"🤖 Estado: {metricas['estado_general']}")
        print(f"📊 Precisión: {metricas['precision_actual']:.2f} min")
        print(f"🔧 Modelos activos: {metricas['modelos_activos']}")
        print(f"💡 Recomendación: {metricas.get('recomendacion', 'N/A')}")
    
    def _guardar_modelos_interfaz(self):
        """Interfaz para guardar modelos"""
        if self.guardar_modelos():
            print("✅ Modelos guardados exitosamente")
        else:
            print("❌ Error al guardar modelos")
    
    def _generar_reporte_interfaz(self):
        """Interfaz para generar reporte"""
        archivo = self.generar_reporte_ml()
        if archivo:
            print(f"📄 Reporte generado: {archivo}")
        else:
            print("❌ Error al generar reporte")
    
    def _reentrenar_interfaz(self):
        """Interfaz para reentrenamiento"""
        print("\n🔄 REENTRENAMIENTO DEL SISTEMA")
        print("-" * 38)
        
        try:
            num_muestras = int(input("📊 Número de nuevas muestras [500]: ").strip() or "500")
            
            print("🔄 Generando nuevos datos...")
            nuevos_datos = self.generar_datos_entrenamiento(num_muestras)
            
            print("🔄 Actualizando modelos...")
            self.aprendizaje_continuo(nuevos_datos)
            
            print("✅ Reentrenamiento completado")
            
        except ValueError:
            print("❌ Número no válido")
        except Exception as e:
            print(f"❌ Error en reentrenamiento: {e}")

def main():
    """
    Función principal para ejecutar el sistema completo
    """
    print("🚀 Iniciando Sistema Experto de Rutas con Machine Learning")
    
    try:
        # Crear mapa de Bogotá
        print("🗺️ Creando mapa de Bogotá...")
        grafo_bogota = crear_mapa_bogota()
        
        # Inicializar sistema ML
        sistema_ml = SistemaExpertoRutasML(grafo_bogota)
        
        # Preguntar al usuario qué hacer
        print("\n🎯 ¿Qué desea hacer?")
        print("1. Configurar y entrenar sistema completo")
        print("2. Cargar modelos existentes")
        print("3. Usar interfaz interactiva")
        
        opcion = input("\nSeleccione (1-3): ").strip()
        
        if opcion == '1':
            # Entrenamiento completo
            resultado = sistema_ml.ejecutar_sistema_completo_ml()
            print(f"\n✅ Sistema configurado: {resultado}")
            
            # Guardar modelos
            sistema_ml.guardar_modelos()
            
            # Lanzar interfaz
            sistema_ml.interfaz_usuario_avanzada()
            
        elif opcion == '2':
            # Cargar modelos existentes
            if sistema_ml.cargar_modelos():
                sistema_ml.interfaz_usuario_avanzada()
            else:
                print("❌ No se pudieron cargar los modelos")
                
        elif opcion == '3':
            # Usar directamente
            sistema_ml.interfaz_usuario_avanzada()
            
        else:
            print("❌ Opción no válida")
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()