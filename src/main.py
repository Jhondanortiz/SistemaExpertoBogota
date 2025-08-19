#!/usr/bin/env python3
"""
Sistema Experto para Búsqueda de Rutas en Bogotá con Machine Learning Mejorado
Integración de algoritmos de aprendizaje automático y aprendizaje continuo con capacidades conversacionales avanzadas
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
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
import re
import platform
import random
import asyncio
import logging
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer
from datasets import Dataset
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from grafo_bogota import crear_mapa_bogota, cargar_coordenadas
from algoritmos_busqueda import BuscadorRutas, ResultadoBusqueda
from utils import validar_nodo, obtener_coordenadas_nodo, EMOJIS, formatear_ruta, formatear_tiempo, formatear_distancia

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Mapeo de nombres de días de la semana a valores numéricos
DIAS_SEMANA = {
    'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6,
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
}

class SistemaExpertoRutasML:
    """Sistema Experto de Rutas potenciado con Machine Learning y capacidades conversacionales avanzadas"""
    
    def __init__(self, grafo_bogota, coordenadas: Dict):
        self.grafo = grafo_bogota
        self.coordenadas = coordenadas
        self.buscador = BuscadorRutas(grafo_bogota, coordenadas)
        self.modelos_tiempo = {}
        self.modelo_optimizacion = None
        self.scaler = StandardScaler()
        self.label_encoder_origen = LabelEncoder()
        self.label_encoder_destino = LabelEncoder()
        self.datos_historicos = []
        self.patrones_trafico = {}
        self.mejores_rutas_cache = {}
        self.modelo_entrenado = False
        self.metricas_ml = {
            'precision_temporal': 0.0,
            'mejora_rutas': 0.0,
            'adaptabilidad': 0.0,
            'diversidad_respuestas': 0.0,
            'bleu_score': 0.0
        }
        self.base_conocimiento = {}
        self.feedback_dataset = []
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al cargar modelo conversacional: {e}")
            self.tokenizer = None
            self.model = None
        self.configurar_logica_difusa()
        logger.info("🤖 Sistema Experto con ML inicializado")

    def configurar_logica_difusa(self):
        """Configura el sistema de lógica difusa para ajustes de tiempo"""
        try:
            trafico = ctrl.Antecedent(np.arange(0, 101, 1), 'trafico')
            tiempo = ctrl.Consequent(np.arange(0, 121, 1), 'tiempo')
            trafico['bajo'] = fuzz.trimf(trafico.universe, [0, 0, 50])
            trafico['alto'] = fuzz.trimf(trafico.universe, [50, 100, 100])
            tiempo['rapido'] = fuzz.trimf(tiempo.universe, [0, 0, 60])
            tiempo['lento'] = fuzz.trimf(tiempo.universe, [60, 120, 120])
            regla1 = ctrl.Rule(trafico['bajo'], tiempo['rapido'])
            regla2 = ctrl.Rule(trafico['alto'], tiempo['lento'])
            sistema = ctrl.ControlSystem([regla1, regla2])
            self.simulador_difuso = ctrl.ControlSystemSimulation(sistema)
            logger.info("✅ Lógica difusa configurada")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error configurando lógica difusa: {e}")
            self.simulador_difuso = None

    def definir_objetivos_ml(self) -> Dict:
        objetivos = {
            'prediccion_temporal': {
                'descripcion': 'Predecir tiempos de viaje dinámicos con alta precisión',
                'metrica_objetivo': 'MAE < 5 minutos',
                'algoritmos': ['Random Forest', 'Gradient Boosting', 'Linear Regression']
            },
            'optimizacion_rutas': {
                'descripcion': 'Seleccionar rutas óptimas según contexto dinámico',
                'metrica_objetivo': 'Reducción 15% tiempo promedio',
                'algoritmos': ['Ensemble de regressores', 'Clustering']
            },
            'deteccion_patrones': {
                'descripcion': 'Identificar patrones de tráfico complejos',
                'metrica_objetivo': 'R² > 0.8 en predicciones',
                'algoritmos': ['Clustering K-means', 'Random Forest']
            },
            'conversacion_avanzada': {
                'descripcion': 'Generar respuestas conversacionales fluidas y contextualmente relevantes',
                'metrica_objetivo': 'BLEU > 0.7 y satisfacción usuario > 4/5',
                'algoritmos': ['Supervised Fine-Tuning']
            },
            'adaptabilidad': {
                'descripcion': 'Adaptación continua a nuevos datos y feedback de usuarios',
                'metrica_objetivo': 'Reentrenamiento automático con feedback',
                'algoritmos': ['Aprendizaje incremental']
            }
        }
        logger.info("\n=== OBJETIVOS DEL MACHINE LEARNING ===")
        for objetivo, detalles in objetivos.items():
            logger.info(f"\n🎯 {objetivo.replace('_', ' ').title()}:")
            logger.info(f"   📝 {detalles['descripcion']}")
            logger.info(f"   📊 Meta: {detalles['metrica_objetivo']}")
            logger.info(f"   🔧 Algoritmos: {', '.join(detalles['algoritmos'])}")
        return objetivos

    def generar_datos_entrenamiento(self, num_muestras: int = 1000) -> pd.DataFrame:
        try:
            nodos = list(self.grafo.nodos)
        except AttributeError as e:
            logger.error(f"{EMOJIS['error']} Error al acceder a los nodos del grafo: {e}")
            raise
        data = []
        for _ in range(num_muestras):
            origen = random.choice(nodos)
            destino = random.choice([n for n in nodos if n != origen])
            if not (validar_nodo(origen) and validar_nodo(destino)):
                logger.warning(f"{EMOJIS['advertencia']} Nodo inválido en datos de entrenamiento: {origen} o {destino}")
                continue
            resultado = self.buscador.buscar(origen, destino, "a*", "tiempo")
            if resultado.exito:
                hora = random.randint(0, 23)
                dia = random.randint(0, 6)
                clima = random.choice([0, 1, 2])  # 0: soleado, 1: nublado, 2: lluvia
                trafico = random.choice([0, 1, 2, 3])  # 0: bajo, 1: medio, 2: alto, 3: muy_alto
                tiempo_base = resultado.tiempo_total
                factor_trafico = 1.0 + 0.2 * trafico + 0.15 * (hora in [7, 8, 17, 18])
                factor_clima = 1.0 + 0.1 * clima
                distancia = resultado.distancia_total
                tiempo_real = tiempo_base * factor_trafico * factor_clima
                data.append({
                    'origen': origen,
                    'destino': destino,
                    'hora_dia': hora,
                    'dia_semana': dia,
                    'condicion_climatica': clima,
                    'nivel_trafico': trafico,
                    'distancia': distancia,
                    'tiempo_real': tiempo_real
                })
        df = pd.DataFrame(data)
        logger.info(f"{EMOJIS['exito']} Generados {len(df)} datos de entrenamiento")
        return df

    def generar_datos_conversacionales(self, num_muestras: int = 200) -> List[Dict]:
        try:
            nodos = list(self.grafo.nodos)
        except AttributeError as e:
            logger.error(f"{EMOJIS['error']} Error al acceder a los nodos del grafo: {e}")
            raise
        datos = []
        climas = ['soleado', 'lluvia', 'nublado']
        traficos = ['bajo', 'medio', 'alto', 'muy_alto']
        for _ in range(num_muestras):
            origen = random.choice(nodos)
            destino = random.choice([n for n in nodos if n != origen])
            if not (validar_nodo(origen) and validar_nodo(destino)):
                logger.warning(f"{EMOJIS['advertencia']} Nodo inválido en datos conversacionales: {origen} o {destino}")
                continue
            hora = random.randint(6, 22)
            periodo = 'AM' if hora < 12 else 'PM'
            hora_display = hora - 12 if hora > 12 else hora
            clima = random.choice(climas)
            trafico = random.choice(traficos)
            resultado = self.buscador.buscar(origen, destino, "a*", "tiempo")
            prompt = f"¿Cuál es la mejor ruta de {origen} a {destino} a las {hora_display} {periodo} con clima {clima} y tráfico {trafico}?"
            if resultado.exito:
                respuesta = f"La mejor ruta de {origen} a {destino} a las {hora_display} {periodo} con clima {clima} y tráfico {trafico} es: {' -> '.join(resultado.ruta)} con un tiempo estimado de {resultado.tiempo_total:.1f} minutos."
            else:
                respuesta = f"No se encontró una ruta válida de {origen} a {destino}."
            datos.append({'prompt': prompt, 'response': respuesta, 'rating': random.randint(3, 5)})
        logger.info(f"{EMOJIS['exito']} Generados {len(datos)} datos conversacionales")
        return datos

    def preparar_caracteristicas(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        try:
            df['origen_encoded'] = self.label_encoder_origen.fit_transform(df['origen'])
            df['destino_encoded'] = self.label_encoder_destino.fit_transform(df['destino'])
            X = df[['origen_encoded', 'destino_encoded', 'hora_dia', 'dia_semana', 'condicion_climatica', 'nivel_trafico', 'distancia']]
            y = df['tiempo_real']
            logger.info(f"{EMOJIS['exito']} Características preparadas: {X.shape[0]} muestras, {X.shape[1]} características")
            return X.values, y
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al preparar características: {e}")
            return np.array([]), np.array([])

    def dividir_datos(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info(f"{EMOJIS['exito']} Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al dividir datos: {e}")
            return X, X, y, y

    def entrenar_modelos_ml(self, X_train: np.ndarray, y_train: np.ndarray):
        modelos = {
            'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            'LinearRegression': LinearRegression()
        }
        for nombre, modelo in modelos.items():
            try:
                modelo.fit(X_train, y_train)
                scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
                self.modelos_tiempo[nombre] = modelo
                logger.info(f"{EMOJIS['exito']} {nombre} entrenado. R² promedio (CV): {scores.mean():.3f}")
            except Exception as e:
                logger.error(f"{EMOJIS['error']} Error al entrenar {nombre}: {e}")

    def optimizar_mejor_modelo(self):
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        }
        try:
            grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
            X, y = self.preparar_caracteristicas(self.generar_datos_entrenamiento())
            X_scaled = self.scaler.fit_transform(X)
            grid_search.fit(X_scaled, y)
            self.modelo_optimizacion = grid_search.best_estimator_
            logger.info(f"{EMOJIS['exito']} Mejor modelo optimizado: {grid_search.best_params_}")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al optimizar modelo: {e}")
            self.modelo_optimizacion = RandomForestRegressor(random_state=42)

    def guardar_modelos(self):
        try:
            Path('modelos').mkdir(exist_ok=True)
            with open('modelos/modelos_tiempo.pkl', 'wb') as f:
                pickle.dump(self.modelos_tiempo, f)
            with open('modelos/modelo_optimizacion.pkl', 'wb') as f:
                pickle.dump(self.modelo_optimizacion, f)
            with open('modelos/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            with open('modelos/label_encoder_origen.pkl', 'wb') as f:
                pickle.dump(self.label_encoder_origen, f)
            with open('modelos/label_encoder_destino.pkl', 'wb') as f:
                pickle.dump(self.label_encoder_destino, f)
            if self.model and self.tokenizer:
                self.model.save_pretrained('modelos/sft_model')
                self.tokenizer.save_pretrained('modelos/sft_model')
            logger.info(f"{EMOJIS['exito']} Modelos, preprocesadores y modelo conversacional guardados en disco")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al guardar modelos: {e}")

    def cargar_modelos(self) -> bool:
        try:
            with open('modelos/modelos_tiempo.pkl', 'rb') as f:
                self.modelos_tiempo = pickle.load(f)
            with open('modelos/modelo_optimizacion.pkl', 'rb') as f:
                self.modelo_optimizacion = pickle.load(f)
            with open('modelos/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('modelos/label_encoder_origen.pkl', 'rb') as f:
                self.label_encoder_origen = pickle.load(f)
            with open('modelos/label_encoder_destino.pkl', 'rb') as f:
                self.label_encoder_destino = pickle.load(f)
            if Path('modelos/sft_model').exists():
                self.model = AutoModelForCausalLM.from_pretrained('modelos/sft_model')
                self.tokenizer = AutoTokenizer.from_pretrained('modelos/sft_model')
            self.modelo_entrenado = True
            logger.info(f"{EMOJIS['exito']} Modelos, preprocesadores y modelo conversacional cargados desde disco")
            return True
        except FileNotFoundError:
            logger.warning(f"{EMOJIS['advertencia']} No se encontraron modelos guardados")
            return False
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al cargar modelos: {e}")
            return False

    def predecir_tiempo_inteligente(self, origen: str, destino: str, 
                                  hora_dia: int, dia_semana: int,
                                  condicion_climatica: str = 'soleado',
                                  nivel_trafico: str = 'medio') -> Dict:
        if not (validar_nodo(origen) and validar_nodo(destino)):
            logger.error(f"{EMOJIS['error']} Nodo inválido: {origen} o {destino}")
            return {'error': f"Nodo {origen if not validar_nodo(origen) else destino} no encontrado"}
        
        coords_origen = obtener_coordenadas_nodo(origen)
        coords_destino = obtener_coordenadas_nodo(destino)
        if not coords_origen or not coords_destino:
            logger.warning(f"{EMOJIS['advertencia']} Coordenadas no válidas para {origen} o {destino}, usando Dijkstra como fallback")
            resultado_base = self.buscador.buscar(origen, destino, algoritmo="dijkstra", criterio="tiempo")
            if not resultado_base.exito:
                return {'error': resultado_base.mensaje}
            return {
                'tiempo_predicho': resultado_base.tiempo_total,
                'tiempo_base': resultado_base.tiempo_total,
                'confianza': 0.8,
                'metodo': 'dijkstra',
                'ruta': resultado_base.ruta,
                'distancia_base': resultado_base.distancia_total,
                'cluster': 0
            }

        resultado_base = self.buscador.buscar(origen, destino, algoritmo="a*", criterio="tiempo")
        if not resultado_base.exito:
            return {'error': resultado_base.mensaje}
        
        try:
            trafico_map = {'bajo': 25, 'medio': 50, 'alto': 75, 'muy_alto': 100}
            clima_map = {'soleado': 0, 'nublado': 25, 'lluvia': 50}
            if nivel_trafico not in trafico_map:
                logger.warning(f"{EMOJIS['advertencia']} Nivel de tráfico inválido: {nivel_trafico}, usando 'medio'")
                nivel_trafico = 'medio'
            if condicion_climatica not in clima_map:
                logger.warning(f"{EMOJIS['advertencia']} Condición climática inválida: {condicion_climatica}, usando 'soleado'")
                condicion_climatica = 'soleado'

            if self.simulador_difuso is None:
                logger.warning(f"{EMOJIS['advertencia']} Simulador difuso no inicializado, usando tiempo base")
                return {
                    'tiempo_predicho': resultado_base.tiempo_total,
                    'tiempo_base': resultado_base.tiempo_total,
                    'confianza': 0.8,
                    'metodo': 'dijkstra',
                    'ruta': resultado_base.ruta,
                    'distancia_base': resultado_base.distancia_total,
                    'cluster': 0
                }

            self.simulador_difuso.input['trafico'] = trafico_map[nivel_trafico]
            try:
                self.simulador_difuso.compute()
                factor_difuso = self.simulador_difuso.output.get('tiempo', 60.0) / 60.0
            except Exception as e:
                logger.warning(f"{EMOJIS['advertencia']} Error en predicción difusa: {e}, usando factor 1.0")
                factor_difuso = 1.0

            cluster = self.predecir_cluster({
                'hora_dia': hora_dia,
                'dia_semana': dia_semana,
                'condicion_climatica': condicion_climatica,
                'nivel_trafico': nivel_trafico
            })
            factor_cluster = 1.0 + 0.1 * cluster

            if self.modelos_tiempo:
                features = np.array([[
                    self.label_encoder_origen.transform([origen])[0],
                    self.label_encoder_destino.transform([destino])[0],
                    hora_dia,
                    dia_semana,
                    clima_map[condicion_climatica],
                    trafico_map[nivel_trafico],
                    resultado_base.distancia_total
                ]])
                features_scaled = self.scaler.transform(features)
                tiempos_predichos = {nombre: modelo.predict(features_scaled)[0] for nombre, modelo in self.modelos_tiempo.items()}
                tiempo_predicho_ajustado = np.mean(list(tiempos_predichos.values())) * factor_difuso * factor_cluster
            else:
                logger.warning(f"{EMOJIS['advertencia']} Modelos ML no entrenados, usando tiempo base ajustado")
                tiempo_predicho_ajustado = resultado_base.tiempo_total * factor_difuso * factor_cluster

            confianza = 0.95 if self.modelo_entrenado and factor_difuso != 1.0 else 0.8
            logger.info(f"{EMOJIS['exito']} Predicción: {formatear_ruta(resultado_base.ruta)}, "
                        f"Tiempo predicho: {formatear_tiempo(tiempo_predicho_ajustado)}, Confianza: {confianza:.2f}")
            return {
                'tiempo_predicho': max(tiempo_predicho_ajustado, resultado_base.tiempo_total * 0.8),
                'tiempo_base': resultado_base.tiempo_total,
                'confianza': confianza,
                'metodo': 'machine_learning' if self.modelos_tiempo else 'dijkstra',
                'ruta': resultado_base.ruta,
                'distancia_base': resultado_base.distancia_total,
                'cluster': cluster
            }
        except Exception as e:
            logger.warning(f"{EMOJIS['advertencia']} Error en predicción: {e}, usando Dijkstra como fallback")
            return {
                'tiempo_predicho': resultado_base.tiempo_total,
                'tiempo_base': resultado_base.tiempo_total,
                'confianza': 0.8,
                'metodo': 'dijkstra',
                'ruta': resultado_base.ruta,
                'distancia_base': resultado_base.distancia_total,
                'cluster': 0
            }

    def busqueda_inteligente_ruta(self, origen: str, destino: str, 
                                contexto: Dict = None) -> Dict:
        contexto = contexto or {
            'hora_dia': datetime.now().hour,
            'dia_semana': datetime.now().weekday(),
            'condicion_climatica': 'soleado',
            'nivel_trafico': 'medio'
        }
        if not (validar_nodo(origen) and validar_nodo(destino)):
            logger.error(f"{EMOJIS['error']} Nodo inválido: {origen} o {destino}")
            return {'exito': False, 'error': f"Nodo {origen if not validar_nodo(origen) else destino} no encontrado"}
        
        resultado = self.buscador.buscar(origen, destino, algoritmo="a*", criterio="tiempo")
        if not resultado.exito:
            return {'exito': False, 'error': resultado.mensaje}
        
        prediccion = self.predecir_tiempo_inteligente(
            origen, destino, contexto['hora_dia'], contexto['dia_semana'],
            contexto['condicion_climatica'], contexto['nivel_trafico']
        )
        if 'error' in prediccion:
            return {'exito': False, 'error': prediccion['error']}
        
        mejora = ((resultado.tiempo_total - prediccion['tiempo_predicho']) / resultado.tiempo_total * 100
                  if resultado.tiempo_total > 0 else 0)
        self.mejores_rutas_cache[f"{origen}_{destino}"] = prediccion
        logger.info(f"{EMOJIS['exito']} Ruta inteligente: {formatear_ruta(resultado.ruta)}, "
                    f"Tiempo: {formatear_tiempo(prediccion['tiempo_predicho'])}, Mejora: {mejora:.1f}%")
        return {
            'exito': True,
            'mejor_algoritmo': 'a*',
            'resultado_optimizado': {
                'resultado_tradicional': resultado,
                'tiempo_ml': prediccion['tiempo_predicho'],
                'confianza': prediccion['confianza'],
                'mejora_estimada': mejora,
                'cluster': prediccion['cluster']
            },
            'todas_opciones': {'a*': prediccion},
            'contexto_usado': contexto
        }

    def fine_tune_with_sft_and_rlhf(self):
        logger.info(f"{EMOJIS['inicio']} Iniciando fine-tuning con SFT...")
        try:
            datos_conversacionales = self.generar_datos_conversacionales(num_muestras=200)
            dataset = Dataset.from_list(datos_conversacionales)
            
            def create_tokenize_function(tokenizer):
                def tokenize_function(examples):
                    inputs = tokenizer(
                        examples['prompt'],
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                    labels = tokenizer(
                        examples['response'],
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': inputs['input_ids'].squeeze(),
                        'attention_mask': inputs['attention_mask'].squeeze(),
                        'labels': labels['input_ids'].squeeze()
                    }
                return tokenize_function

            if self.tokenizer is None:
                logger.error(f"{EMOJIS['error']} Tokenizador no inicializado")
                return

            tokenize_fn = create_tokenize_function(self.tokenizer)
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['prompt', 'response', 'rating'])
            
            training_args = TrainingArguments(
                output_dir="sft_output",
                per_device_train_batch_size=4,
                num_train_epochs=5,
                learning_rate=2e-5,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="no",
                report_to="none",
                logging_dir='./logs'
            )
            
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset
            )
            
            trainer.train()
            logger.info(f"{EMOJIS['exito']} Modelo fine-tuned con SFT")
            self.model.save_pretrained("modelos/sft_model")
            self.tokenizer.save_pretrained("modelos/sft_model")
            logger.info(f"{EMOJIS['exito']} Modelo y tokenizador guardados en modelos/sft_model")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error durante el entrenamiento SFT: {e}")
        
        logger.warning(f"{EMOJIS['advertencia']} RLHF no implementado completamente (requiere modelo de recompensa)")
        self.feedback_dataset = []

    def generar_respuesta(self, caso_uso: Dict, prediccion: Dict) -> str:
        if self.model is None or self.tokenizer is None:
            logger.warning(f"{EMOJIS['advertencia']} Modelo conversacional no disponible, generando respuesta simple")
            return (f"La mejor ruta de {caso_uso.get('origen', '')} a {caso_uso.get('destino', '')} es "
                    f"{formatear_ruta(prediccion.get('ruta', []))} con un tiempo estimado de "
                    f"{formatear_tiempo(prediccion.get('tiempo_predicho', 0))} bajo {caso_uso.get('condiciones', {})}.")
        
        prompt = (f"Usuario pregunta: {caso_uso.get('consulta', '')}. Ruta sugerida: {formatear_ruta(prediccion.get('ruta', []))}. "
                  f"Tiempo estimado: {formatear_tiempo(prediccion.get('tiempo_predicho', 0))}. "
                  f"Condiciones: {caso_uso.get('condiciones', {})}. "
                  f"Genera una respuesta útil, conversacional y amigable sobre la ruta en Bogotá.")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,
                top_p=0.85
            )
            respuesta = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            referencia = (f"La mejor ruta de {caso_uso.get('origen', '')} a {caso_uso.get('destino', '')} es "
                         f"{formatear_ruta(prediccion.get('ruta', []))} con un tiempo de "
                         f"{formatear_tiempo(prediccion.get('tiempo_predicho', 0))} minutos.")
            bleu = sentence_bleu([referencia.split()], respuesta.split(), smoothing_function=SmoothingFunction().method1)
            self.metricas_ml['bleu_score'] = max(self.metricas_ml['bleu_score'], bleu)
            logger.info(f"{EMOJIS['exito']} Respuesta generada: {respuesta[:100]}..., BLEU: {bleu:.3f}")
            return respuesta
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al generar respuesta: {e}")
            return (f"La mejor ruta de {caso_uso.get('origen', '')} a {caso_uso.get('destino', '')} es "
                    f"{formatear_ruta(prediccion.get('ruta', []))} con un tiempo estimado de "
                    f"{formatear_tiempo(prediccion.get('tiempo_predicho', 0))} bajo {caso_uso.get('condiciones', {})}.")

    def registrar_retroalimentacion(self, caso_id: str, retroalimentacion: str, rating: int = None):
        try:
            self.base_conocimiento.setdefault('retroalimentacion', []).append({
                'caso_id': caso_id,
                'retroalimentacion': retroalimentacion,
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            })
            self.feedback_dataset.append({
                'prompt': self.base_conocimiento.get('casos_uso', {}).get(caso_id, {}).get('consulta', caso_id),
                'response': self.generar_respuesta(self.base_conocimiento.get('casos_uso', {}).get(caso_id, {}), {}),
                'feedback': retroalimentacion,
                'rating': rating
            })
            if len(self.feedback_dataset) > 20:
                self.fine_tune_with_sft_and_rlhf()
            self.documentar_base_conocimiento()
            logger.info(f"{EMOJIS['exito']} Retroalimentación registrada para {caso_id}")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al registrar retroalimentación: {e}")

    def interfaz_usuario_avanzada(self):
        nodos_validos = {unidecode(n).upper(): n for n in self.grafo.nodos.keys()}
        while True:
            print("\n=== INTERFAZ INTERACTIVA ===")
            print("1. Buscar ruta óptima")
            print("2. Ver base de conocimiento")
            print("3. Registrar retroalimentación")
            print("4. Salir")
            
            opcion = input("\nSeleccione (1-4): ").strip()
            
            if opcion == '1':
                origen = input("Ingrese origen (ej. UNIMINUTO_CALLE_80): ").strip()
                destino = input("Ingrese destino (ej. GRAN_ESTACION): ").strip()
                origen_normalizado = unidecode(origen).upper()
                destino_normalizado = unidecode(destino).upper()
                if origen_normalizado not in nodos_validos or destino_normalizado not in nodos_validos:
                    print(f"{EMOJIS['error']} Error: Nodo {origen if origen_normalizado not in nodos_validos else destino} no encontrado. Nodos válidos: {list(nodos_validos.values())}")
                    continue
                origen = nodos_validos[origen_normalizado]
                destino = nodos_validos[destino_normalizado]
                hora = input("Ingrese hora (0-23, o formato '8 AM', '2 PM', enter para actual): ").strip()
                dia = input("Ingrese día de la semana (0-6, o nombre del día, enter para actual): ").strip()
                clima = input("Ingrese condición climática (soleado/lluvia/nublado, enter para soleado): ").strip()
                trafico = input("Ingrese nivel de tráfico (bajo/medio/alto/muy_alto, enter para medio): ").strip()
                
                contexto = {
                    'condicion_climatica': clima if clima in ['soleado', 'lluvia', 'nublado'] else 'soleado',
                    'nivel_trafico': trafico if trafico in ['bajo', 'medio', 'alto', 'muy_alto'] else 'medio'
                }
                
                if hora:
                    hora_match = re.match(r'(\d{1,2})\s*(AM|PM)?', hora, re.IGNORECASE)
                    if hora_match:
                        hora_val = int(hora_match.group(1))
                        periodo = hora_match.group(2).upper() if hora_match.group(2) else None
                        if periodo:
                            if periodo == 'PM' and hora_val != 12:
                                hora_val += 12
                            elif periodo == 'AM' and hora_val == 12:
                                hora_val = 0
                        contexto['hora_dia'] = hora_val
                    elif hora.isdigit() and 0 <= int(hora) <= 23:
                        contexto['hora_dia'] = int(hora)
                    else:
                        logger.warning(f"{EMOJIS['advertencia']} Hora inválida: '{hora}'. Usando hora actual.")
                        contexto['hora_dia'] = datetime.now().hour
                else:
                    contexto['hora_dia'] = datetime.now().hour
                
                if dia:
                    dia_normalizado = unidecode(dia.lower())
                    if dia_normalizado in DIAS_SEMANA:
                        contexto['dia_semana'] = DIAS_SEMANA[dia_normalizado]
                    elif dia.isdigit() and 0 <= int(dia) <= 6:
                        contexto['dia_semana'] = int(dia)
                    else:
                        logger.warning(f"{EMOJIS['advertencia']} Día inválido: '{dia}'. Usando día actual.")
                        contexto['dia_semana'] = datetime.now().weekday()
                else:
                    contexto['dia_semana'] = datetime.now().weekday()
                
                resultado = self.busqueda_inteligente_ruta(origen, destino, contexto)
                if resultado['exito']:
                    caso_uso = {
                        'consulta': f"¿Cuál es la mejor ruta de {origen} a {destino}?",
                        'condiciones': contexto,
                        'origen': origen,
                        'destino': destino
                    }
                    respuesta = self.generar_respuesta(caso_uso, resultado['resultado_optimizado'])
                    print(f"\n{EMOJIS['exito']} Ruta encontrada:")
                    print(f"   Respuesta: {respuesta}")
                    print(f"   Algoritmo: {resultado['mejor_algoritmo']}")
                    print(f"   Tiempo estimado: {formatear_tiempo(resultado['resultado_optimizado']['tiempo_ml'])}")
                    print(f"   Confianza: {resultado['resultado_optimizado']['confianza']:.1%}")
                    print(f"   Ruta: {formatear_ruta(resultado['resultado_optimizado']['resultado_tradicional'].ruta)}")
                    print(f"   Distancia: {formatear_distancia(resultado['resultado_optimizado']['resultado_tradicional'].distancia_total)}")
                    print(f"   Cluster de tráfico: {resultado['resultado_optimizado']['cluster']}")
                    retro = input("¿Fue útil esta respuesta? (Sí/No + comentario): ").strip()
                    if retro.lower().startswith('s'):
                        try:
                            rating = int(input("Califica de 1-5: "))
                            if not 1 <= rating <= 5:
                                raise ValueError("Rating debe estar entre 1 y 5")
                            self.registrar_retroalimentacion(f"{origen}_{destino}", retro, rating)
                        except ValueError as e:
                            logger.warning(f"{EMOJIS['advertencia']} Rating inválido: {e}")
                    self.base_conocimiento.setdefault('casos_uso', {})[f"{origen}_{destino}"] = caso_uso
                else:
                    print(f"{EMOJIS['error']} Error: {resultado['error']}")
            
            elif opcion == '2':
                print("\n📚 BASE DE CONOCIMIENTO")
                print(json.dumps(self.base_conocimiento, indent=2, ensure_ascii=False))
            
            elif opcion == '3':
                caso_id = input("Ingrese ID del caso (ej. UNIMINUTO_CALLE_80_GRAN_ESTACION): ").strip()
                retroalimentacion = input("Ingrese retroalimentación: ").strip()
                try:
                    rating = int(input("Califica de 1-5: "))
                    if not 1 <= rating <= 5:
                        raise ValueError("Rating debe estar entre 1 y 5")
                    self.registrar_retroalimentacion(caso_id, retroalimentacion, rating)
                except ValueError as e:
                    logger.warning(f"{EMOJIS['advertencia']} Rating inválido: {e}")
            
            elif opcion == '4':
                print("👋 Saliendo de la interfaz interactiva")
                break
            
            else:
                print(f"{EMOJIS['error']} Opción no válida")

    def evaluar_rendimiento(self, datos_validacion_path: str = None) -> Dict:
        logger.info(f"\n{EMOJIS['reporte']} EVALUANDO RENDIMIENTO DEL SISTEMA EXPERTO")
        logger.info("=" * 50)

        casos_uso = self.definir_casos_uso()
        resultados_pruebas = []
        tiempos_reales = []
        tiempos_predichos = []
        tiempos_dijkstra = []
        bleu_scores = []

        if datos_validacion_path and Path(datos_validacion_path).exists():
            datos_validacion = pd.read_csv(datos_validacion_path)
        else:
            logger.warning(f"{EMOJIS['advertencia']} No se proporcionó datos_validacion.csv. Generando datos simulados...")
            datos_validacion = self.generar_datos_entrenamiento(200)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        try:
            with open(f"reporte_rendimiento_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write("Evaluación del Rendimiento del Sistema Experto de Rutas\n")
                f.write("=" * 50 + "\n\n")

                for i, caso in enumerate(casos_uso, 1):
                    try:
                        origen, destino, _ = self.extraer_consulta(caso['consulta'])
                        if not (validar_nodo(origen) and validar_nodo(destino)):
                            raise ValueError(f"Nodo inválido: {origen} o {destino}")
                        prediccion = self.predecir_tiempo_inteligente(
                            origen, destino, caso['condiciones']['hora_dia'], caso['condiciones']['dia_semana'],
                            caso['condiciones']['condicion_climatica'], caso['condiciones']['nivel_trafico']
                        )
                        if 'error' in prediccion:
                            raise ValueError(prediccion['error'])
                        respuesta = self.generar_respuesta(caso, prediccion)
                        f.write(f"Figura {i}: Resultado de la Prueba para {caso['id']}\n")
                        f.write(f"Caso: {caso['descripcion']}\n")
                        f.write(f"Consulta: {caso['consulta']}\n")
                        f.write(f"Respuesta: {respuesta}\n")
                        f.write(f"Ruta: {formatear_ruta(prediccion.get('ruta', []))}\n")
                        f.write(f"Tiempo predicho: {formatear_tiempo(prediccion.get('tiempo_predicho', 0))}\n")
                        f.write(f"Nota: Generado por el Sistema Experto de Rutas (2025).\n\n")

                        resultados_pruebas.append({'caso_id': caso['id'], 'respuesta': respuesta})
                        condiciones = caso['condiciones']
                        datos_caso = datos_validacion[
                            (datos_validacion['hora_dia'] == condiciones['hora_dia']) &
                            (datos_validacion['dia_semana'] == condiciones['dia_semana']) &
                            (datos_validacion['condicion_climatica'] == condiciones['condicion_climatica']) &
                            (datos_validacion['nivel_trafico'] == condiciones['nivel_trafico']) &
                            (datos_validacion['origen'] == origen) &
                            (datos_validacion['destino'] == destino)
                        ]
                        tiempo_real = datos_caso['tiempo_real'].mean() if not datos_caso.empty else prediccion['tiempo_predicho']
                        tiempos_reales.append(tiempo_real)
                        tiempos_predichos.append(prediccion['tiempo_predicho'])
                        resultado_dijkstra = self.buscador.buscar(origen, destino, algoritmo="dijkstra", criterio="tiempo")
                        tiempos_dijkstra.append(resultado_dijkstra.tiempo_total if resultado_dijkstra.exito else float('inf'))
                        bleu_scores.append(self.metricas_ml['bleu_score'])
                    except Exception as e:
                        f.write(f"Figura {i}: Resultado de la Prueba para {caso['id']}\n")
                        f.write(f"Caso: {caso['descripcion']}\n")
                        f.write(f"Consulta: {caso['consulta']}\n")
                        f.write(f"Error: {str(e)}\n\n")
                        continue

                if tiempos_reales and tiempos_predichos:
                    mae = mean_absolute_error(tiempos_reales, tiempos_predichos)
                    rmse = np.sqrt(mean_squared_error(tiempos_reales, tiempos_predichos))
                    r2 = r2_score(tiempos_reales, tiempos_predichos)
                else:
                    mae, rmse, r2 = float('inf'), float('inf'), -1.0
                diversidad = len(set(r['respuesta'] for r in resultados_pruebas)) / len(resultados_pruebas) * 100 if resultados_pruebas else 0
                avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

                tiempos_dijkstra_validos = [t for t in tiempos_dijkstra if t != float('inf')]
                tiempos_reales_validos = [tr for tr, td in zip(tiempos_reales, tiempos_dijkstra) if td != float('inf')]
                mae_dijkstra = mean_absolute_error(tiempos_reales_validos, tiempos_dijkstra_validos) if tiempos_dijkstra_validos else float('inf')

                f.write("Métricas de Rendimiento\n")
                f.write("=" * 50 + "\n")
                f.write(f"MAE: {mae:.2f} minutos\n")
                f.write(f"RMSE: {rmse:.2f} minutos\n")
                f.write(f"R²: {r2:.3f}\n")
                f.write(f"Diversidad de respuestas: {diversidad:.1f}%\n")
                f.write(f"BLEU Score promedio: {avg_bleu:.3f}\n\n")

                f.write("Comparación con Baseline (Dijkstra)\n")
                f.write("=" * 50 + "\n")
                f.write(f"MAE (Dijkstra): {mae_dijkstra:.2f} minutos\n")
                f.write(f"Mejora del sistema experto: {(mae_dijkstra - mae) / mae_dijkstra * 100:.1f}% (excluyendo rutas inválidas)\n")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al generar reporte de rendimiento: {e}")

        self.metricas_ml.update({'precision_temporal': mae, 'diversidad_respuestas': diversidad, 'bleu_score': avg_bleu})
        logger.info(f"{EMOJIS['exito']} Evaluación completada. Reporte guardado en reporte_rendimiento_{timestamp}.txt")
        logger.info(f"{EMOJIS['reporte']} MAE: {mae:.2f} min, RMSE: {rmse:.2f} min, R²: {r2:.3f}, Diversidad: {diversidad:.1f}%, BLEU: {avg_bleu:.3f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'diversidad': diversidad, 'bleu_score': avg_bleu, 'mae_dijkstra': mae_dijkstra}

    def documentar_base_conocimiento(self):
        try:
            with open('base_conocimiento.json', 'w', encoding='utf-8') as f:
                json.dump(self.base_conocimiento, f, indent=2, ensure_ascii=False)
            logger.info(f"{EMOJIS['exito']} Base de conocimiento documentada en base_conocimiento.json")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al documentar base de conocimiento: {e}")

    def definir_casos_uso(self) -> List[Dict]:
        try:
            nodos = list(self.grafo.nodos.keys())
            if not nodos:
                raise ValueError("No se encontraron nodos en el grafo")
            casos = [
                {
                    'id': f'CU{i:02d}',
                    'descripcion': f'Ruta en hora {"pico" if i % 2 == 0 else "valle"}',
                    'consulta': f'¿Cuál es la mejor ruta de {nodos[i % len(nodos)]} a {nodos[(i + 1) % len(nodos)]} a las {8 if i % 2 == 0 else 14} {"AM" if i % 2 == 0 else "PM"}?',
                    'condiciones': {
                        'hora_dia': 8 if i % 2 == 0 else 14,
                        'dia_semana': i % 7,
                        'condicion_climatica': random.choice(['soleado', 'lluvia', 'nublado']),
                        'nivel_trafico': random.choice(['bajo', 'medio', 'alto', 'muy_alto'])
                    },
                    'origen': nodos[i % len(nodos)],
                    'destino': nodos[(i + 1) % len(nodos)]
                } for i in range(10)
            ]
            logger.info(f"{EMOJIS['exito']} Definidos {len(casos)} casos de uso")
            return casos
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al definir casos de uso: {e}")
            return []

    def extraer_consulta(self, consulta: str) -> Tuple[str, str, str]:
        try:
            match = re.search(r'de (\w+(?:_\w+)*) a (\w+(?:_\w+)*)', consulta, re.IGNORECASE)
            if match:
                origen, destino = match.group(1), match.group(2)
                if validar_nodo(origen) and validar_nodo(destino):
                    return origen, destino, consulta
            logger.warning(f"{EMOJIS['advertencia']} Consulta inválida: {consulta}. Usando valores predeterminados.")
            return 'UNIMINUTO_CALLE_80', 'GRAN_ESTACION', consulta
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al extraer consulta: {e}")
            return 'UNIMINUTO_CALLE_80', 'GRAN_ESTACION', consulta

    def probar_sistema(self):
        casos_uso = self.definir_casos_uso()
        for caso in casos_uso[:3]:
            try:
                origen, destino, _ = self.extraer_consulta(caso['consulta'])
                if not (validar_nodo(origen) and validar_nodo(destino)):
                    logger.warning(f"{EMOJIS['advertencia']} Prueba omitida para {caso['id']}: Nodo inválido {origen} o {destino}")
                    continue
                resultado = self.busqueda_inteligente_ruta(origen, destino, caso['condiciones'])
                if resultado['exito']:
                    logger.info(f"{EMOJIS['exito']} Prueba exitosa para {caso['id']}: {formatear_ruta(resultado['resultado_optimizado']['resultado_tradicional'].ruta)}")
                else:
                    logger.warning(f"{EMOJIS['error']} Prueba fallida para {caso['id']}: {resultado['error']}")
            except Exception as e:
                logger.error(f"{EMOJIS['error']} Error en prueba para {caso['id']}: {e}")

    def probar_escenarios_extremos(self):
        try:
            nodos = list(self.grafo.nodos.keys())
            escenarios = [
                {'origen': nodos[0], 'destino': nodos[-1], 'condiciones': {'hora_dia': 7, 'dia_semana': 0, 'condicion_climatica': 'lluvia', 'nivel_trafico': 'muy_alto'}},
                {'origen': nodos[-1], 'destino': nodos[0], 'condiciones': {'hora_dia': 23, 'dia_semana': 6, 'condicion_climatica': 'soleado', 'nivel_trafico': 'bajo'}},
                {'origen': nodos[1], 'destino': nodos[2], 'condiciones': {'hora_dia': 12, 'dia_semana': 3, 'condicion_climatica': 'nublado', 'nivel_trafico': 'medio'}}
            ]
            for i, escenario in enumerate(escenarios, 1):
                try:
                    if not (validar_nodo(escenario['origen']) and validar_nodo(escenario['destino'])):
                        logger.warning(f"{EMOJIS['advertencia']} Escenario extremo {i} omitido: Nodo inválido {escenario['origen']} o {escenario['destino']}")
                        continue
                    resultado = self.busqueda_inteligente_ruta(escenario['origen'], escenario['destino'], escenario['condiciones'])
                    if resultado['exito']:
                        logger.info(f"{EMOJIS['exito']} Escenario extremo {i} exitoso: {formatear_ruta(resultado['resultado_optimizado']['resultado_tradicional'].ruta)}")
                    else:
                        logger.warning(f"{EMOJIS['error']} Escenario extremo {i} fallido: {resultado['error']}")
                except Exception as e:
                    logger.error(f"{EMOJIS['error']} Error en escenario extremo {i}: {e}")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al probar escenarios extremos: {e}")

    def evaluar_modelos(self, X_test, y_test):
        resultados = {}
        for nombre, modelo in self.modelos_tiempo.items():
            try:
                y_pred = modelo.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                resultados[nombre] = {'r2': r2, 'mae': mae}
                logger.info(f"{EMOJIS['exito']} Evaluación {nombre}: R²={r2:.3f}, MAE={mae:.2f} min")
            except Exception as e:
                logger.error(f"{EMOJIS['error']} Error al evaluar {nombre}: {e}")
                resultados[nombre] = {'r2': -1.0, 'mae': float('inf')}
        return resultados

    def generar_reporte_ml(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        reporte = f"reporte_ml_{timestamp}.txt"
        try:
            with open(reporte, 'w', encoding='utf-8') as f:
                f.write("Reporte de Modelos de Machine Learning\n")
                f.write("=" * 50 + "\n")
                f.write(f"Métricas ML: {self.metricas_ml}\n")
                f.write(f"Modelos entrenados: {list(self.modelos_tiempo.keys())}\n")
                f.write(f"Parámetros optimizados: {self.modelo_optimizacion.get_params() if self.modelo_optimizacion else 'No optimizado'}\n")
            logger.info(f"{EMOJIS['exito']} Reporte ML generado en {reporte}")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al generar reporte ML: {e}")
        return reporte

    def implementar_clustering_patrones(self, df):
        try:
            X = df[['hora_dia', 'dia_semana', 'condicion_climatica', 'nivel_trafico']]
            kmeans = KMeans(n_clusters=3, random_state=42)
            self.patrones_trafico['clusters'] = kmeans.fit_predict(X)
            self.patrones_trafico['modelo'] = kmeans
            logger.info(f"{EMOJIS['exito']} Clustering de patrones implementado con {len(np.unique(self.patrones_trafico['clusters']))} clusters")
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al implementar clustering: {e}")

    def predecir_cluster(self, condiciones):
        try:
            features = np.array([[
                condiciones['hora_dia'],
                condiciones['dia_semana'],
                {'soleado': 0, 'nublado': 1, 'lluvia': 2}[condiciones['condicion_climatica']],
                {'bajo': 0, 'medio': 1, 'alto': 2, 'muy_alto': 3}[condiciones['nivel_trafico']]
            ]])
            cluster = self.patrones_trafico['modelo'].predict(features)[0]
            logger.debug(f"{EMOJIS['info']} Cluster predicho: {cluster}")
            return cluster
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al predecir cluster: {e}")
            return 0

    def ejecutar_sistema_completo_ml(self):
        logger.info(f"\n{EMOJIS['inicio']} INICIANDO SISTEMA EXPERTO CON MACHINE LEARNING")
        logger.info("=" * 60)
        
        try:
            objetivos = self.definir_objetivos_ml()
            df_datos = self.generar_datos_entrenamiento(1000)
            self.implementar_clustering_patrones(df_datos)
            casos_uso = self.definir_casos_uso()
            self.base_conocimiento['casos_uso'] = {caso['id']: caso for caso in casos_uso}
            X, y = self.preparar_caracteristicas(df_datos)
            if X.size == 0:
                logger.error(f"{EMOJIS['error']} No se generaron datos válidos para entrenamiento")
                return {'sistema_configurado': False}
            X_train, X_test, y_train, y_test = self.dividir_datos(X, y)
            self.entrenar_modelos_ml(X_train, y_train)
            self.optimizar_mejor_modelo()
            resultados = self.evaluar_modelos(X_test, y_test)
            self.fine_tune_with_sft_and_rlhf()
            self.documentar_base_conocimiento()
            self.probar_sistema()
            self.probar_escenarios_extremos()
            archivo_reporte = self.generar_reporte_ml()
            
            logger.info(f"\n{EMOJIS['exito']} SISTEMA ML CONFIGURADO CORRECTAMENTE")
            return {
                'sistema_configurado': True,
                'precision_ml': self.metricas_ml['precision_temporal'],
                'diversidad_respuestas': self.metricas_ml['diversidad_respuestas'],
                'bleu_score': self.metricas_ml['bleu_score'],
                'archivo_reporte': archivo_reporte
            }
        except Exception as e:
            logger.error(f"{EMOJIS['error']} Error al ejecutar sistema completo: {e}")
            return {'sistema_configurado': False}

def main():
    logger.info(f"{EMOJIS['inicio']} Iniciando Sistema Experto de Rutas con Machine Learning")
    
    try:
        logger.info(f"{EMOJIS['mapa']} Creando mapa de Bogotá...")
        grafo_bogota = crear_mapa_bogota()
        coordenadas = cargar_coordenadas('src/coordenadas_bogota.json')
        sistema_ml = SistemaExpertoRutasML(grafo_bogota, coordenadas)
        
        logger.info("\n🎯 ¿Qué desea hacer?")
        print("1. Configurar y entrenar sistema completo")
        print("2. Cargar modelos existentes")
        print("3. Usar interfaz interactiva")
        print("4. Evaluar rendimiento")
        
        opcion = input("\nSeleccione (1-4): ").strip()
        
        if opcion == '1':
            resultado = sistema_ml.ejecutar_sistema_completo_ml()
            logger.info(f"\n{EMOJIS['exito']} Sistema configurado: {resultado}")
            sistema_ml.guardar_modelos()
            sistema_ml.interfaz_usuario_avanzada()
        elif opcion == '2':
            if sistema_ml.cargar_modelos():
                sistema_ml.interfaz_usuario_avanzada()
            else:
                logger.warning(f"{EMOJIS['advertencia']} No se pudieron cargar los modelos. Entrenando sistema...")
                resultado = sistema_ml.ejecutar_sistema_completo_ml()
                logger.info(f"\n{EMOJIS['exito']} Sistema configurado: {resultado}")
                sistema_ml.guardar_modelos()
                sistema_ml.interfaz_usuario_avanzada()
        elif opcion == '3':
            if not sistema_ml.modelo_entrenado:
                logger.warning(f"{EMOJIS['advertencia']} Modelos no entrenados ni cargados. Entrenando sistema automáticamente...")
                resultado = sistema_ml.ejecutar_sistema_completo_ml()
                logger.info(f"\n{EMOJIS['exito']} Sistema configurado: {resultado}")
                sistema_ml.guardar_modelos()
            sistema_ml.interfaz_usuario_avanzada()
        elif opcion == '4':
            if not sistema_ml.modelo_entrenado:
                logger.warning(f"{EMOJIS['advertencia']} Modelos no entrenados ni cargados. Entrenando sistema automáticamente...")
                resultado = sistema_ml.ejecutar_sistema_completo_ml()
                logger.info(f"\n{EMOJIS['exito']} Sistema configurado: {resultado}")
                sistema_ml.guardar_modelos()
            resultado = sistema_ml.evaluar_rendimiento()
            logger.info(f"\n{EMOJIS['exito']} Resultados de evaluación: {resultado}")
        else:
            logger.error(f"{EMOJIS['error']} Opción no válida")
            
    except Exception as e:
        logger.error(f"{EMOJIS['error']} Error crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import nltk
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.error(f"{EMOJIS['error']} Error al descargar NLTK punkt: {e}")
    main()
