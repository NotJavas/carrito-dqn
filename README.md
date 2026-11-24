# Carrito autónomo con DQN

Proyecto final de Tópicos de Inteligencia Artificial: un carrito aprende a completar una vuelta en una pista 2D de Pygame. El repositorio reúne el código ejecutable, modelos entrenados, resultados reales, bitácora y un video de demostración.

**Estado al 24 de noviembre de 2026:** el simulador funciona en CPU; hay siete corridas históricas, seis corridas controladas de recompensa y 180 episodios adicionales que comparan DQN, reglas y azar. Aún faltan integrar el informe final del equipo, preparar las diapositivas y ensayar la exposición de 20 minutos. El [checklist de entregables](CHECKLIST_ENTREGABLES.txt) permite seguir ese trabajo.

El punto de partida fue el paquete docente `Carrito_DQN_Profesor`, que ya incluía simulador, agente DQN, entrenamiento y controladores. El trabajo documentado comprende instalación, comparación CPU/GPU, ejecución y evaluación de modelos, ajustes de recompensa y parada, correcciones dinámicas de mensajes y gráficas, análisis y organización de evidencia. Se conserva el [README del paquete docente](docs/README_profesor.md). No se atribuye la implementación completa del algoritmo al equipo.

## Ejecutar en Windows

Requisito: Python 3.11 instalado. Abre PowerShell en la raíz de este repositorio.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

No es necesario activar el entorno al usar su intérprete directamente. El proyecto selecciona CPU explícitamente en `train.py` y `evaluate.py`. La animación necesita una sesión local con escritorio.

Para observar un modelo ya entrenado:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model results_seed76/final_model.pt --episodes 1
```

Para evaluar sin ventana:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model results_seed76/final_model.pt --episodes 10 --no-render
```

Los archivos `.pt` son checkpoints binarios de PyTorch y se cargan con los scripts; no se abren como texto en el editor. Usa los checkpoints de este proyecto cuya procedencia conoces.

## Entrenar una corrida nueva

Usa una salida nueva para conservar las corridas existentes:

```powershell
.\.venv\Scripts\python.exe train.py --episodes 600 --seed 42 --output runs/prueba_seed42
```

El límite efectivo viene de `--episodes`: el valor predeterminado de la interfaz es 600, aunque `TrainConfig.episodes` esté escrito como 5000. La parada temprana puede terminar antes. Actualmente exige 99 éxitos en los últimos 100 episodios. Los mensajes, la ventana de la gráfica y su línea de meta reflejan esos valores de configuración.

Configuración actual del código raíz:

| Elemento | Valor |
|---|---|
| Observación | Cinco sensores, velocidad, seno/coseno del error angular y posición lateral: 9 valores |
| Acciones | 9 combinaciones discretas de inercia, aceleración, freno y giro |
| Red | 9 → 64 → 64 → 9 |
| Aprendizaje | `learning_rate=0.0003`, `gamma=0.99`, lote 128 |
| Memoria | 50 000 transiciones; aprendizaje desde 1000 pasos |
| Exploración | Epsilon de 1.0 a 0.05; escala de reducción de 45 000 pasos |
| Estabilización | `tau=0.005`, recorte de gradiente 10 |
| Episodio | Una vuelta, máximo 1800 pasos, `dt=1/30 s` |
| Recompensa | `progress_reward + speed_reward - 0.02 - control_cost`; se conservan los premios/castigos de choque y vuelta |

Cambiar el término constante de recompensa implica editar `simulator_core.py`. Esto cambia también el retorno de la evaluación; no convierte automáticamente a un modelo guardado en otro modelo entrenado. Cada ejecución de `train.py` crea un agente nuevo: este paquete no implementa reanudar un entrenamiento desde un checkpoint.

`best_model.pt` se elige por retorno móvil de entrenamiento. La evaluación automática al terminar corresponde a `final_model.pt`; los dos archivos pueden representar comportamientos diferentes.

## Resultados y alcance

### Comparación de políticas

El modelo histórico final de semilla 76 se eligió como demostración favorable. Se evaluó sin exploración frente al controlador por reglas y al azar, con la misma física, recompensa actual y semillas de inicio. Hubo 30 episodios por política con inicio fijo y otros 30 con pequeñas perturbaciones de inicio.

| Inicio | Política | Vueltas completas | Colisiones | Pasos medios en vueltas exitosas |
|---|---|---:|---:|---:|
| Fijo | DQN final seed76 | 30/30 | 0/30 | 288.00 |
| Fijo | Reglas | 30/30 | 0/30 | 547.00 |
| Fijo | Azar | 0/30 | 20/30 | No aplica |
| Perturbado | DQN final seed76 | 30/30 | 0/30 | 288.10 |
| Perturbado | Reglas | 30/30 | 0/30 | 546.83 |
| Perturbado | Azar | 0/30 | 21/30 | No aplica |

Los episodios restantes del azar agotaron el límite de pasos. En el inicio fijo, 288 pasos equivalen a 9.60 segundos simulados. El video añade 2.5 segundos de resultado final y dura 12.1 segundos. No incluye narración.

El inicio perturbado solo varía el ángulo inicial en ±0.08 radianes y la orientación en ±0.04, sobre la misma pista. Las repeticiones fijas de políticas deterministas recorren el mismo escenario. Estos resultados no prueban desempeño en pistas nuevas ni representan el promedio de todas las semillas de entrenamiento.

- [Video de una vuelta real](Avance_24nov/Demostracion_DQN_seed76.mp4)
- [Comparación de políticas en Word](Avance_24nov/Comparacion_Politicas_Carrito.docx)
- [Los 180 episodios de evaluación](Avance_24nov/evaluacion_politicas_episodios.csv)
- [Resumen de políticas](Avance_24nov/resumen_politicas.csv)

Para repetir la comparación completa y guardar nuevos CSV/fotogramas:

```powershell
.\.venv\Scripts\python.exe comparar_politicas.py --output runs/comparacion_nueva
```

### Comparación controlada de recompensa

Se comparó la constante original `+0.01` frente a `-0.02`, con semillas 42, 66 y 76, hasta 600 episodios por corrida, ventana 100 y meta 99 %. Las seis corridas agotaron 600 episodios. Todas usaron CPU con un hilo y snapshots aislados de fuentes; diez evaluaciones fijas por modelo final.

| Constante | Semillas con evaluación exitosa | Éxito medio entre semillas | Desviación estándar muestral |
|---|---:|---:|---:|
| +0.01 | 1/3 | 33.33 % | 57.74 puntos porcentuales |
| -0.02 | 0/3 | 0.00 % | 0.00 puntos porcentuales |

Se conservaron los fallos. Bajo ese presupuesto no se observó una mejora consistente al introducir el costo. El modelo histórico de la demostración pertenece a otro experimento; no altera este resultado. Los retornos de fórmulas de recompensa diferentes no se comparan directamente como una clasificación de calidad.

- [Bitácora completa en Word](Bitacora/Bitacora_Carrito_DQN.docx)
- [Bitácora en texto](Bitacora/Bitacora_Carrito_DQN.txt)
- [Resumen de las seis corridas](Bitacora/Evidencias_controladas/resumen_controlados.json)
- [Reproducir las condiciones controladas](Bitacora/Evidencias_controladas/LEEME_REPRODUCIR.txt)
- [Versiones exactas del entorno utilizado](Bitacora/Evidencias_controladas/versiones_entorno_cpu.txt)

Las corridas históricas no guardaban una copia completa de la recompensa ni de la física. Los coeficientes históricos reconstruidos se identifican como inferencias condicionadas en la bitácora. Los metadatos preservan las rutas y fechas reales originales; los comandos de este README son los portables para ejecutar desde un clon nuevo.

## Cómo se organiza el código

| Archivo o carpeta | Para qué sirve |
|---|---|
| `config.py` | Configuración de pista, vehículo y entrenamiento |
| `simulator_core.py` | Física simplificada, sensores, recompensa y finalización |
| `car_env.py` | Interfaz Gymnasium y dibujo en Pygame |
| `dqn_agent.py` | Red Q, replay, selección de acciones y actualización DQN |
| `train.py` | Entrenamiento, CSV, curvas y checkpoints |
| `evaluate.py` | Carga y evaluación visual o sin render |
| `manual_drive.py` | Conducción manual |
| `random_baseline.py`, `heuristic_baseline.py` | Controladores de referencia del paquete docente |
| `comparar_politicas.py` | Protocolo común de 180 episodios usado en el avance |
| `check_environment.py` | Validación del entorno, espacios, transiciones y render |
| `analisis_entrenamiento.ipynb` | Análisis de registros y curvas |
| `results*` | Siete corridas históricas, incluidos resultados fallidos |
| `Bitacora/` | Narrativa, registros y las seis corridas controladas |
| `Avance_24nov/` | Comparación de políticas y video |
| `CHECKLIST_ENTREGABLES.txt` | Avance y tareas pendientes del equipo |

Las llamadas originales a `random_baseline.py` y `heuristic_baseline.py` usan protocolos propios. Para comparar las políticas con las mismas condiciones, usa `comparar_politicas.py`.

Los entornos `.venv` y `.venv-amd`, cachés y eventos de TensorBoard no se incluyen. Los modelos y resultados guardados sí están versionados para que el equipo pueda revisarlos.

## Pendientes del equipo

Consultar el [checklist](CHECKLIST_ENTREGABLES.txt). El siguiente trabajo es integrar el informe Word final, preparar las diapositivas con el video y las conclusiones, repartir la exposición entre tres personas y ensayar los 20 minutos. También queda revisar los contrastes adicionales del README docente y las entregas de Classroom por separado.

El proyecto demuestra conducción aprendida en una simulación cinemática simplificada. La evidencia no demuestra control de un vehículo físico.
