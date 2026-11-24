# Proyecto final Carrito autónomo con DQN

Este proyecto implementa una pista 2D, un automóvil con dinámica simplificada y
un agente de aprendizaje por refuerzo profundo. La red recibe nueve variables y
produce un valor Q para cada una de nueve acciones discretas derivadas de los
controles acelerar, frenar, girar a la izquierda y girar a la derecha.

## Instalación recomendada en Windows 11

1. Instala Python 3.11 o crea un entorno de Anaconda con esa versión.
2. Abre una terminal en esta carpeta.
3. Crea y activa un entorno virtual:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

4. Instala las dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Comprueba el entorno:

```powershell
python check_environment.py
```

## Secuencia de uso

```powershell
python manual_drive.py
python random_baseline.py
python heuristic_baseline.py
python train.py --episodes 600
python evaluate.py --model results/best_model.pt
```

El entrenamiento se realiza sin ventana para aprovechar mejor el procesador.
La evaluación abre Pygame y muestra los sensores, la velocidad, la acción y el
avance. Los archivos `training_log.csv`, `training_curves.png` y los modelos
entrenados se guardan en `results`.

## Qué deben comparar los estudiantes

- Agente aleatorio frente a DQN.
- Cinco sensores frente a tres sensores.
- Red de 32, 64 o 128 neuronas por capa.
- Descuento gamma y duración de la exploración epsilon-greedy.
- Recompensa original frente a una modificación documentada.
- Resultados en entrenamiento frente a diez episodios de evaluación sin azar.

No debe calificarse una sola ejecución aislada. El aprendizaje por refuerzo es
estocástico y las comparaciones deben repetirse con al menos tres semillas.

