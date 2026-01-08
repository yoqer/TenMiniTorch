# Manual de Extensibilidad y Profundización: TenMiNiTorch Lite



Este manual sirve como una guía avanzada y práctica para desarrolladores que deseen extender y profundizar en la funcionalidad de TenMiNiTorch Lite. Se centra en la arquitectura modular de la librería, explicando cómo añadir nuevas funciones y capas, y analizando las posibilidades de despliegue avanzado.

## 1. Guía de Extensibilidad: Funciones de Activación y Capas

La arquitectura de TenMiNiTorch Lite está diseñada para ser extensible. La clave para añadir nuevas funcionalidades reside en dos clases base: `minitorch_lite.autograd.Function` (para operaciones matemáticas) y `minitorch_lite.nn.Module` (para capas de red).

### 1.1. Implementación de Nuevas Funciones de Activación (Ejemplo: Swish)

Las funciones de activación son operaciones matemáticas que se aplican a los tensores. Para que el entrenamiento funcione, deben ser diferenciables, lo que significa que deben implementarse como subclases de `Function`.

#### **Ejemplo Detallado: Swish (SiLU)**

La función Swish (o SiLU) se define como $f(x) = x \cdot \sigma(x)$, donde $\sigma(x)$ es la función sigmoide.

| Aspecto | Detalle |
| :--- | :--- |
| **Funcionalidad** | Introduce no-linealidad en la red, mejorando el rendimiento en modelos profundos en comparación con ReLU. |
| **Fórmula** | $f(x) = x \cdot \frac{1}{1 + e^{-x}}$ |
| **Derivada** | $\frac{df}{dx} = f(x) + \sigma(x) \cdot (1 - f(x))$ |
| **Implementación** | Requiere implementar las operaciones de multiplicación (`Mul`), exponenciación (`Exp`), suma (`Add`) y división (`Div`) en `autograd.py`. |

**Proceso de Implementación (Diseño):**

1.  **Crear la Clase `Swish`:** Heredar de `minitorch_lite.autograd.Function`.
2.  **Método `forward(ctx, x)`:**
    *   Calcular $\sigma(x) = 1 / (1 + e^{-x})$.
    *   Calcular la salida $y = x \cdot \sigma(x)$.
    *   Guardar $x$ y $y$ en el contexto (`ctx.save_for_backward(x, y)`).
    *   Devolver $y$.
3.  **Método `backward(ctx, grad_output)`:**
    *   Recuperar $x$ y $y$ del contexto.
    *   Calcular el gradiente $\frac{df}{dx}$ usando la fórmula de la derivada.
    *   Multiplicar por el gradiente de salida (`grad_output`) y devolver el resultado.

#### **Otras Posibilidades de Activación (Índice)**

| Función de Activación | Fórmula | Requisitos de `autograd` |
| :--- | :--- | :--- |
| **Sigmoid** | $\sigma(x) = 1 / (1 + e^{-x})$ | `Exp`, `Add`, `Div` |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | `Exp`, `Sub`, `Add`, `Div` |
| **Leaky ReLU** | $f(x) = \max(ax, x)$ | `Max`, `Mul` (por escalar) |
| **GELU** | $f(x) = x \cdot \Phi(x)$ | `Mul`, `Erf` (función de error) |

### 1.2. Implementación de Nuevas Capas (Ejemplo: Convolucional 1D)

Las capas se implementan como subclases de `minitorch_lite.nn.Module`.

#### **Ejemplo Detallado: Convolucional 1D (`Conv1D`)**

Una capa convolucional 1D se utiliza para procesar secuencias de datos (ej. series de tiempo, texto).

| Aspecto | Detalle |
| :--- | :--- |
| **Funcionalidad** | Aplica un filtro (kernel) a lo largo de una dimensión de la entrada para extraer características locales. |
| **Parámetros** | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`. |
| **Implementación** | Requiere la implementación de una operación de **Convolución 1D** en `autograd.py` (la más compleja). |

**Proceso de Implementación (Diseño):**

1.  **Crear la Clase `Conv1D`:** Heredar de `minitorch_lite.nn.Module`.
2.  **Método `__init__`:**
    *   Crear los pesos (`self.weight`) y sesgos (`self.bias`) como `Parameter`s con las dimensiones correctas.
3.  **Método `forward(x)`:**
    *   Llamar a la operación de convolución 1D implementada en `autograd.py`.
    *   $Y = \text{Conv1D.apply}(X, W) + b$.

#### **Otras Posibilidades de Capas (Índice)**

| Capa | Funcionalidad | Requisitos de `autograd` |
| :--- | :--- | :--- |
| **Conv2D** | Convolución para imágenes (2D). | Operación de Convolución 2D. |
| **MaxPool1D/2D** | Reducción de dimensionalidad (submuestreo). | Operación de MaxPool (requiere guardar índices). |
| **BatchNorm** | Normalización de activaciones. | Operaciones de media, varianza, división, etc. |
| **Embedding** | Mapeo de índices a vectores densos (texto). | Operación de Indexación. |

## 2. Profundización de Funciones: Estado Actual y Próximos Pasos

MiniTorch Lite es un prototipo funcional en sus componentes básicos, pero requiere trabajo para ser completamente funcional y optimizado.

### 2.1. Funcionalidad Actual (Implementada)

| Módulo | Funcionalidad | Estado |
| :--- | :--- | :--- |
| **`tensor`** | Creación de tensores, almacenamiento de datos (NumPy), metadatos de gradiente. | **Funcional** |
| **`autograd`** | Retropropagación para operaciones básicas (`Add`, `Mul`, `Sum`). | **Funcional** |
| **`nn`** | Módulos base (`Module`, `Sequential`), Capa `Linear`, `MSELoss`. | **Funcional** |
| **`optim`** | Optimizador `SGD`. | **Funcional** |

### 2.2. Lo que Falta para la Funcionalidad Completa (Próximos Pasos)

El principal obstáculo para el entrenamiento de redes neuronales complejas es la falta de la operación de multiplicación matricial (`MatMul`) en el motor de `autograd`.

| Funcionalidad Faltante | Impacto | Implementación Requerida |
| :--- | :--- | :--- |
| **`MatMul` en `autograd`** | **Crítico.** La capa `Linear` y todas las capas convolucionales dependen de esto. Sin él, el entrenamiento real no es posible (solo la simulación). | Crear `MatMul(Function)` e implementar su `backward` (que es la regla de la cadena para la multiplicación matricial). |
| **Operaciones de *Broadcasting*** | **Importante.** Asegurar que los gradientes se reduzcan correctamente cuando las formas de los tensores no coinciden (ej. sumar un vector a una matriz). | Lógica de reducción de gradientes en el `backward` de `Add`, `Mul`, etc. |
| **Módulo `lite`** | **Esencial** para el objetivo "Lite". | Implementar la clase `Quantizer` y el `LiteEngine` para la inferencia cuantizada. |

## 3. Profundización de Librerías: El Ecosistema de MiniTorch Lite

### 3.1. NumPy: El Corazón Temporal

NumPy es el *backend* de cálculo. La clase `Tensor` delega todas las operaciones numéricas a NumPy.

*   **Uso:** Las operaciones como `self.data + other.data` o `np.matmul(self.data, other.data)` se realizan directamente sobre los *arrays* de NumPy.
*   **Detalles de Implementación:** La clase `Function` en `autograd.py` utiliza NumPy para calcular tanto el `forward` como el `backward` (la derivada).
*   **Posibilidades:** Mientras se use NumPy, se puede acceder a cualquier función de NumPy para crear nuevas operaciones, siempre y cuando se implemente su derivada correspondiente en `autograd.py`.

### 3.2. Pandas: El Preprocesamiento Inteligente

Pandas no es una librería de *Deep Learning*, sino de manipulación de datos.

*   **Función:** Carga, limpieza, transformación y análisis exploratorio de datos tabulares.
*   **Uso:** Se utiliza **antes** de MiniTorch Lite. Los datos se cargan en un `DataFrame`, se normalizan, se manejan los valores perdidos, y finalmente se extraen las columnas relevantes.
*   **Implementación:** El paso final es convertir el `DataFrame` o las Series a un *array* de NumPy (`df.values` o `df.to_numpy()`) y luego envolverlo en un `Tensor` de MiniTorch Lite.

```python
import pandas as pd
from minitorch_lite import tensor

# 1. Preprocesamiento con Pandas
df = pd.read_csv('datos.csv')
X_np = df[['feature1', 'feature2']].to_numpy(dtype=np.float32)
Y_np = df['target'].to_numpy(dtype=np.float32)

# 2. Uso en MiniTorch Lite
X = tensor(X_np)
Y = tensor(Y_np)
```

## 4. Estrategias Avanzadas de Despliegue

Para un dispositivo de borde, la conectividad puede ser intermitente. Se pueden emplear estrategias híbridas para maximizar la eficiencia y la robustez.

### 4.1. Acceso a RAG (Retrieval-Augmented Generation) Antes de Inferir

El RAG se utiliza para mejorar la respuesta de un modelo de lenguaje con información externa. En el contexto de MiniTorch Lite, esto se puede adaptar para **validación de datos o pre-inferencia inteligente**.

| Concepto | Aplicación en MiniTorch Lite | Cómo Implementarlo |
| :--- | :--- | :--- |
| **RAG en Pre-Inferencia** | El dispositivo de borde (con el modelo cuantizado) se conecta a Internet para **validar** o **enriquecer** los datos de entrada antes de pasarlos al modelo local. | **1. Conexión:** El dispositivo usa una conexión Wi-Fi/Celular. **2. Consulta:** Envía una consulta ligera (ej. "Temperatura actual en la zona") a un servidor RAG. **3. Respuesta:** El servidor devuelve un dato validado o un *flag* de alerta. **4. Decisión:** El dispositivo decide si procede con la inferencia local o si la respuesta del RAG es suficiente. |
| **Modelo de Respaldo (Fallback)** | Si el modelo local cuantizado detecta una entrada de baja calidad o fuera de su rango de confianza, el dispositivo consulta un modelo más grande y sin cuantizar alojado en un servidor. | **1. Detección de Confianza:** El modelo local calcula una métrica de confianza. **2. Consulta Remota:** Si la confianza es baja, el dispositivo envía los datos de entrada al servidor de *hosting*. **3. Inferencia Remota:** El servidor ejecuta el modelo `float32` (sin cuantizar) para obtener una predicción de alta precisión. **4. Respuesta:** El dispositivo utiliza la predicción remota. |

### 4.2. Uso del Modelo sin Cuantizar desde Dispositivo de Hosting

Para usar el modelo `float32` (sin cuantizar) como respaldo o para re-entrenamiento en línea, se sigue el siguiente flujo:

1.  **Hosting:** El modelo `float32` se aloja en un servidor (ej. AWS Lambda, Google Cloud Run) con una API REST simple.
2.  **Dispositivo de Borde:** El dispositivo de borde (ej. Raspberry Pi) utiliza una librería de red (ej. `requests` en Python) para enviar los datos de entrada al *endpoint* del servidor.
3.  **Ventajas:** Permite obtener la máxima precisión cuando la conectividad lo permite, sin sobrecargar el dispositivo de borde con el modelo grande.

Este enfoque híbrido maximiza la eficiencia en el borde (modelo cuantizado) y la precisión/robustez (modelo de respaldo en la nube).

***

## Referencias

[1] Swish: A Self-Gated Activation Function. [Online].
[2] PyTorch. *Extending PyTorch*. [Online].
[3] Google AI Edge. *Model optimization: Post-training quantization*. [Online].
