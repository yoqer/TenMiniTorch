# Asignación - MiniTorch


![tema4](https://github.com/user-attachments/assets/28fbca50-79db-4a4e-be06-fb83fa3c3abb)



## Redes Neuronales

Ahora tenemos una biblioteca de aprendizaje profundo completamente funcional con la mayoría de las características de un sistema industrial real como Torch. Para aprovechar este duro trabajo, este módulo está completamente basado en usar el framework de software. En particular, vamos a construir un sistema de reconocimiento de imágenes. Haremos esto construyendo la infraestructura para una versión de LeNet en MNIST: una red neuronal convolucional (CNN) clásica para el reconocimiento de dígitos, y para una convolución 1D para clasificación de sentimientos de NLP.

Necesitas los archivos de asignaciones anteriores, así que asegúrate de traerlos a tu nuevo repositorio. Te recomendamos que te familiarices con `tensor.py`, ya que podrías encontrar algunas de esas funciones útiles para implementar este Módulo.

## Guías

- [Convolución](https://minitorch.github.io/module4/convolution/)
- [Pooling](https://minitorch.github.io/module4/pooling/)
- [Softmax](https://minitorch.github.io/module4/softmax/)

## Tarea 4.1: Convolución 1D

Implementarás la convolución 1D en Numba. Esta función se usa en el pase hacia adelante (`forward`) y hacia atrás (`backward`) de conv1d.

### Por hacer

Completa la siguiente función en `minitorch/fast_conv.py`, y pasa las pruebas marcadas como `task4_1`.

- `_tensor_conv1d`

## Tarea 4.2: Convolución 2D

Implementarás la convolución 2D en Numba. Esta función se usa en el pase hacia adelante (`forward`) y hacia atrás (`backward`) de conv2d.

### Por hacer

Completa la siguiente función en `minitorch/fast_conv.py`, y pasa las pruebas marcadas como `task4_2`.

- `_tensor_conv2d`

## Tarea 4.3: Pooling

Implementarás pooling 2D en tensores con una operación de promedio.

### Por hacer

Completa la siguiente función en `minitorch/nn.py`, y pasa las pruebas marcadas como `task4_3`. Úsala para implementar `avgpool2d`.

#### `minitorch.tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]`

Reforma un tensor de imagen para pooling 2D

---

python
```python
input: batch x channel x height x width
kernel: height x width del pooling

```


____________________________________________________________________________________________________________________________________________________________________________________________________________





Python

```python

Tensor de tamaño batch x channel x new_height x new_width x (kernel_height * kernel_width) así como los valores new_height y new_width.

```



____________________________________________________________________






![tema4-2](https://github.com/user-attachments/assets/4e092352-2dea-44d2-9a1b-8738813a4465)




## Tarea 4.4: Softmax y Dropout

Implementarás max, softmax y log softmax en tensores, así como las operaciones de dropout y max-pooling.

### Por hacer

- Completa las siguientes funciones en `minitorch/nn.py`, y pasa las pruebas marcadas como `task4_4`.

- Agrega pruebas de propiedad para la función en `test/test_nn.py` y asegúrate de entender su computación de gradiente.

- `minitorch.max`
- `minitorch.softmax`
- `minitorch.logsoftmax`
- `minitorch.maxpool2d`
- `minitorch.dropout`

Implementar convolución y pooling eficientemente es crítico para el reconocimiento de imágenes a gran escala. Sin embargo, ambos son un poco más difíciles que algunas de las funciones CUDA básicas que hemos escrito hasta ahora. Para esta tarea, agrega un archivo extra `cuda_conv.py` que implemente `conv1d` y `conv2d` en CUDA. Muestra la salida en colab.

## Tarea 4.5: Entrenando un Clasificador de Imágenes

Si tu código funciona, ahora deberías poder pasar a los scripts de entrenamiento de NLP y CV en `project/run_sentiment.py` y `project/run_mnist_multiclass.py`. Este script tiene la misma configuración básica de entrenamiento que el `módulo3`, pero ahora adaptado a clasificación de sentimientos e imágenes. Necesitas implementar `Conv1D`, `Conv2D` y `Network` para ambos archivos.

Recomendamos ejecutar en la línea de comandos cuando pruebes. Pero también puedes usar la visualización de Streamlit para ver los estados ocultos de tu modelo, como lo siguiente:

### Por hacer

- Entrena un modelo en Sentiment (SST2), y agrega los registros de impresión de entrenamiento como un archivo de texto `sentiment.txt` al repositorio. Debe mostrar pérdida de entrenamiento, precisión de entrenamiento y precisión de validación. (El modelo debe obtener >70% de precisión de validación óptima.)

- Entrena un modelo en Clasificación de Dígitos (MNIST) y agrega los registros como un archivo de texto `mnist.txt` al repositorio. Debe mostrar pérdida de entrenamiento y precisión de validación de 16 clases.



