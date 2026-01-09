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

```python
input: batch x channel x height x width
kernel: height x width del pooling
