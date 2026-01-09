# Asignación - MiniTorch

## Eficiencia

Además de ayudar a simplificar el código, los tensores proporcionan una base para acelerar la computación. De hecho, son realmente la única forma de escribir código de aprendizaje profundo de manera eficiente en un lenguaje lento como Python. Sin embargo, nada de lo que hemos hecho hasta ahora realmente hace que algo sea más rápido que `module0`. Este módulo se enfoca en aprovechar los tensores para escribir código rápido, primero en CPUs estándar y luego usando GPUs.

Necesitas los archivos de asignaciones anteriores, así que asegúrate de traerlos a tu nuevo repositorio.

## Guías

- [Paralelismo](https://minitorch.github.io/module3/parallel/)
- [Multiplicación de Matrices](https://minitorch.github.io/module3/matrixmult/)
- [CUDA](https://minitorch.github.io/module3/cuda/)

Para esta asignación, necesitarás acceso a una GPU. Recomendamos ejecutar comandos en un entorno de Google Colab. Sigue estas instrucciones para la [configuración de Colab](https://colab.research.google.com/drive/1QLWrwMANAAUc-gouHwYqluLa9CKBsDFc?usp=sharing).

Esta asignación no requiere que modifiques el objeto tensor principal. En su lugar, solo cambiarás el código de operaciones de orden superior central.

- `fast_ops.py`: Operaciones de bajo nivel para CPU
- `cuda_ops.py`: Operaciones de bajo nivel para GPU

## Tarea 3.1: Paralelización

**Nota**

Esta tarea requiere familiaridad básica con `prange` de Numba.

Asegúrate de leer muy cuidadosamente la sección sobre paralelismo, [Numba](https://numba.pydata.org/numba-doc/latest/user/parallel.html).

El backend principal para nuestra base de código son las tres funciones `map`, `zip` y `reduce`. Si podemos acelerar estas tres, todo lo que hemos construido hasta ahora mejorará. Este ejercicio te pide que utilices Numba y la función `njit` para acelerar estas funciones. En particular, si puedes utilizar la paralelización a través de `prange` puedes obtener grandes mejoras. ¡Pero ten cuidado! La paralelización puede llevar a errores extraños.

Para ayudarte a depurar este código, hemos creado un script de análisis paralelo para ti:

```bash
python project/parallel_check.py

