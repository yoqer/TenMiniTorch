# Asignación - MiniTorch

![kimi-17679474646094266329487416457710](https://github.com/user-attachments/assets/ba94a295-7a48-448a-b911-0af897f1be2d)



## Eficiencia

Además de ayudar a simplificar el código, los tensores proporcionan una base para acelerar la computación. De hecho, son realmente la única forma de escribir código de aprendizaje profundo de manera eficiente en un lenguaje lento como Python. Sin embargo, nada de lo que hemos hecho hasta ahora realmente hace que algo sea más rápido que: `Module-0`


Este módulo se enfoca en aprovechar los tensores para escribir código rápido, primero en CPUs estándar y luego usando GPUs.

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

Esta tarea requiere familiaridad básica con: 
`prange` 

de Numba.

Asegúrate de leer muy cuidadosamente la sección sobre paralelismo, [Numba](https://numba.pydata.org/numba-doc/latest/user/parallel.html).

El backend principal para nuestra base de código son las tres funciones: `map`, `zip` y `reduce`

Si podemos acelerar estas tres, todo lo que hemos construido hasta ahora mejorará. Este ejercicio te pide que utilices Numba y la función `njit` 

para acelerar estas funciones. En particular, si puedes utilizar la paralelización a través de `prange` 

puedes obtener grandes mejoras. ¡Pero ten cuidado! La paralelización puede llevar a errores extraños.

Para ayudarte a depurar este código, hemos creado un script de análisis paralelo para ti:

```bash
python project/parallel_check.py

```

________________________________________________________________________________________________________________________________________





Ejecutar este script ejecutará diagnósticos de NUMBA en tus funciones.

Por hacer

Completa lo siguiente en `minitorch/fast_ops.py` y pasa las pruebas marcadas como `task3_1`.

- Incluye la salida de diagnósticos del script anterior en tu README.
- Asegúrate de que el código implemente las optimizaciones especificadas en los docstrings. Verificaremos esto explícitamente.

`minitorch.fast_ops.tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]`

Función tensor_map de bajo nivel de NUMBA. Ver `tensor_ops.py` para la descripción.

Optimizaciones:

- Bucle principal en paralelo
- Todos los índices usan buffers de numpy
- Cuando `out` e `in` están alineados en stride, evita la indexación

---

```python
fn: función de mapeo float-a-float para aplicar.
```

---

`minitorch.fast_ops.tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]`

Función tensor zip de orden superior de NUMBA. Ver `tensor_ops.py` para la descripción.

Optimizaciones:

- Bucle principal en paralelo
- Todos los índices usan buffers de numpy
- Cuando `out`, `a`, `b` están alineados en stride, evita la indexación

---

```python
fn: función que mapea dos floats a float para aplicar.
```

---

`minitorch.fast_ops.tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]`

Función tensor reduce de orden superior de NUMBA. Ver `tensor_ops.py` para la descripción.

Optimizaciones:

- Bucle principal en paralelo
- Todos los índices usan buffers de numpy
- El bucle interno no debe llamar ninguna función o escribir variables no locales

---

```python
fn: función de reducción que mapea dos floats a float.
```

---

Tarea 3.2: Multiplicación de Matrices

La multiplicación de matrices es clave para todos los modelos que hemos entrenado hasta ahora. En el último módulo, calculamos la multiplicación de matrices usando broadcasting. En esta tarea, te pedimos que la implementes directamente como una función. Haz tu mejor esfuerzo para hacer la función eficiente, pero por ahora lo único que importa es que produces correctamente una función multiply que pase nuestras pruebas y tenga algún paralelismo.

Para usar esta función, también necesitarás agregar una nueva función `MatMul` a `tensor_functions.py`. Hemos agregado una versión en el código inicial que puedes copiar. También podría ser útil agregar una `matrix_multiply` lenta con broadcasting a `tensor_ops.py` para depuración.

Para ayudarte a depurar este código, puedes usar el script de análisis paralelo.

Después de terminar esta tarea, puedes saltar a la 3.5 y experimentar con el entrenamiento en la tarea real bajo condiciones de velocidad.

Por hacer

Completa la siguiente función en `minitorch/fast_ops.py`. Pasa las pruebas marcadas como `task3_2`.

- Incluye la salida de diagnósticos del script anterior en tu README.
- Asegúrate de que el código implemente las optimizaciones especificadas en los docstrings. Verificaremos esto explícitamente.

`minitorch.fast_ops._tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None`

Función de multiplicación de matrices tensor de NUMBA.

Debe funcionar para cualquier forma de tensor que haga broadcast siempre que:

```python
assert a_shape[-1] == b_shape[-2]
```

Optimizaciones:

- Bucle externo en paralelo
- Sin buffers de índice ni llamadas a funciones
- El bucle interno no debe tener escrituras globales, 1 multiplicación.

---

```python
out (Storage): almacenamiento para el tensor `out`
out_shape (Shape): forma para el tensor `out`
out_strides (Strides): strides para el tensor `out`
a_storage (Storage): almacenamiento para el tensor `a`
a_shape (Shape): forma para el tensor `a`
a_strides (Strides): strides para el tensor `a`
b_storage (Storage): almacenamiento para el tensor `b`
b_shape (Shape): forma para el tensor `b`
b_strides (Strides): strides para el tensor `b`
```

---

Tarea 3.3: Operaciones CUDA

Podemos hacer aún mejor que la paralelización si tenemos acceso a hardware especializado. Esta tarea te pide que construyas una implementación GPU de las operaciones del backend. Será difícil igualar lo que hace PyTorch, pero si eres inteligente puedes hacer estas computaciones realmente rápidas (apunta a 2x de la tarea 3.1).

Reduce es una función particularmente desafiante. Proporcionamos guías y una función de práctica simple para ayudarte a comenzar.

Por hacer

Completa las siguientes funciones en `minitorch/cuda_ops.py`, y pasa las pruebas marcadas como `task3_3`.

`minitorch.cuda_ops.tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]`

Función tensor map de orden superior de CUDA. ::

```python
fn_map = tensor_map(fn)
fn_map(out, ... )
```

---

```python
fn: función de mapeo float-a-float para aplicar.
```

---

`minitorch.cuda_ops.tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]`

Función tensor zipWith (o map2) de orden superior de CUDA ::

```python
fn_zip = tensor_zip(fn)
fn_zip(out, ...)
```

---

```python
fn: función que mapea dos floats a float para aplicar.
```

---

`minitorch.cuda_ops._sum_practice(out: Storage, a: Storage, size: int) -> None`

Este es un kernel de suma de práctica para prepararte para reduce.

Dado un array de longitud \\(n\\) y out de tamaño \\(n // ext{blockDIM}\\)
debe sumar cada blockDim valores en una celda out.

\\\\(\\[a_1, a_2, ..., a{100}\\]\\\\)

\\|

\\\\(\\[a_1 +...+ a{31}, a{32} + ... + a{64}, ... ,\\]\\\\)

Nota: ¡Cada bloque debe hacer la suma usando memoria compartida!

---

```python
out (Storage): almacenamiento para el tensor `out`.
a (Storage): almacenamiento para el tensor `a`.
size (int): longitud de a.
```

---

`minitorch.cuda_ops.tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]`

Función tensor reduce de orden superior de CUDA.

---

```python
fn: función de reducción que mapea dos floats a float.
```

---

Tarea 3.4: Multiplicación de Matrices CUDA

Finalmente podemos combinar ambos enfoques e implementar `matmul` en CUDA. Esta operación es probablemente la más importante en todo el aprendizaje profundo y es central para hacer que los modelos sean rápidos. Nuevamente, primero nos esforzamos por la precisión, pero cuanto más rápido puedas hacerla, mejor.

Implementar multiplicación de matrices y reducción eficientemente es enormemente importante para muchas tareas de aprendizaje profundo. Sigue las guías proporcionadas en clase para implementar estas funciones.

Debes documentar tu código para mostrarnos que entiendes cada línea. Muéstranos que esto conduce a aceleraciones en operaciones de matrices grandes haciendo una gráfica comparándolas con operaciones ingenuas.

Por hacer

Implementa `minitorch/cuda_ops.py` con CUDA, y pasa las pruebas marcadas como `task3_4`. Sigue los requisitos especificados en la documentación.

`minitorch.cuda_ops._mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None`

Este es un kernel MM cuadrado de práctica para prepararte para matmul.

Dado un almacenamiento `out` y dos almacenamientos `a` y `b`. Donde sabemos
que ambos tienen forma \\[size, size\\] con strides \\[size, 1\\].

Size es siempre < 32.

Requisitos:

- Todos los datos deben moverse primero a memoria compartida.
- Solo lee cada celda en `a` y `b` una vez.
- Solo escribe a memoria global una vez por kernel.

Compute

```python
for i:
    for j:
        for k:
            out[i, j] += a[i, k] * b[k, j]
```

---

```python
out (Storage): almacenamiento para el tensor `out`.
a (Storage): almacenamiento para el tensor `a`.
b (Storage): almacenamiento para el tensor `b`.
size (int): tamaño del cuadrado
```

---

`minitorch.cuda_ops._tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides, out_size: int, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None`

Función de multiplicación de matrices tensor de CUDA.

Requisitos:

- Todos los datos deben moverse primero a memoria compartida.
- Solo lee cada celda en `a` y `b` una vez.
- Solo escribe a memoria global una vez por kernel.

Debe funcionar para cualquier forma de tensor que haga broadcast siempre que ::

```python
assert a_shape[-1] == b_shape[-2]
```

Returns:
None: Llena `out`

Tarea 3.5: Entrenamiento

Si tu código funciona, ahora deberías poder pasar al script de entrenamiento tensor en `project/run_fast_tensor.py`. Este código es la misma configuración básica de entrenamiento que `module2`, pero ahora utiliza tu código tensor rápido. Hemos dejado la capa `matmul` en blanco para que la implementes con tu código tensor.

Aquí está el comando ::

```bash
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

Por hacer

- Implementa las funciones faltantes en `project/run_fast_tensor.py`. Estas deben hacer exactamente lo mismo que las funciones correspondientes en `project/run_tensor.py`, pero ahora usar el backend más rápido

- Entrena un modelo tensor y agrega tus resultados para todos los conjuntos de datos al README.

- Ejecuta un modelo más grande y registra el tiempo por época reportado por el entrenador.

Entrena un modelo tensor y agrega tus resultados para los tres conjuntos de datos al README. También registra el tiempo por época reportado por el entrenador. (Como referencia, nuestra implementación paralela dio una aceleración de 10x).
En una configuración estándar de GPU de Colab, apunta a que tu CPU esté por debajo de 2 segundos por época y tu GPU por debajo de 1 segundo por época. (Con algo de ingenio puedes hacerlo mucho mejor.)

```




