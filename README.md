# Introducción al curso de Machine Learning Aplicado con Python

Bienvenidos al **Curso de Machine Learning Aplicado con Python**, en este curso veremos cómo utilizar Machine Learning con distintas librerías de Python. Particularmente estaremos utilizando **_Scikit-Learn_**, que es una de las más utilizadas de la industria.

En este curso también nos enfocaremos es en entender todo el flujo de trabajo que se hace cuando se resuelve un problema de **Machine Learning**.

Además de entender muy bien los algoritmos de **Machine Learning** que estaremos viendo, veremos otras disciplinas que son tanto o más importantes como el **Feature Engineering** y la selección de modelos.

> **Machine Learning = Programación + Estadística**

## Importancia de definir el problema en Machine Learning

**Errores comunes** que se ven cuando no se define bien el problema y se comienza a codear:

- No hay problemas por resolver.
- Existen soluciones más simples.
- No se puede medir el impacto del modelo.
- No se sabe si el problema ya ha sido resuelto antes.
- El problema es imposible de resolver.

**Preguntas clave** para reconocer el **tipo de aprendizaje** que se necesita:

- ¿Qué beneficio se puede generar y para quién?
- ¿Cuál de las siguientes funcionalidades sería más útil para lograr el objetivo?

**a)** Predecir alguna métrica (Aprendizaje supervisado)
**b)** Predecir una etiqueta (Aprendizaje supervisado)
**c)** Agrupar elementos similares.
**d)** Optimizar un proceso con prueba y error.

**Preguntas clave** para aterrizar el **problema de aprendizaje supervisado**:

- ¿De qué tipo es el valor que se quiere predecir?

  - a) Continuo
  - b) Discreto

- ¿Cuál es la definición de éxito en una predicción?
- ¿Con qué datos se contaría para hacer esa predicción?
- ¿La pregunta que se está tratando de resolver pertenece a alguna disciplina en particular?
- Considerando nuestra intuición en la disciplina ¿Los datos nos permiten predecir el objetivo?

## Terminología de Machine Learning

- **Datos tabulares** = Datos en dos dimensiones.
- **Líneas** = Ejemplos
- **Columna** = Feauture. Éstas son importantes porque nos van a ayudar a predecir cosas gracias a los modelos que usemos de Machine Learning.
- **Cantidad de columnas** = Dimensión de los datos
- **Output de un algoritmo de Machine Learning (ML)** = Modelo
  Variable objetivo = Target

Los materiales del curso se encuentran en el repositorio [Machine Learning Platzi](https://github.com/JuanPabloMF/machine-learning-platzi/tree/master)

## El ciclo de Machine Learning

Muchas veces pensamos que hacer **Machine Learning** corresponde solamente a implementar un algoritmo de cualquiera de las librerías y con ello ya existe la solución a un problema. Pero en realidad existe todo un ciclo de trabajo donde los algoritmos de **Machine Learning** son solo una etapa, sin embargo, las demás etapas también son muy importantes y toman su tiempo para lograr los resultados que esperamos.

Hacer **Machine Learning** corresponde a trabajar en un ciclo, ir trabajando varias etapas e ir iterando.

**Ciclo de Machine Learning**:

- Definición del problema.
  - Aprendizaje Supervisado?
  - Variable Objetivo (Labels)
  - Métrica de evaluación
- Preparación de los datos.
  - Obtención de datos
  - Join de BDDS
  - Limpieza de datos
- Representación de los datos.
  - Análisis exploratorio de dayos
  - Extracción manual de features
  - Extracción automática
  - Selección de features
- Modelamiento / Algoritmos de ML.
  - Selección del modelo
  - Fiteo de algoritmo
  - Predicción
- Evaluación.
  - Cross Validation
  - Underfill o Overtilting
  - Optimización de hyperparámetros

Este no es el final del proceso, se debe iterar hasta que en alguna de las iteraciones salga la solución al problema.

- Producción (Fin del proceso).

> **Navaja de Ockans o pasimonia** Si se tine 2 soluciones a un problema y una es más simple, simpre optar por la más sencilla y poco a poco se va iterar el algoritmo para hacerlo más complejo.

> El **Feature Engineering** es muy importante; algunos expertos mencionan que todo el arte del Machine Learning está en definir bien los features

## Montar un ambiente de trabajo Pydata

En este curso utilizaremos **Jupyter Notebooks** y el **stack Pydata**, que incluye las librerías _Numpy, Pandas, Matplotlib y Scikit-learn_.

En lo que sigue te presento 3 opciones para trabajar con estas tecnologías:

1. Trabajar en el cloud gracias a la excelente herramienta Google Colab.
2. Trabajar en local en tu computadora gracias a Anaconda.
3. Trabajar en tu computadora virtualizando a través de contenedores con Docker.

> Google Colab es la opción mas facil de usar, así que esta es la opción oficial recomendada por el curso.

_Anaconda permite que trabajes en local, y Docker es bueno para usos en producción_

### :star: Google Colab

Google Colab es muy similar en su uso a Google Docs. Puedes ir directamente a su pagina https://colab.research.google.com/ o abrir Google Drive, clickear en el botón “+ Nuevo” y desde ahí elegir la opción “Mas” y clickear en Colaboratory (como se ve en la imagen)

![Google Colab](https://i.imgur.com/2JxZrAD.png)

_Con esto tendrás un notebook de Google Colab a tu disposición!_

![Google Colab Notebook](https://i.imgur.com/3VuujFB.png)

:fire: Un notebook se compone de celdas de codigo y de texto. Una celda de codigo es ejecutable tecleando **“Control + Enter”**. Para probar que todo esta ok puedes ejecutar el siguiente codigo:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
```

Veras arriba a la derecha que colab se esta conectando a una instancia de computo en la nube (esto colab lo hace solo, no te preocupes!). Esta conexión la realiza solo la primera vez.

![Connect Google Colab](https://i.imgur.com/GjChCKJ.png)

Una vez conectado, el código que ingresaste se ejecutara, y veras que efectivamente los imports se harán sin ningún problema por lo que ya tienes todo lo que necesitas para trabajar!

Un solo punto adicional que queda por aclarar es que como estas trabajando en la nube no tienes directamente los archivos de la clase a mano para poder leerlos desde el notebook.

Para poder acceder los archivos del curso puedes hacer lo siguiente:

1. Encuentra el link hacia el archivo que quieres cargar en el repositorio de github https://github.com/JuanPabloMF/datasets-platzi-course
2. Con el link del archivo csv puedes llamar la función de pandas read_csv de la siguiente manera:

```py
import pandas as pd
pd.read_csv(url)
```

Ya puedes entrar de llenos al curso y empezar a implementar tus primeros modelos de machine learning!

Para más información ver la guía [Configuración del ambiente de trabajo con Google Collab](setup-environment.pdf). Abrirla en un navegador.
