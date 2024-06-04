# Sabor al Dato: Evaluación de Ingredientes y Saborizantes para una Vida Saludable
Este proyecto consiste en una aplicación web que permite a cualquier usuario mediante una interfaz evaluar la calidad alimenticia de un producto. Para ello, se ha desarrollado como back-end de la aplicación una **red-neuronal** que recibirá como entrada un vector formado a apartir de la lista de ingredientes insertada por el usuario y devolverá una predicción de los valores de NutriScore y NovaScore.

## Despliegue de la aplicación web

### Requisitos
[Python 3.12.1](https://www.python.org/downloads/release/python-3121/)

### Descarga e instalación

+ Instalar la versión de [Python 3.12.1](https://www.python.org/downloads/release/python-3121/)
+ Clonar el repositorio y entrar dentro del directorio principal:
  ```bash
  git clone https://github.com/TitorSpace/TFG.git
  cd tfgWEB
  ```
+ Se crea un entorno virtual:
  +   Linux:
    ```bash
      source env/bin/activate
    ```
  +   Windows:
    ```bash
      .\env\Scripts\Activate
    ```
+ Se instalan todas las dependencias:
  ```bash
      pip install -r requirements.txt
  ```
+ Los datasets usados ocupan bastante espacio asi que se han almacenado en Google Drive donde se podrá descargar el .zip con [el dataset de inicio](https://drive.google.com/file/d/1XF19QlivTv6jolIUjg8r1fi9pANTsZS7/view?usp=sharing): 
+ Descomprimir el zip y guardar el archivo **openFoodFactsProducts.csv** en la carpeta **datasets**.

+ Para que todo se ejecute de forma correcta hay que seguir cierto orden: se ejecuta inicialmente el archivo *mining.py* para elaborar los datos de entrada de la red. Posteriormente se ejecuta el archivo **trainning_cross_val.py** para generar los modelos neuronales. Eventualmente se ejecuta en la carpeta *app-web-project* el script **app.py** que será el responsable de iniciar la sesión para la página web.
+ Una vez ejecutado el último script mencionado, enlace para acceder a la web de forma local es [este](http://127.0.0.1:5000)
+ Las rutas de exportación y de importación de los archivos está en el script **config.py**. Si se desea modificar alguna ruta solo hay que modificar la variable asociada a esa ruta.
