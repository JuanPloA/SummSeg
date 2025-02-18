import argparse
import os
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import math
import csv

borde_size = 80

# Mapa de colores según la etiqueta (Label)
label_to_color = {
    "wall": (0,0,0),
    "good": (0,255,0),
    "cleaning": (0,0,255),
    "tools": (255,128,255),
    "poor": (255,255,0),
    "none": (128,128,128)
}

color_to_label = {
    (0,0,0): "wall" ,
    (0,255,0): "good",
    (0,0,255): "cleaning",
    (255,128,255): "tools",
    (255,255,0): "poor",
    (128,128,128): "white"
}

def traducir_csv_a_intervalos_y_colores(csv_file):
    intervalos = []
    colores = []
    
    # Leer el archivo CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        frames = []
        labels = []
        
        # Leer las filas del CSV
        for row in reader:
            frames.append(int(row['Frame']))
            labels.append(row['Label'])
        
        # Generar los intervalos
        start_frame = 0
        for i, frame in enumerate(frames):
            end_frame = frame
            intervalos.append((start_frame, end_frame))
            colores.append(label_to_color.get(labels[i], "black"))  # "black" es el color por defecto si no hay coincidencia
            start_frame = end_frame + 1  # El siguiente intervalo empieza después del frame actual

    return intervalos, colores


def obtener_archivos(directorio, indices):
    archivos = sorted(os.listdir(directorio))
    #minitest
    # last = 0
    # for archivo in archivos:
    #     # Divide el string por el carácter de subrayado ("_") y selecciona la última parte
    #     number_str = archivo.split('_')[-1]

    #     # Elimina el ".png" y convierte a número entero
    #     number = int(number_str.replace(".png", ""))

    #     if number != last + 1:
    #         print(number, last)
    #     last = number


    # Filtrar los archivos por los índices proporcionados
    archivos_filtrados = [archivos[indice - 1] for indice in indices]
    return archivos_filtrados

def crear_mosaico(directorio, archivos, intervalos, frames, colores, mosaico_color):

    # Calcular el número de imágenes
    num_imagenes = len(archivos)


    
    # Calcular filas y columnas para que sea lo más cuadrado posible
    columnas = math.ceil(math.sqrt(num_imagenes))
    filas = math.ceil(num_imagenes / columnas)

    # Cargar la primera imagen para obtener el tamaño de la celda
    imagen_prueba = Image.open(os.path.join(directorio, archivos[0]))
    ancho_original, alto_original = imagen_prueba.size
    
    # Reducir el tamaño de la imagen para hacer espacio para el borde interno
    ancho_reducido = ancho_original - 2 * borde_size
    alto_reducido = alto_original - 2 * borde_size

    nuevo_ancho, nuevo_alto = 100, 100
    
    # Crear un mosaico vacío del tamaño adecuado
    mosaico = Image.new('RGB', (columnas * ancho_original, filas * alto_original))
    # mosaico = Image.new('RGB', (columnas * (nuevo_ancho + 2 * borde_size), filas * (nuevo_alto + 2 * borde_size)))

    if mosaico_color:
        inter_actual = intervalos[0]
        inter_i = 0
    # Pegar cada imagen en su posición en el mosaico (se cargan una por una)
    for i, archivo in enumerate(archivos):
        if mosaico_color:
            while frames[i] > inter_actual[1]:
                inter_i+=1
                inter_actual = intervalos[inter_i]
            
        imagen_path = os.path.join(directorio, archivo)
        imagen = Image.open(imagen_path)

        if mosaico_color:
            # Redimensionar la imagen para hacer espacio para el borde
            imagen_reducida = imagen.resize((ancho_reducido, alto_reducido))
            
            # # Añadir el borde rojo a la imagen reducida
            imagen = ImageOps.expand(imagen_reducida, border=borde_size, fill=colores[inter_i])
            
        # Calcular posición x e y en el mosaico
        x = (i % columnas) * ancho_original
        y = (i // columnas) * alto_original

        mosaico.paste(imagen,(x,y))
        
        # Cerrar la imagen para liberar memoria
        imagen.close()
    return  mosaico.resize((ancho_original, alto_original))


def crear_barra(N, posiciones, intervalos, colores, gt, color_relleno=(255, 0, 0), altura=50):
    reduc = 10
    N = int(N/10)
    """
    Crea una barra blanca de tamaño N y rellena ciertas posiciones con un color especificado.
    
    Args:
    - N: Ancho total de la barra (en píxeles).
    - posiciones: Lista de posiciones (índices) que deben ser coloreadas.
    - color_relleno: Color con el que rellenar las posiciones (por defecto es rojo).
    - altura: Altura de la barra (en píxeles).
    
    Returns:
    - Una imagen de la barra con las posiciones coloreadas.
    """
    # Crear una imagen blanca del tamaño especificado
    barra = Image.new('RGB', (N, altura), color='white')
    draw = ImageDraw.Draw(barra)
    
    if gt:
        inter_actual = intervalos[0]
        inter_i = 0

    # Rellenar las posiciones especificadas con el color
    for pos in posiciones:
        if gt:
            while pos > inter_actual[1]:
                inter_i+=1
                inter_actual = intervalos[inter_i]
        
        pos = int(pos/reduc)
        if 0 <= pos < N:
            # Rellenar la posición específica (como un píxel o un rectángulo)
            if gt:
                draw.line([(pos, 0), (pos, altura)], fill=colores[inter_i], width=1)
            else:
                draw.line([(pos, 0), (pos, altura)], fill=(0,0,0), width=1)
    
    return barra






def fila_mosaico(directorio, archivos, intervalos, frames, colores):
    # Calcular el número de imágenes
    num_imagenes = len(archivos)
    
    # Calcular filas y columnas para que sea lo más cuadrado posible
    columnas = num_imagenes
    filas = 1

    # Cargar la primera imagen para obtener el tamaño de la celda
    imagen_prueba = Image.open(os.path.join(directorio, archivos[0]))
    ancho_original, alto_original = imagen_prueba.size
    
    # Reducir el tamaño de la imagen para hacer espacio para el borde interno
    ancho_reducido = ancho_original - 2 * borde_size
    alto_reducido = alto_original - 2 * borde_size
    
    # Crear un mosaico vacío del tamaño adecuado
    mosaico = Image.new('RGB', (columnas * ancho_original, filas * alto_original))
    # mosaico = Image.new('RGB', (columnas * (nuevo_ancho + 2 * borde_size), filas * (nuevo_alto + 2 * borde_size)))

    inter_actual = intervalos[0]
    inter_i = 0
    # Pegar cada imagen en su posición en el mosaico (se cargan una por una)
    for i, archivo in enumerate(archivos):
        while frames[i] > inter_actual[1]:
            inter_i+=1
            inter_actual = intervalos[inter_i]
            
        imagen_path = os.path.join(directorio, archivo)
        imagen = Image.open(imagen_path)
        
        # Redimensionar la imagen para hacer espacio para el borde
        imagen_reducida = imagen.resize((ancho_reducido, alto_reducido))
        
        # Añadir el borde rojo a la imagen reducida
        imagen_con_borde = ImageOps.expand(imagen_reducida, border=borde_size, fill=colores[inter_i])
        
        # Calcular posición x e y en el mosaico
        x = (i % columnas) * ancho_original
        y = (i // columnas) * alto_original
        
        # Pegar la imagen con borde en el mosaico
        mosaico.paste(imagen_con_borde, (x, y))
        
        # Cerrar la imagen para liberar memoria
        imagen.close()
    return mosaico.resize((ancho_original, alto_original))

def mosaico_tocho(directorio, archivos, intervalos, frames, colores):
    # Calcular el número de imágenes
    num_imagenes = len(archivos)
    
    imagen_prueba = Image.open(os.path.join(directorio, archivos[0]))
    ancho_original, alto_original = imagen_prueba.size
    
    # Calcular filas y columnas para que sea lo más cuadrado posible
    columnas = math.ceil(math.sqrt(num_imagenes))
    filas = math.ceil(num_imagenes / columnas)

    print(columnas, filas)

    mosaico = Image.new('RGB', (columnas * ancho_original, filas * alto_original))

    print("aaaaaaa")

    for i in range(filas-1):
        fila = fila_mosaico(directorio, archivos[(i*columnas):((i+1)*columnas)], intervalos, frames, colores)
    


    exit()