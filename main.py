
import torch
import torch.nn as nn
from torchvision import models, transforms
from tensorflow.keras.applications import ResNet50
from PIL import Image
import h5py
import os
from tqdm import tqdm
import yaml
import subprocess
import json
import argparse
import numpy as np
from utils.reducers import *
from utils.visuals import *
import matplotlib.pyplot as plt
import time
import re

from model.encoder import *

total_time = 0



# Eliminar la capa de clasificación final para obtener el tamaño de salida 1024
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Eliminar la capa final del modelo original
        self.features = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x



# Transformaciones para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para cargar una imagen y extraer características
def ft_extract(image_path):
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    image = image.to(device)

    # Extraer características
    with torch.no_grad():
        features = feature_extractor(image)
    
    return features



title_0 = '''
   ___     _              _    _ _   _  _____ _    _  __      __     _____   _____ 
  / _ \   | |        /\  | |  | | \ | |/ ____| |  | | \ \    / /\   |  __ \ / ____|
 | | | |  | |       /  \ | |  | |  \| | |    | |__| |  \ \  / /  \  | |__) | (___  
 | | | |  | |      / /\ \| |  | | . ` | |    |  __  |   \ \/ / /\ \ |  _  / \___ \ 
 | |_| |  | |____ / ____ \ |__| | |\  | |____| |  | |    \  / ____ \| | \ \ ____) |
  \___(_) |______/_/    \_\____/|_| \_|\_____|_|  |_|     \/_/    \_\_|  \_\_____/ '''

print(title_0)

start_time = time.time()
with open("config.yml", 'r') as archivo:
    datos = yaml.safe_load(archivo)
                                                                                   
    # Directorio que contiene los frames
    directorio_frames = datos['directorio_frames']

    sequencia = os.path.basename(directorio_frames)
    # Extraer el nombre del último directorio
    nombre_seq = datos['nombre']

    evaluation = datos['evaluation']

    correct = datos['correct']
                                                                                    
    extract_features = datos['extract_features']

    inference = datos['inference']

    discard_size = datos['discard_size']

    max_compress = datos['max_compress']

    mosaico_bin = datos['mosaico_bin']

    method = datos['method']

    gpu = datos['gpu']

    N = datos['N']

    visualization = datos['visualization']

    percentile = datos['percentile']

    mosaico = datos['mosaico']

    batches = datos['batches']

    bin_segment = datos['bin_segment']


    #Voy a cargar la segmentacion en funcion del nombre

    seq_num = nombre_seq.split("_")[1]

    if method == "SummSegInv":
        batches = 1
        corrected_bin = True
    else:
        corrected_bin = False

    segment = "data/etiq_class_" + str(seq_num) + ".txt"

    if corrected_bin:
        bin_segment = "data/etiq_correct_" +  str(seq_num) + "_"+ str(correct) + ".txt"
    else:
        bin_segment = "data/etiq_bin_" +  str(seq_num) + ".txt"

    ground_truth = "data/etiq_seq_" + str(seq_num) + ".csv"

    end_time = time.time()

    if gpu:
         print("GPU avaliable:", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

    print("directorio_frames:", directorio_frames)
    print("nombre_seq:", nombre_seq)
    print("inference:", inference)
    print("method:", method)
    print("percentile:", percentile)
    print("N:", N)
    print("ground_truth:", ground_truth)
    print("visualization:", visualization)
    print("batches:", batches)

    cargar_variables_time = end_time - start_time
   
    
if max_compress:
    if corrected_bin:
        res_dir = "results/" + sequencia + "_" + str(method) + "_" + str(percentile) + "_" + str(batches) + "_max_compress" + str(correct)   + "_" + str(discard_size)
    else:
        res_dir = "results/" + sequencia + "_" + str(method) + "_" + str(percentile) + "_" + str(batches) + "_max_compress"  + "_" + str(discard_size)
else:
    if corrected_bin:
        res_dir = "results/" + sequencia + "_" + str(method) + "_" + str(percentile) + "_" + str(batches) + "_" + str(N) + "_" + str(correct) + "_" + str(discard_size)
    else:
        res_dir = "results/" + sequencia + "_" + str(method) + "_" + str(percentile) + "_" + str(batches) + "_" + str(N) + "_" + str(discard_size)
os.makedirs(res_dir, exist_ok=True)



if extract_features:
    
    title_1 = '''
    __     ______ ______       _______ _    _ _____  ______   ________   _________ _____            _____ _______ 
    /_ |   |  ____|  ____|   /\|__   __| |  | |  __ \|  ____| |  ____\ \ / /__   __|  __ \     /\   / ____|__   __|
    | |   | |__  | |__     /  \  | |  | |  | | |__) | |__    | |__   \ V /   | |  | |__) |   /  \ | |       | |   
    | |   |  __| |  __|   / /\ \ | |  | |  | |  _  /|  __|   |  __|   > <    | |  |  _  /   / /\ \| |       | |   
    | |_  | |    | |____ / ____ \| |  | |__| | | \ \| |____  | |____ / . \   | |  | | \ \  / ____ \ |____   | |   
    |_(_) |_|    |______/_/    \_\_|   \____/|_|  \_\______| |______/_/ \_\  |_|  |_|  \_\/_/    \_\_____|  |_| '''

    print(title_1)  



    # SON FEATURES DE GOOGLENET DE 1024
    model = models.googlenet(pretrained=True)

    # model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    model.eval()

    # Crear el extractor de características
    feature_extractor = FeatureExtractor(model)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Establecer en modo de evaluación


    # Lista para almacenar las características
    features_list = []
    i = 0
    ii = 0
    features_acum = None

    #Vamos a cambiar la lista sobre la que itera frame_name
    imagenes = sorted(os.listdir(directorio_frames)) #nombre de las imagenes del directorio

    # if guided: #Vamos a filtrar imagenes
    #     intervalos_buenos = []

    #     with open(segment, "r") as file:
    #         for linea in file:
    #             partes = linea.split()  # Dividir la línea por espacios
    #             inicio, fin, etiqueta = int(partes[0]), int(partes[1]), partes[2]
    #             if etiqueta == "good":
    #                 intervalos_buenos.append((inicio, fin))
    #     print(intervalos_buenos[:5])

    #     # Filtrar el array usando los intervalos buenos
    #     imagenes_filtradas = [
    #         elemento for i, elemento in enumerate(imagenes)
    #         if any(inicio <= (i+1) <= fin for inicio, fin in intervalos_buenos)
    #     ]
    #     imagenes = imagenes_filtradas
    start_time = time.time()
    # Procesar cada frame en el directorio
    for frame_name in tqdm(imagenes, desc="Extrayendo features", unit="imagenes"):
        frame_path = os.path.join(directorio_frames, frame_name)
        if frame_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            #tenemos un frame
            i += 1
            # Extraer características del frame
            features = ft_extract(frame_path)
            features = features.to("cpu")
            if ii < batches: # Tenemos que acumular
                if ii == 0:
                    features_acum = features
                else:
                    features_acum += features
                ii += 1
            else: # Hemos acumulado ya
                features_acum = features_acum / batches
                features_list.append(features_acum.squeeze(0))  # Eliminar dimensión de batch
                features_acum = features
                ii = 1

    
    # Guardar las características en el fichero npz
    end_time = time.time()
    all_features = np.array(features_list)
    print(all_features.shape)

    np.save("data/features/" + sequencia + "_" + str(batches) + ".npy", all_features)
   

    extract_time = end_time - start_time

        
    
    total_time += end_time-start_time
    print("Características guardadas en features/" + sequencia + ".npy")
    print(f"Tiempo: {end_time - start_time:.4f} segundos")


if inference:
    
    title_2 = '''
    ___      _____ _   _ ______ ______ _____  ______ _   _  _____ ______ 
   |__ \    |_   _| \ | |  ____|  ____|  __ \|  ____| \ | |/ ____|  ____|
      ) |     | | |  \| | |__  | |__  | |__) | |__  |  \| | |    | |__   
     / /      | | | . ` |  __| |  __| |  _  /|  __| | . ` | |    |  __|  
    / /_ _   _| |_| |\  | |    | |____| | \ \| |____| |\  | |____| |____ 
   |____(_) |_____|_| \_|_|    |______|_|  \_\______|_| \_|\_____|______|
                                                                        
                                                                        
    '''

    print(title_2)

    features = list(np.load("data/features/" + sequencia + "_" + str(batches)   + ".npy", allow_pickle=True))

    features = torch.stack(features)

    


    model = SumCapEncoder()
        
    model.eval()
    checkpoint_path = "model/epoch=054.ckpt"
    checkpoint = torch.load(checkpoint_path)
    encoder_ckpt = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith("bert")}
    encoder_ckpt = {k.replace("encoder.", ""): v for k, v in encoder_ckpt.items()}
    model.load_state_dict(encoder_ckpt)


    start_time = time.time()
    if corrected_bin:
        all_scores = []
        os.makedirs("data/inferences/" + sequencia + "_" + str(batches) + "_subscores", exist_ok=True)
        corr_intervalos, _ = procesar_fichero(bin_segment)
        i = -1
        for inicio, fin in corr_intervalos:
            i+=1
            sub_tensor = features[inicio:fin,:]  # Extraer columnas del intervalo
            tensor = sub_tensor.unsqueeze(0)
            enc_output, scores = model(tensor)
            print(scores.shape)
            scores = scores.detach().cpu().numpy()
            all_scores.append(scores)
            np.save("data/inferences/" + sequencia + "_" + str(batches) + "_subscores/" + str(i) + ".npy", scores)
        
        end_time = time.time()
    else:
        tensor = features.unsqueeze(0)
        enc_output, scores = model(tensor)
    

        # Convertir el tensor a un array de NumPy
        scores = scores.detach().cpu().numpy()
        end_time = time.time()
    

        # Guardar el array en un archivo .npz
        np.save("data/inferences/" + sequencia + "_" + str(batches)  + "_scores.npy", scores)
    
    print("Scores guardados en data/inferences/" + sequencia + "_" + str(batches)  + "_scores.npy" )
    inference_time = end_time -start_time

    print("INFERNECE TIME:", inference_time)
        
    total_time += end_time-start_time
    scores = scores[0]

    with open("data/inferences/" + sequencia + "_" + str(batches)   + "_times.txt", "w") as f:
        f.write(f"Extract features: {extract_time:.4f} segundos\n")
        f.write(f"Inference: {inference_time:.4f} segundos\n")
        f.write(f"Total: {total_time:.4f} segundos\n")
    


title_3='''
  ____     _____   ____   _____ _______ _____  _____   ____   _____ ______  _____ _____ 
 |___ \   |  __ \ / __ \ / ____|__   __|  __ \|  __ \ / __ \ / ____|  ____|/ ____/ ____|
   __) |  | |__) | |  | | (___    | |  | |__) | |__) | |  | | |    | |__  | (___| (___  
  |__ <   |  ___/| |  | |\___ \   | |  |  ___/|  _  /| |  | | |    |  __|  \___ \\___ \ 
  ___) |  | |    | |__| |____) |  | |  | |    | | \ \| |__| | |____| |____ ____) |___) |
 |____(_) |_|     \____/|_____/   |_|  |_|    |_|  \_\\____/ \_____|______|_____/_____/ 
                                                                                        
                                                                                        
'''
start_time = time.time()

print(title_3)

if not inference: #Hay que cargar los scores

    if not corrected_bin:
        # Cargar el archivo .npz
        scores = list(np.load("data/inferences/" + sequencia + "_" + str(batches)  + "_scores.npy"))[0]

        print(len(scores))

    else:
        all_scores = []
        # Cuenta solo los archivos, excluyendo directorios
        directorio = "data/inferences/" + sequencia + "_" + str(batches) + "_subscores" 
        numero_archivos = len([f for f in os.listdir(directorio) if os.path.isfile(os.path.join(directorio, f))])
        for i in range(numero_archivos):
            scores = list(np.load("data/inferences/" + sequencia + "_" + str(batches) + "_subscores/" + str(i) + ".npy"))[0]
            all_scores.append(scores)
if not corrected_bin:
    max = np.max(scores)

   
    

    norm_scores = scores/max

    print("media scores:", np.mean(norm_scores))

    # ###############################
    # full_len = len(norm_scores) * batches
    # percent = N/full_len

    # percentile = 100 - (batches*percent*100)
    # ##################################

    th = np.percentile(norm_scores, percentile)

    print("TH:", th)

    i = 0

    resumen = []
    for sc in norm_scores:
        if sc > th:
            # print(i, sc)
            resumen.append(i)
        i+=1
    print("num frames:", len(resumen)*batches)

    frames = []
    for seg in resumen:
        for i in range(batches):
            frames.append((seg*batches)+i)



    # print(len(frames))
    # print(frames)


    resumen_ubiss = frames

else:
    resumen = []
    offset = 0
    intervalos, etiqs = procesar_fichero(bin_segment)
    # Combinar intervalos, etiquetas y números en una estructura única
    data_combinada = list(zip(intervalos, etiqs, all_scores))

    # Ordenar por etiqueta (buenos primero) y tamaño del intervalo (fin - inicio)
    data_ordenada = sorted(
        data_combinada,
        key=lambda x: (x[1], x[0][1] - x[0][0])  # Ordena por etiqueta y luego por tamaño
    )

    # Separar los intervalos, etiquetas y números en listas individuales ordenadas
    intervalos_ordenados = [item[0] for item in data_ordenada][::-1]
    etiquetas_ordenadas = [item[1] for item in data_ordenada][::-1]
    scores_ordenados = [item[2] for item in data_ordenada][::-1]

    frames_per_interval = [0 for i in range(len(intervalos_ordenados))]
    if max_compress == False:
        faltan = N
        while faltan > 0:
            for i,inter in enumerate(intervalos_ordenados):

                if faltan == 0:
                    break
                if etiquetas_ordenadas[i] == 1:
                    frames_per_interval[i] += 1
                    faltan-=1
                else:
                    if frames_per_interval[i] == 0:
                        frames_per_interval[i] = 1
                        faltan -= 1
    else:
        for i,inter in enumerate(intervalos_ordenados):
            if etiquetas_ordenadas[i] == 1:
                frames_per_interval[i] = 1
            else:
                if frames_per_interval[i] == 0:
                    frames_per_interval[i] = 1

    
    for i, n_frames in enumerate(frames_per_interval):
        if n_frames != 0:
            inter = intervalos_ordenados[i]
            scores = scores_ordenados[i]
            inicio, fin = inter
            max = np.max(scores)

            norm_scores = scores/max
            indices_mas_grandes = np.argsort(norm_scores)[-n_frames:][::-1]

            for indice in indices_mas_grandes:
                resumen.append(inicio + indice)




    resumen.sort()

        
    frames = resumen
        

with open(res_dir + '/resumen_ubiss.txt', 'w') as f:
    for numero in frames:
        f.write(f"{numero}\n")

end_time = time.time()
print("Resumen guardado en" +  res_dir + '/resumen_ubiss.txt')

summ_gen_time = end_time - start_time
    
total_time += end_time-start_time

title_4 = '''
  _  _      _____ _    _ __  __ __  __          _______     __    _____  ______ _____  _    _  _____ ______ 
 | || |    / ____| |  | |  \/  |  \/  |   /\   |  __ \ \   / /   |  __ \|  ____|  __ \| |  | |/ ____|  ____|
 | || |_  | (___ | |  | | \  / | \  / |  /  \  | |__) \ \_/ /    | |__) | |__  | |  | | |  | | |    | |__   
 |__   _|  \___ \| |  | | |\/| | |\/| | / /\ \ |  _  / \   /     |  _  /|  __| | |  | | |  | | |    |  __|  
    | |_   ____) | |__| | |  | | |  | |/ ____ \| | \ \  | |      | | \ \| |____| |__| | |__| | |____| |____ 
    |_(_) |_____/ \____/|_|  |_|_|  |_/_/    \_\_|  \_\ |_|      |_|  \_\______|_____/ \____/ \_____|______|
                                                                                                            
                                                                                                            
'''

print(title_4)
start_time = time.time()

metrics = [0, 0]
if method == 'uniform':
    
    resumen_reducido = uniform_reduce(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen_uniform', N, False)
    print("Resumen reducido guardado en" +  res_dir + '/resumen_uniform.txt')
elif method == 'Ubiss_Uniform':
    resumen_reducido = uniform_reduce_v2(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen_uniform', N, norm_scores, batches)
    print("Resumen reducido guardado en" +  res_dir + '/resumen_uniform.txt')
elif method == "guided":
    resumen_reducido = guided_reduce(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen_guided', N, bin_segment, False)
    print("Resumen reducido guardado en" +  res_dir + '/resumen_guided.txt')
elif method == "guided_2":
    resumen_reducido = guided_reduce_v2(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen', N, bin_segment, False)
    print("Resumen reducido guardado en" +  res_dir + '/resumen.txt')
elif method == "guided_3":
    mask, resumen_reducido = guided_reduce_v3(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen', N, bin_segment, False)
    print("Resumen reducido guardado en" +  res_dir + '/resumen.txt')
elif method == "guided_4":
    metrics, mask, resumen_reducido = guided_reduce_v4(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen', N, bin_segment, discard_size)
    print("Resumen reducido guardado en" +  res_dir + '/resumen.txt')
elif method == "guided_5":
    metrics, mask, resumen_reducido = guided_reduce_v5(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen', N, segment, bin_segment, discard_size)
    print("Resumen reducido guardado en" +  res_dir + '/resumen.txt')
elif method == "SummSeg":
    metrics, mask, resumen_reducido = guided_reduce_v6(res_dir + '/resumen_ubiss.txt', res_dir + '/resumen', N, segment, bin_segment, discard_size, max_compress)
    print("Resumen reducido guardado en" +  res_dir + '/resumen.txt')
elif method == "SummSegInv":
    resumen_reducido = resumen

end_time = time.time()
total_time += end_time-start_time


# GUARDA TIEMPOS
tiempo_extract = 0.0
tiempo_inference = 0.0

# Abrir el archivo y leerlo
with open("data/inferences/" + sequencia + "_" + str(batches)   + "_times.txt", 'r') as f:
    # Leer cada línea del archivo
    for linea in f:
        # Usar una expresión regular para buscar los números en cada línea
        if 'Extract features' in linea:
            # Extraer el número y convertirlo a float
            tiempo_extract = float(re.search(r'(\d+\.\d+)', linea).group(1))
        elif 'Inference' in linea:
            # Extraer el número y convertirlo a float
            tiempo_inference = float(re.search(r'(\d+\.\d+)', linea).group(1))

total_time += tiempo_extract + tiempo_inference


with open(res_dir + "/times.txt", "w") as f:
    f.write(f"Extract features: {tiempo_extract:.4f} segundos\n")
    f.write(f"Inference: {tiempo_inference:.4f} segundos\n")
    f.write(f"Summary generation: {summ_gen_time:.4f} segundos\n")
    f.write(f"Summary reduction: {end_time - start_time:.4f} segundos\n")
    f.write(f"Total: {total_time:.4f} segundos\n\n")






title_5 = '''
  _____    ________      __     _     _    _      _______ _____ ____  _   _ 
 | ____|  |  ____\ \    / /\   | |   | |  | |  /\|__   __|_   _/ __ \| \ | |
 | |__    | |__   \ \  / /  \  | |   | |  | | /  \  | |    | || |  | |  \| |
 |___ \   |  __|   \ \/ / /\ \ | |   | |  | |/ /\ \ | |    | || |  | | . ` |
  ___) |  | |____   \  / ____ \| |___| |__| / ____ \| |   _| || |__| | |\  |
 |____(_) |______|   \/_/    \_\______\____/_/    \_\_|  |_____\____/|_| \_|
                                                                            
                                                                            
'''
if evaluation:
    print(title_5)

    intervalos, colores = traducir_csv_a_intervalos_y_colores(ground_truth)


    frames = resumen_reducido

    frames_per_interval = []
    conteos = []
    ratio_per_interval = []

    wall_count= 0
    good_count= 0
    tool_count= 0
    poor_count= 0
    cleaning_count= 0

    wall_ratios = []
    good_ratios = []
    tool_ratios = []
    poor_ratios = []
    cleaning_ratios = []

    wall_total_interval = 0
    good_total_interval = 0
    tool_total_interval = 0
    poor_total_interval = 0
    cleaning_total_interval = 0

    wall_interval_count= 0
    good_interval_count= 0
    tool_interval_count= 0
    poor_interval_count= 0
    cleaning_interval_count= 0


    # Iteramos sobre cada intervalo
    for i,intervalo in enumerate(intervalos):

        if color_to_label[colores[i]] == "wall":
            wall_total_interval += 1
        elif color_to_label[colores[i]] == "good":
            good_total_interval += 1
        elif color_to_label[colores[i]] == "tools":
            tool_total_interval += 1
        elif color_to_label[colores[i]] == "cleaning":
            cleaning_total_interval += 1
        elif color_to_label[colores[i]] == "poor":
            poor_total_interval += 1


        inicio, fin = intervalo
        frames_per_interval.append(fin-inicio)
        # Contamos cuántos números están en el intervalo [inicio, fin]
        cuenta = sum(inicio <= num <= fin for num in frames)
        conteos.append(cuenta)
        ratio = cuenta/(fin-inicio)
        ratio_per_interval.append(ratio)

        





    ratios = []
    compress = 0
    for i,ratio in enumerate(ratio_per_interval):
        ratio = ratio * 100
        # if 1 > ratio:
        #     ratio = 1
        ratios.append(ratio)
        compress += ratio


        if color_to_label[colores[i]] == "wall":
            if conteos[i] != 0:
                wall_interval_count += 1
                wall_count += conteos[i]
                wall_ratios.append(ratio)
        elif color_to_label[colores[i]] == "good":
            if conteos[i] != 0:
                good_interval_count += 1
                good_count += conteos[i]
                good_ratios.append(ratio)
        elif color_to_label[colores[i]] == "tools":
            if conteos[i] != 0:
                tool_interval_count += 1
                tool_count += conteos[i]
                tool_ratios.append(ratio)
        elif color_to_label[colores[i]] == "poor":
            if conteos[i] != 0:
                poor_interval_count += 1
                poor_count += conteos[i]
                poor_ratios.append(ratio)
        else:
            if conteos[i] != 0:
                cleaning_interval_count += 1
                cleaning_count += conteos[i]
                cleaning_ratios.append(ratio)


    wall_ratio = np.mean(wall_ratios)
    good_ratio = np.mean(good_ratios)
    tool_ratio = np.mean(tool_ratios)
    poor_ratio = np.mean(poor_ratios)
    cleaning_ratio = np.mean(cleaning_ratios)





    # Filtramos los intervalos que tienen al menos un número
    intervalos_con_numeros = [i for i, cuenta in enumerate(conteos) if cuenta > 0]

    fulfilled_intervals = len(intervalos_con_numeros)

    compress = compress/(fulfilled_intervals)

    print(f"Se cumplen {fulfilled_intervals} de los {len(intervalos)} intervalos.")
    print(str(fulfilled_intervals/len(intervalos) * 100) + "%")

    with open(res_dir + "/metrics.txt", "w") as f:
        f.write(f"Se cumplen {fulfilled_intervals} de los {len(intervalos)} intervalos.\n")
        f.write(f"Tool ratio:{tool_ratio}\n")
        f.write(f"Good ratio:{good_ratio}\n")
        f.write(f"Poor ratio:{poor_ratio}\n")
        f.write(f"Cleaning ratio:{cleaning_ratio}\n")
        f.write(f"Wall ratio:{wall_ratio}\n")

        f.write(f"\nTool count:{tool_count}\n")
        f.write(f"Good count:{good_count}\n")
        f.write(f"Poor count:{poor_count}\n")
        f.write(f"Cleaning count:{cleaning_count}\n")
        f.write(f"Wall count:{wall_count}\n")


        f.write(f"\nTool interval count:{tool_interval_count}")
        f.write(f"\nTool total interval:{tool_total_interval}\n\n")

        f.write(f"\nGood interval count:{good_interval_count}")
        f.write(f"\nGood total interval:{good_total_interval}\n\n")

        f.write(f"\nPoor interval count:{poor_interval_count}")
        f.write(f"\nPoor total interval:{poor_total_interval}\n\n")

        f.write(f"\nCleaning interval count:{cleaning_interval_count}")
        f.write(f"\nCleaning total interval:{cleaning_total_interval}\n\n")
        
        f.write(f"\nWall interval count:{wall_interval_count}")
        f.write(f"\nWall total interval:{wall_total_interval}\n\n")
        


        
        
    
        
        
        # f.write(f"Compresion:{compress}\n")

        if method != "uniform" and method != "SummSegInv":
            f.write(f"\nIntervalos buenos descartados: {metrics[0]} \n")
            f.write(f"Intervalos malos descartados: {metrics[1]} \n")
    

if visualization:

    title_7 = '''

    ______  __      _______  _____ _    _         _      _____ ______      _______ _____ ____  _   _ 
    |____  | \ \    / /_   _|/ ____| |  | |  /\   | |    |_   _|___  /   /\|__   __|_   _/ __ \| \ | |
        / /   \ \  / /  | | | (___ | |  | | /  \  | |      | |    / /   /  \  | |    | || |  | |  \| |
       / /     \ \/ /   | |  \___ \| |  | |/ /\ \ | |      | |   / /   / /\ \ | |    | || |  | | . ` |
      / /       \  /   _| |_ ____) | |__| / ____ \| |____ _| |_ / /__ / ____ \| |   _| || |__| | |\  |
     /_(_)       \/   |_____|_____/ \____/_/    \_\______|_____/_____/_/    \_\_|  |_____\____/|_| \_|
                                                                                                    
                                                                                                    
    '''

    print(title_7)

    

    # MOSAICO


    # Obtener los archivos de imagen seleccionados
    archivos_seleccionados = obtener_archivos(directorio_frames, resumen_reducido)

    # Crear el mosaico de manera eficiente, cargando cada imagen solo al momento de usarla
    if mosaico:
        if mosaico_bin:
            intervalos, colores = procesar_fichero(bin_segment)
            colores_correct = []
            for color in colores:
                if color == 1:
                    colores_correct.append((0,255,0))
                else:
                    colores_correct.append((255,0,0))
            colores = colores_correct

        if evaluation:
            mosaico = crear_mosaico(directorio_frames, archivos_seleccionados, intervalos, resumen_reducido, colores, True)
        else:
            #coger los colores de la segmentacion
            intervalos, etiq, colores = procesar_fichero_class_seg(segment)
            mosaico = crear_mosaico(directorio_frames, archivos_seleccionados, intervalos, resumen_reducido, colores, True)
        mosaico.save(res_dir + "/mosaico.png")  # Guardar el mosaico
    #BARRA

    print(len(os.listdir(directorio_frames)))
    if evaluation:
        imagen_barra_red = crear_barra(len(os.listdir(directorio_frames)), resumen_reducido, intervalos, colores, True)
    else:
        #coger los colores de la segmentacion
        intervalos, etiq, colores = procesar_fichero_class_seg(segment)
        imagen_barra_red = crear_barra(len(os.listdir(directorio_frames)), resumen_reducido, intervalos, colores, True)
    # imagen_barra_todo = crear_barra(len(os.listdir(directorio_frames)), resumen_ubiss, intervalos, colores)

    # imagen_barra_mask = crear_barra(len(os.listdir(directorio_frames)), mask, intervalos, colores)
    # imagen_barra_mask.save(res_dir + "/barra_mask.png")  # Guardar el mosaico

    imagen_barra_red.save(res_dir + "/barra_red.png")  # Guardar el mosaico
    # imagen_barra_todo.save(res_dir + "/barra_todo.png")  # Guardar el mosaico
   

    
    # if method == "uniform":
    #     # mosaico.save(res_dir + "/uniform/mosaico.png")  # Guardar el mosaico
        
    # elif method == "guided":
    #     # mosaico.save(res_dir + "/guided/mosaico.png")  # Guardar el mosaico
    #     imagen_barra.save(res_dir + "/guided/barra.png")  # Guardar el mosaico
    # elif method == "guided_2":
    #     # mosaico.save(res_dir + "/guided/mosaico.png")  # Guardar el mosaico
    #     imagen_barra.save(res_dir + "/guided/barra.png")  # Guardar el mosaico
    # elif method == "guided_3":
    #     # mosaico.save(res_dir + "/guided/mosaico.png")  # Guardar el mosaico
    #     imagen_barra.save(res_dir + "/guided/barra.png")  # Guardar el mosaico