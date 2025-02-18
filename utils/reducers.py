import numpy as np

def procesar_fichero(ruta_fichero):
    # Arrays para almacenar los intervalos y las etiquetas
    intervalos = []
    etiquetas = []

    # Leer el fichero línea por línea
    with open(ruta_fichero, 'r') as fichero:
        for linea in fichero:
            # Dividir la línea en componentes
            partes = linea.strip().split()
            
            # Extraer los límites del intervalo y la etiqueta
            inicio = int(partes[0])
            fin = int(partes[1])
            etiqueta = partes[2]
            
            # Añadir el intervalo como una tupla (inicio, fin)
            intervalos.append((inicio, fin))
            
            # Convertir "good" a 1 y "bad" a 0
            if etiqueta == "good":
                etiquetas.append(1)
            elif etiqueta == "bad":
                etiquetas.append(0)
    
    return intervalos, etiquetas

label_to_color = {
    "wall": (0,0,0),
    "good": (0,255,0),
    "cleaning": (0,0,255),
    "tools": (255,128,255),
    "poor": (255,255,0),
    "none": (128,128,128)
}

def procesar_fichero_class_seg(ruta_fichero):
    # Arrays para almacenar los intervalos y las etiquetas
    intervalos = []
    etiquetas = []
    colores = []

    # Leer el fichero línea por línea
    with open(ruta_fichero, 'r') as fichero:
        for linea in fichero:
            # Dividir la línea en componentes
            partes = linea.strip().split()
            
            # Extraer los límites del intervalo y la etiqueta
            inicio = int(partes[0])
            fin = int(partes[1])
            etiqueta = partes[2]
            
            # Añadir el intervalo como una tupla (inicio, fin)
            intervalos.append((inicio, fin))
            
            # Convertir "good" a 1 y "bad" a 0
            colores.append(label_to_color[etiqueta])
            etiquetas.append(etiqueta)
    
    return intervalos, etiquetas, colores

def clasificar_numeros(numeros, intervalos, etiquetas):
    clasificacion = []
    
    # Clasificar cada número según su intervalo
    for numero in numeros:
        # Por defecto el número no tiene clasificación (None)
        etiqueta_asignada = None
        
        # Buscar en qué intervalo cae el número
        for i, (inicio, fin) in enumerate(intervalos):
            if inicio <= numero <= fin:
                etiqueta_asignada = etiquetas[i]
                break
        
        # Añadir la clasificación (1 para 'good' o 0 para 'bad')
        clasificacion.append(etiqueta_asignada)
    
    return clasificacion


def uniform_reduce(input_dir, output_dir, N, fullfill):
    
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]


    n_frames = len(frames)
    step = int(n_frames/N) + 1


    print(n_frames, step)

    
    # Seleccionar uno de cada N frames
    selected_frames = frames[::step]
    if fullfill:
        faltan = (len(selected_frames) - N) * -1
        print(faltan)

        for i in range(1,faltan+1):
            selected_frames.append(frames[(i*step)+1])

        selected_frames.sort()

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return selected_frames

def uniform_reduce_v2(input_dir, output_dir, N, norm_scores, batches):
    full_len = len(norm_scores) * batches
    percent = N/full_len

    percentile = 100 - (batches*percent*100)

    th = np.percentile(norm_scores, percentile)

    print("TH:", th)

    i = 0

    resumen = []
    for sc in norm_scores:
        if sc > th:
            # print(i, sc)
            resumen.append(i)
        i+=1
    print("num frames:", len(resumen))

    selected_frames = []

    for i in resumen:
        selected_frames.append(i*batches)
    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return selected_frames
    


def guided_reduce(input_dir, output_dir, N, segment, fullfill):
    
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    inter, etiqs = procesar_fichero(segment)

    clasi = clasificar_numeros(frames, inter, etiqs)

    malos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 0]

    seg_malos = []
    current_segment = []
    for i, frame in enumerate(malos):
        if i == len(malos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_malos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != malos[i+1] - 1: #Se acabo
                seg_malos.append(current_segment)
                current_segment = []

    print(len(seg_malos))

    buenos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 1]

   
    seg_buenos = []
    current_segment = []
    for i, frame in enumerate(buenos):
        if i == len(buenos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_buenos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != buenos[i+1] - 1: #Se acabo
                seg_buenos.append(current_segment)
                current_segment = []

    print(len(seg_buenos))

    resumen = []
    divid = 2

    for seg in seg_buenos:
        # print("\n",seg)
        size = len(seg)
        n_frames = int(size/divid + 0.999)
        # print(size, n_frames)
        divs = []
        divs.append(0)
        for i in range(n_frames-1): 
            divs.append(divid*(i+1) - 1)
            divs.append(divid*(i+1))
        divs.append(size-1)
        # print(divs)
        # if n_frames == 1:
        #     print(divs)
        #     input()
        for i in range(n_frames):
            ini = divs[2*i]
            fin = divs[2*i+1]
            med = int((ini+fin)/2)
            resumen.append(seg[med])
        
    print(len(resumen))

    # Vamos a reducir los que esten muy juntos
    cercanos = []
    actual = []
    starting = resumen[0]
    for i,frame in enumerate(resumen):
        if frame > starting + 32:
            cercanos.append(actual)
            actual = [frame]
            starting = frame
        else:
            actual.append(frame)

    reduced_resumen = []
    for seg in cercanos:
        # print(seg)
        reduced_resumen.append(int(np.mean(seg)))
    
    resumen = reduced_resumen
    print(len(resumen))

    fullfill = False

    if fullfill:
        faltan = N - len(resumen)

        
        print("FALTAN:", faltan)

        divid = 4
        bad_resumen = []
        malos_ordenados = sorted(seg_malos, key=len)
        malos_ordenados = malos_ordenados[::-1]
        for seg in malos_ordenados:
            if faltan < 0:
                break
            else:
                size = len(seg)
                n_frames = int(size/divid + 0.999)
                divs = []
                divs.append(0)
                for i in range(n_frames-1): 
                    divs.append(divid*(i+1) - 1)
                    divs.append(divid*(i+1))
                divs.append(size-1)
                for i in range(n_frames):
                    faltan-=1
                    if faltan < 0:
                        break
                    ini = divs[2*i]
                    fin = divs[2*i+1]
                    med = int((ini+fin)/2)
                    bad_resumen.append(seg[med])
                    resumen.append(seg[med])

        print(bad_resumen)
        resumen.sort()
        print(len(resumen))

    selected_frames = resumen

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return selected_frames




def guided_reduce_v2(input_dir, output_dir, N, segment, fullfill):
    
    intervalos, etiquetas = procesar_fichero(segment)



    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    # frames = frames[::2]
    resultado = [[] for _ in intervalos]

    # Asignar cada número al intervalo correspondiente
    for numero in frames:
        for i, (inicio, fin) in enumerate(intervalos):
            if inicio <= numero <= fin:
                resultado[i].append(numero)
                break  # Salir del bucle de intervalos una vez que el número se ha asignado

    
    
    resumen = []
    for i,interval in enumerate(resultado):
        print(interval)
        if len(interval) != 0:
            if etiquetas[i] == 0: # Es tramo malo asi que metemos solo uno
                resumen.append(interval[int(len(interval) * 1/2)])
            else: # Es tramo bueno vamos a meter 2
                subtramos = np.array_split(interval, 2)
                # print(interval)
                for tramo in subtramos:
                    if len(tramo) != 0:
                        resumen.append(tramo[int(len(tramo) * 1/2)])
            # resumen.append(interval[int(len(interval)/2)])
    exit()
    #Hay segmentos buenos que no estan representados, vamos a incluirlos
    indices_no_representados = []

    # Comprobar cada intervalo
    for i, (inicio, fin) in enumerate(intervalos):
        # Comprobar si algún número está dentro del intervalo actual
        representado = any(inicio <= numero <= fin for numero in frames)
        
        # Si el intervalo no tiene ningún número representado, añadir el índice
        if not representado:
            indices_no_representados.append(i)

    faltan = N - len(resumen)
    print("Faltan:", faltan)
    good_norep = 0
    for i in indices_no_representados:
        if faltan == 0:
            break
        if etiquetas[i] == 1: #Es bueno, lo añadimos
            good_norep+=1
            tramo = intervalos[i]
            faltan -=1
            resumen.append(tramo[int(len(tramo) * 1/2)])

    bad_added = 0     
    for i in indices_no_representados: # Miramos a ver si faltan metemos alguno malo
        if faltan == 0:
            break
        if etiquetas[i] == 0: 
            tramo = intervalos[i]
            faltan -=1
            bad_added += 1
            resumen.append(tramo[int(len(tramo) * 1/2)])
    
    print("good no rep", good_norep)
    print("bad no rep", bad_added)

    resumen.sort()

    with open(output_dir + ".txt", "w") as f:
        for n in resumen:
            f.write(str(n) + "\n")

    return resumen



def guided_reduce_v3(input_dir, output_dir, N, segment, fullfill):
    
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    inter, etiqs = procesar_fichero(segment)

    clasi = clasificar_numeros(frames, inter, etiqs)

    malos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 0]

    seg_malos = []
    current_segment = []
    for i, frame in enumerate(malos):
        if i == len(malos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_malos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != malos[i+1] - 1: #Se acabo
                seg_malos.append(current_segment)
                current_segment = []

    print(len(seg_malos))

    buenos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 1]

   
    seg_buenos = []
    current_segment = []
    for i, frame in enumerate(buenos):
        if i == len(buenos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_buenos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != buenos[i+1] - 1: #Se acabo
                seg_buenos.append(current_segment)
                current_segment = []

    print(len(seg_buenos))


    resumen = []
    divid = 2

    for seg in seg_buenos:
        # print("\n",seg)
        size = len(seg)
        n_frames = int(size/divid + 0.999)
        # print(size, n_frames)
        divs = []
        divs.append(0)
        for i in range(n_frames-1): 
            divs.append(divid*(i+1) - 1)
            divs.append(divid*(i+1))
        divs.append(size-1)
        # print(divs)
        # if n_frames == 1:
        #     print(divs)
        #     input()
        for i in range(n_frames):
            ini = divs[2*i]
            fin = divs[2*i+1]
            med = int((ini+fin)/2)
            resumen.append(seg[med])
        
    print(len(resumen))

    mask = resumen.copy()

    # Vamos a reducir los que esten muy juntos
    cercanos = []
    actual = []
    starting = resumen[0]
    for i,frame in enumerate(resumen):
        if frame > starting + 32:
            cercanos.append(actual)
            actual = [frame]
            starting = frame
        else:
            actual.append(frame)

    reduced_resumen = []
    for seg in cercanos:
        # print(seg)
        reduced_resumen.append(int(np.mean(seg)))
    
    resumen = reduced_resumen
    print(len(resumen))

    #Hay segmentos buenos que no estan representados, vamos a incluirlos
    indices_no_representados = []

    # Comprobar cada intervalo
    for i, (inicio, fin) in enumerate(inter):
        # Comprobar si algún número está dentro del intervalo actual
        representado = any(inicio <= numero <= fin for numero in resumen)
        
        # Si el intervalo no tiene ningún número representado, añadir el índice
        if not representado:
            indices_no_representados.append(i)


    print(indices_no_representados)
    faltan = N - len(resumen)
    print("Faltan:", faltan)
    good_norep = 0
    for i in indices_no_representados:
        if faltan <= 0:
            break
        if etiqs[i] == 1: #Es bueno, lo añadimos
            good_norep+=1
            tramo = inter[i]
            faltan -=1
            resumen.append(tramo[int(len(tramo) * 1/2)])

    bad_added = 0     
    for i in indices_no_representados: # Miramos a ver si faltan metemos alguno malo
        if faltan <= 0:
            break
        if etiqs[i] == 0: 
            tramo = inter[i]
            faltan -=1
            bad_added += 1
            resumen.append(tramo[int(len(tramo) * 1/2)])
    
    print("good no rep", good_norep)
    print("bad no rep", bad_added)

    resumen.sort()

    selected_frames = resumen

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return mask, selected_frames



def guided_reduce_v4(input_dir, output_dir, N, segment, discard_size):
    

    metrics = []
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    inter, etiqs = procesar_fichero(segment)

    
    n_buenos = 0
    n_malos = 0

    for i in etiqs:
        if i == 1:
            n_buenos += 1
        else:
            n_malos += 1



    clasi = clasificar_numeros(frames, inter, etiqs)

    malos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 0]

    seg_malos = []
    current_segment = []
    for i, frame in enumerate(malos):
        if i == len(malos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_malos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != malos[i+1] - 1: #Se acabo
                seg_malos.append(current_segment)
                current_segment = []


    buenos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 1]

   
    seg_buenos = []
    current_segment = []
    for i, frame in enumerate(buenos):
        if i == len(buenos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_buenos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != buenos[i+1] - 1: #Se acabo
                seg_buenos.append(current_segment)
                current_segment = []


    resumen = []
    divid = 2

    for seg in seg_buenos:
        size = len(seg)
        n_frames = int(size/divid + 0.999)
        divs = []
        divs.append(0)
        for i in range(n_frames-1): 
            divs.append(divid*(i+1) - 1)
            divs.append(divid*(i+1))
        divs.append(size-1)
        for i in range(n_frames):
            ini = divs[2*i]
            fin = divs[2*i+1]
            med = int((ini+fin)/2)
            resumen.append(seg[med])

    

    print("Pre reduc:", len(resumen))

    # Vamos a reducir los que esten muy juntos
    cercanos = []
    actual = []
    starting = resumen[0]
    for i,frame in enumerate(resumen):
        if frame > starting + discard_size:
            cercanos.append(actual)
            actual = [frame]
            starting = frame
        else:
            actual.append(frame)

    reduced_resumen = []
    for seg in cercanos:
        # print(seg)
        reduced_resumen.append(int(np.mean(seg)))
    
    resumen = reduced_resumen
    mask = resumen.copy()

    faltan = N - len(resumen)
    print("Faltan:", faltan)

    
    # VAMOS A RELLENAR DE OTRA MANERA, tengo la longitud de los intervalos de las predicciones



    # Combina los intervalos con sus etiquetas
    intervalos_con_etiquetas = list(zip(inter, etiqs))

    # Ordena los intervalos (y sus etiquetas) de acuerdo a la clave
    intervalos_ordenados = sorted(intervalos_con_etiquetas, key=lambda x: x[0][1] - x[0][0], reverse=True)

    # Separa los intervalos y etiquetas ordenados
    inter = [i[0] for i in intervalos_ordenados]  # Intervalos ordenados
    etiqs = [i[1] for i in intervalos_ordenados]

    
    len_segs = []

    for intervalo in inter:
        len_segs.append(intervalo[1] - intervalo[0])

    # good_perc = 0.03 * int(N/200)
    intervalos_contribucion = 0
    for i,intervalo in enumerate(inter):
        if faltan <= 0:
            break
        if etiqs[i] == 1: #Intervalo bueno
            if len_segs[i] > 12: # Intervalo bueno relativamente grande
                inicio, fin = intervalo
                frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)

                rate = frames_in_interval/len_segs[i]
                
                frames_to_1per = np.ceil(len_segs[i] * 0.03)

                frames_to_add = frames_to_1per - frames_in_interval
                

                if frames_to_add > 0:
                    frames_added = np.linspace(int(inicio), int(fin), int(frames_to_add) + 2)[1:-1]
                    frames_added = np.unique(np.round(frames_added).astype(int))

                    for frame in frames_added:
                        if faltan <= 0:
                            break
                        if frame not in resumen:
                            resumen.append(frame)
                        else:
                            print("OJO")
                            resumen.append(frame + 2)
                    
                        faltan -= 1
                    
                intervalos_contribucion += 1

                print(i, rate, frames_to_1per, frames_in_interval, frames_added)
    

    metrics.append(n_buenos - intervalos_contribucion)

    intervalos_bad_contribucion = 0
    print("Faltan:", faltan)
    ii = 0
    first_round = True
    while faltan > 0:
        print("Faltan:", faltan)
        ii += 1
        for i,intervalo in enumerate(inter):
            if faltan <= 0:
                for i in range(127, len(inter)):
                    if etiqs[i] == 0:
                        print(len_segs[i])
                break
            if etiqs[i] == 0: #Intervalo bueno
                if len_segs[i] > 36: # Intervalo malo relativamente grande
                    inicio, fin = intervalo
                    frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                    if frames_in_interval == 0 or  not first_round:
                        intervalos_bad_contribucion += 1
                        resumen.append(int((inicio+fin)/2) + ii)
                        faltan -= 1
        first_round = False

    metrics.append(n_malos - intervalos_bad_contribucion)
    print("Faltan:", faltan)
            
    resumen.sort()


    selected_frames = resumen

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return metrics, mask, selected_frames







def guided_reduce_v5(input_dir, output_dir, N, segment, bin_segment, discard_size):
    
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    inter, etiqs, colores = procesar_fichero_class_seg(segment)

    for i in range(len(inter)):
        print(inter[i], etiqs[i])

    n_buenos = 0
    n_malos = 0

    for i in etiqs:
        if i == "tools" or i == "good" or i == "none" or i == "poor":
            n_buenos += 1
        else:
            n_malos += 1

    # Diccionario con el orden de las etiquetas
    orden_etiquetas = {'tools': 0, 'good': 1, 'poor': 2, 'none': 3, 'cleaning': 4, 'wall': 5}

    # Ordenar los intervalos y sus etiquetas en función del orden de las etiquetas
    intervalos_ordenados = [x for _, x in sorted(zip(etiqs, inter), key=lambda x: orden_etiquetas[x[0]])]

    # También puedes obtener los arrays de intervalos y etiquetas ordenados por separado si lo necesitas
    etiquetas_ordenadas = sorted(etiqs, key=lambda x: orden_etiquetas[x])

    for i in range(len(inter)):
        print(intervalos_ordenados[i], etiquetas_ordenadas[i])

    len_segs = []

    for intervalo in intervalos_ordenados:
        len_segs.append(intervalo[1] - intervalo[0])
    

    bin_inter, bin_etiq = procesar_fichero(bin_segment)

    clasi = clasificar_numeros(frames, bin_inter, bin_etiq)

    malos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 0]

    seg_malos = []
    current_segment = []
    for i, frame in enumerate(malos):
        if i == len(malos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_malos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != malos[i+1] - 1: #Se acabo
                seg_malos.append(current_segment)
                current_segment = []


    buenos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 1]

   
    seg_buenos = []
    current_segment = []
    for i, frame in enumerate(buenos):
        if i == len(buenos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_buenos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != buenos[i+1] - 1: #Se acabo
                seg_buenos.append(current_segment)
                current_segment = []

    resumen = []
    divid = 2

    for seg in seg_buenos:
        size = len(seg)
        n_frames = int(size/divid + 0.999)
        divs = []
        divs.append(0)
        for i in range(n_frames-1): 
            divs.append(divid*(i+1) - 1)
            divs.append(divid*(i+1))
        divs.append(size-1)
        for i in range(n_frames):
            ini = divs[2*i]
            fin = divs[2*i+1]
            med = int((ini+fin)/2)
            resumen.append(seg[med])

    


    # Vamos a reducir los que esten muy juntos
    cercanos = []
    actual = []
    starting = resumen[0]
    for i,frame in enumerate(resumen):
        if frame > starting + discard_size:
            cercanos.append(actual)
            actual = [frame]
            starting = frame
        else:
            actual.append(frame)

    reduced_resumen = []
    for seg in cercanos:
        # print(seg)
        reduced_resumen.append(int(np.mean(seg)))
    
    resumen = reduced_resumen

    faltan = N - len(resumen)
    print("Faltan:", faltan)
    mask = resumen.copy()

    
    # VAMOS A RELLENAR DE OTRA MANERA, tengo la longitud de los intervalos de las predicciones y

    metrics = []


    
    inter = intervalos_ordenados

    etiqs = etiquetas_ordenadas

    # good_perc = 0.03

    len_segs = []

    for intervalo in inter:
        len_segs.append(intervalo[1] - intervalo[0])


    intervalos_contribucion = 0
    for i,intervalo in enumerate(inter):
        if faltan <= 0:
            break
        if etiqs[i] == "tools" or etiqs[i] == "good" or etiqs[i] == "poor" or etiqs[i] == "none": #Intervalo bueno
            if len_segs[i] > 12: # Intervalo bueno relativamente grande
                inicio, fin = intervalo
                frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)

                rate = frames_in_interval/len_segs[i]
                
                frames_to_1per = np.ceil(len_segs[i] * 0.03)

                frames_to_add = frames_to_1per - frames_in_interval
                

                if frames_to_add > 0:
                    frames_added = np.linspace(int(inicio), int(fin), int(frames_to_add) + 2)[1:-1]
                    frames_added = np.unique(np.round(frames_added).astype(int))

                    for frame in frames_added:
                        if faltan <= 0:
                            break
                        if frame not in resumen:
                            resumen.append(frame)
                        else:
                            print("OJO")
                            resumen.append(frame + 2)
                    
                        faltan -= 1

                intervalos_contribucion += 1
                print(i, rate, frames_to_1per, frames_in_interval, frames_added)
    

    metrics.append(n_buenos - intervalos_contribucion)
    intervalos_bad_contribucion = 0
    print("Faltan:", faltan)
    ii = 0
    first_round = True
    while faltan > 0:
        print("Faltan:", faltan)
        ii += 1
        for i,intervalo in enumerate(inter):
            if faltan <= 0:
                for i in range(127, len(inter)):
                    if etiqs[i] == 0:
                        print(len_segs[i])
                break
            if etiqs[i] == "wall" or etiqs[i] == "cleaning": #Intervalo malo
                if len_segs[i] > 36: # Intervalo malo relativamente grande
                    inicio, fin = intervalo
                    frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                    if frames_in_interval == 0 or  not first_round:
                        intervalos_bad_contribucion += 1
                        resumen.append(int((inicio+fin)/2) + ii)
                        faltan -= 1
        first_round = False

    print("Faltan:", faltan)
            
    resumen.sort()
    metrics.append(n_malos - intervalos_bad_contribucion)

    selected_frames = resumen

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return metrics, mask, selected_frames


def guided_reduce_v6(input_dir, output_dir, N, segment, bin_segment, discard_size, max_reduce):
    
    with open(input_dir, 'r') as f:
        frames = [int(linea.strip()) for linea in f]

    inter, etiqs, colores = procesar_fichero_class_seg(segment)

    n_buenos = 0
    n_malos = 0

    for i in etiqs:
        if i == "tools" or i == "good" or i == "none" or i == "poor":
            n_buenos += 1
        else:
            n_malos += 1

    # Diccionario con el orden de las etiquetas
    orden_etiquetas = {'tools': 0, 'good': 1, 'poor': 2, 'none': 3, 'cleaning': 4, 'wall': 5}

    # Ordenar los intervalos y sus etiquetas en función del orden de las etiquetas
    intervalos_ordenados = [x for _, x in sorted(zip(etiqs, inter), key=lambda x: orden_etiquetas[x[0]])]

    # También puedes obtener los arrays de intervalos y etiquetas ordenados por separado si lo necesitas
    etiquetas_ordenadas = sorted(etiqs, key=lambda x: orden_etiquetas[x])

    len_segs = []

    for intervalo in intervalos_ordenados:
        len_segs.append(intervalo[1] - intervalo[0])
    

    bin_inter, bin_etiq = procesar_fichero(bin_segment)

    clasi = clasificar_numeros(frames, bin_inter, bin_etiq)

    malos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 0]

    seg_malos = []
    current_segment = []
    for i, frame in enumerate(malos):
        if i == len(malos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_malos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != malos[i+1] - 1: #Se acabo
                seg_malos.append(current_segment)
                current_segment = []


    buenos = [num for num, etiqueta in zip(frames, clasi) if etiqueta == 1]

   
    seg_buenos = []
    current_segment = []
    for i, frame in enumerate(buenos):
        if i == len(buenos) - 1: #es el ultimo
            current_segment.append(frame)
            seg_buenos.append(current_segment)
        else:
            current_segment.append(frame)
            if frame != buenos[i+1] - 1: #Se acabo
                seg_buenos.append(current_segment)
                current_segment = []

    resumen = []
    divid = 8

    for seg in seg_buenos:
        size = len(seg)
        n_frames = int(size/divid + 0.999)
        divs = []
        divs.append(0)
        for i in range(n_frames-1): 
            divs.append(divid*(i+1) - 1)
            divs.append(divid*(i+1))
        divs.append(size-1)
        for i in range(n_frames):
            ini = divs[2*i]
            fin = divs[2*i+1]
            med = int((ini+fin)/2)
            resumen.append(seg[med])

    


    # Vamos a reducir los que esten muy juntos
    cercanos = []
    actual = []
    starting = resumen[0]
    for i,frame in enumerate(resumen):
        if frame > starting + discard_size:
            cercanos.append(actual)
            actual = [frame]
            starting = frame
        else:
            actual.append(frame)

    reduced_resumen = []
    for seg in cercanos:
        # print(seg)
        reduced_resumen.append(int(np.mean(seg)))
    
    resumen = reduced_resumen

    faltan = N - len(resumen)
    print("Faltan:", faltan)
    while faltan < 0:
        discard_size += 1
        cercanos = []
        actual = []
        starting = resumen[0]
        for i,frame in enumerate(resumen):
            if frame > starting + discard_size:
                cercanos.append(actual)
                actual = [frame]
                starting = frame
            else:
                actual.append(frame)

        reduced_resumen = []
        for seg in cercanos:
            # print(seg)
            reduced_resumen.append(int(np.mean(seg)))
        
        resumen = reduced_resumen

        faltan = N - len(resumen)
        print("Faltan:", faltan) 


    mask = resumen.copy()

    selected_frames = resumen

    
    # VAMOS A RELLENAR DE OTRA MANERA, tengo la longitud de los intervalos de las predicciones y

    metrics = []


    
    inter = intervalos_ordenados

    etiqs = etiquetas_ordenadas

    # good_perc = 0.03

    len_segs = []

    frames_to_add = []

    for intervalo in inter:
        frames_to_add.append(0)
        len_segs.append(intervalo[1] - intervalo[0])

    pre_faltan = faltan

    frames_in_interval = []
    for i,intervalo in enumerate(inter):
        inicio, fin = intervalo
        fs = sum(1 for num in resumen if inicio <= num <= fin)
        frames_in_interval.append(fs)

    intervalos_contribucion = 0
    if not max_reduce:
        while faltan > 0:
            for i,intervalo in enumerate(inter):
                if faltan <= 0:
                    break        
                if etiqs[i] == "tools" or etiqs[i] == "good" or etiqs[i] == "poor" or etiqs[i] == "none": #Intervalo bueno
                    if len_segs[i] > 12: # Intervalo bueno relativamente grande
                        # inicio, fin = intervalo
                        # frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                        if frames_in_interval[i] > 0:
                            frames_in_interval[i] -= 1
                        else:
                            frames_to_add[i] += 1
                            faltan -= 1
                else:
                    if len_segs[i] > 36: # Intervalo malo relativamente grande
                        # inicio, fin = intervalo
                        # frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                        if frames_in_interval[i] == 0:
                            if frames_to_add[i] == 0:
                                frames_to_add[i] = 1
                                faltan -= 1
    else:
        for i,intervalo in enumerate(inter):       
                if etiqs[i] == "tools" or etiqs[i] == "good" or etiqs[i] == "poor" or etiqs[i] == "none": #Intervalo bueno
                    if len_segs[i] > 12: # Intervalo bueno relativamente grande
                        # inicio, fin = intervalo
                        # frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                        if frames_in_interval[i] > 0:
                            frames_in_interval[i] -= 1
                        else:
                            frames_to_add[i] += 1
                else:
                    if len_segs[i] > 36: # Intervalo malo relativamente grande
                        # inicio, fin = intervalo
                        # frames_in_interval = sum(1 for num in resumen if inicio <= num <= fin)
                        if frames_in_interval[i] == 0:
                            if frames_to_add[i] == 0:
                                frames_to_add[i] = 1

    for i, f in enumerate(frames_to_add):
        if f > 0:
            inicio, fin = inter[i]
            frames_added = np.linspace(int(inicio), int(fin), int(f) + 2)[1:-1]
            frames_added = np.unique(np.round(frames_added).astype(int))
            for frame in frames_added:
                # if faltan <= 0:
                #     break
                if frame not in resumen:
                    resumen.append(frame)
                else:
                    print("OJO")
                    resumen.append(frame + 2)

                # faltan -= 1

    print(len(resumen))
    resumen.sort()
    metrics.append(0)
    metrics.append(0)

    # selected_frames = resumen

    
    with open(output_dir + ".txt", "w") as f:
        for n in selected_frames:
            
            f.write(str(n) + "\n")

    return metrics, mask, selected_frames