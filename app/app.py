import pandas as pd
import networkx as nx
import os
from pyvis.network import Network
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random
from collections import deque

app = Flask(__name__)

file_path = os.path.join(r"C:\Users\andre\code\peruvianmusic\BASE_DATE_MUSIC.xlsx")

if os.path.exists(file_path):
    data = pd.read_excel(file_path, engine='openpyxl', usecols=["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                                                               "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION"])
else:
    print(f"Archivo no se encuentra en la ruta: {file_path}")

def cargar_canciones(file_path):
    print(f"Base de datos: {file_path}")
    data = pd.read_excel(file_path, engine='openpyxl', usecols=["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                                                               "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION"])
    data['PUNTUACION'] = pd.to_numeric(data['PUNTUACION'], errors='coerce')
    data['IMPACTO SOCIAL'] = pd.to_numeric(data['IMPACTO SOCIAL'], errors='coerce')

    return data

def calcular_similitudes(canciones):
    # Asegurarnos de que las columnas sean numéricas
    canciones['PUNTUACION'] = pd.to_numeric(canciones['PUNTUACION'], errors='coerce')
    canciones['IMPACTO SOCIAL'] = pd.to_numeric(canciones['IMPACTO SOCIAL'], errors='coerce')
    
    # Manejar valores nulos, si existen
    canciones['PUNTUACION'] = canciones['PUNTUACION'].fillna(canciones['PUNTUACION'].mean())
    canciones['IMPACTO SOCIAL'] = canciones['IMPACTO SOCIAL'].fillna(canciones['IMPACTO SOCIAL'].mean())
    
    # Escalar las características
    scaler = MinMaxScaler()
    canciones_features = canciones[['PUNTUACION', 'IMPACTO SOCIAL']].values
    canciones_normalizadas = scaler.fit_transform(canciones_features)

    # Calcular similitudes
    similitudes = cosine_similarity(canciones_normalizadas)
    return similitudes

# Obtener recomendaciones
def obtener_recomendaciones(canciones, genero, puntuacion_referencia):
    # Filtra por género y puntuación mínima
    recomendaciones = canciones[
        (canciones['GENERO'].str.lower() == genero.lower()) & 
        (canciones['PUNTUACION'] >= puntuacion_referencia)
    ]

    return recomendaciones

#construye grafo principal
def construir_grafo(canciones):
    G = nx.Graph()

    similitudes = calcular_similitudes(canciones)
    
    #posición para cada nodo 
    posiciones = {idx: (idx // 40, idx % 40) for idx in range(len(canciones))}

    for idx, row in canciones.iterrows():
        cancion = row['CANCION']
        grupo = row['GRUPO/ARTISTA']
        genero = row['GENERO']
        puntuacion = row['PUNTUACION']
        impacto_social = row['IMPACTO SOCIAL']

        #info emergente 
        info_tooltip = (
            f"Canción: {cancion}\n"
            f"Artista: {grupo}\n"
            f"Género: {genero}\n"
            f"Puntuación: {puntuacion}\n"
            f"Impacto Social: {impacto_social}\n"
        )


        #agrega nodo con la información emergente (tooltip)
        G.add_node(idx, label=cancion, pos=posiciones[idx], 
                genero=genero, tipo_impacto=row['TIPO DE IMPACTO'], title=info_tooltip)

    print('Aca nfor')  
    for i in range(len(canciones) - 1):
        for j in range(i + 1, len(canciones)):
            # ver que compartan el mismo género y tipo de impacto
            if (canciones['GENERO'][i] == canciones['GENERO'][j] and 
                canciones['TIPO DE IMPACTO'][i] == canciones['TIPO DE IMPACTO'][j]):

                #ver si la similitud entre las canciones supera un umbral
                if similitudes[i][j] > 0.5: 
                    if len(list(G.neighbors(i))) < 3 and len(list(G.neighbors(j))) < 3:
                        G.add_edge(i, j, weight=similitudes[i][j])
    print('Aca n')

    net = Network(height="750px", width="100%", bgcolor="#fab802", font_color="white")
    print('Aca n1')
    for node, pos in nx.get_node_attributes(G, 'pos').items():
        net.add_node(node, label=G.nodes[node]['label'], 
                    title=G.nodes[node]['title'],  #TOOLTIP
                    x=pos[0] * 300, y=pos[1] * 100, fixed=True)
    print('Aca n1')
    for edge in G.edges:
        net.add_edge(*edge, width=6)
    print('Aca n2')

    return net.generate_html()


def construir_grafo_recomendaciones(canciones_filtradas):
    grafo = nx.Graph()

    similitudes = calcular_similitudes(canciones_filtradas)
    print(f"Estas son las similitudes: {similitudes}")

    lista_indices = []

    for idx, row in canciones_filtradas.iterrows():
        grafo.add_node(idx, 
                       cancion=row['CANCION'], 
                       artista=row['GRUPO/ARTISTA'], 
                       genero=row['GENERO'], 
                       tipo_impacto=row['TIPO DE IMPACTO'])
        lista_indices.append(idx)  

    for i in range(len(canciones_filtradas)):
        for j in range(i + 1, len(canciones_filtradas)):  
            if similitudes[i][j] > 0.9995:  
                grafo.add_edge(lista_indices[i], lista_indices[j], peso=similitudes[i][j])
                
    print(f"Aristas: {grafo.edges(data=True)}")  

    return grafo


def bfs_recomendaciones(grafo, nodo_inicial, max_recomendaciones=5, tipo_impacto=None):
    cola = deque([nodo_inicial])  
    visitados = set([nodo_inicial]) 
    recomendaciones = []  #almacenar las recomendaciones

    print(f"Este es el grafo del bfs antes: {grafo.nodes(data=True)}")  

    if nodo_inicial not in grafo:
        print(f"El nodo inicial {nodo_inicial} no existe en el grafo.")
        return []  
    else:
        print(f"El nodo inicial {nodo_inicial} sí está en el grafo.")
    
    while cola:
        nodo_actual = cola.popleft()  #primer nodo de la cola

        atributos = grafo.nodes[nodo_actual]
        print(f"Nodo actual: {nodo_actual}, atributos: {atributos}")

        if tipo_impacto is None or atributos.get('tipo_impacto') == tipo_impacto:
            print(f"Nodo {nodo_actual} cumple con el filtro de impacto: {tipo_impacto}")
            recomendaciones.append(nodo_actual) 

        for vecino in grafo.neighbors(nodo_actual):
            if vecino not in visitados:
                visitados.add(vecino)
                print(f"Agregando vecino {vecino} a la cola.")
                cola.append(vecino)  

    print(recomendaciones)
    random.shuffle(recomendaciones)
    print(f"Recomendaciones desordenadas: {recomendaciones}")

    return recomendaciones[:max_recomendaciones]


#pagina 1
@app.route("/")
def index():
    canciones = cargar_canciones(file_path)
    grafo_html = construir_grafo(canciones)
    return render_template("index.html", grafo_html=grafo_html, data={
        "titulo": "Grafo de Canciones",
        "bienvenida": "BIENVENIDO A WEOLD"
    })

#página 2
@app.route("/uno")
def mostrar_pagina_dos():
    return render_template("uno.html")

#página 3 con recomendaciones
@app.route("/dos", methods=["GET", "POST"])
def mostrar_pagina_dos_actual():
    canciones = cargar_canciones(file_path)
    grafo_html = ""  
    recomendaciones = []  
    error = None

    if request.method == "POST":
        genero = request.form.get("genero", "").strip()
        puntuacion_referencia = request.form.get("puntuacion_minima", 200)
        tipo_impacto = request.form.get("tipo_impacto", "").strip()

        if not genero or not tipo_impacto:
            error = "Por favor, selecciona un género y un tipo de impacto."
        else:
            try:
                puntuacion_referencia = int(puntuacion_referencia)
                if puntuacion_referencia < 200 or puntuacion_referencia > 999:
                    raise ValueError
            except ValueError:
                error = "Por favor, ingresa una puntuación válida entre 200 y 999."

            if not error:
                recomendaciones_df = obtener_recomendaciones(canciones, genero, puntuacion_referencia)

                if recomendaciones_df.empty:
                    error = "No se encontraron canciones para los criterios seleccionados."
                else:
                    recomendaciones_df = recomendaciones_df.reset_index(drop=True)

                    grafo_recomendaciones = construir_grafo_recomendaciones(recomendaciones_df)

                    nodo_inicial = None
                    while not nodo_inicial:
                        nodo_inicial = random.choice(recomendaciones_df.index)
                        if recomendaciones_df.loc[nodo_inicial, "TIPO DE IMPACTO"] == tipo_impacto:
                            break
                        else:
                            nodo_inicial = None

                    print(f"Nodo inicial seleccionado: {nodo_inicial}")

                    nodos_recomendados = bfs_recomendaciones(
                        grafo_recomendaciones,
                        nodo_inicial,
                        max_recomendaciones=5,
                        tipo_impacto=tipo_impacto
                    )
                    print(f"Nodos recomendados por BFS: {nodos_recomendados}")

                    net = Network(height="750px", width="100%", bgcolor="#FF6347", font_color="black")
                    for nodo in nodos_recomendados:
                        nodo = int(nodo)  #nodo a entero
                        cancion = recomendaciones_df.loc[nodo]
                        net.add_node(
                            nodo,
                            label=cancion["CANCION"],
                            title=f"Artista: {cancion['GRUPO/ARTISTA']}\nImpacto: {cancion['TIPO DE IMPACTO']}\nPuntuación: {cancion['PUNTUACION']}",
                            color="#fab802"
                        )
                    for i in range(len(nodos_recomendados)):
                        for j in range(i + 1, len(nodos_recomendados)):
                            net.add_edge(nodos_recomendados[i], nodos_recomendados[j])

                    grafo_html = net.generate_html()
                    recomendaciones = [
                        {
                            "cancion": recomendaciones_df.loc[nodo, "CANCION"],
                            "artista": recomendaciones_df.loc[nodo, "GRUPO/ARTISTA"],
                            "genero": recomendaciones_df.loc[nodo, "GENERO"],
                            "puntuacion": recomendaciones_df.loc[nodo, "PUNTUACION"],
                            "tipo_impacto": recomendaciones_df.loc[nodo, "TIPO DE IMPACTO"],
                            "impacto_social": recomendaciones_df.loc[nodo, "IMPACTO SOCIAL"],
                            "audio_url": "#"
                        }
                        for nodo in nodos_recomendados
                    ]

    return render_template("dos.html", grafo_html=grafo_html, recomendaciones=recomendaciones, error=error)

if __name__ == "__main__":
    app.run(debug=True)