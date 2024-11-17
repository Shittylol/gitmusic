import pandas as pd
import networkx as nx
import os
from pyvis.network import Network
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

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

#calcular similitudes
def calcular_similitudes(canciones):
    scaler = MinMaxScaler()
    canciones_features = canciones[['PUNTUACION', 'IMPACTO SOCIAL']].values
    canciones_normalizadas = scaler.fit_transform(canciones_features)

    similitudes = cosine_similarity(canciones_normalizadas)
    return similitudes

def agrupar_canciones(canciones, n_clusters=5):
    #KMeans basado en PUNTUACION e IMPACTO SOCIAL.
    features = canciones[['PUNTUACION', 'IMPACTO SOCIAL']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    canciones['CLUSTER'] = kmeans.fit_predict(features)

    return canciones

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

    print('Aca nfor')  #optimiza
    for i in range(len(canciones) - 1):
        for j in range(i + 1, len(canciones)):
            # ver que compartan el mismo género y tipo de impacto
            if (canciones['GENERO'][i] == canciones['GENERO'][j] and 
                canciones['TIPO DE IMPACTO'][i] == canciones['TIPO DE IMPACTO'][j]):

                #ver si la similitud entre las canciones supera un umbral
                if similitudes[i][j] > 0.5: 
                    #no más de 3 conexiones por canción
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

def obtener_recomendaciones(canciones, cluster, genero, puntuacion_referencia, rango=50):
   
    #el rango de puntuación
    puntuacion_min = puntuacion_referencia - rango
    puntuacion_max = puntuacion_referencia + rango

    #filtra por cluster, género y rango de puntuación (1 filtro)
    recomendaciones = canciones[
        (canciones['CLUSTER'] == cluster) &
        (canciones['GENERO'].str.lower() == genero.lower()) &
        (canciones['PUNTUACION'] >= puntuacion_min) &
        (canciones['PUNTUACION'] <= puntuacion_max)
    ]

    return recomendaciones

#grafo resultante
def construir_grafo_recomendaciones(canciones_filtradas):
    
    grafo = nx.Graph()

    #nodos en el grafo
    for idx, row in canciones_filtradas.iterrows():
        grafo.add_node(idx, 
                       cancion=row['CANCION'], 
                       artista=row['GRUPO/ARTISTA'], 
                       genero=row['GENERO'], 
                       tipo_impacto=row['TIPO DE IMPACTO'])

    #criterios adicionales tipo de impacto(letras)
    for i, row_i in canciones_filtradas.iterrows():
        for j, row_j in canciones_filtradas.iterrows():
            if i != j:
                if row_i['GRUPO/ARTISTA'] == row_j['GRUPO/ARTISTA'] or row_i['TIPO DE IMPACTO'] == row_j['TIPO DE IMPACTO']:
                    grafo.add_edge(i, j)
    
    return grafo

def bfs_recomendaciones(grafo, nodo_inicial, max_recomendaciones=5, tipo_impacto=None, rango_anio=None):
    
    visitados = set()
    recomendaciones = []
    cola = [nodo_inicial]

    while cola and len(recomendaciones) < max_recomendaciones:
        nodo = cola.pop(0) 
        if nodo not in visitados:
            visitados.add(nodo)

            #atributos del nodo actual y coge tipo de impacto y año
            atributos = grafo.nodes[nodo]
            cumple_impacto = tipo_impacto is None or atributos.get('tipo_impacto') == tipo_impacto
            cumple_anio = rango_anio is None or (
                rango_anio[0] <= atributos.get('anio', 0) <= rango_anio[1]
            )

            if cumple_impacto and cumple_anio:
                recomendaciones.append(nodo)

            for vecino in grafo.neighbors(nodo):
                if vecino not in visitados:
                    cola.append(vecino)

    return recomendaciones

# Página principal
@app.route("/")
def index():
    canciones = cargar_canciones(file_path)
    grafo_html = construir_grafo(canciones)
    return render_template("index.html", grafo_html=grafo_html, data={
        "titulo": "Grafo de Canciones",
        "bienvenida": "BIENVENIDO A WEOLD"
    })

#muestra pagina 2
@app.route("/uno")
def mostrar_pagina_dos():
    return render_template("uno.html")

#muestra pagina 3
@app.route("/dos", methods=["GET", "POST"])
def mostrar_pagina_dos_actual():
    #carga canciones desde el archivo y agruparlas en clusters
    canciones = cargar_canciones(file_path)
    canciones = agrupar_canciones(canciones, n_clusters=5)

    if request.method == "POST":
        genero = request.form.get("genero", None)
        puntuacion_referencia = request.form.get("puntuacion_minima", 200)
        tipo_impacto = request.form.get("tipo_impacto", None)
        anio_min = request.form.get("anio_min", None)
        anio_max = request.form.get("anio_max", None)

        if not genero or genero.strip() == "":
            return render_template(
                "dos.html", error="Por favor, selecciona un género.", grafo_html=None, recomendaciones=[]
            )

        try:
            puntuacion_referencia = int(puntuacion_referencia)
            if puntuacion_referencia < 200 or puntuacion_referencia > 999:
                raise ValueError
            anio_min = int(anio_min) if anio_min else None
            anio_max = int(anio_max) if anio_max else None
        except ValueError:
            return render_template(
                "dos.html",
                error="Por favor, ingresa valores válidos para puntuación y año.",
                grafo_html=None,
                recomendaciones=[]
            )

        cluster_relevante = canciones.loc[
            canciones['GENERO'].str.lower() == genero.lower(), 'CLUSTER'
        ].mode()

        if cluster_relevante.empty:
            return render_template(
                "dos.html",
                error="No se encontraron recomendaciones para el género seleccionado, SÉ MÁS ESPECÍFICO :))",
                grafo_html=None,
                recomendaciones=[]
            )

        #canciones recomendadas con rango de puntaje dinámico
        cluster = cluster_relevante.iloc[0]
        recomendaciones_df = obtener_recomendaciones(canciones, cluster, genero, puntuacion_referencia, rango=100)

        if recomendaciones_df.empty:
            return render_template(
                "dos.html",
                error="No se encontraron canciones para los criterios seleccionados.",
                grafo_html=None,
                recomendaciones=[]
            )

        #grafo para las canciones filtradas
        grafo_recomendaciones = construir_grafo_recomendaciones(recomendaciones_df)

        #selecciona el nodo inicial más cercano a la puntuación de referencia
        nodo_inicial = recomendaciones_df.index[0]

        #recomendaciones con BFS 
        nodos_recomendados = bfs_recomendaciones(
            grafo_recomendaciones, 
            nodo_inicial, 
            max_recomendaciones=5, 
            tipo_impacto=tipo_impacto, 
            rango_anio=(anio_min, anio_max) if anio_min and anio_max else None
        )

        #lista de recomendaciones finales
        recomendaciones = [
            {
                "cancion": recomendaciones_df.loc[nodo, "CANCION"],
                "artista": recomendaciones_df.loc[nodo, "GRUPO/ARTISTA"],
                "genero": recomendaciones_df.loc[nodo, "GENERO"],
                "puntuacion": recomendaciones_df.loc[nodo, "PUNTUACION"],
                "tipo_impacto": recomendaciones_df.loc[nodo, "TIPO DE IMPACTO"],
                "audio_url": "#"  
            }
            for nodo in nodos_recomendados
        ]

        return render_template("dos.html", genero=genero, grafo_html=None, recomendaciones=recomendaciones)

    return render_template("dos.html", grafo_html=None, recomendaciones=[])


if __name__ == "__main__":
    app.run(debug=True, port=5000)