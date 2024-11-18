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
    
    #inicializa
    canciones['CLUSTER'] = -1

    #itera sobre cada género único en la columna genero
    for genero in canciones['GENERO'].unique():
        #filtra canciones que pertenecen al género actual
        subset = canciones[canciones['GENERO'] == genero]

        #no agrupa si hay una cancion
        if len(subset) > 1:  
            
            features = subset[['PUNTUACION']]
            
            #número de clusters no mayor al número de canciones en el subconjunto
            kmeans = KMeans(n_clusters=min(n_clusters, len(subset)), random_state=42)
            
            #los clusters para las canciones del género actual
            clusters = kmeans.fit_predict(features)
            
            #actualiza cluster a canciones de genero y puntuacion actual
            canciones.loc[subset.index, 'CLUSTER'] = clusters

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

def obtener_recomendaciones(canciones, cluster, genero, puntuacion_referencia, rango=100):
   
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

    for idx, row in canciones_filtradas.iterrows():
        grafo.add_node(idx, 
                       cancion=row['CANCION'], 
                       artista=row['GRUPO/ARTISTA'], 
                       genero=row['GENERO'], 
                       tipo_impacto=row['TIPO DE IMPACTO'])

    for i, row_i in canciones_filtradas.iterrows():
        for j, row_j in canciones_filtradas.iterrows():
            if i != j and (row_i['GRUPO/ARTISTA'] == row_j['GRUPO/ARTISTA'] or row_i['TIPO DE IMPACTO'] == row_j['TIPO DE IMPACTO']):
                grafo.add_edge(i, j)
    
    return grafo

def bfs_recomendaciones(grafo, nodo_inicial, max_recomendaciones=5, tipo_impacto=None, impacto_social_min=0):
    """
    Realiza un recorrido BFS en el grafo para ajustar recomendaciones basadas en tipo de impacto e impacto social.

    Args:
        grafo (nx.Graph): Grafo de canciones filtradas.
        nodo_inicial (int): Nodo inicial para iniciar el BFS.
        max_recomendaciones (int): Máximo número de canciones recomendadas.
        tipo_impacto (str): Tipo de impacto a considerar en las recomendaciones.
        impacto_social_min (int): Mínimo nivel de impacto social requerido.

    Returns:
        list: Lista de IDs de nodos recomendados.
    """
    visitados = set()
    recomendaciones = []
    cola = [nodo_inicial]

    while cola and len(recomendaciones) < max_recomendaciones:
        nodo = cola.pop(0)
        if nodo not in visitados:
            visitados.add(nodo)

            # Verificar atributos del nodo actual
            atributos = grafo.nodes[nodo]
            cumple_impacto = tipo_impacto is None or atributos.get('tipo_impacto') == tipo_impacto
            cumple_impacto_social = atributos.get('impacto_social', 0) >= impacto_social_min

            if cumple_impacto and cumple_impacto_social:
                recomendaciones.append(nodo)

            # Agregar vecinos a la cola
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
    # Cargar las canciones desde el archivo y agruparlas en clusters
    canciones = cargar_canciones(file_path)
    canciones = agrupar_canciones(canciones, n_clusters=5)

    # Valores predeterminados para el renderizado
    error = None
    grafo_html = None
    recomendaciones = []
    genero = None  # Inicializar genero con None

    if request.method == "POST":
        genero = request.form.get("genero", None)
        puntuacion_referencia = request.form.get("puntuacion_minima", 200)
        tipo_impacto = request.form.get("tipo_impacto", None)
        impacto_social_min = request.form.get("impacto_social_min", 0)

        if not genero or genero.strip() == "":
            error = "Por favor, selecciona un género."
        else:
            try:
                puntuacion_referencia = int(puntuacion_referencia)
                impacto_social_min = int(impacto_social_min)
                if puntuacion_referencia < 200 or puntuacion_referencia > 999:
                    raise ValueError
            except ValueError:
                error = "Por favor, ingresa valores válidos para puntuación e impacto social."

            if not error:
                cluster_relevante = canciones.loc[
                    canciones['GENERO'].str.lower() == genero.lower(), 'CLUSTER'
                ].mode()

                if cluster_relevante.empty:
                    error = "No se encontraron recomendaciones para el género seleccionado."
                else:
                    # Obtener canciones recomendadas con rango dinámico
                    cluster = cluster_relevante.iloc[0]
                    recomendaciones_df = obtener_recomendaciones(canciones, cluster, genero, puntuacion_referencia, rango=100)

                    if recomendaciones_df.empty:
                        error = "No se encontraron canciones para los criterios seleccionados."
                    else:
                        # Construir el grafo para las canciones filtradas
                        grafo_recomendaciones = construir_grafo_recomendaciones(recomendaciones_df)

                        # Seleccionar el nodo inicial más cercano a la puntuación de referencia
                        nodo_inicial = recomendaciones_df.index[0]

                        # Recomendaciones con BFS
                        nodos_recomendados = bfs_recomendaciones(
                            grafo_recomendaciones,
                            nodo_inicial,
                            max_recomendaciones=5,
                            tipo_impacto=tipo_impacto,
                            impacto_social_min=impacto_social_min,
                        )

                        # Lista de recomendaciones finales
                        recomendaciones = [
                            {
                                "cancion": recomendaciones_df.loc[nodo, "CANCION"],
                                "artista": recomendaciones_df.loc[nodo, "GRUPO/ARTISTA"],
                                "genero": recomendaciones_df.loc[nodo, "GENERO"],
                                "puntuacion": recomendaciones_df.loc[nodo, "PUNTUACION"],
                                "tipo_impacto": recomendaciones_df.loc[nodo, "TIPO DE IMPACTO"],
                                "impacto_social": recomendaciones_df.loc[nodo, "IMPACTO SOCIAL"],
                                "audio_url": "#"  # Aquí podrías añadir una URL real para la canción
                            }
                            for nodo in nodos_recomendados
                        ]

                        # Construir el grafo de recomendaciones para el HTML
                        net = Network(height="750px", width="100%", font_color="black", bgcolor="#FF6347")
                        for nodo in nodos_recomendados:
                            datos = grafo_recomendaciones.nodes[nodo]
                            net.add_node(
                                str(nodo),
                                label=datos["cancion"],
                                title=f"{datos['cancion']} - {datos['artista']} ({datos['tipo_impacto']})"
                            )
                        for u, v in grafo_recomendaciones.edges():
                            if u in nodos_recomendados and v in nodos_recomendados:
                                net.add_edge(str(u), str(v))

                        # Generar HTML del grafo
                        grafo_html = net.generate_html()

    return render_template("dos.html", genero=genero, grafo_html=grafo_html, recomendaciones=recomendaciones, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)