import pandas as pd
import networkx as nx
import os
from pyvis.network import Network
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Asegúrate de que la ruta del archivo sea correcta
file_path = os.path.join(r"C:\Users\andre\code\peruvianmusic\BASE_DATE_MUSIC.xlsx")

# Verifica que el archivo exista
if os.path.exists(file_path):
    # Lee todas las columnas necesarias del archivo Excel
    data = pd.read_excel(file_path, engine='openpyxl', usecols=["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                                                               "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION"])
else:
    print(f"El archivo no se encuentra en la ruta: {file_path}")

# Cargar todos los nodos de canciones desde el archivo Excel
def cargar_canciones(file_path):
    print(f"Esta es la base de datos: {file_path}")
    data = pd.read_excel(file_path, engine='openpyxl', usecols=["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                                                               "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION"])
    data['PUNTUACION'] = pd.to_numeric(data['PUNTUACION'], errors='coerce')
    data['IMPACTO SOCIAL'] = pd.to_numeric(data['IMPACTO SOCIAL'], errors='coerce')

    # Devolver todas las canciones
    return data

# Normalizar características y calcular similitudes
def calcular_similitudes(canciones):
    # Normalizar las características
    scaler = MinMaxScaler()
    canciones_features = canciones[['PUNTUACION', 'IMPACTO SOCIAL']].values
    canciones_normalizadas = scaler.fit_transform(canciones_features)

    # Calcular la similitud entre canciones
    similitudes = cosine_similarity(canciones_normalizadas)
    return similitudes

def construir_grafo(canciones):
    G = nx.Graph()

    # Calcular similitudes
    similitudes = calcular_similitudes(canciones)
    
    # Generar una posición ficticia para cada nodo (canción)
    posiciones = {idx: (idx // 40, idx % 40) for idx in range(len(canciones))}

    for idx, row in canciones.iterrows():
        cancion = row['CANCION']
        grupo = row['GRUPO/ARTISTA']
        genero = row['GENERO']
        puntuacion = row['PUNTUACION']
        impacto_social = row['IMPACTO SOCIAL']

        # Información emergente para el tooltip
        info_tooltip = (
            f"Canción: {cancion}\n"
            f"Artista: {grupo}\n"
            f"Género: {genero}\n"
            f"Puntuación: {puntuacion}\n"
            f"Impacto Social: {impacto_social}\n"
        )


        # Agregar nodo con la información emergente (tooltip)
        G.add_node(idx, label=cancion, pos=posiciones[idx], 
                genero=genero, tipo_impacto=row['TIPO DE IMPACTO'], title=info_tooltip)

    print('Aca nfor')  # OPTIMIZAR
    for i in range(len(canciones) - 1):
        for j in range(i + 1, len(canciones)):
            # Verificar que compartan el mismo género y tipo de impacto
            if (canciones['GENERO'][i] == canciones['GENERO'][j] and 
                canciones['TIPO DE IMPACTO'][i] == canciones['TIPO DE IMPACTO'][j]):

                # Verificar si la similitud entre las canciones supera un umbral
                if similitudes[i][j] > 0.5:  # Puedes ajustar el umbral de similitud
                    # Verificar que no haya más de 3 conexiones por canción
                    if len(list(G.neighbors(i))) < 3 and len(list(G.neighbors(j))) < 3:
                        G.add_edge(i, j, weight=similitudes[i][j])
    print('Aca n')

    # Generando el grafo visualmente
    net = Network(height="750px", width="100%", bgcolor="#fab802", font_color="white")
    print('Aca n1')
    for node, pos in nx.get_node_attributes(G, 'pos').items():
        net.add_node(node, label=G.nodes[node]['label'], 
                    title=G.nodes[node]['title'],  # Tooltip con información
                    x=pos[0] * 300, y=pos[1] * 100, fixed=True)
    print('Aca n1')
    for edge in G.edges:
        net.add_edge(*edge, width=6)
    print('Aca n2')

    return net.generate_html()

# Página principal
@app.route("/")
def index():
    canciones = cargar_canciones(file_path)
    grafo_html = construir_grafo(canciones)
    return render_template("index.html", grafo_html=grafo_html, data={
        "titulo": "Grafo de Canciones",
        "bienvenida": "BIENVENIDO A WEOLD"
    })

@app.route("/uno")
def mostrar_pagina_dos():
    return render_template("uno.html")

@app.route("/dos", methods=["GET", "POST"])
def mostrar_pagina_dos_actual():
    if request.method == "POST":
        # datos del formulario
        genero = request.form.get("genero", "Sin género seleccionado")
        return render_template("dos.html", genero=genero)
    return render_template("dos.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

