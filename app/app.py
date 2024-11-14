'''
import pandas as pd
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template, request

app = Flask(__name__)

# Función para cargar los datos
def cargar_canciones(file_path="BASE_DATE_MUSIC_COMPLEJIDAD.csv"):
    # Cargar el archivo CSV con las columnas correctas
    data = pd.read_csv(file_path, encoding='latin-1')
    data.columns = ["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                    "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION", "AÑO"]
    
    # Convertir las columnas numéricas
    data['PUNTUACION'] = pd.to_numeric(data['PUNTUACION'], errors='coerce')
    data['IMPACTO SOCIAL'] = pd.to_numeric(data['IMPACTO SOCIAL'], errors='coerce')
    
    # Devolver una muestra de datos
    return data.sample(n=min(13, len(data)), random_state=42)

# Crear el grafo con forma de guitarra peruana
def construir_grafo_guitarra(file_path):
    # Cargar canciones
    canciones = cargar_canciones(file_path)

    # Crear el grafo
    G = nx.Graph()

    # Definir las posiciones de los nodos en forma de guitarra peruana
    posiciones = {
        # Cuerpo
        0: (0, 0), 1: (2, 0), 2: (1, 1), 3: (0, 2), 4: (2, 2),
        # Mástil
        5: (1, 3), 6: (1, 4), 7: (1, 5),
        # Cuerdas
        8: (0.8, 3), 9: (1.2, 3), 10: (0.8, 4), 11: (1.2, 4), 12: (0.8, 5), 13: (1.2, 5),
    }

    nodos = list(posiciones.keys())
    for idx, nodo in enumerate(nodos):
        if idx < len(canciones):
            cancion = canciones.iloc[idx]['CANCION']
            G.add_node(nodo, label=cancion, pos=posiciones[nodo])
        else:
            G.add_node(nodo, label=f"Nodo {nodo}", pos=posiciones[nodo])

    # Conexiones para formar la guitarra
    conexiones = [
        # Cuerpo
        (0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4),
        # Mástil
        (2, 5), (5, 6), (6, 7),
        # Cuerdas
        (5, 8), (5, 9), (6, 10), (6, 11), (7, 12), (7, 13),
    ]
    G.add_edges_from([(a, b) for a, b in conexiones if a in G.nodes and b in G.nodes])

    # Crear el grafo interactivo
    net = Network(height="750px", width="100%", bgcolor="#FF6347", font_color="white")
    for node, pos in nx.get_node_attributes(G, 'pos').items():
        net.add_node(node, label=G.nodes[node]['label'], x=pos[0] * 300, y=-pos[1] * 300, fixed=True)
    for edge in G.edges:
        net.add_edge(*edge, width=6) 
    return net.generate_html()


@app.route("/")
def index():
    file_path = "BASE_DATE_MUSIC_COMPLEJIDAD.csv"
    grafo_html = construir_grafo_guitarra(file_path)
    data= {
        "titulo": "Grafo con forma de Guitarra",
        "bienvenida": "BIENVENIDO A WEOLD"
    }
    return render_template("index.html", grafo_html=grafo_html, data=data)
@app.route("/uno")
def mostrar_pagina_dos():
    return render_template("uno.html")

@app.route("/dos", methods=["GET", "POST"])
def mostrar_recomendaciones():
    if request.method == "GET":
        return render_template("dos.html", recomendaciones=None, grafo_html=None)

    #POST, procesa el género seleccionado
    genero = request.form.get("genero")

    #rafo basado en el género seleccionado
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    #nodos y aristas basados en el género
    canciones = {
        "cumbia": [("Cumbia1", "Cumbia2"), ("Cumbia2", "Cumbia3"), ("Cumbia3", "Cumbia1")],
        "huayno": [("Huayno1", "Huayno2"), ("Huayno2", "Huayno3"), ("Huayno1", "Huayno3")],
        "marinera": [("Marinera1", "Marinera2"), ("Marinera2", "Marinera3")],
        "vals": [("Vals1", "Vals2"), ("Vals2", "Vals3")],
        "criollo": [("Criollo1", "Criollo2"), ("Criollo2", "Criollo3")],
    }

    #nodos y conexiones al grafo interactivo
    for edge in canciones.get(genero, []):
        net.add_node(edge[0], label=edge[0])
        net.add_node(edge[1], label=edge[1])
        net.add_edge(edge[0], edge[1])

    #grafo como HTML
    grafo_html = net.generate_html()

    #Top 3 
    recomendaciones = [f"{genero} - Canción {i}" for i in range(1, 4)]

    #página  renderizada con el grafo y las recomendaciones
    return render_template("dos.html", recomendaciones=recomendaciones, grafo_html=grafo_html)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
'''
import pandas as pd
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def cargar_canciones(file_path="BASE_DATE_MUSIC_COMPLEJIDAD.csv"):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.columns = ["CANCION", "GRUPO/ARTISTA", "GENERO", "PUNTUACION", 
                    "IMPACTO SOCIAL", "TIPO DE IMPACTO", "DESCRIPCION", "AÑO"]
    
    data['PUNTUACION'] = pd.to_numeric(data['PUNTUACION'], errors='coerce')
    data['IMPACTO SOCIAL'] = pd.to_numeric(data['IMPACTO SOCIAL'], errors='coerce')
    
    return data.sample(n=min(13, len(data)), random_state=42)

def construir_grafo_guitarra(file_path):
    canciones = cargar_canciones(file_path)

    G = nx.Graph()

    posiciones = {
        #cuerpo
        0: (0, 0), 1: (2, 0), 2: (1, 1), 3: (0, 2), 4: (2, 2),
        #mastil
        5: (1, 3), 6: (1, 4), 7: (1, 5),
        #cuerdas
        8: (0.8, 3), 9: (1.2, 3), 10: (0.8, 4), 11: (1.2, 4), 12: (0.8, 5), 13: (1.2, 5),
    }

    nodos = list(posiciones.keys())
    for idx, nodo in enumerate(nodos):
        if idx < len(canciones):
            cancion = canciones.iloc[idx]['CANCION']
            G.add_node(nodo, label=cancion, pos=posiciones[nodo])
        else:
            G.add_node(nodo, label=f"Nodo {nodo}", pos=posiciones[nodo])

    conexiones = [
        (0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4),
        (2, 5), (5, 6), (6, 7),
        (5, 8), (5, 9), (6, 10), (6, 11), (7, 12), (7, 13),
    ]
    G.add_edges_from([(a, b) for a, b in conexiones if a in G.nodes and b in G.nodes])

    net = Network(height="750px", width="100%", bgcolor="#FF6347", font_color="white")
    for node, pos in nx.get_node_attributes(G, 'pos').items():
        net.add_node(node, label=G.nodes[node]['label'], x=pos[0] * 300, y=-pos[1] * 300, fixed=True)
    for edge in G.edges:
        net.add_edge(*edge, width=6) 
    return net.generate_html()

@app.route("/")
def index():
    file_path = "BASE_DATE_MUSIC_COMPLEJIDAD.csv"
    grafo_html = construir_grafo_guitarra(file_path)
    data = {
        "titulo": "Grafo con forma de Guitarra",
        "bienvenida": "WEOLD"
    }
    return render_template("index.html", grafo_html=grafo_html, data=data)

@app.route("/uno")
def mostrar_pagina_dos():
    return render_template("uno.html")

@app.route("/dos")
def mostrar_pagina_dos_actual():
    return render_template("dos.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)

