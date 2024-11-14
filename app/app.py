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

@app.route("/dos", methods=["GET", "POST"])
def mostrar_pagina_dos_actual():
    if request.method == "POST":
        # datos del formulario
        genero = request.form.get("genero", "Sin género seleccionado")
        return render_template("dos.html", genero=genero)
    return render_template("dos.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

