from flask import Flask, render_template, request
from pyvis.network import Network
import networkx as nx

app = Flask(__name__)

@app.route("/")
def mostrar_grafo():
    # Crear el grafo
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C", "D", "E"])
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")])

    # Crear una red interactiva con pyvis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    grafo_html = net.generate_html()

    # Variables adicionales
    data = {
        "titulo": "Mi Grafo Interactivo",
        "bienvenida": "Bienvenido a la visualización del grafo interactivo."
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
