"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"



@st.cache_data
def cargar():
    DATA_DIR = Path(__file__).resolve().parent / "data"

    catalogo = pd.read_csv(DATA_DIR / "catalogo.csv")
    hist = pd.read_csv(DATA_DIR / "ordenes_historial.csv")
    hold = pd.read_csv(DATA_DIR / "ordenes_holdout.csv")
    exp = pd.read_csv(DATA_DIR / "experimento.csv")
    clientes = pd.read_csv(DATA_DIR / "clientes.csv")

    productos = list(catalogo["item"])

    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0
    )

    X = M.values.astype(np.float32)

    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0
    ).values

    NUEVO = H * (1 - X)
    NUEVO[:, productos.index("Agua")] = 0

    es_control = (
        exp.set_index("customer_id").loc[M.index, "grupo"] == "control"
    ).values

    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)


def puntajes(metodo, x_vec, K):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it
    # TODO 3: filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    vecinos = np.argsort(-sims)[:K]
    return sims[vecinos] @ X[vecinos]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption("Ayuda a identificar qué productos recomendar a cada cliente según sus compras y las preferencias de clientes similares.")

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox(
    "Método de recomendación",
    ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]
)
    K = st.slider(
    "Número de vecinos",
    min_value=5,
    max_value=100,
    value=30
)

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    st.write("Ya compró:", [productos[j] for j in np.where(x_vec > 0)[0]])

    # TODO 5: muestren las 3 recomendaciones con su puntaje y, para cada una, el "porque"
    #         - por ítem: "porque compró X (similitud 0.87)"
    #         - por cliente: "N de sus K vecinos más parecidos lo compraron"
    #         - popularidad: "es de lo más comprado"
    #         (y qué hacen con un cliente SIN historial: los métodos colaborativos dan puntaje 0)
    if x_vec.sum() == 0 and metodo != "Popularidad":
        scores = popularidad.astype(float)
        metodo_usado = "Popularidad"
        st.info("Cliente sin historial: recomendamos productos populares.")
    else:
        scores = puntajes(metodo, x_vec, K)
        metodo_usado = metodo

    recomendaciones = top3(scores, x_vec)

    st.subheader("Top 3 recomendaciones")

    for j in recomendaciones:

        if metodo_usado == "Popularidad":
            motivo = "Es uno de los productos más comprados."

        elif metodo_usado == "Filtrado por ítem":
            comprados = np.where(x_vec > 0)[0]
            similar = comprados[np.argmax(S_it[j, comprados])]

            motivo = (
                f"Porque compró {productos[similar]} "
                f"(similitud {S_it[j, similar]:.2f})."
            )

        else:
            sims = cosine_similarity(x_vec[None, :], X)[0]
            vecinos = np.argsort(-sims)[:K]

            cantidad = int(X[vecinos, j].sum())

            motivo = (
                f"{cantidad} de sus {K} vecinos más parecidos "
                "compraron este producto."
            )

        st.write(f"**{productos[j]}** — Puntaje: {scores[j]:.2f}")
        st.caption(motivo)   
with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    elegibles = np.where(es_control & (NUEVO.sum(axis=1) > 0))[0]

    n_eval = st.slider(
        "Número de clientes a evaluar",
        min_value=50,
        max_value=len(elegibles),
        value=300
    )

    muestra = np.random.default_rng(42).choice(
        elegibles, size=n_eval, replace=False
    )

    resultados = []

    for nombre in ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]:

        aciertos = []
        recomendados = set()

        for u in muestra:
            scores = puntajes(nombre, X[u], K)
            top = top3(scores, X[u])

            aciertos.append(NUEVO[u, top].sum() > 0)
            recomendados.update(top.tolist())

        resultados.append({
            "Método": nombre,
            "Hit Rate@3 (%)": round(np.mean(aciertos) * 100, 2),
            "Cobertura": len(recomendados)
        })

    tabla = pd.DataFrame(resultados)

    st.subheader("Comparación de recomendadores")
    st.dataframe(tabla, use_container_width=True)

    st.subheader("Hit Rate@3 por método")
    st.bar_chart(tabla.set_index("Método")["Hit Rate@3 (%)"])