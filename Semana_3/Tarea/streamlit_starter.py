"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    return pd.read_csv(base_dir / "data" / "uplift_campaign.csv")

@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(
        train_t[FEATURES],
        train_t["responded_60d"]
    )
    modelo_control = LogisticRegression(max_iter=1000).fit(
        train_c[FEATURES],
        train_c["responded_60d"]
    )


    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption(
    "Esta app permite evaluar qué porcentaje de clientes conviene contactar "
    "para maximizar el valor incremental de la campaña."
)

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
    "Porcentaje de la base a contactar",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(
        score_col,
        ascending=False
    ).reset_index(drop=True)

    es_tratado = (
        orden["treatment_group"] == "treatment"
    ).values

    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(
        np.where(es_tratado, respondio, 0)
    )

    acumulado_control = np.cumsum(
        np.where(~es_tratado, respondio, 0)
    )

    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)

    xs, ys = [0.0], [0.0]

    for i in np.linspace(1, n, steps).astype(int):
        nt = n_tratados_acum[i - 1]
        nc = n_control_acum[i - 1]

        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0

        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)

    return np.array(xs), np.array(ys)


xs_uplift, ys_uplift = curva_qini(
    test,
    "uplift_estimado"
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    xs_uplift,
    ys_uplift,
    color=GREEN,
    linewidth=3,
    label="T-Learner"
)

ax.plot(
    [0, 1],
    [0, ys_uplift[-1]],
    color=GRAY,
    linestyle="--",
    label="Al azar"
)

ax.set_xlabel("% de la base contactada")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Curva Qini del T-Learner")
ax.legend()

plt.tight_layout()

st.pyplot(fig)


# Ordenar clientes por uplift estimado
orden_uplift = test.sort_values(
    "uplift_estimado",
    ascending=False
)

# Construir tabla de políticas
filas = []

for pct in range(10, 101, 10):

    n_contactar = int(len(test) * pct / 100)

    seleccionados = orden_uplift.head(
        n_contactar
    ).index

    valor_cliente, cantidad = valor_de_la_politica(
        test,
        seleccionados
    )

    filas.append({
        "% Contactado": pct,
        "Clientes": cantidad,
        "Valor por cliente": valor_cliente,
        "Valor total": valor_cliente * cantidad
    })

tabla_politicas = pd.DataFrame(filas)

st.subheader("Tabla de políticas")

st.dataframe(
    tabla_politicas.style.format({
        "% Contactado": "{:.0f}%",
        "Clientes": "{:,.0f}",
        "Valor por cliente": "${:,.2f}",
        "Valor total": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Resultado correspondiente al porcentaje seleccionado con el slider
n_elegido = int(len(test) * pct_contactar / 100)

idx_elegido = orden_uplift.head(
    n_elegido
).index

valor_elegido, cantidad_elegida = valor_de_la_politica(
    test,
    idx_elegido
)

st.subheader("Política seleccionada")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Clientes a contactar",
    f"{cantidad_elegida:,}"
)

col2.metric(
    "Valor por cliente",
    f"${valor_elegido:,.2f}"
)

col3.metric(
    "Valor incremental total",
    f"${valor_elegido * cantidad_elegida:,.2f}"
)



