
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
)
from kmodes.kprototypes import KPrototypes


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Marketing Analytics - Segmentación",
    page_icon="📊",
    layout="wide",
)

CARPETA = Path(__file__).resolve().parent

VARIABLES_NUMERICAS = [
    "ingreso_anual",
    "gasto_total",
    "compras_tienda",
    "compras_web",
    "compras_catalogo",
    "visitas_web_mes",
    "recencia_dias",
    "hijos_pequenos",
    "edad",
]

VARIABLES_CATEGORICAS = [
    "educacion",
    "estado_civil",
]


# =========================================================
# 2. FUNCIONES AUXILIARES
# =========================================================

@st.cache_data
def cargar_csv(ruta):
    return pd.read_csv(ruta)


def crear_perfil(datos, etiquetas):

    tabla = datos.copy()

    tabla["Segmento"] = [
        "Ruido" if int(x) == -1 else f"C{int(x) + 1}"
        for x in etiquetas
    ]

    perfil = (
        tabla.groupby("Segmento", as_index=False)
        .agg(
            clientes=("gasto_total", "size"),
            gasto_medio=("gasto_total", "mean"),
            gasto_total=("gasto_total", "sum"),
        )
    )

    perfil["% clientes"] = (
        perfil["clientes"] / len(tabla) * 100
    )

    perfil["% gasto"] = (
        perfil["gasto_total"]
        / perfil["gasto_total"].sum() * 100
    )

    return perfil


def mostrar_tabla_valor(tabla):

    st.dataframe(
        tabla.style.format(
            {
                "gasto_medio": "US$ {:,.2f}",
                "gasto_total": "US$ {:,.2f}",
                "% clientes": "{:.2f}%",
                "% gasto": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(show_spinner=False)
def preparar_datos(datos):

    faltantes = [
        columna
        for columna in (
            VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
        )
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas: " + ", ".join(faltantes)
        )

    numeros = datos[VARIABLES_NUMERICAS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    validos = numeros.notna().all(axis=1)

    base = (
        datos.loc[validos]
        .copy()
        .reset_index(drop=True)
    )

    numeros = numeros.loc[validos].reset_index(
        drop=True
    )

    if len(base) < 3:
        raise ValueError(
            "No hay suficientes clientes con datos válidos."
        )

    base[VARIABLES_NUMERICAS] = numeros

    categorias = (
        base[VARIABLES_CATEGORICAS]
        .fillna("Sin dato")
        .astype(str)
        .to_numpy()
    )

    escalador = StandardScaler()

    X = escalador.fit_transform(
        numeros.to_numpy(dtype=float)
    )

    return base, X, categorias


@st.cache_data(
    show_spinner="Entrenando los cuatro modelos..."
)
def entrenar_modelos(X, categorias, k):

    # K-Means
    modelo_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    seg_kmeans = modelo_kmeans.fit_predict(X)

    # Clustering jerárquico Ward
    modelo_ward = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )

    seg_ward = modelo_ward.fit_predict(X)

    # Gaussian Mixture Model
    modelo_gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=3,
        reg_covar=1e-6,
    )

    seg_gmm = modelo_gmm.fit_predict(X)

    # K-Prototypes: numéricas + categóricas
    n_numericas = X.shape[1]

    X_mixto = np.empty(
        (len(X), n_numericas + categorias.shape[1]),
        dtype=object,
    )

    X_mixto[:, :n_numericas] = X
    X_mixto[:, n_numericas:] = categorias

    indices_categoricos = list(
        range(n_numericas, X_mixto.shape[1])
    )

    modelo_kproto = KPrototypes(
        n_clusters=k,
        init="Cao",
        n_init=3,
        max_iter=100,
        random_state=42,
        verbose=0,
    )

    seg_kproto = modelo_kproto.fit_predict(
        X_mixto,
        categorical=indices_categoricos,
    )

    return {
        "K-Means": seg_kmeans,
        "Jerárquico (Ward)": seg_ward,
        "GMM": seg_gmm,
        "K-Prototypes": seg_kproto,
    }


@st.cache_data(show_spinner=False)
def entrenar_dbscan(X, eps, min_samples):

    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
    )

    return modelo.fit_predict(X)


@st.cache_data(show_spinner=False)
def calcular_silueta(X, etiquetas, excluir_ruido=False):

    etiquetas = np.asarray(etiquetas)

    if excluir_ruido:
        mascara = etiquetas != -1
        X = X[mascara]
        etiquetas = etiquetas[mascara]

    n_grupos = len(np.unique(etiquetas))

    if n_grupos < 2 or n_grupos >= len(etiquetas):
        return None

    return float(
        silhouette_score(X, etiquetas)
    )


def texto_silueta(valor):
    if valor is None or pd.isna(valor):
        return "No definida"

    return f"{valor:.4f}"


# =========================================================
# 3. ENCABEZADO Y NAVEGACIÓN
# =========================================================

st.title("Segmentación de Clientes")

st.write(
    "Laboratorio de Marketing Analytics - Semana 6"
)

tab_visor, tab_comparador = st.tabs(
    [
        "Visor de segmentos",
        "Comparador de técnicas",
    ]
)


# =========================================================
# 4. VISOR DE SEGMENTOS
# =========================================================

with tab_visor:

    st.header("Visor de segmentos")

    archivo = st.file_uploader(
        "Sube un archivo CSV con los segmentos",
        type=["csv"],
    )

    if archivo is not None:
        archivo.seek(0)
        df_visor = pd.read_csv(archivo)

    else:
        ruta_default = (
            CARPETA / "segmentos_notebook.csv"
        )

        if not ruta_default.exists():
            st.error(
                "No se encontró segmentos_notebook.csv"
            )
            st.stop()

        df_visor = cargar_csv(str(ruta_default))

    # Detectar columnas de segmentación
    opciones_segmento = [
        columna
        for columna in [
            "Cluster",
            "seg_kmeans",
            "seg_gmm",
            "seg_kproto",
        ]
        if columna in df_visor.columns
    ]

    if not opciones_segmento:
        st.error(
            "El CSV no contiene una columna de segmentos."
        )
        st.stop()

    columna_grupo = st.selectbox(
        "Selecciona la segmentación que deseas visualizar",
        opciones_segmento,
    )

    if "gasto_total" not in df_visor.columns:
        st.error(
            "El archivo debe contener gasto_total."
        )
        st.stop()

    df_visor = df_visor.copy()

    df_visor["gasto_total"] = pd.to_numeric(
        df_visor["gasto_total"],
        errors="coerce",
    )

    df_visor = df_visor.dropna(
        subset=[columna_grupo, "gasto_total"]
    )

    if df_visor.empty:
        st.warning("No hay registros válidos.")
        st.stop()

    df_visor["Segmento"] = (
        df_visor[columna_grupo].astype(str)
    )

    # Indicadores principales
    st.subheader("Resumen general")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total de clientes",
        f"{len(df_visor):,}",
    )

    col2.metric(
        "Segmentos",
        df_visor["Segmento"].nunique(),
    )

    col3.metric(
        "Gasto total",
        f"US$ {df_visor['gasto_total'].sum():,.2f}",
    )

    # Perfil económico
    resumen = (
        df_visor.groupby("Segmento", as_index=False)
        .agg(
            clientes=("gasto_total", "size"),
            gasto_medio=("gasto_total", "mean"),
            gasto_total=("gasto_total", "sum"),
        )
    )

    resumen["% clientes"] = (
        resumen["clientes"] / len(df_visor) * 100
    )

    resumen["% gasto"] = (
        resumen["gasto_total"]
        / resumen["gasto_total"].sum() * 100
    )

    # Respuesta a última campaña
    if "respondio_ultima_campana" in df_visor.columns:

        respuesta = pd.to_numeric(
            df_visor["respondio_ultima_campana"],
            errors="coerce",
        )

        df_visor["_respuesta"] = respuesta

        tasas = (
            df_visor.groupby("Segmento")["_respuesta"]
            .mean() * 100
        )

        resumen["% respondió"] = (
            resumen["Segmento"].map(tasas)
        )

    st.subheader("Valor por segmento")

    columnas_formato = {
        "gasto_medio": "US$ {:,.2f}",
        "gasto_total": "US$ {:,.2f}",
        "% clientes": "{:.2f}%",
        "% gasto": "{:.2f}%",
    }

    if "% respondió" in resumen.columns:
        columnas_formato["% respondió"] = "{:.2f}%"

    st.dataframe(
        resumen.style.format(columnas_formato),
        use_container_width=True,
        hide_index=True,
    )

    # Gráfico de clientes
    st.subheader("Distribución de clientes por segmento")

    fig_clientes = px.bar(
        resumen,
        x="Segmento",
        y="clientes",
        color="Segmento",
        text="clientes",
        title="Cantidad de clientes por segmento",
    )

    fig_clientes.update_traces(
        textposition="outside"
    )

    fig_clientes.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_clientes,
        use_container_width=True,
    )

    # Gráficos adicionales
    st.subheader("Análisis visual de segmentos")

    graf1, graf2 = st.columns(2)

    with graf1:

        fig_gasto = px.bar(
            resumen,
            x="Segmento",
            y="% gasto",
            color="Segmento",
            text_auto=".2f",
            title="Participación del gasto por segmento",
        )

        fig_gasto.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_gasto,
            use_container_width=True,
        )

    with graf2:

        if "ingreso_anual" in df_visor.columns:

            df_visor["ingreso_anual"] = pd.to_numeric(
                df_visor["ingreso_anual"],
                errors="coerce",
            )

            fig_scatter = px.scatter(
                df_visor,
                x="ingreso_anual",
                y="gasto_total",
                color="Segmento",
                opacity=0.65,
                title="Relación entre ingresos y gasto",
                labels={
                    "ingreso_anual": "Ingreso anual (US$)",
                    "gasto_total": "Gasto total (US$)",
                },
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True,
            )

    # Descargar resumen
    st.download_button(
        "Descargar resumen por segmento",
        data=resumen.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumen_segmentos.csv",
        mime="text/csv",
    )

    with st.expander("Ver base de clientes"):
        st.dataframe(
            df_visor,
            use_container_width=True,
        )


# =========================================================
# 5. COMPARADOR DE LAS CINCO TÉCNICAS
# =========================================================

with tab_comparador:

    st.header("Comparador de técnicas")

    st.info(
        "El visor utiliza el CSV seleccionado. "
        "El comparador entrena las cinco técnicas "
        "sobre clientes_retail.csv."
    )

    ruta_base = CARPETA / "clientes_retail.csv"

    if not ruta_base.exists():
        st.error(
            "No se encontró clientes_retail.csv "
            "en la carpeta Semana_6."
        )
        st.stop()

    df_base = cargar_csv(str(ruta_base))

# Corregir los nombres de columnas exportadas por Orange
    df_base = df_base.rename(columns={
    "m#id_cliente": "id_cliente",
    "D#educacion": "educacion",
    "D#estado_civil": "estado_civil",
    "mD#respondio_ultima_campana": "respondio_ultima_campana"
})


    try:
        base, X_scaled, categorias = preparar_datos(
            df_base
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.write(
        f"Clientes disponibles para el análisis: "
        f"**{len(base):,}**"
    )

    # Parámetros interactivos
    st.subheader("Configuración de algoritmos")

    c1, c2, c3 = st.columns(3)

    with c1:
        k = st.slider(
            "Número de segmentos",
            min_value=2,
            max_value=8,
            value=3,
        )

    with c2:
        eps = st.selectbox(
            "DBSCAN: valor de eps",
            [1.5, 2.0, 3.0],
        )

    with c3:
        min_samples = st.number_input(
            "DBSCAN: mínimo de muestras",
            min_value=2,
            max_value=20,
            value=4,
        )

    # Entrenar algoritmos
    try:
        etiquetas = entrenar_modelos(
            X_scaled,
            categorias,
            k,
        )
    except Exception as error:
        st.error(
            f"Error al entrenar los modelos: {error}"
        )
        st.stop()

    etiquetas["DBSCAN"] = entrenar_dbscan(
        X_scaled,
        eps,
        min_samples,
    )

    # =====================================================
    # 6. TABLA COMPARATIVA
    # =====================================================

    st.subheader(
        "Comparación general de las cinco técnicas"
    )

    comparacion = []

    etiquetas_kmeans = etiquetas["K-Means"]

    for nombre, grupos in etiquetas.items():

        grupos = np.asarray(grupos)

        mascara = grupos != -1

        cantidad_grupos = len(
            np.unique(grupos[mascara])
        )

        cantidad_ruido = int(
            np.sum(~mascara)
        )

        silueta = calcular_silueta(
            X_scaled,
            grupos,
            excluir_ruido=(nombre == "DBSCAN"),
        )

        ari = adjusted_rand_score(
            etiquetas_kmeans,
            grupos,
        )

        conteos = (
            pd.Series(grupos[mascara])
            .value_counts()
            .sort_index()
        )

        tamanos = ", ".join(
            f"C{int(grupo) + 1}: {int(n)}"
            for grupo, n in conteos.items()
        )

        comparacion.append(
            {
                "Técnica": nombre,
                "Grupos": cantidad_grupos,
                "Tamaños": tamanos,
                "Ruido": cantidad_ruido,
                "Silueta": silueta,
                "ARI vs K-Means": ari,
            }
        )

    tabla_comparacion = pd.DataFrame(
        comparacion
    )

    st.dataframe(
        tabla_comparacion.style.format(
            {
                "Silueta": texto_silueta,
                "ARI vs K-Means": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Las siluetas utilizan las nueve variables "
        "numéricas estandarizadas. DBSCAN excluye "
        "el ruido del cálculo de silueta. "
        "K-Prototypes también incorpora educación "
        "y estado civil para formar sus grupos."
    )

    # =====================================================
    # 7. COINCIDENCIA ENTRE TÉCNICAS (ARI)
    # =====================================================

    st.subheader("Coincidencia entre técnicas (ARI)")

    nombres = list(etiquetas.keys())

    matriz_ari = pd.DataFrame(
        index=nombres,
        columns=nombres,
        dtype=float,
    )

    for nombre_a in nombres:
        for nombre_b in nombres:

            matriz_ari.loc[nombre_a, nombre_b] = (
                adjusted_rand_score(
                    etiquetas[nombre_a],
                    etiquetas[nombre_b],
                )
            )

    st.dataframe(
        matriz_ari.style.format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "ARI = 1 indica coincidencia perfecta. "
        "Valores cercanos a 0 indican poca coincidencia."
    )

    # =====================================================
    # 8. EXPLORACIÓN POR TÉCNICA
    # =====================================================

    st.subheader("Explorar resultados por técnica")

    tecnica_elegida = st.selectbox(
        "Selecciona una técnica para explorar",
        nombres,
    )

    grupos_elegidos = etiquetas[tecnica_elegida]

    perfil = crear_perfil(
        base,
        grupos_elegidos,
    )

    st.write(
        f"Resultados de **{tecnica_elegida}**"
    )

    mostrar_tabla_valor(perfil)

    col_g1, col_g2 = st.columns(2)

    with col_g1:

        fig_tamanos = px.bar(
            perfil,
            x="Segmento",
            y="clientes",
            color="Segmento",
            text="clientes",
            title="Tamaño de los grupos",
        )

        fig_tamanos.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_tamanos,
            use_container_width=True,
        )

    with col_g2:

        fig_valor = px.bar(
            perfil,
            x="Segmento",
            y="% gasto",
            color="Segmento",
            text_auto=".2f",
            title="Porcentaje del gasto por grupo",
        )

        fig_valor.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig_valor,
            use_container_width=True,
        )

    # Scatter Plot
    datos_grafico = base.copy()

    datos_grafico["Segmento"] = [
        "Ruido" if int(x) == -1
        else f"C{int(x) + 1}"
        for x in grupos_elegidos
    ]

    fig_dispersion = px.scatter(
        datos_grafico,
        x="ingreso_anual",
        y="gasto_total",
        color="Segmento",
        opacity=0.65,
        title=(
            f"Ingreso vs gasto - {tecnica_elegida}"
        ),
    )

    st.plotly_chart(
        fig_dispersion,
        use_container_width=True,
    )

    # =====================================================
    # 9. DESCARGAR RESULTADOS
    # =====================================================

    st.subheader("Exportar segmentaciones")

    resultados = base.copy()

    nombres_columnas = {
        "K-Means": "seg_kmeans",
        "Jerárquico (Ward)": "seg_ward",
        "GMM": "seg_gmm",
        "K-Prototypes": "seg_kproto",
        "DBSCAN": "seg_dbscan",
    }

    for nombre, columna in nombres_columnas.items():
        resultados[columna] = etiquetas[nombre]

    st.download_button(
        "Descargar comparación de clientes",
        data=resultados.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="comparacion_segmentaciones.csv",
        mime="text/csv",
    )

    # =====================================================
    # 10. RECOMENDACIÓN FINAL
    # =====================================================

    st.subheader("Recomendación para Marketing")

    perfil_kmeans = crear_perfil(
        base,
        etiquetas["K-Means"],
    )

    segmento_valioso = (
        perfil_kmeans.sort_values(
            "gasto_total",
            ascending=False,
        )
        .iloc[0]
    )

    st.markdown(
        f"""
1. Recomendamos **K-Means con {k} segmentos**
   por su facilidad de interpretación y aplicación.
2. Priorizaríamos el segmento
   **{segmento_valioso['Segmento']}**, que genera
   **{segmento_valioso['% gasto']:.2f}% del gasto total**.
3. Aplicaríamos estrategias de fidelización y
   promociones personalizadas, evaluando su
   rentabilidad antes de ampliar la inversión.
"""
    )

    st.caption(
        "El gasto histórico no equivale al CLV ni "
        "al beneficio incremental de una campaña."
    )

    with st.expander("Ver datos utilizados en el comparador"):
        st.dataframe(
            resultados,
            use_container_width=True,
        )
