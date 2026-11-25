
"""Agente de Marketing Analytics.
Backend de la aplicación de Streamlit.
"""

# ============================================================
# 1. CONEXIÓN CON GEMINI
# ============================================================

import os
import re
import json
import time
import glob
import difflib
import unicodedata

import numpy as np
import pandas as pd

from google import genai
from google.genai import types, errors

try:
    from google.colab import userdata
    API_KEY = userdata.get("GEMINI_API_KEY")
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

excluir = (
    "tts", "image", "live", "audio",
    "embedding", "preview", "exp"
)

modelos = []

for m in client.models.list():
    nombre = m.name.replace("models/", "")

    if "generateContent" in (m.supported_actions or []) and "gemini" in nombre:
        if not any(x in nombre for x in excluir):
            modelos.append(nombre)

lite = sorted(m for m in modelos if "flash-lite" in m)

MODELO = lite[-1] if lite else sorted(modelos)[-1]

print("Modelo:", MODELO)

D_MENSUAL = 0.008


# ============================================================
# 2. FUENTES DE DATOS
# ============================================================

FUENTES = {
    "precios": {
        "descripcion": "Pruebas de precio: a cada precio, cuántas unidades se vendieron al mes (caso Nube Fit)",
        "ruta": "datos/pruebas_precio.csv",
        "columnas": {
            "segmento": "segmento",
            "precio": "precio",
            "unidades": "unidades"
        }
    },

    "tickets": {
        "descripcion": "Tickets de compra: un renglón por producto comprado (Semana 4, cafetería)",
        "ruta": "datos/tickets.csv",
        "columnas": {
            "ticket": "order_id",
            "producto": "item"
        }
    },

    "clientes": {
        "descripcion": "Clientes de una tienda con ingreso, gasto, compras por canal y edad (Semana 6)",
        "ruta": "datos/clientes_retail.csv",
        "columnas": {
            "id": "id_cliente",
            "gasto": "gasto_total"
        },
        "variables": [
            "ingreso_anual",
            "gasto_total",
            "compras_tienda",
            "compras_web",
            "compras_catalogo",
            "visitas_web_mes",
            "recencia_dias",
            "hijos_pequenos",
            "edad"
        ]
    }
}

CARPETA_APUNTES = "apuntes"

_BASE = os.path.dirname(os.path.abspath(__file__))

for _conf in FUENTES.values():
    if (
        not _conf["ruta"].startswith(("http://", "https://", "/"))
        and not os.path.isabs(_conf["ruta"])
    ):
        _conf["ruta"] = os.path.join(_BASE, _conf["ruta"])

if not os.path.isabs(CARPETA_APUNTES):
    CARPETA_APUNTES = os.path.join(_BASE, CARPETA_APUNTES)

_CACHE = {}


def cargar_datos(nombre):
    """Lee una fuente y devuelve sus columnas estandarizadas."""

    if nombre not in FUENTES:
        raise ValueError(
            f"No existe la fuente '{nombre}'. "
            f"Fuentes disponibles: {list(FUENTES)}"
        )

    if nombre in _CACHE:
        return _CACHE[nombre].copy()

    conf = FUENTES[nombre]
    ruta = conf["ruta"]

    extension = ruta.lower().split("?")[0].rsplit(".", 1)[-1]

    try:
        if extension in ("xlsx", "xls"):
            df = pd.read_excel(ruta)

        elif extension == "parquet":
            df = pd.read_parquet(ruta)

        elif extension == "json":
            df = pd.read_json(ruta)

        else:
            df = pd.read_csv(ruta)

    except Exception as e:
        raise ValueError(
            f"No pude leer '{ruta}' para la fuente '{nombre}': {e}"
        )

    df.columns = [
        str(c).split("#")[-1].strip()
        for c in df.columns
    ]

    faltan = [
        su_col
        for su_col in conf["columnas"].values()
        if su_col not in df.columns
    ]

    faltan += [
        v for v in conf.get("variables", [])
        if v not in df.columns
    ]

    if faltan:
        raise ValueError(
            f"La fuente '{nombre}' no tiene las columnas {faltan}. "
            f"Columnas del archivo: {list(df.columns)}"
        )

    for estandar, su_col in conf["columnas"].items():
        df[estandar] = df[su_col]

    _CACHE[nombre] = df

    return df.copy()


# ============================================================
# 3. HERRAMIENTAS INICIALES
# ============================================================

def listar_fuentes() -> dict:
    """Lista las fuentes de datos disponibles, su descripción,
    columnas y ejemplos de segmentos o productos."""

    resumen = {}

    for nombre, conf in FUENTES.items():

        try:
            df = cargar_datos(nombre)
            ejemplos = {}

            for estandar in conf["columnas"]:

                if not pd.api.types.is_numeric_dtype(df[estandar]):

                    ejemplos[estandar] = [
                        str(x)
                        for x in df[estandar].value_counts().index[:30]
                    ]

            resumen[nombre] = {
                "descripcion": conf["descripcion"],
                "filas": len(df),
                "columnas": list(conf["columnas"]),
                "ejemplos": ejemplos
            }

        except ValueError as e:

            resumen[nombre] = {
                "descripcion": conf["descripcion"],
                "error": str(e)
            }

    return resumen


def calcular_clv(
    margen_mensual: float,
    churn_mensual: float
) -> dict:
    """Calcula el CLV con la fórmula del curso:
    CLV = m × (1 - c) / (c + d).
    margen_mensual: margen por cliente al mes en US$.
    churn_mensual: proporción mensual, por ejemplo 0.04 = 4%.
    La tasa de descuento mensual es 0.008."""

    if churn_mensual <= 0 or churn_mensual >= 1:
        return {
            "error": "El churn debe estar entre 0 y 1 (0% y 100%). "
                     "Si dijeron 4, pregunta si es 4% (0.04)."
        }

    if margen_mensual <= 0:
        return {
            "error": "El margen debe ser mayor que 0."
        }

    m = margen_mensual
    c = churn_mensual
    d = D_MENSUAL

    return {
        "clv": round(m * (1 - c) / (c + d), 2)
    }


def churn_maximo_tolerable(
    margen_nuevo: float,
    clv_actual: float
) -> dict:
    """Calcula el churn máximo mensual que permite mantener
    el CLV después de un cambio de precio.
    margen_nuevo: margen mensual nuevo por cliente en US$.
    clv_actual: valor de vida actual del cliente en US$."""

    if margen_nuevo <= 0 or clv_actual <= 0:
        return {
            "error": "El margen nuevo y el CLV actual deben ser mayores que 0."
        }

    c = (
        (margen_nuevo - clv_actual * D_MENSUAL)
        / (margen_nuevo + clv_actual)
    )

    if c < 0:
        return {
            "error": "Con ese margen nuevo ningún churn mantiene el CLV."
        }

    return {
        "churn_maximo": round(c, 4)
    }


def valor_campana_retencion(
    clientes_contactados: int,
    uplift: float,
    clv: float,
    costo_por_contacto: float
) -> dict:
    """Calcula el beneficio esperado de una campaña de retención.
    clientes_contactados: cantidad de clientes a contactar.
    uplift: proporción incremental de retención (0.05 = 5%).
    clv: valor de vida del cliente en US$.
    costo_por_contacto: costo individual del contacto."""

    if uplift < 0 or uplift > 1:
        return {
            "error": "El uplift debe estar entre 0 y 1."
        }

    if (
        clientes_contactados <= 0
        or clv <= 0
        or costo_por_contacto < 0
    ):
        return {
            "error": "Clientes y CLV deben ser mayores que 0, "
                     "y el costo no puede ser negativo."
        }

    retenidos = clientes_contactados * uplift
    valor = retenidos * clv
    costo_total = clientes_contactados * costo_por_contacto

    beneficio = valor - costo_total

    return {
        "clientes_retenidos_extra": round(retenidos, 1),
        "valor_generado": round(valor, 2),
        "costo_total": round(costo_total, 2),
        "beneficio_neto": round(beneficio, 2),
        "conviene": beneficio > 0,
        "uplift_minimo_para_no_perder": round(
            costo_por_contacto / clv, 4
        )
    }


# ============================================================
# 4. HERRAMIENTAS DESARROLLADAS
# ============================================================

def relacion_clv_cac(
    clv: float,
    cac: float,
    margen_mensual: float
) -> dict:
    """Calcula la relación CLV/CAC y el tiempo de recuperación.
    clv: valor de vida del cliente en US$.
    cac: costo de adquisición del cliente en US$.
    margen_mensual: margen mensual por cliente en US$."""

    if clv <= 0 or cac <= 0 or margen_mensual <= 0:
        return {
            "error": "CLV, CAC y margen mensual deben ser mayores que 0."
        }

    razon = clv / cac
    meses = cac / margen_mensual

    if razon >= 3:
        lectura = "sano (3 o más)"

    elif razon >= 1:
        lectura = "rentable pero ajustado (entre 1 y 3)"

    else:
        lectura = (
            "pierde dinero (menos de 1): "
            "cuesta más traer al cliente de lo que deja"
        )

    return {
        "clv_sobre_cac": round(razon, 2),
        "meses_para_recuperar_cac": round(meses, 1),
        "lectura": lectura
    }


def estimar_elasticidad(
    segmento: str = "",
    fuente: str = "precios"
) -> dict:
    """Estima la elasticidad precio mediante una regresión log-log.
    segmento: nombre del segmento; vacío utiliza todos los datos.
    fuente: nombre de la fuente de pruebas de precio."""

    try:
        df = cargar_datos(fuente)

    except ValueError as e:
        return {"error": str(e)}

    if segmento:

        if "segmento" not in df.columns:
            return {
                "error": f"La fuente '{fuente}' no tiene columna de segmento."
            }

        disponibles = sorted(
            df["segmento"].astype(str).unique()
        )

        df = df[
            df["segmento"].astype(str).str.lower()
            == segmento.lower()
        ]

        if df.empty:
            return {
                "error": f"No existe el segmento '{segmento}'. "
                         f"Segmentos disponibles: {disponibles}"
            }

    df = df[
        (df["precio"] > 0)
        & (df["unidades"] > 0)
    ]

    if df["precio"].nunique() < 3:
        return {
            "error": "Se necesitan pruebas con al menos 3 precios distintos."
        }

    eps = float(
        np.polyfit(
            np.log(df["precio"]),
            np.log(df["unidades"]),
            1
        )[0]
    )

    if eps < -1:
        lectura = "elástica: las ventas reaccionan más que el precio"

    elif eps < 0:
        lectura = "inelástica: las ventas reaccionan menos que el precio"

    else:
        lectura = "positiva: revisen los datos, no es lo normal"

    return {
        "segmento": segmento or "todos",
        "elasticidad": round(eps, 2),
        "pruebas_usadas": int(len(df)),
        "lectura": lectura
    }


def proponer_precio(
    precio_actual: float,
    costo: float,
    elasticidad: float,
    unidades_actuales: float
) -> dict:
    """Calcula el precio óptimo teórico y el precio propuesto.
    Respeta un cambio máximo de ±10% por ciclo,
    un precio mínimo de 1.2 veces el costo
    y aprobación humana para cambios superiores al 5%.

    precio_actual: precio actual en US$.
    costo: costo unitario en US$.
    elasticidad: elasticidad negativa de la demanda.
    unidades_actuales: ventas mensuales actuales."""

    p, C, e, Q = (
        precio_actual,
        costo,
        elasticidad,
        unidades_actuales
    )

    if p <= 0 or C <= 0 or Q <= 0:
        return {
            "error": "Precio, costo y unidades deben ser mayores que 0."
        }

    if e >= 0:
        return {
            "error": "La elasticidad debe ser negativa. Revisa el signo."
        }

    if e < -1:
        p_optimo = C * e / (1 + e)

    else:
        p_optimo = None

    propuesto = p * 1.10 if p_optimo is None else p_optimo

    propuesto = min(propuesto, p * 1.10)
    propuesto = max(propuesto, p * 0.90)
    propuesto = max(propuesto, 1.2 * C)

    cambio = propuesto / p - 1

    def utilidad(precio):
        return (
            (precio - C)
            * Q
            * (precio / p) ** e
        )

    return {
        "precio_optimo": (
            round(p_optimo, 2)
            if p_optimo is not None
            else "No hay óptimo finito: conviene subir hasta el tope."
        ),
        "precio_propuesto": round(propuesto, 2),
        "cambio_pct": round(cambio * 100, 1),
        "utilidad_mensual_hoy": round(utilidad(p)),
        "utilidad_mensual_propuesta": round(utilidad(propuesto)),
        "requiere_aprobacion_humana": abs(cambio) > 0.05
    }


def metricas_regla_asociacion(
    producto_a: str,
    producto_b: str,
    fuente: str = "tickets"
) -> dict:
    """Calcula soporte, confianza y lift de una regla A → B.
    producto_a: nombre exacto del producto antecedente.
    producto_b: nombre exacto del producto consecuente.
    fuente: nombre de la fuente de tickets.
    No sustituir nombres sin confirmación del usuario."""

    try:
        df = cargar_datos(fuente)

    except ValueError as e:
        return {"error": str(e)}

    productos = sorted(
        df["producto"].astype(str).unique()
    )

    for nombre in (producto_a, producto_b):

        if nombre not in productos:

            parecidos = difflib.get_close_matches(
                nombre,
                productos,
                n=5,
                cutoff=0.3
            )

            return {
                "error": f"No existe el producto '{nombre}'.",
                "quizas_quisiste_decir": parecidos
            }

    if producto_a == producto_b:
        return {
            "error": "A y B deben ser productos distintos."
        }

    canastas = df.groupby("ticket")["producto"].apply(set)

    n = len(canastas)

    con_a = canastas.apply(
        lambda s: producto_a in s
    )

    con_b = canastas.apply(
        lambda s: producto_b in s
    )

    soporte_a = con_a.sum() / n
    soporte_b = con_b.sum() / n
    soporte_ab = (con_a & con_b).sum() / n

    confianza = soporte_ab / soporte_a
    lift = confianza / soporte_b

    return {
        "tickets": int(n),
        "soporte_a": round(float(soporte_a), 4),
        "soporte_b": round(float(soporte_b), 4),
        "soporte_ab": round(float(soporte_ab), 4),
        "confianza": round(float(confianza), 4),
        "lift": round(float(lift), 2),
        "lectura": (
            "vale la pena recomendar B a quien compra A"
            if lift > 1
            else "no se compran juntos más de lo normal"
        )
    }


def segmentar_clientes(
    k: int,
    fuente: str = "clientes"
) -> dict:
    """Agrupa clientes mediante k-means y variables estandarizadas.
    Devuelve cantidad de clientes, participación del gasto
    y promedios por segmento.
    k: número de segmentos entre 2 y 6.
    fuente: nombre de la fuente de clientes."""

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    try:
        df = cargar_datos(fuente)

    except ValueError as e:
        return {"error": str(e)}

    if k < 2 or k > 6:
        return {
            "error": "k debe estar entre 2 y 6 segmentos."
        }

    variables = FUENTES[fuente].get("variables", [])

    if len(variables) < 2:
        return {
            "error": f"Definan al menos 2 variables numéricas en FUENTES['{fuente}']."
        }

    datos = df[variables].dropna()

    X = StandardScaler().fit_transform(datos)

    etiquetas = KMeans(
        n_clusters=int(k),
        n_init=10,
        random_state=0
    ).fit_predict(X)

    datos = datos.assign(segmento=etiquetas)

    gasto = (
        df.loc[datos.index, "gasto"]
        if "gasto" in df.columns
        else None
    )

    segmentos = []

    for s, grupo in datos.groupby("segmento"):

        fila = {
            "clientes": int(len(grupo)),
            "pct_clientes": round(
                len(grupo) / len(datos) * 100,
                1
            )
        }

        if gasto is not None:

            fila["pct_gasto"] = round(
                float(
                    gasto[grupo.index].sum()
                    / gasto.sum()
                    * 100
                ),
                1
            )

        fila["promedios"] = {
            v: round(float(grupo[v].mean()), 1)
            for v in variables
        }

        segmentos.append(fila)

    segmentos.sort(
        key=lambda f: f.get("pct_gasto", 0),
        reverse=True
    )

    for i, fila in enumerate(segmentos, 1):
        fila["segmento"] = f"S{i}"

    return {
        "k": int(k),
        "clientes_usados": int(len(datos)),
        "segmentos": segmentos,
        "nota": "ordenados de mayor a menor % del gasto"
    }


# ============================================================
# 5. MINI RAG
# ============================================================

def _sin_acentos(texto):

    return "".join(
        c for c in unicodedata.normalize(
            "NFD",
            texto.lower()
        )
        if unicodedata.category(c) != "Mn"
    )


def buscar_apunte(tema: str) -> dict:
    """Busca párrafos relacionados con un tema en los apuntes.
    Devuelve el contenido y el archivo correspondiente.
    Úsala antes de explicar conceptos o fórmulas del curso.
    tema: palabras clave, por ejemplo uplift grupo de control."""

    archivos = sorted(
        glob.glob(
            os.path.join(CARPETA_APUNTES, "*.md")
        )
        +
        glob.glob(
            os.path.join(CARPETA_APUNTES, "*.txt")
        )
    )

    if not archivos:
        return {
            "error": f"No encontré apuntes en '{CARPETA_APUNTES}'."
        }

    palabras_ignorar = {
        "del", "las", "los", "una", "uno", "unos",
        "para", "por", "que", "con", "como",
        "cual", "segun", "sobre", "curso",
        "apuntes", "formula", "dime", "explica"
    }

    palabras = [
        p for p in re.findall(
            r"\w+",
            _sin_acentos(tema)
        )
        if len(p) >= 3 and p not in palabras_ignorar
    ]

    if not palabras:
        return {
            "error": "Escribe al menos una palabra clave de 3 letras o más."
        }

    resultados = []

    for ruta in archivos:

        with open(ruta, encoding="utf-8") as archivo:
            texto = archivo.read()

        for parrafo in re.split(r"\n\s*\n", texto):

            palabras_parrafo = re.findall(
                r"\w+",
                _sin_acentos(parrafo)
            )

            puntos = sum(
                palabras_parrafo.count(p)
                for p in palabras
            )

            if puntos > 0:

                resultados.append((
                    puntos,
                    os.path.basename(ruta),
                    parrafo.strip()[:600]
                ))

    if not resultados:
        return {
            "aviso": "No encontré nada sobre ese tema en los apuntes.",
            "archivos": [
                os.path.basename(a)
                for a in archivos
            ]
        }

    resultados.sort(reverse=True)

    return {
        "resultados": [
            {
                "archivo": a,
                "texto": t
            }
            for _, a, t in resultados[:3]
        ]
    }


# ============================================================
# 6. REGISTRO COMPLETO DE HERRAMIENTAS
# ============================================================

HERRAMIENTAS = {
    "listar_fuentes": listar_fuentes,
    "calcular_clv": calcular_clv,
    "churn_maximo_tolerable": churn_maximo_tolerable,
    "valor_campana_retencion": valor_campana_retencion,
    "relacion_clv_cac": relacion_clv_cac,
    "estimar_elasticidad": estimar_elasticidad,
    "proponer_precio": proponer_precio,
    "metricas_regla_asociacion": metricas_regla_asociacion,
    "segmentar_clientes": segmentar_clientes,
    "buscar_apunte": buscar_apunte
}


# ============================================================
# 7. INSTRUCCIONES DEL AGENTE
# ============================================================

INSTRUCCIONES = """
Eres un tutor experto en Marketing Analytics para estudiantes de maestría.
Tu objetivo es proporcionar respuestas correctas, verificables y fundamentadas.

REGLAS OBLIGATORIAS:

1. UTILIZACIÓN DE HERRAMIENTAS
- Para realizar cálculos, utiliza siempre la herramienta correspondiente.
- Nunca calcules de memoria ni inventes resultados.
- Los resultados de Python son la fuente principal de verdad numérica.
- Si una herramienta necesaria no está disponible, informa la limitación.

2. VALIDACIÓN DE DATOS
- Si falta un dato obligatorio, pregunta al usuario; no lo supongas.
- Respeta los mensajes de error devueltos por las herramientas.
- No sustituyas valores inválidos sin autorización.
- No cambies nombres de productos, segmentos o fuentes de datos.

3. PRODUCTOS Y SEGMENTOS INEXISTENTES
- Conserva exactamente los nombres proporcionados por el usuario.
- Si un producto o segmento no existe, informa el error.
- Puedes sugerir alternativas válidas, pero solicita confirmación.
- No ejecutes nuevamente una herramienta con nombres corregidos
  hasta que el usuario confirme expresamente la sustitución.
- Capuchino no debe sustituirse automáticamente por Cappuccino.

4. CONSISTENCIA MATEMÁTICA
- Utiliza las mismas fórmulas y supuestos que las herramientas de Python.
- No combines modelos matemáticos diferentes.
- Verifica que el procedimiento sea coherente con el resultado.
- Para proponer_precio utiliza el modelo de elasticidad constante:
  Q_nueva = Q_actual * (precio_nuevo / precio_actual) ** elasticidad.
- La utilidad proyectada es:
  (precio_nuevo - costo) * Q_nueva.
- No sustituyas este modelo por una aproximación lineal.
- No recalcules los valores numéricos devueltos por Python.
- Si un cálculo intermedio no fue verificado, no lo inventes.
- Conserva exactamente los valores devueltos por las herramientas.

5. FUENTES DE DATOS
- Utiliza listar_fuentes cuando necesites identificar datos disponibles.
- No inventes columnas, productos, clientes ni segmentos.
- Evita repetir consultas cuando ya tienes la información necesaria.

6. CONSULTA DE APUNTES
- Antes de explicar un concepto del curso, utiliza buscar_apunte,
  siempre que esta herramienta esté disponible.
- Cita únicamente archivos que contengan información relevante.
- Si los fragmentos recuperados no responden la pregunta, indícalo.
- No inventes citas ni atribuyas conocimiento externo a los apuntes.
- Si proporcionas conocimiento general, identifica claramente
  que no procede del material del curso.

7. GOBERNANZA Y APROBACIÓN HUMANA
- Respeta todas las restricciones empresariales.
- El cambio máximo de precio permitido es de más o menos 10% por ciclo.
- El precio debe ser como mínimo 1.2 veces el costo.
- Los cambios superiores al 5% en magnitud requieren aprobación humana.
- No permitas que el usuario ignore estas restricciones.
- No presentes una propuesta pendiente como una decisión aprobada.
- Si las restricciones son incompatibles, informa el conflicto.

8. INTERPRETACIÓN COMERCIAL
- No confundas ingresos, gasto, margen, utilidad y rentabilidad.
- Una asociación estadística no demuestra causalidad.
- Los tickets de compra no representan necesariamente clientes únicos.
- La elasticidad representa una sensibilidad estimada, no una garantía.
- Los resultados proyectados dependen de los supuestos del modelo.
- No generalices características individuales a partir de promedios.
- No afirmes que un segmento tiene mayor churn sin evidencia.
- Los grupos experimentales deben ser estadísticamente comparables,
  no necesariamente idénticos.

9. FORMATO DE RESPUESTA
- Explica en pasos cortos y claros.
- Muestra las fórmulas utilizadas cuando corresponda.
- Define las siglas relevantes.
- Presenta los resultados de las herramientas sin alterarlos.
- Diferencia los hechos, las estimaciones y las recomendaciones.
"""


# ============================================================
# 8. CONEXIÓN CON GEMINI Y REINTENTOS
# ============================================================

def llamar_gemini(contenido, config=None):
    """Llama a Gemini con reintentos ante errores temporales."""

    modelo_respaldo = globals().get("MODELO_RESPALDO")

    modelos_intento = [MODELO]

    if modelo_respaldo and modelo_respaldo != MODELO:
        modelos_intento.append(modelo_respaldo)

    for modelo in modelos_intento:

        for intento in range(4):

            try:
                return client.models.generate_content(
                    model=modelo,
                    contents=contenido,
                    config=config
                )

            except errors.APIError as e:

                if e.code not in (429, 500, 502, 503, 504):
                    raise

                if intento == 3:
                    ultimo_error = e
                    break

                espera = (
                    40
                    if e.code == 429
                    else 5 * (2 ** intento)
                )

                print(
                    f"Error {e.code} en {modelo}. "
                    f"Reintento en {espera} segundos..."
                )

                time.sleep(espera)

    raise ultimo_error


# ============================================================
# 9. RESPUESTA DETERMINISTA DE PRICING
# ============================================================

def respuesta_pricing_verificada(pasos):
    """Construye una respuesta directamente con los resultados
    de Python para evitar alteraciones numéricas del LLM.

    Se utiliza cuando existe una única propuesta de precio válida
    y las herramientas ejecutadas corresponden al análisis de pricing.
    """

    propuestas = [
        p for p in pasos
        if p["herramienta"] == "proponer_precio"
    ]

    herramientas_pricing = {
        "listar_fuentes",
        "estimar_elasticidad",
        "proponer_precio",
        "buscar_apunte"
    }

    if len(propuestas) != 1:
        return None

    if not all(
        p["herramienta"] in herramientas_pricing
        for p in pasos
    ):
        return None

    datos = propuestas[0]["datos"]
    resultado = propuestas[0]["resultado"]

    if "error" in resultado:
        return None

    optimo = resultado["precio_optimo"]

    if isinstance(optimo, (int, float)):
        optimo = f"US${optimo:,.2f}"

    aprobacion = (
        "Sí. Requiere aprobación humana antes de implementarse."
        if resultado["requiere_aprobacion_humana"]
        else "No requiere aprobación según el umbral de cambio."
    )

    utilidad_hoy = resultado["utilidad_mensual_hoy"]
    utilidad_propuesta = resultado["utilidad_mensual_propuesta"]

    # Cálculos realizados directamente por Python
    diferencia = utilidad_propuesta - utilidad_hoy

    if diferencia > 0:
        interpretacion = (
            f"La utilidad mensual proyectada aumenta "
            f"US${diferencia:,.2f} respecto a la situación actual."
        )

    elif diferencia < 0:
        interpretacion = (
            f"La utilidad mensual proyectada disminuye "
            f"US${abs(diferencia):,.2f} respecto a la situación actual."
        )

    else:
        interpretacion = (
            "La utilidad mensual proyectada se mantiene sin cambios."
        )

    contexto_segmento = ""

    for paso in pasos:

        if paso["herramienta"] == "estimar_elasticidad":

            estimacion = paso["resultado"]

            if "error" not in estimacion:

                contexto_segmento = (
                    f"- Segmento analizado: {estimacion['segmento']}\n"
                    f"- Pruebas de precio utilizadas: "
                    f"{estimacion['pruebas_usadas']}\n"
                )

    respuesta = f"""
### Análisis de optimización de precios

Los resultados fueron obtenidos mediante las herramientas de Python.

### 1. Datos utilizados

{contexto_segmento}- Precio actual: US${datos['precio_actual']:,.2f}
- Costo unitario: US${datos['costo']:,.2f}
- Elasticidad: {datos['elasticidad']}
- Unidades actuales: {datos['unidades_actuales']:,.0f}

### 2. Resultados verificados

| Indicador | Resultado |
|---|---:|
| Precio óptimo teórico | {optimo} |
| Precio propuesto | US${resultado['precio_propuesto']:,.2f} |
| Cambio porcentual | {resultado['cambio_pct']}% |
| Utilidad mensual actual | US${utilidad_hoy:,.2f} |
| Utilidad mensual proyectada | US${utilidad_propuesta:,.2f} |
| Diferencia de utilidad | US${diferencia:,.2f} |

### 3. Modelo utilizado

El precio óptimo teórico se calcula mediante:

$$P^* = C \\times \\frac{{\\varepsilon}}{{1 + \\varepsilon}}$$

Cuando la elasticidad es menor que -1.

La demanda proyectada utiliza el modelo de elasticidad constante:

$$Q(p) = Q_0 \\times \\left(\\frac{{p}}{{p_0}}\\right)^{{\\varepsilon}}$$

La utilidad proyectada se calcula como:

$$U(p) = (p-C) \\times Q(p)$$

### 4. Recomendación

**Precio recomendado: US${resultado['precio_propuesto']:,.2f}.**

{interpretacion}

La proyección depende de los supuestos del modelo de elasticidad
constante y no constituye una garantía de ventas futuras.

### 5. Aprobación humana

**{aprobacion}**

*Los valores numéricos de la propuesta provienen directamente de
Python y no fueron recalculados por Gemini.*
"""

    return respuesta.strip()


# ============================================================
# 10. AGENTE CON LLAMADAS A HERRAMIENTAS
# ============================================================

def preguntar(
    pregunta,
    historial=None,
    max_pasos=8,
    mostrar=True,
    herramientas=None
):
    """Realiza una consulta al agente.

    Devuelve:
        respuesta: texto final.
        historial: conversación.
        pasos: herramientas, parámetros y resultados.

    herramientas: menú disponible para el agente.
    """

    menu = (
        herramientas
        if herramientas is not None
        else HERRAMIENTAS
    )

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCCIONES,
        tools=list(menu.values()),
        temperature=0.2,
        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    historial = list(historial or [])

    historial.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=pregunta
                )
            ]
        )
    )

    pasos = []

    for paso in range(1, max_pasos + 1):

        # 1. Gemini analiza la consulta

        r = llamar_gemini(
            historial,
            config
        )

        historial.append(
            r.candidates[0].content
        )

        # 2. Si no solicita herramientas, termina

        if not r.function_calls:

            # Protección determinista para pricing

            respuesta_verificada = (
                respuesta_pricing_verificada(pasos)
            )

            if respuesta_verificada is not None:

                historial[-1] = types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=respuesta_verificada
                        )
                    ]
                )

                return (
                    respuesta_verificada,
                    historial,
                    pasos
                )

            # Respuesta normal para las demás consultas

            return r.text, historial, pasos

        resultados = []

        # 3. Ejecutar las herramientas solicitadas

        for pedido in r.function_calls:

            datos = dict(pedido.args)

            try:

                salida = menu[pedido.name](
                    **datos
                )

            except Exception as e:

                salida = {
                    "error": f"{type(e).__name__}: {e}"
                }

            pasos.append({
                "herramienta": pedido.name,
                "datos": datos,
                "resultado": salida
            })

            if mostrar:

                print(
                    f"Paso {paso}: "
                    f"{pedido.name}({datos}) "
                    f"→ {str(salida)[:160]}"
                )

            resultados.append(
                types.Part.from_function_response(
                    name=pedido.name,
                    response={
                        "resultado": salida
                    }
                )
            )

        # 4. Devolver los resultados a Gemini

        historial.append(
            types.Content(
                role="user",
                parts=resultados
            )
        )

    return (
        "Me detuve: llegué al máximo de pasos.",
        historial,
        pasos
    )


# ============================================================
# 11. VERIFICACIÓN DE RESULTADOS
# ============================================================

def revisar(
    descripcion,
    obtenido,
    esperado,
    tolerancia=0.01
):
    """Compara un resultado con el valor esperado."""

    if (
        isinstance(obtenido, (int, float))
        and abs(obtenido - esperado)
        <= tolerancia * max(1, abs(esperado))
    ):

        print(
            "OK       ",
            descripcion,
            "→",
            obtenido
        )

    else:

        print(
            "REVISAR  ",
            descripcion,
            "→ obtuvo",
            obtenido,
            "· esperado",
            esperado
        )
