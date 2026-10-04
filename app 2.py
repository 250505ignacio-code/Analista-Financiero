# ==============================================================================
# WEB APP INTERACTIVA DE ANÁLISIS FINANCIERO Y CUANTITATIVO CON IA
# Framework: Streamlit
# Datos de mercado: Yahoo Finance (yfinance)
# Inteligencia Artificial: Groq API (GPT-OSS 120B)
#
# Instalación:
#   pip install streamlit yfinance groq pandas plotly
# Ejecución:
#   streamlit run app.py
# ==============================================================================

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from groq import Groq
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Analista Financiero IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Dashboard & Analista Financiero Cuantitativo con IA")
st.markdown(
    "Genera análisis técnicos, métricas de mercado y proyecciones de inversión "
    "impulsadas por IA a partir de datos de Yahoo Finance."
)

# Nota: Groq retiró llama-3.3-70b-versatile y llama-3.1-8b-instant (16/08/2026).
# Si algún modelo deja de funcionar, revisa https://console.groq.com/docs/models
MODELOS_GROQ = {
    "GPT-OSS 120B (más preciso)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (más rápido)": "openai/gpt-oss-20b",
    "Qwen 3.6 27B (preview)": "qwen/qwen3.6-27b",
}


# ------------------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ------------------------------------------------------------------------------
def clave_groq_por_defecto() -> str:
    """Busca la API key en st.secrets o en variables de entorno (opcional)."""
    try:
        clave = st.secrets.get("GROQ_API_KEY", "")
        if clave:
            return clave
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


@st.cache_data(ttl=300, show_spinner=False)
def obtener_historial(ticker: str, periodo: str) -> pd.DataFrame:
    """Descarga el historial de precios desde Yahoo Finance (cache 5 min)."""
    try:
        return yf.Ticker(ticker).history(period=periodo, auto_adjust=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_info(ticker: str) -> dict:
    """Datos fundamentales básicos (pueden faltar según el activo)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    campos = [
        "shortName", "sector", "industry", "marketCap",
        "trailingPE", "forwardPE", "priceToBook", "dividendYield", "beta",
    ]
    return {c: info.get(c) for c in campos}


def calcular_rsi(cierre: pd.Series, n: int = 14) -> pd.Series:
    delta = cierre.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))


def retorno(cierre: pd.Series, dias: int):
    if len(cierre) > dias:
        return (cierre.iloc[-1] / cierre.iloc[-1 - dias] - 1) * 100
    return None


def formatear_usd(n) -> str:
    if n is None or pd.isna(n):
        return "N/D"
    for sufijo, factor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(n) >= factor:
            return f"${n / factor:.2f}{sufijo}"
    return f"${n:,.0f}"


def fmt(valor, decimales=2, sufijo="") -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    return f"{valor:,.{decimales}f}{sufijo}"


def calcular_metricas(ticker: str):
    """Calcula precio, variación e indicadores técnicos con 1 año de datos."""
    h = obtener_historial(ticker, "1y")
    if h.empty or len(h) < 2:
        return None

    cierre = h["Close"]
    precio = float(cierre.iloc[-1])
    previo = float(cierre.iloc[-2])
    info = obtener_info(ticker)

    return {
        "ticker": ticker,
        "nombre": info.get("shortName") or ticker,
        "sector": info.get("sector") or "N/D",
        "precio": precio,
        "cambio_pct": (precio / previo - 1) * 100,
        "max_dia": float(h["High"].iloc[-1]),
        "min_dia": float(h["Low"].iloc[-1]),
        "max_52s": float(h["High"].max()),
        "min_52s": float(h["Low"].min()),
        "ret_1m": retorno(cierre, 21),
        "ret_3m": retorno(cierre, 63),
        "ret_1a": retorno(cierre, len(cierre) - 1),
        "sma20": float(cierre.rolling(20).mean().iloc[-1]) if len(cierre) >= 20 else None,
        "sma50": float(cierre.rolling(50).mean().iloc[-1]) if len(cierre) >= 50 else None,
        "rsi": float(calcular_rsi(cierre).iloc[-1]) if len(cierre) >= 15 else None,
        "volatilidad": float(cierre.pct_change().std() * (252 ** 0.5) * 100),
        "cap": info.get("marketCap"),
        "per": info.get("trailingPE"),
        "per_fwd": info.get("forwardPE"),
        "beta": info.get("beta"),
    }


def grafico_velas(ticker: str, periodo: str) -> go.Figure | None:
    h = obtener_historial(ticker, periodo)
    if h.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=h.index, open=h["Open"], high=h["High"],
            low=h["Low"], close=h["Close"], name="Precio",
        ),
        row=1, col=1,
    )
    if len(h) >= 20:
        fig.add_trace(
            go.Scatter(x=h.index, y=h["Close"].rolling(20).mean(),
                       name="SMA 20", line=dict(width=1.5)),
            row=1, col=1,
        )
    if len(h) >= 50:
        fig.add_trace(
            go.Scatter(x=h.index, y=h["Close"].rolling(50).mean(),
                       name="SMA 50", line=dict(width=1.5)),
            row=1, col=1,
        )
    fig.add_trace(
        go.Bar(x=h.index, y=h["Volume"], name="Volumen", opacity=0.5),
        row=2, col=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def grafico_comparativo(tickers: list[str], periodo: str) -> go.Figure | None:
    """Rendimiento normalizado (base 100) para comparar varios activos."""
    fig = go.Figure()
    for t in tickers:
        h = obtener_historial(t, periodo)
        if h.empty:
            continue
        base = h["Close"] / h["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=h.index, y=base, name=t, mode="lines"))
    if not fig.data:
        return None
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Base 100",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def construir_resumen_prompt(metricas: dict) -> str:
    lineas = []
    for t, m in metricas.items():
        lineas.append(
            f"• {t} ({m['nombre']}, sector: {m['sector']}): ${m['precio']:.2f} USD "
            f"({m['cambio_pct']:+.2f}% hoy) | Rango día: ${m['min_dia']:.2f}-${m['max_dia']:.2f} | "
            f"Rango 52 sem: ${m['min_52s']:.2f}-${m['max_52s']:.2f} | "
            f"Ret. 1m: {fmt(m['ret_1m'], 1, '%')}, 3m: {fmt(m['ret_3m'], 1, '%')}, "
            f"1a: {fmt(m['ret_1a'], 1, '%')} | "
            f"SMA20: {fmt(m['sma20'])}, SMA50: {fmt(m['sma50'])}, RSI14: {fmt(m['rsi'], 1)} | "
            f"Volatilidad anualizada: {fmt(m['volatilidad'], 1, '%')} | "
            f"Cap. mercado: {formatear_usd(m['cap'])}, PER: {fmt(m['per'], 1)}, "
            f"PER fwd: {fmt(m['per_fwd'], 1)}, Beta: {fmt(m['beta'])}"
        )
    return "\n".join(lineas)


def generar_informe(api_key: str, modelo: str, metricas: dict) -> str:
    tickers = list(metricas.keys())
    resumen = construir_resumen_prompt(metricas)

    prompt = f"""
Actúa como un analista financiero sénior y estratega cuantitativo de Wall Street.
Revisa los siguientes datos de mercado (fecha: {datetime.now().strftime('%d/%m/%Y')}):

{resumen}

Genera un informe completo y profesional, en español, con esta estructura en Markdown:

1. **Visión General**: sentimiento global y contexto particular de las empresas ({', '.join(tickers)}).
2. **Análisis Técnico/Operativo**: interpreta variaciones, medias móviles, RSI, volatilidad y posición dentro del rango de 52 semanas.
3. **Proyección a 6-12 Meses y Probabilidad de Ocurrencia**:
   - Proyección técnica y fundamental para cada empresa ({', '.join(tickers)}) a 6 y 12 meses.
   - Asigna un porcentaje (%) explícito de probabilidad a cada escenario (alcista / lateral / bajista), sumando 100%.
   - Fundamenta cada probabilidad con valoración, momentum, salud financiera, catalizadores del sector y entorno macro.
4. **Estrategia e Inversión**: recomendación concreta de asignación de activos a mediano/largo plazo.
5. **Riesgos y Limitaciones**: aclara que las probabilidades son estimaciones cualitativas, no cálculos estadísticos calibrados.

Reglas clave:
- Basa tus conclusiones en los datos entregados; no inventes cifras ni noticias que no conozcas con certeza.
- Usa negritas, listas, tablas Markdown cuando aplique y emojis financieros para lectura rápida.
- Mantén un tono técnico, profesional y riguroso.
"""

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Eres un analista financiero cuantitativo de nivel institucional."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return completion.choices[0].message.content


# ------------------------------------------------------------------------------
# 3. BARRA LATERAL
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Obtén tu API Key gratuita en [console.groq.com](https://console.groq.com)")

    groq_api_key = st.text_input(
        "Groq API Key (gsk_...):",
        type="password",
        value=clave_groq_por_defecto(),
        help="Clave gratuita de Groq. También puedes definir GROQ_API_KEY en st.secrets o como variable de entorno.",
    )

    nombre_modelo = st.selectbox("Modelo de IA:", list(MODELOS_GROQ.keys()), index=0)

    st.divider()
    st.markdown("**Parámetros de análisis**")
    periodo_historial = st.selectbox(
        "Periodo histórico para gráficos:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )

    st.divider()
    st.caption(
        "⚠️ Contenido informativo y educativo. No constituye asesoría financiera "
        "ni recomendación de inversión."
    )

# ------------------------------------------------------------------------------
# 4. ENTRADA DE USUARIO
# ------------------------------------------------------------------------------
st.subheader("🔎 Selección de Acciones")
col_input, col_button = st.columns([3, 1])

with col_input:
    acciones_input = st.text_input(
        "Símbolos / tickers (separados por coma):",
        value="TTWO, NVDA, AAPL",
        placeholder="Ejemplo: TTWO, NVDA, TSLA, MSFT",
    )

with col_button:
    st.write("")
    st.write("")
    boton_analizar = st.button("🚀 Generar Análisis", use_container_width=True, type="primary")

# ------------------------------------------------------------------------------
# 5. PROCESAMIENTO (al pulsar el botón)
# ------------------------------------------------------------------------------
if boton_analizar:
    tickers = list(dict.fromkeys(
        t.strip().upper() for t in acciones_input.split(",") if t.strip()
    ))

    if not groq_api_key:
        st.error("⚠️ Ingresa tu clave API de Groq en la barra lateral para continuar.")
    elif not tickers:
        st.warning("⚠️ Debes ingresar al menos un símbolo de acción.")
    else:
        metricas = {}
        with st.spinner("📡 Descargando datos de Yahoo Finance..."):
            for t in tickers:
                m = calcular_metricas(t)
                if m:
                    metricas[t] = m
                else:
                    st.warning(f"No se encontraron datos para **{t}**. Verifica el símbolo.")

        if not metricas:
            st.error("No se pudo obtener datos de ningún ticker.")
        else:
            reporte, error = None, None
            try:
                with st.spinner("🧠 Generando informe con IA en Groq..."):
                    reporte = generar_informe(
                        groq_api_key, MODELOS_GROQ[nombre_modelo], metricas
                    )
            except Exception as e:
                error = str(e)

            # Se guarda en session_state para que no desaparezca al interactuar
            st.session_state["resultado"] = {
                "metricas": metricas,
                "reporte": reporte,
                "error": error,
                "fecha": datetime.now(),
            }

# ------------------------------------------------------------------------------
# 6. VISUALIZACIÓN DE RESULTADOS
# ------------------------------------------------------------------------------
resultado = st.session_state.get("resultado")

if resultado:
    metricas = resultado["metricas"]
    tickers_ok = list(metricas.keys())

    tab_mercado, tab_informe = st.tabs(["📊 Datos de Mercado", "🤖 Informe de IA"])

    # ---- Pestaña 1: datos de mercado ----
    with tab_mercado:
        st.caption(f"Datos al {resultado['fecha'].strftime('%d/%m/%Y %H:%M')} (Yahoo Finance)")

        # Tarjetas de métricas, en filas de máximo 4
        for i in range(0, len(tickers_ok), 4):
            fila = tickers_ok[i:i + 4]
            columnas = st.columns(len(fila))
            for col, t in zip(columnas, fila):
                m = metricas[t]
                col.metric(
                    label=f"{t} · {m['nombre']}",
                    value=f"${m['precio']:,.2f}",
                    delta=f"{m['cambio_pct']:+.2f}%",
                )

        # Tabla comparativa
        st.markdown("#### Comparativa de indicadores")
        tabla = pd.DataFrame([
            {
                "Ticker": t,
                "Precio": m["precio"],
                "Var. hoy %": m["cambio_pct"],
                "Ret. 1m %": m["ret_1m"],
                "Ret. 3m %": m["ret_3m"],
                "Ret. 1a %": m["ret_1a"],
                "RSI 14": m["rsi"],
                "Volat. anual %": m["volatilidad"],
                "PER": m["per"],
                "Cap. mercado": formatear_usd(m["cap"]),
            }
            for t, m in metricas.items()
        ]).set_index("Ticker")
        st.dataframe(tabla.round(2), use_container_width=True)

        # Comparativo normalizado
        if len(tickers_ok) > 1:
            st.markdown(f"#### Rendimiento comparado ({periodo_historial}, base 100)")
            fig_comp = grafico_comparativo(tickers_ok, periodo_historial)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)

        # Gráficos de velas por ticker
        st.markdown("#### Gráficos históricos")
        subtabs = st.tabs(tickers_ok)
        for sub, t in zip(subtabs, tickers_ok):
            with sub:
                fig = grafico_velas(t, periodo_historial)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sin datos históricos para este periodo.")

    # ---- Pestaña 2: informe IA ----
    with tab_informe:
        if resultado["error"]:
            st.error(f"❌ Error al conectar con la API de Groq: `{resultado['error']}`")
        elif resultado["reporte"]:
            st.markdown(resultado["reporte"])
            st.download_button(
                label="📥 Descargar Informe (.md)",
                data=resultado["reporte"],
                file_name=f"analisis_financiero_{resultado['fecha'].strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
            st.caption(
                "Las probabilidades son estimaciones generadas por un modelo de lenguaje, "
                "no cálculos estadísticos. No son asesoría financiera."
            )
else:
    st.info("👆 Ingresa tus tickers y pulsa **Generar Análisis** para comenzar.")
