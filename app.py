# ==============================================================================
# TERMINAL FINANCIERO: ANÁLISIS CUANTITATIVO + ESCÁNER DE ACCIONES INFRAVALORADAS
# Framework: Streamlit
# Datos de mercado: Yahoo Finance (yfinance)
# Inteligencia Artificial: Groq API (GPT-OSS / Qwen)
#
# requirements.txt:
#   streamlit
#   yfinance
#   groq
#   pandas
#   numpy
#   plotly
# ==============================================================================

import os
import re
from datetime import datetime

import numpy as np
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
    page_title="Terminal Financiero & Escáner IA",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💎 Terminal Financiero & Escáner de Acciones Infravaloradas con IA")
st.markdown(
    "Analiza activos individuales o escanea universos de mercado "
    "para detectar acciones infravaloradas mediante métricas fundamentales y análisis IA."
)

# Groq retiró Llama 3.3 70B, Llama 3.1 8B y Mixtral (los de 2025). Si algún modelo
# deja de funcionar, revisa los vigentes en https://console.groq.com/docs/models
MODELOS_GROQ = {
    "GPT-OSS 120B (recomendado)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (ultrarrápido)": "openai/gpt-oss-20b",
    "Qwen 3.6 27B (preview)": "qwen/qwen3.6-27b",
}

# Listas estáticas de referencia: la composición real de los índices cambia con el tiempo.
UNIVERSOS_MERCADO = {
    "S&P 500 (Selección Top 30)": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JNJ", "JPM", "V",
        "PG", "XOM", "UNH", "HD", "MA", "BAC", "ABBV", "PFE", "KO", "DIS",
        "CSCO", "PEP", "TMO", "COST", "CVX", "WMT", "MCD", "INTC", "QCOM", "WFC",
    ],
    "Nasdaq 100 (Tecnología y Crecimiento)": [
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "TSLA", "AVGO", "ASML", "COST",
        "PEP", "AZN", "CSCO", "TMUS", "TXN", "AMD", "QCOM", "INTC", "AMGN", "INTU",
        "HON", "AMAT", "BKNG", "SBUX", "ADP", "GILD", "MDLZ", "ADI", "LRCX", "MU",
    ],
    "Dow Jones (Valor y Gigantes Industriales)": [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
        "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
    ],
}

CAMPOS_INFO = [
    "shortName", "longName", "sector", "industry", "marketCap",
    "trailingPE", "forwardPE", "priceToBook", "beta",
    "pegRatio", "enterpriseToEbitda", "profitMargins", "freeCashflow",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice", "recommendationKey",
    "currentPrice", "regularMarketPrice", "dividendRate", "trailingAnnualDividendYield",
]


# ------------------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES DE DATOS Y MÉTRICAS
# ------------------------------------------------------------------------------
def ancho_completo(fn, *args, **kwargs):
    """Muestra un elemento a todo el ancho, compatible con versiones nuevas y
    antiguas de Streamlit (width="stretch" vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def clave_groq_por_defecto() -> str:
    try:
        clave = st.secrets.get("GROQ_API_KEY", "")
        if clave:
            return clave
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# Las funciones con prefijo "_" lanzan error si no hay datos, así Streamlit
# NO guarda en caché los fallos (por ejemplo, si Yahoo limita las consultas).
@st.cache_data(ttl=300, show_spinner=False)
def _descargar_historial(ticker: str, periodo: str) -> pd.DataFrame:
    h = yf.Ticker(ticker).history(period=periodo, auto_adjust=True)
    if h is None or h.empty:
        raise ValueError(f"Sin historial para {ticker}")
    return h


def obtener_historial(ticker: str, periodo: str) -> pd.DataFrame:
    try:
        return _descargar_historial(ticker, periodo)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _descargar_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info or {}
    nombre = info.get("shortName") or info.get("longName")
    if not nombre:
        raise ValueError(f"Sin información para {ticker}")
    res = {c: info.get(c) for c in CAMPOS_INFO}
    res["shortName"] = nombre
    res["ticker"] = ticker
    return res


def obtener_info_completa(ticker: str) -> dict:
    try:
        return _descargar_info(ticker)
    except Exception:
        return {"ticker": ticker}


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


def redondear(valor, decimales=2):
    """Redondea y devuelve NaN si el dato falta (mantiene las columnas numéricas)."""
    if valor is None or pd.isna(valor):
        return np.nan
    return round(float(valor), decimales)


def rendimiento_dividendo(info: dict) -> float:
    """Dividend yield en %. Se calcula con dividendo anual / precio, porque Yahoo
    cambió el formato del campo 'dividendYield' (fracción vs porcentaje)."""
    precio = info.get("currentPrice") or info.get("regularMarketPrice")
    tasa = info.get("dividendRate")
    if tasa and precio:
        return float(tasa) / float(precio) * 100
    y = info.get("trailingAnnualDividendYield")
    return float(y) * 100 if y else 0.0


def calcular_metricas(ticker: str):
    h = obtener_historial(ticker, "1y")
    if h.empty or len(h) < 2:
        return None

    cierre = h["Close"]
    precio = float(cierre.iloc[-1])
    previo = float(cierre.iloc[-2])
    info = obtener_info_completa(ticker)

    ret_std = cierre.pct_change().dropna().std()
    volatilidad = float(ret_std * (252 ** 0.5) * 100) if not pd.isna(ret_std) else 0.0

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
        "volatilidad": volatilidad,
        "cap": info.get("marketCap"),
        "per": info.get("trailingPE"),
        "per_fwd": info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "beta": info.get("beta"),
    }


# ------------------------------------------------------------------------------
# 3. MOTOR DE ESCANEO DE ACCIONES INFRAVALORADAS
# ------------------------------------------------------------------------------
def calcular_score_infravaloracion(info: dict) -> float:
    """Puntaje heurístico de 0 a 100 que estima el descuento fundamental."""
    score = 50.0

    # PER
    pe = info.get("trailingPE")
    if pe and pe > 0:
        if pe < 12:
            score += 15
        elif pe < 18:
            score += 8
        elif pe > 30:
            score -= 15

    # P/B
    pb = info.get("priceToBook")
    if pb and pb > 0:
        if pb < 1.2:
            score += 15
        elif pb < 2.0:
            score += 8
        elif pb > 5.0:
            score -= 10

    # PEG
    peg = info.get("pegRatio")
    if peg and peg > 0:
        if peg < 1.0:
            score += 20
        elif peg < 1.5:
            score += 10
        elif peg > 2.5:
            score -= 15

    # EV / EBITDA
    ev_ebitda = info.get("enterpriseToEbitda")
    if ev_ebitda and ev_ebitda > 0:
        if ev_ebitda < 10:
            score += 15
        elif ev_ebitda < 14:
            score += 5
        elif ev_ebitda > 22:
            score -= 10

    # Margen neto
    margin = info.get("profitMargins")
    if margin and margin > 0.15:
        score += 10
    elif margin is not None and margin < 0.0:
        score -= 15

    return float(np.clip(score, 0.0, 100.0))


def escanear_universo(tickers, max_per, max_pb, max_peg, min_div, avance=None):
    """Devuelve (DataFrame con las acciones que cumplen, lista de tickers sin datos)."""
    resultados, sin_datos = [], []
    total = len(tickers)

    for i, t in enumerate(tickers, start=1):
        if avance:
            avance(i, total, t)

        info = obtener_info_completa(t)
        if not info.get("shortName"):
            sin_datos.append(t)
            continue

        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        peg = info.get("pegRatio")
        ev_ebitda = info.get("enterpriseToEbitda")
        div = rendimiento_dividendo(info)

        cumple_pe = pe is not None and 0 < pe <= max_per
        cumple_pb = pb is not None and 0 < pb <= max_pb
        cumple_peg = peg is None or 0 < peg <= max_peg  # sin dato de PEG no descarta
        cumple_div = div >= min_div

        if cumple_pe and cumple_pb and cumple_peg and cumple_div:
            precio = info.get("currentPrice") or info.get("regularMarketPrice")
            target = info.get("targetMeanPrice")
            upside = (target / precio - 1) * 100 if (target and precio) else None

            resultados.append({
                "Ticker": t,
                "Nombre": info.get("shortName"),
                "Sector": info.get("sector") or "N/D",
                "Score Valor": redondear(calcular_score_infravaloracion(info), 1),
                "Precio": redondear(precio),
                "PER": redondear(pe),
                "P/B": redondear(pb),
                "PEG": redondear(peg),
                "EV/EBITDA": redondear(ev_ebitda),
                "Div Yield %": redondear(div),
                "Cap. Mercado": formatear_usd(info.get("marketCap")),
                "Target Analistas": redondear(target),
                "Upside Analistas %": redondear(upside, 1),
            })

    df = pd.DataFrame(resultados)
    if not df.empty:
        df = df.sort_values(by="Score Valor", ascending=False).reset_index(drop=True)
    return df, sin_datos


# ------------------------------------------------------------------------------
# 4. GRÁFICOS
# ------------------------------------------------------------------------------
def grafico_velas(ticker: str, periodo: str):
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
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def grafico_comparativo(tickers, periodo: str):
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


# ------------------------------------------------------------------------------
# 5. GENERACIÓN DE INFORMES CON IA (GROQ)
# ------------------------------------------------------------------------------
def consultar_groq(api_key: str, modelo: str, rol: str, prompt: str) -> str:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": rol},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,  # los modelos con razonamiento gastan parte en "pensar"
    )
    texto = completion.choices[0].message.content or ""
    # Algunos modelos devuelven su razonamiento entre <think>...</think>
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    if not texto:
        raise RuntimeError(
            "El modelo no devolvió texto (posible límite de tokens). "
            "Vuelve a intentar o prueba otro modelo."
        )
    return texto


def generar_informe(api_key: str, modelo: str, metricas: dict) -> str:
    resumen = []
    for t, m in metricas.items():
        resumen.append(
            f"• {t} ({m['nombre']}, Sector: {m['sector']}): ${m['precio']:.2f} USD "
            f"({m['cambio_pct']:+.2f}% hoy) | "
            f"Rango 52 sem: ${m['min_52s']:.2f}-${m['max_52s']:.2f} | "
            f"Cap. mercado: {formatear_usd(m['cap'])} | "
            f"PER: {fmt(m['per'])}, PER fwd: {fmt(m['per_fwd'])}, P/B: {fmt(m['pb'])}, "
            f"PEG: {fmt(m['peg'])}, EV/EBITDA: {fmt(m['ev_ebitda'])}, Beta: {fmt(m['beta'])} | "
            f"RSI14: {fmt(m['rsi'], 1)}, SMA20: {fmt(m['sma20'])}, SMA50: {fmt(m['sma50'])} | "
            f"Ret 1m: {fmt(m['ret_1m'], 1, '%')}, 3m: {fmt(m['ret_3m'], 1, '%')}, "
            f"1a: {fmt(m['ret_1a'], 1, '%')} | Volatilidad anual: {fmt(m['volatilidad'], 1, '%')}"
        )

    prompt = f"""
Actúa como un estratega cuantitativo y gestor de fondos de inversión de Wall Street.
Analiza la siguiente información técnica y fundamental (fecha: {datetime.now().strftime('%d/%m/%Y')}):

{chr(10).join(resumen)}

Escribe un informe institucional en español con el siguiente formato Markdown:

1. **Diagnóstico Cuantitativo y Valoración**:
   - Evalúa si los múltiplos (PER, P/B, PEG, EV/EBITDA) indican sobrevaloración o descuento relativo.
2. **Análisis Técnico y Momentum**:
   - Interpreta medias móviles (SMA20/50), RSI14, volatilidad y posición dentro del rango de 52 semanas.
3. **Proyección de Escenarios (6-12 Meses)**:
   - Escenarios Alcista, Base y Bajista para cada activo, con porcentajes explícitos que sumen 100%.
   - Fundamenta cada probabilidad con valoración, momentum, salud financiera, catalizadores y entorno macro.
4. **Dictamen Estratégico**:
   - Recomendación clara de asignación de capital y gestión del riesgo.
5. **Riesgos y Limitaciones**:
   - Aclara que las probabilidades son estimaciones cualitativas, no cálculos estadísticos calibrados.

Reglas clave:
- Basa tus conclusiones en los datos entregados; no inventes cifras ni noticias que no conozcas con certeza.
- Usa negritas, listas, tablas Markdown cuando aplique y emojis financieros para lectura rápida.
- Mantén un tono técnico, profesional y riguroso.
"""
    return consultar_groq(
        api_key, modelo,
        "Eres un analista financiero cuantitativo de nivel institucional.",
        prompt,
    )


def generar_informe_infravaloradas(api_key: str, modelo: str, df_infra: pd.DataFrame) -> str:
    datos_tabla = df_infra.fillna("N/D").to_csv(index=False, sep="|")

    prompt = f"""
Actúa como un analista especialista en Value Investing (estilo Benjamin Graham / Warren Buffett).
Un escáner cuantitativo detectó las siguientes empresas con múltiplos de descuento
(tabla separada por "|"; "Score Valor" es una heurística propia de 0 a 100, no un dato oficial):

{datos_tabla}

Elabora una auditoría de valoración en español estructurada así:

1. **Análisis de Oportunidades y Margen de Seguridad**:
   - Identifica cuáles presentan un margen de seguridad genuino según PER, P/B, PEG y EV/EBITDA.
2. **Filtro de "Trampas de Valor" (Value Traps)**:
   - Advierte dónde el bajo múltiplo podría reflejar deterioro estructural del negocio o del sector.
3. **Catalizadores Fundamentales**:
   - Qué factores macro/microeconómicos podrían impulsar una revalorización hacia el valor intrínseco.
4. **Ranking y Selección Final**:
   - Clasifica las 3 mejores oportunidades por relación riesgo/beneficio.

Reglas: basa el análisis en la tabla; no inventes cifras ni noticias que no conozcas con certeza.
Recuerda que son filtros cuantitativos de un instante, no una recomendación de inversión.
"""
    return consultar_groq(
        api_key, modelo,
        "Eres un experto en Value Investing y valoración intrínseca de empresas.",
        prompt,
    )


# ------------------------------------------------------------------------------
# 6. INTERFAZ Y NAVEGACIÓN PRINCIPAL
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Obtén tu API Key gratuita en [console.groq.com](https://console.groq.com)")
    groq_api_key = st.text_input(
        "Groq API Key (gsk_...):",
        type="password",
        value=clave_groq_por_defecto(),
        help="Clave gratuita de Groq. También puedes definir GROQ_API_KEY en Secrets.",
    )
    nombre_modelo = st.selectbox("Modelo de IA:", list(MODELOS_GROQ.keys()), index=0)
    periodo_historial = st.selectbox(
        "Periodo histórico:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
    )
    st.divider()
    st.caption("⚠️ Uso estrictamente informativo. No constituye asesoría de inversión.")

tab_analisis, tab_escaner = st.tabs([
    "📈 Dashboard y Análisis Cuantitativo",
    "🔍 Escáner de Acciones Infravaloradas",
])

# ------------------------------------------------------------------------------
# TAB 1: DASHBOARD INDIVIDUAL / MULTI-ACTIVO
# ------------------------------------------------------------------------------
with tab_analisis:
    st.subheader("🔎 Análisis de Activos Específicos")
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
        boton_analizar = ancho_completo(st.button, "🚀 Analizar Tickers", type="primary")

    if boton_analizar:
        tickers = list(dict.fromkeys(
            t.strip().upper() for t in acciones_input.split(",") if t.strip()
        ))
        if not groq_api_key:
            st.error("⚠️ Ingresa tu clave API de Groq en la barra lateral.")
        elif not tickers:
            st.warning("⚠️ Debes ingresar al menos un símbolo.")
        else:
            metricas, sin_datos = {}, []
            with st.spinner("📡 Descargando datos de Yahoo Finance..."):
                for t in tickers:
                    m = calcular_metricas(t)
                    if m:
                        metricas[t] = m
                    else:
                        sin_datos.append(t)

            if not metricas:
                st.error("No se encontraron datos para los tickers ingresados.")
            else:
                reporte, error = None, None
                try:
                    with st.spinner("🧠 Generando informe con IA en Groq..."):
                        reporte = generar_informe(
                            groq_api_key, MODELOS_GROQ[nombre_modelo], metricas
                        )
                except Exception as e:
                    error = str(e)

                # Se guarda para que no desaparezca al interactuar con la página
                st.session_state["res_analisis"] = {
                    "metricas": metricas,
                    "sin_datos": sin_datos,
                    "reporte": reporte,
                    "error": error,
                    "fecha": datetime.now(),
                }

    res = st.session_state.get("res_analisis")
    if res:
        m_dict = res["metricas"]
        t_list = list(m_dict.keys())

        st.caption(f"Datos al {res['fecha'].strftime('%d/%m/%Y %H:%M')} (Yahoo Finance)")
        if res["sin_datos"]:
            st.warning("Sin datos para: " + ", ".join(res["sin_datos"]) + ". Verifica el símbolo.")

        # Tarjetas de precio, en filas de máximo 4
        for i in range(0, len(t_list), 4):
            fila = t_list[i:i + 4]
            for col, t in zip(st.columns(len(fila)), fila):
                m = m_dict[t]
                col.metric(
                    f"{t} · {m['nombre']}",
                    f"${m['precio']:,.2f}",
                    f"{m['cambio_pct']:+.2f}%",
                )

        # Tabla comparativa
        st.markdown("#### Tabla de Múltiplos e Indicadores")
        df_comp = pd.DataFrame([
            {
                "Ticker": t, "Precio": m["precio"], "Var %": m["cambio_pct"],
                "Ret 1m %": m["ret_1m"], "Ret 3m %": m["ret_3m"], "Ret 1a %": m["ret_1a"],
                "PER": m["per"], "P/B": m["pb"], "PEG": m["peg"], "EV/EBITDA": m["ev_ebitda"],
                "RSI": m["rsi"], "Volatilidad %": m["volatilidad"],
                "Cap. Mercado": formatear_usd(m["cap"]),
            }
            for t, m in m_dict.items()
        ]).set_index("Ticker")
        ancho_completo(st.dataframe, df_comp.round(2))

        # Gráfico comparativo (base 100) y velas por ticker
        if len(t_list) > 1:
            st.markdown(f"#### Rendimiento comparado ({periodo_historial}, base 100)")
            fig_c = grafico_comparativo(t_list, periodo_historial)
            if fig_c:
                ancho_completo(st.plotly_chart, fig_c)

        st.markdown("#### Gráficos de Velas")
        for tab, t in zip(st.tabs(t_list), t_list):
            with tab:
                fig_v = grafico_velas(t, periodo_historial)
                if fig_v:
                    ancho_completo(st.plotly_chart, fig_v)
                else:
                    st.info("Sin datos históricos para este periodo.")

        # Informe IA
        st.markdown("#### 🤖 Informe Institucional IA")
        if res["error"]:
            st.error(f"❌ Error al conectar con la API de Groq: `{res['error']}`")
        elif res["reporte"]:
            st.markdown(res["reporte"])
            st.download_button(
                label="📥 Descargar Informe (.md)",
                data=res["reporte"],
                file_name=f"analisis_financiero_{res['fecha'].strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
            st.caption(
                "Las probabilidades son estimaciones generadas por un modelo de lenguaje, "
                "no cálculos estadísticos. No son asesoría financiera."
            )

# ------------------------------------------------------------------------------
# TAB 2: ESCÁNER DE ACCIONES INFRAVALORADAS
# ------------------------------------------------------------------------------
with tab_escaner:
    st.subheader("🔍 Filtro Algorítmico de Acciones Infravaloradas")
    st.markdown(
        "Aplica filtros fundamentales sobre universos de empresas para aislar "
        "oportunidades de inversión por valor."
    )

    col_u, col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1, 1])
    with col_u:
        universo_sel = st.selectbox("Universo de Mercado:", list(UNIVERSOS_MERCADO.keys()))
    with col_f1:
        max_per_val = st.slider("PER Máximo:", 5.0, 40.0, 20.0)
    with col_f2:
        max_pb_val = st.slider("P/B Máximo:", 0.5, 10.0, 3.0)
    with col_f3:
        max_peg_val = st.slider("PEG Máximo:", 0.5, 3.0, 1.5)
    with col_f4:
        min_div_val = st.slider("Dividendo mín. %:", 0.0, 6.0, 0.0, step=0.5)

    with st.expander("ℹ️ ¿Cómo funcionan el filtro y el Score Valor?"):
        st.markdown(
            "- **Filtro:** una acción pasa si cumple PER, P/B y dividendo mínimo, "
            "y su PEG está bajo el máximo (si Yahoo no tiene PEG, no se descarta).\n"
            "- **Score Valor (0-100):** parte en 50 y suma o resta puntos según PER, P/B, "
            "PEG, EV/EBITDA y margen neto. Es una **heurística propia**, no una valoración oficial.\n"
            "- Un múltiplo bajo no siempre es una oportunidad: puede ser una *trampa de valor*."
        )

    col_btn_scan, _ = st.columns([1, 3])
    with col_btn_scan:
        boton_escanear = ancho_completo(
            st.button, "🔎 Escanear Mercado", type="primary"
        )

    if boton_escanear:
        lista_tickers = UNIVERSOS_MERCADO[universo_sel]
        barra = st.progress(0.0, text="Iniciando escaneo...")

        def actualizar_barra(i, total, ticker):
            barra.progress(i / total, text=f"Analizando {ticker} ({i}/{total})")

        df_resultados, sin_datos_esc = escanear_universo(
            tickers=lista_tickers,
            max_per=max_per_val,
            max_pb=max_pb_val,
            max_peg=max_peg_val,
            min_div=min_div_val,
            avance=actualizar_barra,
        )
        barra.empty()

        st.session_state["res_escaner"] = {
            "df": df_resultados,
            "sin_datos": sin_datos_esc,
            "universo": universo_sel,
            "total": len(lista_tickers),
            "fecha": datetime.now(),
        }
        # El informe anterior ya no corresponde a esta nueva tabla
        st.session_state.pop("reporte_valor", None)

    res_esc = st.session_state.get("res_escaner")
    if res_esc is not None:
        df_esc = res_esc["df"]
        st.caption(
            f"Universo: {res_esc['universo']} · {res_esc['total']} activos · "
            f"{res_esc['fecha'].strftime('%d/%m/%Y %H:%M')}"
        )
        if res_esc["sin_datos"]:
            st.warning(
                f"Yahoo Finance no devolvió datos para {len(res_esc['sin_datos'])} activo(s): "
                + ", ".join(res_esc["sin_datos"])
                + ". Si son muchos, puede ser un límite temporal de consultas: espera un minuto y reintenta."
            )

        if df_esc.empty:
            st.warning(
                "No se encontraron empresas que cumplan todos los filtros. "
                "Prueba relajar los límites de PER, P/B o PEG."
            )
        else:
            st.success(f"Se encontraron **{len(df_esc)}** acciones que cumplen los filtros.")
            ancho_completo(st.dataframe, df_esc, hide_index=True)

            st.divider()
            st.subheader("🧠 Auditoría de Valoración con IA")
            st.markdown(
                "Genera un análisis para detectar oportunidades reales y descartar trampas de valor."
            )

            if st.button("🤖 Auditar Resultados con IA", type="secondary"):
                if not groq_api_key:
                    st.error("Ingresa tu API Key de Groq en la barra lateral.")
                else:
                    with st.spinner("Analizando múltiplos y fundamentos con la IA..."):
                        try:
                            st.session_state["reporte_valor"] = generar_informe_infravaloradas(
                                groq_api_key, MODELOS_GROQ[nombre_modelo], df_esc
                            )
                        except Exception as e:
                            st.error(f"❌ Error generando informe: `{e}`")

            if "reporte_valor" in st.session_state:
                st.markdown(st.session_state["reporte_valor"])
                st.download_button(
                    label="📥 Descargar Auditoría de Valoración (.md)",
                    data=st.session_state["reporte_valor"],
                    file_name=f"auditoria_infravaloradas_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )
                st.caption(
                    "Análisis generado por un modelo de lenguaje a partir de filtros cuantitativos. "
                    "No es asesoría financiera."
                )
