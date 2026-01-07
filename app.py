import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Visual system (cores + tema)
# -----------------------------
COLOR_SEQ = ["#56B4E9", "#009E73", "#E69F00", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]

PLOTLY_TEMPLATE_LIGHT = "simple_white"
PLOTLY_TEMPLATE_DARK = "plotly_dark"

st.set_page_config(
    page_title="Dashboard Interativo (SOTA)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def _guess_datetime_cols(df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in df.columns:
        if df[c].dtype == "datetime64[ns]":
            candidates.append(c)
            continue
        if df[c].dtype == "object":
            # tentativa rápida de parse (amostra)
            sample = df[c].dropna().astype(str).head(50)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:
                candidates.append(c)
    return candidates

def _coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
    return out

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _categorical_cols(df: pd.DataFrame) -> list[str]:
    cats = []
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            cats.append(c)
        elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cats.append(c)
    return cats

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Use CSV ou Excel.")
    # limpa colunas
    df.columns = [str(c).strip() for c in df.columns]
    return df

def kpi_row(df: pd.DataFrame, metric_col: str):
    total = df[metric_col].sum(skipna=True)
    avg = df[metric_col].mean(skipna=True)
    med = df[metric_col].median(skipna=True)
    cnt = df[metric_col].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c2.metric("Média", f"{avg:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c3.metric("Mediana", f"{med:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c4.metric("N (válidos)", f"{cnt:,}".replace(",", "."))

def plot_timeseries(df: pd.DataFrame, date_col: str, metric_col: str, agg: str, template: str):
    if date_col not in df.columns:
        return None

    dff = df.dropna(subset=[date_col]).copy()
    if dff.empty:
        return None

    # granularidade diária (pode ajustar depois)
    dff["__date"] = pd.to_datetime(dff[date_col]).dt.date
    if agg == "Soma":
        g = dff.groupby("__date", as_index=False)[metric_col].sum()
        ytitle = f"{metric_col} (soma)"
    else:
        g = dff.groupby("__date", as_index=False)[metric_col].mean()
        ytitle = f"{metric_col} (média)"

    fig = px.line(
        g,
        x="__date",
        y=metric_col,
        markers=False,
        template=template,
        title="Tendência no tempo",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Data",
        yaxis_title=ytitle,
        showlegend=False,
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>" + ytitle + ": %{y:,.2f}<extra></extra>")
    return fig

def plot_category_rank(df: pd.DataFrame, cat_col: str, metric_col: str, agg: str, top_n: int, template: str):
    if cat_col not in df.columns:
        return None
    dff = df.dropna(subset=[cat_col]).copy()
    if dff.empty:
        return None

    if agg == "Soma":
        g = dff.groupby(cat_col, as_index=False)[metric_col].sum()
        title = f"Top {top_n} por {cat_col} (soma de {metric_col})"
        val = "sum"
    else:
        g = dff.groupby(cat_col, as_index=False)[metric_col].mean()
        title = f"Top {top_n} por {cat_col} (média de {metric_col})"
        val = "mean"

    g = g.sort_values(metric_col, ascending=False).head(top_n)

    fig = px.bar(
        g,
        x=metric_col,
        y=cat_col,
        orientation="h",
        template=template,
        title=title,
        color=metric_col,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,
    )
    fig.update_traces(hovertemplate=f"<b>%{{y}}</b><br>{metric_col}: %{{x:,.2f}}<extra></extra>")
    return fig

def plot_distribution(df: pd.DataFrame, metric_col: str, template: str):
    dff = df[metric_col].dropna()
    if dff.empty:
        return None

    fig = px.histogram(
        dff,
        x=metric_col,
        nbins=40,
        template=template,
        title="Distribuição (histograma)",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=metric_col,
        yaxis_title="Contagem",
        showlegend=False,
    )
    fig.update_traces(hovertemplate=f"{metric_col}: %{{x:,.2f}}<br>Contagem: %{{y}}<extra></extra>")
    return fig

def plot_corr(df: pd.DataFrame, template: str):
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num.shape[1] < 3:
        return None

    corr = num.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        template=template,
        title="Correlação (numéricos)",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_showscale=True,
    )
    return fig

def quick_story(df: pd.DataFrame, metric_col: str, date_col: str | None, cat_col: str | None, agg: str):
    bullets = []

    # 1) maior contribuição por categoria
    if cat_col and cat_col in df.columns:
        dff = df.dropna(subset=[cat_col]).copy()
        if not dff.empty:
            if agg == "Soma":
                g = dff.groupby(cat_col)[metric_col].sum().sort_values(ascending=False)
                label = "contribuição (soma)"
            else:
                g = dff.groupby(cat_col)[metric_col].mean().sort_values(ascending=False)
                label = "performance (média)"

            if len(g) > 0:
                top = g.index[0]
                bullets.append(f"• **{top}** lidera em **{label}** para **{metric_col}**.")

    # 2) tendência recente (últimos 14 pontos)
    if date_col and date_col in df.columns:
        dff = df.dropna(subset=[date_col]).copy()
        if not dff.empty:
            dff["__date"] = pd.to_datetime(dff[date_col]).dt.date
            if agg == "Soma":
                g = dff.groupby("__date", as_index=False)[metric_col].sum()
            else:
                g = dff.groupby("__date", as_index=False)[metric_col].mean()

            g = g.sort_values("__date")
            if len(g) >= 8:
                tail = g.tail(14)
                first, last = tail[metric_col].iloc[0], tail[metric_col].iloc[-1]
                if pd.notna(first) and first != 0:
                    delta = (last - first) / abs(first)
                    direction = "alta" if delta > 0 else "queda"
                    bullets.append(f"• Nos últimos pontos de tempo, houve **{direction}** de **{delta:.1%}** em **{metric_col}**.")

    # 3) outlier simples
    s = df[metric_col].dropna()
    if len(s) > 20:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            bullets.append(f"• Há **{len(outliers)}** possíveis outliers (IQR). Vale checar causas e qualidade do dado.")

    if not bullets:
        bullets = ["• Ajuste filtros/métrica para gerar uma narrativa mais forte."]

    return bullets

# -----------------------------
# UI
# -----------------------------
st.title("📊 Dashboard Interativo (bonito, rápido, acionável)")
st.caption("Suba seu CSV/Excel, escolha a métrica e explore com filtros. Sem tabelas feias, sem gráfico morto.")

with st.sidebar:
    st.header("Controles")
    uploaded = st.file_uploader("Upload do dataset (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    theme = st.toggle("Dark mode", value=False)
    template = PLOTLY_TEMPLATE_DARK if theme else PLOTLY_TEMPLATE_LIGHT

if not uploaded:
    st.info("Faça upload de um arquivo para começar.")
    st.stop()

try:
    df_raw = load_data(uploaded)
except Exception as e:
    st.error(f"Erro ao carregar arquivo: {e}")
    st.stop()

# Detecta colunas
num_cols = _numeric_cols(df_raw)
cat_cols = _categorical_cols(df_raw)
dt_guess = _guess_datetime_cols(df_raw)

if len(num_cols) == 0:
    st.error("Não encontrei colunas numéricas para usar como métrica (KPI).")
    st.stop()

with st.sidebar:
    metric_col = st.selectbox("Métrica (KPI)", options=num_cols, index=0)

    date_col = st.selectbox("Coluna de data (opcional)", options=["(nenhuma)"] + dt_guess, index=0)
    date_col = None if date_col == "(nenhuma)" else date_col

    cat_col = st.selectbox("Dimensão (categoria) (opcional)", options=["(nenhuma)"] + cat_cols, index=0)
    cat_col = None if cat_col == "(nenhuma)" else cat_col

    agg = st.radio("Agregação", options=["Soma", "Média"], horizontal=True)
    top_n = st.slider("Top N (ranking)", min_value=5, max_value=30, value=10, step=1)

# Coerção de data se existir
df = df_raw.copy()
if date_col:
    df = _coerce_datetime(df, date_col)

# -----------------------------
# Filtros no sidebar (minimalistas)
# -----------------------------
with st.sidebar:
    st.divider()
    st.subheader("Filtros")

    # filtro de data
    if date_col and date_col in df.columns and df[date_col].notna().any():
        min_d = df[date_col].min()
        max_d = df[date_col].max()
        if pd.notna(min_d) and pd.notna(max_d):
            d1, d2 = st.date_input(
                "Intervalo de datas",
                value=(min_d.date(), max_d.date()),
                min_value=min_d.date(),
                max_value=max_d.date(),
            )
            df = df[(df[date_col].dt.date >= d1) & (df[date_col].dt.date <= d2)]

    # filtro de categoria
    if cat_col and cat_col in df.columns:
        options = sorted([x for x in df[cat_col].dropna().unique().tolist()])[:5000]
        selected = st.multiselect(f"Filtrar {cat_col}", options=options, default=options[: min(len(options), 12)]))
        if selected:
            df = df[df[cat_col].isin(selected)]

    # filtro de faixa numérica
    if metric_col in df.columns and df[metric_col].notna().any():
        lo = float(df[metric_col].quantile(0.01))
        hi = float(df[metric_col].quantile(0.99))
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            r = st.slider(f"Faixa de {metric_col} (p1–p99)", min_value=lo, max_value=hi, value=(lo, hi))
            df = df[df[metric_col].between(r[0], r[1], inclusive="both")]

# -----------------------------
# Layout principal
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Visão executiva")
    kpi_row(df, metric_col)

    story = quick_story(df, metric_col, date_col, cat_col, agg)
    st.markdown("### O que os dados estão dizendo")
    for b in story:
        st.markdown(b)

with right:
    st.subheader("Preview do dado (limpo)")
    st.dataframe(df.head(50), use_container_width=True, height=320)

st.divider()

c1, c2 = st.columns(2)

with c1:
    fig_ts = plot_timeseries(df, date_col, metric_col, agg, template) if date_col else None
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Selecione uma coluna de data para ver tendência temporal.")

with c2:
    fig_rank = plot_category_rank(df, cat_col, metric_col, agg, top_n, template) if cat_col else None
    if fig_rank:
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.info("Selecione uma dimensão categórica para ver ranking por categoria.")

c3, c4 = st.columns(2)

with c3:
    fig_dist = plot_distribution(df, metric_col, template)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)

with c4:
    fig_corr = plot_corr(df, template)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlação aparece quando houver pelo menos 3 colunas numéricas úteis.")

st.caption("Dica: quando você me disser o público e a decisão que o dashboard precisa suportar, eu adapto layout, KPIs e narrativa (storytelling) para isso.")
