
# app.py
# Banestes • Itacibá — Carteira de Crédito
# Dashboard Streamlit + Plotly (dark premium) com parsing fiel ao CSV "wide"
# Ajustes incluídos:
# - Tema dark elegante + alto contraste
# - Layout: 2 gráficos empilhados (Saldo em cima, Rendas embaixo)
# - Orçado = barras / Realizado = linha (sem “cair pra zero” quando não há dados: NaN para e a linha para)
# - Hover limpo (sem %{legendgroup})
# - “Legenda contextual” com produtos selecionados em chip
# - KPIs: Saldo é snapshot (último mês com dado) + gap vs orçado (mesmo mês); Rendas é acumulado
# - Removidas infos de debug (amostra/expander) e download desnecessário

import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ==========================================================
# BANESTES DESIGN TOKENS (dark premium + contraste)
# ==========================================================
BRAND = {
    "blue": "#1E0AE8",
    "green": "#00AB16",
    "ink": "#F2F6FF",
    "muted": "#B8C6E6",
    "bg": "#0A1020",
    "card": "#0F1830",
    "border": "rgba(242, 246, 255, 0.12)",
    "grid": "rgba(242, 246, 255, 0.08)",
}

st.set_page_config(
    page_title="Banestes | Itacibá — Carteira de Crédito",
    page_icon="📊",
    layout="wide",
)

# ==========================================================
# UI / CSS (premium, menos “quadrado”)
# ==========================================================
def inject_css() -> None:
  st.markdown('<p class="section-title">KPIs</p>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Realizado (último mês com dado)</div>
        <p class="kpi-value">{fmt_br(saldo_real_last)}</p>
        {badge_html(saldo_farol_txt, saldo_farol_color)}
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Saldo • Gap vs Orçado (mesmo mês)</div>
        <p class="kpi-value">{fmt_br(saldo_gap)}</p>
        <div style="margin-top:6px; color: rgba(242,246,255,0.70); font-size:12px;">
          Base: <b>{saldo_last_real_dt.strftime("%b/%Y") if saldo_last_real_dt is not None else "—"}</b>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Rendas • Orçado (acumulado no recorte)</div>
        <p class="kpi-value">{fmt_br(rendas_orc)}</p>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Rendas • Realizado (acumulado no recorte)</div>
        <p class="kpi-value">{fmt_br(rendas_real)}</p>
        {badge_html(rendas_farol_txt, rendas_farol_color)}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Plotly template (compatível e dark)
# ==========================================================
def make_plotly_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["card"],
        font=dict(family="Inter, system-ui, sans-serif", color=BRAND["ink"], size=13),
        xaxis=dict(
            gridcolor=BRAND["grid"],
            zeroline=False,
            showline=False,
            ticks="outside",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(color=BRAND["muted"]),
            title=dict(text="", font=dict(color=BRAND["muted"])),
        ),
        yaxis=dict(
            gridcolor=BRAND["grid"],
            zeroline=False,
            showline=False,
            ticks="outside",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(color=BRAND["muted"]),
            title=dict(text="", font=dict(color=BRAND["muted"])),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=BRAND["muted"]),
        ),
        margin=dict(l=14, r=14, t=48, b=14),
    )
    return t


# ==========================================================
# Helpers de formatação e parsing (BR numbers)
# ==========================================================
def fmt_br(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def br_to_float(x) -> float:
    """Converte string BR (1.234.567,89) -> float. Suporta (123,45) negativo."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("\xa0", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return np.nan
    return -v if neg else v

def kpi_status(real: float, plan: float, eps: float = 1e-9):
    """
    Retorna (label, color) para farol.
    - verde: real >= plan
    - vermelho: real < plan
    - neutro: se não tiver dados
    """
    if real is None or plan is None or (isinstance(real, float) and np.isnan(real)) or (isinstance(plan, float) and np.isnan(plan)):
        return ("Sem dado", "rgba(242,246,255,0.55)")
    return ("Cumprido", BRAND["green"]) if (real + eps) >= plan else ("Não cumprido", "#FF4D6D")


def badge_html(text: str, color: str) -> str:
    return f"""
    <span style="
      display:inline-flex; align-items:center; gap:6px;
      padding:5px 10px; border-radius:999px;
      border:1px solid {BRAND['border']};
      background: rgba(255,255,255,0.06);
      color:{color}; font-weight:800; font-size:12px;">
      ● {text}
    </span>
    """



MONTH_MAP = {"Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
             "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12}


@st.cache_data(show_spinner=False)
def parse_relatorio(uploaded_file) -> pd.DataFrame:
    """
    Parser fiel ao relatório no formato wide:
      - 2025: blocos [Orçado, Realizado] por mês
      - 2026: apenas Orçado por mês
    Retorna tidy:
      tipo, data, ano, mes, produto_cod, produto_desc, produto, orcado, realizado
    """
    raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")
    cols = list(raw.columns)

    # ----- 2025: colunas padrão do export atual -----
    code_2025 = "2025"
    desc_2025 = "Unnamed: 1"

    blocks_2025 = []
    for start in range(2, min(50, len(cols)), 4):
        mlabel = raw.iloc[0, start] if start < len(cols) else None
        if pd.isna(mlabel):
            continue
        mlabel = str(mlabel).strip()
        if mlabel in MONTH_MAP and (start + 1) < len(cols):
            blocks_2025.append((MONTH_MAP[mlabel], cols[start], cols[start + 1]))

    # ----- 2026: colunas padrão do export atual -----
    code_2026 = "Unnamed: 50"
    desc_2026 = "Unnamed: 51"

    blocks_2026 = []
    for idx in range(53, 65):
        colname = f"Unnamed: {idx}"
        if colname not in raw.columns:
            continue
        mlabel = raw.iloc[0, raw.columns.get_loc(colname)]
        if pd.isna(mlabel):
            continue
        mlabel = str(mlabel).strip()
        if mlabel in MONTH_MAP:
            blocks_2026.append((MONTH_MAP[mlabel], colname))

    records = []

    # 2025 rows (tipicamente começam na linha 2)
    for i in range(2, len(raw)):
        code = raw.at[i, code_2025] if code_2025 in raw.columns else None
        if pd.isna(code):
            continue
        code_str = str(code).strip()
        if not code_str.isdigit():
            continue

        desc = raw.at[i, desc_2025] if desc_2025 in raw.columns else ""
        desc_str = re.sub(r"^\s*\d+\s*-\s*", "", str(desc).replace("\xa0", " ")).strip()
        desc_str = re.sub(r"\s+", " ", desc_str)

        tipo = "Saldo" if code_str.startswith("18201") else ("Rendas" if code_str.startswith("18202") else None)
        if tipo is None:
            continue

        for m, c_orc, c_real in blocks_2025:
            records.append({
                "tipo": tipo,
                "data": pd.Timestamp(2025, m, 1),
                "ano": 2025,
                "mes": m,
                "produto_cod": code_str,
                "produto_desc": desc_str,
                "orcado": br_to_float(raw.at[i, c_orc]),
                "realizado": br_to_float(raw.at[i, c_real]),
            })

    # 2026 rows (tipicamente começam na linha 1)
    for i in range(1, len(raw)):
        code = raw.at[i, code_2026] if code_2026 in raw.columns else None
        if pd.isna(code):
            continue
        code_str = str(code).strip()
        if not code_str.isdigit():
            continue

        desc = raw.at[i, desc_2026] if desc_2026 in raw.columns else ""
        desc_str = re.sub(r"^\s*\d+\s*-\s*", "", str(desc).replace("\xa0", " ")).strip()
        desc_str = re.sub(r"\s+", " ", desc_str)

        tipo = "Saldo" if code_str.startswith("18201") else ("Rendas" if code_str.startswith("18202") else None)
        if tipo is None:
            continue

        for m, colname in blocks_2026:
            records.append({
                "tipo": tipo,
                "data": pd.Timestamp(2026, m, 1),
                "ano": 2026,
                "mes": m,
                "produto_cod": code_str,
                "produto_desc": desc_str,
                "orcado": br_to_float(raw.at[i, colname]),
                "realizado": np.nan,  # 2026 sem realizado
            })

    tidy = pd.DataFrame.from_records(records)
    tidy["produto"] = tidy["produto_cod"] + " - " + tidy["produto_desc"]
    return tidy


def safe_drop_totals(selected_codes: set[str]) -> tuple[set[str], list[str]]:
    """
    Evita dupla contagem: se selecionar TOTAL (18201/18202) e também subprodutos,
    removemos o TOTAL automaticamente.
    """
    warnings = []
    out = set(selected_codes)
    for total in ["18201", "18202"]:
        has_total = total in out
        has_children = any((c != total and c.startswith(total)) for c in out)
        if has_total and has_children:
            out.remove(total)
            warnings.append(
                f"Removi automaticamente o TOTAL {total} para evitar dupla contagem (subprodutos também selecionados)."
            )
    return out, warnings


def selected_products_label(products_df: pd.DataFrame, selected_codes: set[str], max_items: int = 2) -> str:
    """Rótulo curto para chip de produtos selecionados."""
    if not selected_codes:
        return "Todos os produtos"
    mp = dict(zip(products_df["produto_cod"], products_df["produto"]))
    names = [mp.get(c, c) for c in sorted(selected_codes)]
    if len(names) <= max_items:
        return " • ".join(names)
    return " • ".join(names[:max_items]) + f"  +{len(names) - max_items} outros"


# ==========================================================
# Série + gráfico (orçado barras, realizado linha)
# ==========================================================
def series_for(dff: pd.DataFrame, tipo: str) -> pd.DataFrame:
    x = dff[dff["tipo"] == tipo].copy()
    if x.empty:
        return x
    g = x.groupby("data", as_index=False).agg(
        orcado=("orcado", "sum"),
        # essencial: se tudo é NaN em uma data (ex: 2026), permanece NaN (linha para)
        realizado=("realizado", lambda s: s.sum(min_count=1)),
    )
    return g.sort_values("data")


def bar_line_figure(df_series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_series["data"],
            y=df_series["orcado"],
            name="Orçado",
            marker=dict(color=BRAND["blue"], line=dict(width=0)),
            opacity=0.72,
            hovertemplate="<b>%{x|%b/%Y}</b><br>Orçado: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_series["data"],
            y=df_series["realizado"],
            name="Realizado",
            mode="lines+markers",
            line=dict(color=BRAND["green"], width=3),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate="<b>%{x|%b/%Y}</b><br>Realizado: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title_text="",  # evita “undefined”
        height=520,
        barmode="overlay",
        bargap=0.28,
        legend_title_text="",
        hovermode="x unified",
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["card"],
        hoverlabel=dict(
            bgcolor="rgba(15,24,48,0.94)",
            bordercolor="rgba(242,246,255,0.15)",
            font=dict(color=BRAND["ink"]),
        ),
    )

    fig.update_xaxes(
        title_text="",
        tickformat="%b\n%Y",
        showgrid=False,
        tickfont=dict(color=BRAND["muted"]),
    )
    fig.update_yaxes(
        title_text="",
        gridcolor=BRAND["grid"],
        tickfont=dict(color=BRAND["muted"]),
        separatethousands=True,
    )
    return fig

def top5_representatividade_rendas(dff: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna dataframe com Top 5 produtos por renda (Realizado se existir; senão Orçado),
    com coluna share (%).
    """
    x = dff[dff["tipo"] == "Rendas"].copy()
    if x.empty:
        return pd.DataFrame()

    # Se existir realizado em algum lugar do recorte, usamos realizado; senão, orçado.
    has_real = x["realizado"].notna().any() and (x["realizado"].sum(min_count=1) is not np.nan)
    metric = "realizado" if has_real else "orcado"

    g = (
        x.groupby(["produto_cod", "produto"], as_index=False)
         .agg(valor=(metric, lambda s: s.sum(min_count=1)))
    )

    g = g.dropna()
    g = g[g["valor"] > 0]
    if g.empty:
        return pd.DataFrame()

    g = g.sort_values("valor", ascending=False)

    top5 = g.head(5).copy()
    rest = g.iloc[5:]["valor"].sum() if len(g) > 5 else 0.0

    if rest > 0:
        top5 = pd.concat(
            [top5, pd.DataFrame([{"produto_cod": "OUTROS", "produto": "Outros", "valor": rest}])],
            ignore_index=True,
        )

    total = float(top5["valor"].sum())
    top5["share"] = top5["valor"] / total
    top5["metric"] = metric
    return top5



# ==========================================================
# App
# ==========================================================
inject_css()
px.defaults.template = make_plotly_template()

# Top bar
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <p class="title">Itacibá • Carteira de Crédito</p>
        <p class="subtitle">Orçado x Realizado (2025) • Orçado (2026) • filtros por período e produto</p>
      </div>
      <div class="pill">
        <span style="color:{BRAND["blue"]}; font-weight:800;">● Orçado</span>
        <span style="color:{BRAND["green"]}; font-weight:800;">● Realizado</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Dados")
    uploaded = st.file_uploader("Anexe o relatório (CSV exportado)", type=["csv"])

    st.divider()
    st.markdown("### Filtros")
    periodo = st.radio("Período", ["Total", "Ano", "Mês"], horizontal=True)

if not uploaded:
    st.info("Anexe o CSV do relatório para montar o dashboard.")
    st.stop()

df = parse_relatorio(uploaded)

# Period controls
years = sorted(df["ano"].unique().tolist())
with st.sidebar:
    ano_sel = None
    mes_sel = None
    if periodo in ["Ano", "Mês"]:
        ano_sel = st.selectbox("Ano", years, index=0)
    if periodo == "Mês":
        mes_sel = st.selectbox("Mês", list(range(1, 13)), index=0)

# Produto controls
products = df[["produto_cod", "produto"]].drop_duplicates().sort_values("produto")
default_codes = [c for c in ["18201", "18202"] if c in set(products["produto_cod"])]

with st.sidebar:
    selected = st.multiselect(
        "Produtos (selecione 1 ou mais)",
        options=products["produto_cod"].tolist(),
        default=default_codes,
        format_func=lambda c: products.loc[products["produto_cod"] == c, "produto"].iloc[0],
    )

selected_set, warn_list = safe_drop_totals(set(selected))
for w in warn_list:
    st.warning(w)

dff = df[df["produto_cod"].isin(selected_set)] if selected_set else df.copy()

# Apply time filter
if periodo == "Ano" and ano_sel is not None:
    dff = dff[dff["ano"] == ano_sel]
if periodo == "Mês" and ano_sel is not None and mes_sel is not None:
    dff = dff[(dff["ano"] == ano_sel) & (dff["mes"] == mes_sel)]

prod_label = selected_products_label(products, selected_set)

# ==========================================================
# KPIs (Saldo snapshot + Rendas acumulado)
# ==========================================================
def sum_at_date(tipo: str, col: str, dt: pd.Timestamp) -> float:
    x = dff[(dff["tipo"] == tipo) & (dff["data"] == dt)][col]
    if col == "realizado":
        return float(x.sum(min_count=1))  # evita NaN -> 0
    return float(x.sum(skipna=True))

def last_real_date_with_data(tipo: str, col: str, eps: float = 0.0001):
    x = dff[(dff["tipo"] == tipo)][["data", col]].copy()
    x = x.dropna()
    # ignora zeros (ex.: mês ainda não fechado)
    x = x[x[col].abs() > eps]
    if x.empty:
        return None
    return pd.to_datetime(x["data"]).max()


# Saldo snapshot no último mês com realizado
saldo_last_real_dt = last_real_date_with_data("Saldo", "realizado")


saldo_real_last = sum_at_date("Saldo", "realizado", saldo_last_real_dt) if saldo_last_real_dt is not None else np.nan
saldo_orc_same = sum_at_date("Saldo", "orcado", saldo_last_real_dt) if saldo_last_real_dt is not None else np.nan
saldo_gap = (saldo_real_last - saldo_orc_same) if (saldo_last_real_dt is not None and not np.isnan(saldo_orc_same)) else np.nan

# Rendas acumulado no recorte
r = dff[dff["tipo"] == "Rendas"]
rendas_orc = float(r["orcado"].sum(skipna=True))
rendas_real = float(r["realizado"].sum(min_count=1))

st.markdown('<p class="section-title">KPIs</p>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Realizado (último mês com dado)</div>
        <p class="kpi-value">{fmt_br(saldo_real_last)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Gap vs Orçado (mesmo mês)</div>
        <p class="kpi-value">{fmt_br(saldo_gap)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Rendas • Orçado (acumulado no recorte)</div>
        <p class="kpi-value">{fmt_br(rendas_orc)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Rendas • Realizado (acumulado no recorte)</div>
        <p class="kpi-value">{fmt_br(rendas_real)}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# Gráfico 1: Saldo (barras + linha)
# ==========================================================
st.markdown('<p class="section-title">Evolução • Saldo da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_saldo = series_for(dff, "Saldo")
if s_saldo.empty:
    st.info("Sem dados para Saldo nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_saldo), use_container_width=True)

# ==========================================================
# Gráfico 2: Rendas (barras + linha)
# ==========================================================
st.markdown('<p class="section-title">Evolução • Rendas da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_rendas = series_for(dff, "Rendas")
if s_rendas.empty:
    st.info("Sem dados para Rendas nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_rendas), use_container_width=True)

saldo_farol_txt, saldo_farol_color = kpi_status(saldo_real_last, saldo_orc_same)
rendas_farol_txt, rendas_farol_color = kpi_status(rendas_real, rendas_orc)


