# app.py
# Banestes • Itacibá — Carteira de Crédito (Dashboard Interativo)
# Streamlit + Plotly (dark premium)

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# Brand tokens (Banestes)
# ==========================================================
BRAND = {
    "blue": "#1E0AE8",
    "green": "#00AB16",
    "red": "#FF4D6D",
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
# CSS
# ==========================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700;800&display=swap');

          :root {{
            --b-blue: {BRAND["blue"]};
            --b-green: {BRAND["green"]};
            --b-red: {BRAND["red"]};
            --b-ink: {BRAND["ink"]};
            --b-muted: {BRAND["muted"]};
            --b-bg: {BRAND["bg"]};
            --b-card: {BRAND["card"]};
            --b-border: {BRAND["border"]};
            --b-grid: {BRAND["grid"]};
          }}

          html, body, [class*="css"] {{
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            color: var(--b-ink);
          }}

          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          .stApp {{ background: var(--b-bg); }}

          section[data-testid="stSidebar"] {{
            background: var(--b-card);
            border-right: 1px solid var(--b-border);
          }}

          .topbar {{
            display:flex; align-items:center; justify-content:space-between; gap:16px;
            padding:16px 18px; background: var(--b-card);
            border: 1px solid var(--b-border); border-radius: 22px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
            margin-bottom: 14px;
          }}
          .title {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800; letter-spacing: -0.02em;
            font-size: 20px; margin:0; line-height:1.1;
            color: rgba(242,246,255,0.98);
          }}
          .subtitle {{
            margin:4px 0 0 0; color: rgba(242,246,255,0.70); font-size: 13px;
          }}

          .section-title {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 18px 0 10px 0;
            color: rgba(242,246,255,0.96);
          }}

          .kpi-card-native {{
            background: var(--b-card);
            border: 1px solid var(--b-border);
            border-radius: 22px;
            padding: 16px 16px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.30);
          }}
          .kpi-label {{
            color: rgba(242,246,255,0.72);
            font-size: 12px;
            margin-bottom: 6px;
          }}
          .kpi-value {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800;
            font-size: 26px;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.05;
            color: rgba(242,246,255,0.98);
          }}
          .kpi-sub {{
            margin-top: 8px;
            color: rgba(242,246,255,0.70);
            font-size: 12px;
          }}

          .pill {{
            display:inline-flex; align-items:center; gap:8px;
            padding: 6px 12px; border-radius: 999px;
            border: 1px solid var(--b-border);
            background: rgba(30, 10, 232, 0.10);
            font-size: 12px;
            color: rgba(242,246,255,0.92);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# Plotly template
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
# Helpers
# ==========================================================
def fmt_br(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def br_to_float(x) -> float:
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
    if real is None or plan is None:
        return ("Sem dado", "rgba(242,246,255,0.55)")
    if (isinstance(real, float) and np.isnan(real)) or (isinstance(plan, float) and np.isnan(plan)):
        return ("Sem dado", "rgba(242,246,255,0.55)")
    return ("Cumprido", BRAND["green"]) if (real + eps) >= plan else ("Não cumprido", BRAND["red"])

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

MONTH_MAP = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
}

# ==========================================================
# Parser do CSV (padrão do export atual)
# ==========================================================
@st.cache_data(show_spinner=False)
def parse_relatorio(uploaded_file) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")
    cols = list(raw.columns)

    code_2025 = "2025"
    desc_2025 = "Unnamed: 1"

    blocks_2025 = []
    for start in range(2, min(60, len(cols)), 4):
        try:
            mlabel = raw.iloc[0, start]
        except Exception:
            continue
        if pd.isna(mlabel):
            continue
        mlabel = str(mlabel).strip()
        if mlabel in MONTH_MAP and (start + 1) < len(cols):
            blocks_2025.append((MONTH_MAP[mlabel], cols[start], cols[start + 1]))

    code_2026 = "Unnamed: 50"
    desc_2026 = "Unnamed: 51"

    blocks_2026 = []
    for idx in range(53, 90):
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

    # 2025
    if code_2025 in raw.columns:
        for i in range(2, len(raw)):
            code = raw.at[i, code_2025]
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

    # 2026 (orçado)
    if code_2026 in raw.columns:
        for i in range(1, len(raw)):
            code = raw.at[i, code_2026]
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
                    "realizado": np.nan,
                })

    tidy = pd.DataFrame.from_records(records)
    if tidy.empty:
        return tidy

    tidy["produto"] = tidy["produto_cod"] + " - " + tidy["produto_desc"]
    return tidy

# ==========================================================
# Regras de filtro de produto (anti-dupla contagem)
# ==========================================================
def safe_drop_totals(selected_codes: set[str]) -> tuple[set[str], list[str]]:
    warnings = []
    out = set(selected_codes)
    for total in ["18201", "18202"]:
        has_total = total in out
        has_children = any((c != total and c.startswith(total)) for c in out)
        if has_total and has_children:
            out.remove(total)
            warnings.append(f"Removi automaticamente o TOTAL {total} para evitar dupla contagem (subprodutos também selecionados).")
    return out, warnings

def selected_products_label(products_df: pd.DataFrame, selected_codes: set[str], max_items: int = 2) -> str:
    if not selected_codes:
        return "Todos os produtos"
    mp = dict(zip(products_df["produto_cod"], products_df["produto"]))
    names = [mp.get(c, c) for c in sorted(selected_codes)]
    if len(names) <= max_items:
        return " • ".join(names)
    return " • ".join(names[:max_items]) + f"  +{len(names) - max_items} outros"

# ==========================================================
# Série e gráficos
# ==========================================================
def series_for(dff: pd.DataFrame, tipo: str) -> pd.DataFrame:
    x = dff[dff["tipo"] == tipo].copy()
    if x.empty:
        return x
    g = x.groupby("data", as_index=False).agg(
        orcado=("orcado", "sum"),
        realizado=("realizado", lambda s: s.sum(min_count=1)),  # mantém NaN se não existir dado
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
        tickformat="%b\n%Y",
        showgrid=False,
        tickfont=dict(color=BRAND["muted"]),
    )
    fig.update_yaxes(
        gridcolor=BRAND["grid"],
        tickfont=dict(color=BRAND["muted"]),
        separatethousands=True,
    )
    return fig

def last_real_date_with_data(dff: pd.DataFrame, tipo: str, col: str, eps: float = 0.0001):
    x = dff[dff["tipo"] == tipo][["data", col]].copy()
    x = x.dropna()
    x = x[x[col].abs() > eps]
    if x.empty:
        return None
    return pd.to_datetime(x["data"]).max()

def sum_at_date(dff: pd.DataFrame, tipo: str, col: str, dt: pd.Timestamp) -> float:
    x = dff[(dff["tipo"] == tipo) & (dff["data"] == dt)][col]
    if col == "realizado":
        return float(x.sum(min_count=1))
    return float(x.sum(skipna=True))

# ==========================================================
# Representatividade (Top 5 + Outros) - Rendas
def top5_representatividade_rendas(df_time: pd.DataFrame) -> pd.DataFrame:
    """
    Top 5 + Outros para Rendas (por produto), no recorte df_time.
    Usa Realizado se houver dado; senão usa Orçado.
    Exclui TOTAL 18202 para evitar distorção.
    """
    x = df_time[df_time["tipo"] == "Rendas"].copy()
    if x.empty:
        return pd.DataFrame()

    x["produto_cod"] = x["produto_cod"].astype(str)

    # mantém apenas linha 18202*
    x = x[x["produto_cod"].str.startswith("18202")]

    # exclui TOTAL 18202
    x = x[x["produto_cod"] != "18202"]
    if x.empty:
        return pd.DataFrame()

    # escolhe métrica: Realizado se existir dado; senão Orçado
    has_real = x["realizado"].notna().any() and (float(x["realizado"].sum(min_count=1)) > 0)
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
    rest = float(g.iloc[5:]["valor"].sum()) if len(g) > 5 else 0.0

    if rest > 0:
        top5 = pd.concat(
            [top5, pd.DataFrame([{"produto_cod": "OUTROS", "produto": "Outros", "valor": rest}])],
            ignore_index=True,
        )

    total = float(top5["valor"].sum())
    top5["share"] = top5["valor"] / total if total > 0 else 0.0
    top5["metric"] = metric
    return top5


def representatividade_figure(rep: pd.DataFrame) -> go.Figure:
    """
    Gráfico premium: barras horizontais (Top 5 + Outros),
    % + valor abreviado, 'Outros' sempre por último.
    """
    def human_millions(v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        abs_v = abs(v)
        if abs_v >= 1_000_000:
            return f"{v/1_000_000:.1f}M".replace(".", ",")
        if abs_v >= 1_000:
            return f"{v/1_000:.0f}k".replace(".", ",")
        return f"{v:.0f}".replace(".", ",")

    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"

    rep2 = rep.copy()
    rep2["is_outros"] = (rep2["produto"] == "Outros").astype(int)
    rep2 = rep2.sort_values(["is_outros", "valor"], ascending=[True, False]).drop(columns=["is_outros"])

    rep_plot = rep2.iloc[::-1].copy()
    rep_plot["pct"] = (rep_plot["share"] * 100).round(1)
    rep_plot["label"] = rep_plot.apply(
        lambda r: f"{r['pct']:.1f}%  •  {human_millions(float(r['valor']))}", axis=1
    )

    grad = ["#2B59FF", "#3550FF", "#3D47FF", "#453DFF", "#4D33FF", "#5B2BD6"]
    colors = []
    k = 0
    for p in rep_plot["produto"].tolist():
        if p == "Outros":
            colors.append("rgba(242,246,255,0.18)")
        else:
            colors.append(grad[min(k, len(grad) - 1)])
            k += 1

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=rep_plot["produto"],
            x=rep_plot["valor"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=rep_plot["label"],
            textposition="outside",
            textfont=dict(color=BRAND["muted"], size=12),
            customdata=rep_plot["pct"],
            hovertemplate="<b>%{y}</b><br>"
                          f"Rendas ({metric_lbl}): " + "%{x:,.2f}<br>"
                          "Share: %{customdata:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["card"],
        margin=dict(l=14, r=18, t=10, b=14),
        showlegend=False,
        xaxis=dict(
            title="",
            gridcolor="rgba(242,246,255,0.04)",
            zeroline=False,
            tickfont=dict(color=BRAND["muted"]),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=BRAND["ink"]),
        ),
    )

    xmax = float(rep_plot["valor"].max()) if len(rep_plot) else 0
    fig.update_xaxes(range=[0, xmax * 1.18])

    return fig



# ==========================================================
st.markdown('<p class="section-title">Representatividade • Produtos (Rendas)</p>', unsafe_allow_html=True)

rep = top5_representatividade_rendas(df_filtered)


if rep.empty:
    st.info("Sem dados suficientes de Rendas para calcular representatividade neste recorte.")
else:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
    st.markdown(
        f'<div class="pill"><span style="opacity:.8">Base:</span> <b>Rendas ({metric_lbl})</b> <span style="opacity:.6">• Top 5 + Outros</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(representatividade_figure(rep), use_container_width=True)
    st.caption("Top 5 + Outros. TOTAL 18202 é excluído para evitar distorção.")

# ==========================================================
# App
# ==========================================================
inject_css()
px.defaults.template = make_plotly_template()

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
if df.empty:
    st.error("Não consegui interpretar o CSV. Verifique se é o relatório exportado no mesmo padrão.")
    st.stop()

# Período
years = sorted(df["ano"].unique().tolist())
with st.sidebar:
    ano_sel = None
    mes_sel = None
    if periodo in ["Ano", "Mês"]:
        ano_sel = st.selectbox("Ano", years, index=0)
    if periodo == "Mês":
        mes_sel = st.selectbox("Mês", list(range(1, 13)), index=0)

df_time = df.copy()
if periodo == "Ano" and ano_sel is not None:
    df_time = df_time[df_time["ano"] == ano_sel]
if periodo == "Mês" and ano_sel is not None and mes_sel is not None:
    df_time = df_time[(df_time["ano"] == ano_sel) & (df_time["mes"] == mes_sel)]

# Produtos
products = df[["produto_cod", "produto"]].drop_duplicates().sort_values("produto")
default_codes = [c for c in ["18201", "18202"] if c in set(products["produto_cod"])]

with st.sidebar:
    selected = st.multiselect(
        "Produtos (multi)",
        options=products["produto_cod"].tolist(),
        default=default_codes,
        format_func=lambda c: products.loc[products["produto_cod"] == c, "produto"].iloc[0],
    )

selected_set, warn_list = safe_drop_totals(set(selected))
for w in warn_list:
    st.warning(w)

dff = df_time[df_time["produto_cod"].isin(selected_set)] if selected_set else df_time.copy()
prod_label = selected_products_label(products, selected_set)

# ==========================================================
# KPIs
# ==========================================================
st.markdown('<p class="section-title">KPIs</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="small")

def render_kpi(col, title, value, badge_text=None, badge_color=None, sub=None):
    with col:
        st.markdown('<div class="kpi-card-native">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="kpi-value">{value}</p>', unsafe_allow_html=True)

        if badge_text:
            st.markdown(
                f"""
                <span style="
                  display:inline-flex; align-items:center; gap:6px;
                  padding:5px 10px; border-radius:999px;
                  border:1px solid {BRAND['border']};
                  background: rgba(255,255,255,0.06);
                  color:{badge_color}; font-weight:800; font-size:12px;">
                  ● {badge_text}
                </span>
                """,
                unsafe_allow_html=True,
            )

        if sub:
            st.markdown(f'<div class="kpi-sub">{sub}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

render_kpi(
    c1,
    "Saldo • Realizado (último mês com dado)",
    fmt_br(saldo_real_last),
    saldo_farol_txt,
    saldo_farol_color,
    sub=f"Base: <b>{base_txt}</b>",
)

render_kpi(
    c2,
    "Saldo • Gap vs Orçado (mesmo mês)",
    fmt_br(saldo_gap),
    sub="Comparação no mês-base do realizado.",
)

render_kpi(
    c3,
    "Rendas • Orçado (acumulado no recorte)",
    fmt_br(rendas_orc),
)

render_kpi(
    c4,
    "Rendas • Realizado (acumulado no recorte)",
    fmt_br(rendas_real),
    rendas_farol_txt,
    rendas_farol_color,
)

st.caption(f"Base do Saldo (último mês com dado): {base_txt}")

# ==========================================================
# Saldo chart
# ==========================================================
st.markdown('<p class="section-title">Evolução • Saldo da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_saldo = series_for(dff, "Saldo")
if s_saldo.empty:
    st.info("Sem dados de Saldo nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_saldo), use_container_width=True)

# ==========================================================
# Rendas chart
# ==========================================================
st.markdown('<p class="section-title">Evolução • Rendas da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_rendas = series_for(dff, "Rendas")
if s_rendas.empty:
    st.info("Sem dados de Rendas nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_rendas), use_container_width=True)

# ==========================================================

# ==========================================================
# Representatividade (Top 5 Rendas)
# ==========================================================
st.markdown('<p class="section-title">Representatividade • Produtos (Rendas)</p>', unsafe_allow_html=True)

rep = top5_representatividade_rendas(df_time)

if rep.empty:
    st.info("Sem dados suficientes de Rendas para calcular representatividade neste recorte.")
else:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
    st.markdown(
        f'<div class="pill"><span style="opacity:.8">Base:</span> <b>Rendas ({metric_lbl})</b> <span style="opacity:.6">• Top 5 + Outros</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(representatividade_figure(rep), use_container_width=True)
    st.caption("Top 5 + Outros. TOTAL 18202 é excluído para evitar distorção.")
