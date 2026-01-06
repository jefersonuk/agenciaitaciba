# app.py
# Banestes • Itacibá — Carteira de Crédito
# Streamlit + Plotly (dark premium)
#
# Inclui:
# - Tema dark elegante e legível (alto contraste)
# - 2 gráficos empilhados: Saldo e Rendas (Orçado=barras | Realizado=linha)
# - Realizado não “cai pra zero” quando não há dado (NaN para e a linha para)
# - Filtros: Período (Total/Ano/Mês) + Produtos (multi) com anti-dupla contagem
# - KPIs executivos:
#   * Saldo: snapshot do último mês com realizado “concreto” (realizado != 0) + gap vs orçado no mesmo mês
#   * Rendas: acumulado no recorte (orçado e realizado)
# - Faróis (verde/vermelho) para cumprimento (Saldo e Rendas)
# - Representatividade (Top 5) por produto considerando apenas Rendas (sem pizza)
#   (usa subprodutos 18202* e exclui o TOTAL 18202)

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
# UI / CSS (premium)
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

          .kpi-grid {{
            display:grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 12px;
          }}
          @media (max-width: 1100px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
          }}
          @media (max-width: 650px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
          }}

          .kpi-card {{
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

          .section-title {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 18px 0 10px 0;
            color: rgba(242,246,255,0.96);
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
# Plotly template (compatível + dark)
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
# Helpers (formatação, farol, parsing BR numbers)
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
    """Farol: verde se real >= plan; vermelho se real < plan; neutro se sem dados."""
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


@st.cache_data(show_spinner=False)
def parse_relatorio(uploaded_file) -> pd.DataFrame:
    """
    Parser fiel ao relatório no formato wide (como exportado):
      - 2025: blocos [Orçado, Realizado] por mês
      - 2026: apenas Orçado por mês
    Retorna formato tidy:
      tipo, data, ano, mes, produto_cod, produto_desc, produto, orcado, realizado
    """
    raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")
    cols = list(raw.columns)

    # 2025 (colunas típicas do export atual)
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

    # 2026 (colunas típicas do export atual)
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

    # 2025 data rows (tipicamente começam em i=2)
    for i in range(2, len(raw)):
        if code_2025 not in raw.columns:
            break
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

    # 2026 data rows (tipicamente começam em i=1)
    for i in range(1, len(raw)):
        if code_2026 not in raw.columns:
            break
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
                "realizado": np.nan,  # 2026 sem realizado
            })

    tidy = pd.DataFrame.from_records(records)
    if tidy.empty:
        return tidy

    tidy["produto"] = tidy["produto_cod"] + " - " + tidy["produto_desc"]
    return tidy


def safe_drop_totals(selected_codes: set[str]) -> tuple[set[str], list[str]]:
    """
    Evita dupla contagem:
    se selecionar TOTAL (18201/18202) e também subprodutos, removemos o TOTAL.
    """
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


def last_real_date_with_data(dff: pd.DataFrame, tipo: str, col: str, eps: float = 0.0001):
    """
    Última data com dado "concreto" (não nulo e != 0).
    Isso evita puxar Dez/2025 = 0,00 quando o mês ainda não fechou.
    """
    x = dff[(dff["tipo"] == tipo)][["data", col]].copy()
    x = x.dropna()
    x = x[x[col].abs() > eps]
    if x.empty:
        return None
    return pd.to_datetime(x["data"]).max()


# ==========================================================
# Série + gráficos
# ==========================================================
def series_for(dff: pd.DataFrame, tipo: str) -> pd.DataFrame:
    x = dff[dff["tipo"] == tipo].copy()
    if x.empty:
        return x
    g = x.groupby("data", as_index=False).agg(
        orcado=("orcado", "sum"),
        realizado=("realizado", lambda s: s.sum(min_count=1)),  # mantém NaN se não houver dados
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
        title_text="",
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


def top5_representatividade_rendas(df_time: pd.DataFrame) -> pd.DataFrame:
    """
    Representatividade por produto usando apenas Rendas.
    - Usa subprodutos 18202* e exclui TOTAL 18202 (para não distorcer)
    - Se existir realizado no recorte, usa realizado; senão, orçado
    - Retorna Top 5 + "Outros"
    """
    x = df_time[df_time["tipo"] == "Rendas"].copy()
    if x.empty:
        return pd.DataFrame()

    # Excluir TOTAL 18202 e manter somente subprodutos (18202XXXX...)
    x = x[x["produto_cod"].astype(str).str.startswith("18202")]
    x = x[x["produto_cod"].astype(str) != "18202"]

    if x.empty:
        return pd.DataFrame()

    has_real = x["realizado"].notna().any() and (not np.isnan(float(x["realizado"].sum(min_count=1))))
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
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"

    # Ordena desc e prepara plot (topo = maior)
    rep_plot = rep.sort_values("valor", ascending=True).copy()

    # Gradiente moderno (azul → roxo), com "Outros" mais neutro
    grad = ["#2B59FF", "#3550FF", "#3D47FF", "#453DFF", "#4D33FF", "#5B2BD6"]
    colors = []
    for p in rep_plot["produto"].tolist():
        if p == "Outros":
            colors.append("rgba(242,246,255,0.22)")
        else:
            colors.append(grad[min(len(colors), len(grad)-1)])

    # Texto (% share) no fim da barra
    perc = (rep_plot["share"] * 100).round(1).astype(str) + "%"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=rep_plot["produto"],
            x=rep_plot["valor"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=perc,
            textposition="outside",
            textfont=dict(color=BRAND["muted"], size=12),
            hovertemplate="<b>%{y}</b><br>"
                          f"Rendas ({metric_lbl}): " + "%{x:,.2f}<br>"
                          "Share: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title_text="",
        height=380,
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["card"],
        margin=dict(l=14, r=14, t=12, b=14),
        xaxis=dict(
            gridcolor="rgba(242,246,255,0.05)",
            tickfont=dict(color=BRAND["muted"]),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color=BRAND["ink"]),
        ),
        showlegend=False,
    )

    # Deixa espaço para os % fora da barra
    fig.update_xaxes(range=[0, rep_plot["valor"].max() * 1.12])

    return fig
 fig


# ==========================================================
# App start
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
    st.error("Não consegui interpretar o CSV. Verifique se é o relatório exportado no mesmo padrão de colunas.")
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

# Aplicar filtro de tempo primeiro (para ser base da representatividade)
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
        "Produtos (selecione 1 ou mais)",
        options=products["produto_cod"].tolist(),
        default=default_codes,
        format_func=lambda c: products.loc[products["produto_cod"] == c, "produto"].iloc[0],
    )

selected_set, warn_list = safe_drop_totals(set(selected))
for w in warn_list:
    st.warning(w)

# Aplicar filtro de produto sobre o recorte de tempo
dff = df_time[df_time["produto_cod"].isin(selected_set)] if selected_set else df_time.copy()
prod_label = selected_products_label(products, selected_set)

# ==========================================================
# KPIs (Saldo snapshot + Rendas acumulado) + faróis
# ==========================================================
def sum_at_date(dff_: pd.DataFrame, tipo: str, col: str, dt: pd.Timestamp) -> float:
    x = dff_[(dff_["tipo"] == tipo) & (dff_["data"] == dt)][col]
    if col == "realizado":
        return float(x.sum(min_count=1))
    return float(x.sum(skipna=True))


# Saldo: último mês com realizado “concreto” (não nulo e != 0)
saldo_last_real_dt = last_real_date_with_data(dff, "Saldo", "realizado", eps=0.0001)
saldo_real_last = sum_at_date(dff, "Saldo", "realizado", saldo_last_real_dt) if saldo_last_real_dt is not None else np.nan
saldo_orc_same = sum_at_date(dff, "Saldo", "orcado", saldo_last_real_dt) if saldo_last_real_dt is not None else np.nan
saldo_gap = (saldo_real_last - saldo_orc_same) if (saldo_last_real_dt is not None and not np.isnan(saldo_orc_same)) else np.nan

# Rendas: acumulado no recorte
r = dff[dff["tipo"] == "Rendas"]
rendas_orc = float(r["orcado"].sum(skipna=True))
rendas_real = float(r["realizado"].sum(min_count=1))

# Faróis
saldo_farol_txt, saldo_farol_color = kpi_status(saldo_real_last, saldo_orc_same)
rendas_farol_txt, rendas_farol_color = kpi_status(rendas_real, rendas_orc)

st.markdown('<p class="section-title">KPIs</p>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Realizado (último mês com dado)</div>
        <p class="kpi-value">{fmt_br(saldo_real_last)}</p>
        {badge_html(saldo_farol_txt, saldo_farol_color)}
        

      <div class="kpi-card">
        <div class="kpi-label">Saldo • Gap vs Orçado (mesmo mês)</div>
        <p class="kpi-value">{fmt_br(saldo_gap)}</p>
        <div class="kpi-sub">Comparação no mês-base do realizado.</div>
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
    unsafe_allow_html=True,
)

base_txt = saldo_last_real_dt.strftime("%b/%Y") if saldo_last_real_dt is not None else "—"
st.caption(f"Base do Saldo (último mês com dado): {base_txt}")

# ==========================================================
# Gráfico 1: Saldo
# ==========================================================
st.markdown('<p class="section-title">Evolução • Saldo da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_saldo = series_for(dff, "Saldo")
if s_saldo.empty:
    st.info("Sem dados para Saldo nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_saldo), use_container_width=True)

# ==========================================================
# Gráfico 2: Rendas
# ==========================================================
st.markdown('<p class="section-title">Evolução • Rendas da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Produtos:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

s_rendas = series_for(dff, "Rendas")
if s_rendas.empty:
    st.info("Sem dados para Rendas nesse recorte.")
else:
    st.plotly_chart(bar_line_figure(s_rendas), use_container_width=True)

# ==========================================================
# Representatividade: Top 5 produtos (Rendas)
# ==========================================================
metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
st.markdown(
    f'<div class="pill"><span style="opacity:.8">Base:</span> <b>Rendas ({metric_lbl})</b></div>',
    unsafe_allow_html=True
)


st.markdown('<p class="section-title">Representatividade • Produtos (Rendas)</p>', unsafe_allow_html=True)

# Aqui usamos a base de tempo (df_time), e não o filtro de produto,
# para mostrar os 5 principais produtos que compõem o resultado de crédito no recorte.
rep = top5_representatividade_rendas(df_time)

if rep.empty:
    st.info("Sem dados suficientes de rendas para calcular representatividade neste recorte.")
else:
    st.plotly_chart(representatividade_figure(rep), use_container_width=True)
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
    st.caption(f"Base: Rendas ({metric_lbl}) no recorte de tempo. Top 5 + Outros. (TOTAL 18202 é excluído).")
