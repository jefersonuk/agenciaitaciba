import re
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ==========================
# Branding (Banestes)
# ==========================
BRAND = {
    "blue": "#1E0AE8",     # Pantone 2728 C
    "green": "#00AB16",    # Pantone 2423 C
    "bg": "#0B1220",
    "bg2": "#0A1020",
    "card": "rgba(255,255,255,0.06)",
    "card2": "rgba(255,255,255,0.04)",
    "border": "rgba(255,255,255,0.10)",
    "grid": "rgba(255,255,255,0.06)",
    "ink": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.70)",
    "muted2": "rgba(255,255,255,0.55)",
    "danger": "#FF4D6D",
    "ok": "#2EE59D",
}


# ==========================
# Streamlit page config
# ==========================
st.set_page_config(
    page_title="Itacibá • Carteira de Crédito",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================
# CSS (dark + modern cards)
# ==========================
st.markdown(
    f"""
<style>
div[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 700px at 20% 10%, rgba(30,10,232,0.18), transparent 60%),
              radial-gradient(900px 600px at 85% 0%, rgba(0,171,22,0.14), transparent 55%),
              linear-gradient(180deg, {BRAND["bg"]}, {BRAND["bg2"]}) !important;
  color: {BRAND["ink"]} !important;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
}}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
  border-right: 1px solid {BRAND["border"]} !important;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

.header-wrap {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border-radius: 18px;
  padding: 18px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 14px 40px rgba(0,0,0,0.25);
}}

.header-title {{
  font-size: 20px;
  font-weight: 900;
  letter-spacing: -0.2px;
  line-height: 1.1;
}}

.header-sub {{
  font-size: 13px;
  opacity: .75;
  margin-top: 4px;
}}

.legend-pill {{
  border: 1px solid {BRAND["border"]};
  background: rgba(255,255,255,0.05);
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 13px;
  display: inline-flex;
  gap: 12px;
}}

.dot {{
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display:inline-block;
}}

.section-title {{
  font-size: 18px;
  font-weight: 900;
  margin: 18px 0 10px 0;
  letter-spacing: -0.2px;
}}

.kpi-card {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 12px 35px rgba(0,0,0,0.25);
  min-height: 110px;
}}

.kpi-label {{
  font-size: 12px;
  opacity: .78;
  font-weight: 800;
}}

.kpi-value {{
  font-size: 20px;
  font-weight: 900;
  margin: 6px 0 8px 0;
  letter-spacing: -0.3px;
}}

.kpi-sub {{
  font-size: 12px;
  opacity: .68;
  margin-top: 6px;
}}

.kpi-sub b {{
  opacity: .95;
  font-weight: 900;
}}

.badge {{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid {BRAND["border"]};
  background: rgba(255,255,255,0.06);
  font-weight: 900;
  font-size: 12px;
}}

.pill {{
  border: 1px solid {BRAND["border"]};
  background: rgba(255,255,255,0.05);
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 12px;
  display:inline-flex;
  gap: 8px;
  align-items: center;
  margin: 2px 0 10px 0;
}}

.chart-card {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
  border-radius: 18px;
  padding: 12px 12px 6px 12px;
  box-shadow: 0 12px 35px rgba(0,0,0,0.25);
}}

hr {{
  border: none;
  border-top: 1px solid {BRAND["border"]};
  margin: 14px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================
# Helpers
# ==========================
PT_MONTH = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
}

MONTH_NUM_TO_LABEL = {v: k for k, v in PT_MONTH.items()}


def parse_ptbr_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan


def fmt_br(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def fmt_compact(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M".replace(".", ",")
    if av >= 1_000:
        return f"{v/1_000:.0f}k".replace(".", ",")
    return f"{v:.0f}".replace(".", ",")


def month_label(dt_):
    if pd.isna(dt_) or dt_ is None:
        return "—"
    return dt_.strftime("%b/%Y")


def type_from_code(code_str: str) -> str:
    if code_str.startswith("18201"):
        return "Saldo"
    if code_str.startswith("18202"):
        return "Rendas"
    return "Outro"


def make_plotly_layout_base():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=30, b=16),
        font=dict(color=BRAND["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=BRAND["muted"]),
        ),
        yaxis=dict(
            gridcolor=BRAND["grid"],
            zeroline=False,
            tickfont=dict(color=BRAND["muted"]),
        ),
    )


# ==========================
# Parser do CSV matricial (2025 + 2026)
# ==========================
@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(file_bytes)
    raw = pd.read_csv(bio, sep=None, engine="python", encoding="latin1")

    # 2025
    col_code_2025 = raw.columns[0]
    col_desc_2025 = raw.columns[1]

    month_start_cols = []
    for c in raw.columns[2:50]:
        if str(raw.loc[0, c]) in PT_MONTH.keys():
            month_start_cols.append(c)

    blocks_2025 = []
    for start in month_start_cols:
        idx = raw.columns.get_loc(start)
        block_cols = raw.columns[idx:idx + 4]
        mon = str(raw.loc[0, start]).strip()
        blocks_2025.append((mon, block_cols.tolist()))

    data_rows_2025 = raw.iloc[2:64].copy()

    recs = []
    for _, r in data_rows_2025.iterrows():
        code = str(r[col_code_2025]).strip()
        if code.lower() == "nan" or code == "" or code == "None":
            continue
        desc = str(r[col_desc_2025]).strip()
        desc = re.sub(r"\s+", " ", desc).strip()

        for mon, cols in blocks_2025:
            orc = parse_ptbr_number(r[cols[0]])
            rea = parse_ptbr_number(r[cols[1]])
            dt_ = datetime(2025, PT_MONTH[mon], 1)

            recs.append(
                {
                    "data": pd.to_datetime(dt_),
                    "ano": 2025,
                    "mes": PT_MONTH[mon],
                    "produto_cod": code,
                    "produto": desc,
                    "tipo": type_from_code(code),
                    "orcado": orc,
                    "realizado": rea,
                }
            )

    df_2025 = pd.DataFrame(recs)

    # 2026
    code_col_2026 = None
    desc_col_2026 = None
    for c in raw.columns:
        if str(raw.loc[0, c]).strip() == "Cód":
            code_col_2026 = c
            idx = raw.columns.get_loc(c)
            if idx + 1 < len(raw.columns):
                desc_col_2026 = raw.columns[idx + 1]
            break

    if code_col_2026 is not None and desc_col_2026 is not None:
        idx_code = raw.columns.get_loc(code_col_2026)
        month_cols_2026 = []
        for c in raw.columns[idx_code + 3:]:
            lab = str(raw.loc[0, c]).strip()
            if lab in PT_MONTH:
                month_cols_2026.append((lab, c))

        data_rows_2026 = raw.iloc[1:64].copy()

        recs2 = []
        for _, r in data_rows_2026.iterrows():
            code = str(r[code_col_2026]).strip()
            if code.lower() == "nan" or code == "" or code == "None":
                continue
            desc = str(r[desc_col_2026]).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            for mon, c in month_cols_2026:
                orc = parse_ptbr_number(r[c])
                dt_ = datetime(2026, PT_MONTH[mon], 1)
                recs2.append(
                    {
                        "data": pd.to_datetime(dt_),
                        "ano": 2026,
                        "mes": PT_MONTH[mon],
                        "produto_cod": code,
                        "produto": desc,
                        "tipo": type_from_code(code),
                        "orcado": orc,
                        "realizado": np.nan,
                    }
                )

        df_2026 = pd.DataFrame(recs2)
    else:
        df_2026 = pd.DataFrame(columns=df_2025.columns)

    df = pd.concat([df_2025, df_2026], ignore_index=True)

    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)

    df = df[~(df["orcado"].isna() & df["realizado"].isna())].copy()
    df.sort_values(["data", "produto_cod"], inplace=True)

    return df


# ==========================
# Charts
# ==========================
def make_evolucao_figure(df_view: pd.DataFrame, tipo: str, title: str, prod_label: str) -> go.Figure:
    d = df_view[df_view["tipo"] == tipo].copy()
    if d.empty:
        fig = go.Figure()
        fig.update_layout(**make_plotly_layout_base(), height=360)
        fig.add_annotation(
            text="Sem dados para este recorte/produtos.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=BRAND["muted"], size=14),
        )
        return fig

    agg = d.groupby("data", as_index=False).agg(
        orcado=("orcado", lambda s: float(np.nansum(s.values))),
        realizado=("realizado", lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan),
    )

    mask_data = agg["realizado"].notna() & (agg["realizado"].abs() > 0)
    if mask_data.any():
        last_dt = agg.loc[mask_data, "data"].max()
        agg.loc[agg["data"] > last_dt, "realizado"] = np.nan
        agg.loc[(agg["data"] > last_dt) & (agg["realizado"] == 0), "realizado"] = np.nan
    else:
        agg["realizado"] = np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=agg["data"],
            y=agg["orcado"],
            name=f"Orçado • {prod_label}",
            marker=dict(color=BRAND["blue"], opacity=0.75),
            hovertemplate="<b>%{x|%b/%Y}</b><br>Orçado: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=agg["data"],
            y=agg["realizado"],
            name=f"Realizado • {prod_label}",
            mode="lines+markers",
            line=dict(color=BRAND["green"], width=3),
            marker=dict(size=7),
            connectgaps=False,
            hovertemplate="<b>%{x|%b/%Y}</b><br>Realizado: %{y:,.2f}<extra></extra>",
        )
    )

    layout = make_plotly_layout_base()
    layout.update(
        height=420,
        title=dict(text=title, x=0, font=dict(size=16, color=BRAND["ink"])),
        barmode="overlay",
        yaxis=dict(**layout["yaxis"], title="Valor"),
        xaxis=dict(**layout["xaxis"], title=""),
    )
    fig.update_layout(**layout)

    return fig


def top5_representatividade_rendas(df_base: pd.DataFrame) -> pd.DataFrame:
    x = df_base[df_base["tipo"] == "Rendas"].copy()
    if x.empty:
        return pd.DataFrame()

    x["produto_cod"] = x["produto_cod"].astype(str)
    x = x[x["produto_cod"].str.startswith("18202")]
    x = x[x["produto_cod"] != "18202"]
    if x.empty:
        return pd.DataFrame()

    has_real = x["realizado"].notna().any() and float(np.nansum(x["realizado"].values)) > 0
    metric = "realizado" if has_real else "orcado"

    g = (
        x.groupby(["produto_cod", "produto"], as_index=False)
        .agg(valor=(metric, lambda s: float(np.nansum(s.values))))
    )
    g = g[g["valor"] > 0].sort_values("valor", ascending=False)

    if g.empty:
        return pd.DataFrame()

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

    rep2 = rep.copy()
    rep2["is_outros"] = (rep2["produto"] == "Outros").astype(int)
    rep2 = rep2.sort_values(["is_outros", "valor"], ascending=[True, False]).drop(columns=["is_outros"])

    rep_plot = rep2.iloc[::-1].copy()
    rep_plot["pct"] = (rep_plot["share"] * 100).round(1)
    rep_plot["label"] = rep_plot.apply(
        lambda r: f"{r['pct']:.1f}%  •  {fmt_compact(float(r['valor']))}", axis=1
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

    layout = make_plotly_layout_base()
    layout.update(
        height=440,
        showlegend=False,
        xaxis=dict(
            title="",
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            tickfont=dict(color=BRAND["muted"]),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=BRAND["ink"]),
        ),
    )
    fig.update_layout(**layout)

    xmax = float(rep_plot["valor"].max()) if len(rep_plot) else 0
    fig.update_xaxes(range=[0, xmax * 1.18 if xmax > 0 else 1])

    return fig


# ==========================
# Sidebar: upload + filtros
# ==========================
st.sidebar.markdown("## Dados")
uploaded = st.sidebar.file_uploader("Anexe o relatório (CSV exportado)", type=["csv"])

if uploaded is None:
    st.sidebar.info("Envie o CSV do relatório para carregar os dados.")
    st.stop()

df = load_report(uploaded.getvalue())

produtos = (
    df[["produto_cod", "produto"]]
    .drop_duplicates()
    .sort_values(["produto_cod"])
)
produtos["label"] = produtos["produto_cod"] + " - " + produtos["produto"]
prod_labels = produtos["label"].tolist()

default_sel = []
for cod in ["18201", "18202"]:
    match = produtos[produtos["produto_cod"] == cod]
    if not match.empty:
        default_sel.append(match["label"].iloc[0])

st.sidebar.markdown("## Filtros")
periodo = st.sidebar.radio("Período", ["Total", "Ano", "Mês"], horizontal=True)

anos = sorted(df["ano"].unique().tolist())
ano_sel = None
mes_sel = None

if periodo in ["Ano", "Mês"]:
    ano_sel = st.sidebar.selectbox("Ano", anos, index=len(anos) - 1)

if periodo == "Mês":
    meses_no_ano = sorted(df[df["ano"] == ano_sel]["mes"].unique().tolist())
    mes_sel = st.sidebar.selectbox(
        "Mês",
        meses_no_ano,
        format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m)),
    )

prod_sel = st.sidebar.multiselect("Produto (multi)", prod_labels, default=default_sel)


# ==========================
# Aplica filtros
# ==========================
# df_period: somente período (sem filtro de produto)
df_period = df.copy()
if periodo == "Ano":
    df_period = df_period[df_period["ano"] == ano_sel]
elif periodo == "Mês":
    df_period = df_period[(df_period["ano"] == ano_sel) & (df_period["mes"] == mes_sel)]

# df_view: período + produto
df_view = df_period.copy()
if prod_sel:
    cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
    df_view = df_view[df_view["produto_cod"].isin(cod_sel)]

if not prod_sel:
    prod_label = "Todos"
else:
    if len(prod_sel) == 1:
        prod_label = prod_sel[0].split(" - ", 1)[1][:28]
    else:
        prod_label = f"Seleção ({len(prod_sel)})"


# ==========================
# Header
# ==========================
st.markdown(
    f"""
<div class="header-wrap">
  <div>
    <div class="header-title">Itacibá • Carteira de Crédito</div>
    <div class="header-sub">Orçado x Realizado (2025) • Orçado (2026) • filtros por período e produto</div>
  </div>
  <div class="legend-pill">
    <span><span class="dot" style="background:{BRAND["blue"]}"></span> Orçado</span>
    <span><span class="dot" style="background:{BRAND["green"]}"></span> Realizado</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================
# KPIs
# ==========================
st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

saldo = df_view[df_view["tipo"] == "Saldo"].copy()
rendas = df_view[df_view["tipo"] == "Rendas"].copy()


def last_real_month(d: pd.DataFrame):
    if d.empty:
        return None
    g = d.groupby("data", as_index=False).agg(
        real=("realizado", lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan)
    )
    mask = g["real"].notna() & (g["real"].abs() > 0)
    if not mask.any():
        return None
    return g.loc[mask, "data"].max()


def value_at_month(d: pd.DataFrame, dt_: pd.Timestamp, col: str):
    if d.empty or dt_ is None:
        return np.nan
    g = d.groupby("data", as_index=False).agg(
        val=(col, lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan)
    )
    row = g[g["data"] == dt_]
    return float(row["val"].iloc[0]) if not row.empty else np.nan


saldo_last_dt = last_real_month(saldo)
saldo_real_last = value_at_month(saldo, saldo_last_dt, "realizado")
saldo_orc_last = value_at_month(saldo, saldo_last_dt, "orcado")
saldo_gap = saldo_real_last - saldo_orc_last

saldo_ok = (
    (not np.isnan(saldo_real_last))
    and (not np.isnan(saldo_orc_last))
    and (saldo_real_last >= saldo_orc_last)
)
saldo_farol_txt = "Cumprido" if saldo_ok else "Não cumprido"
saldo_farol_color = BRAND["ok"] if saldo_ok else BRAND["danger"]
base_txt = month_label(saldo_last_dt)


def acumulado(d: pd.DataFrame, col: str, upto_dt: pd.Timestamp = None):
    if d.empty:
        return 0.0
    x = d.copy()
    if upto_dt is not None:
        x = x[x["data"] <= upto_dt]
    if x.empty:
        return 0.0
    g = x.groupby("data", as_index=False).agg(
        val=(col, lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan)
    )
    return float(np.nansum(g["val"].values))


rendas_last_dt = last_real_month(rendas)
rendas_orc = acumulado(rendas, "orcado", upto_dt=None)
rendas_real = acumulado(rendas, "realizado", upto_dt=None)

rendas_orc_base = acumulado(rendas, "orcado", upto_dt=rendas_last_dt) if rendas_last_dt is not None else np.nan
rendas_real_base = acumulado(rendas, "realizado", upto_dt=rendas_last_dt) if rendas_last_dt is not None else np.nan

rendas_ok = False
if rendas_last_dt is not None and (not np.isnan(rendas_orc_base)) and (not np.isnan(rendas_real_base)):
    rendas_ok = rendas_real_base >= rendas_orc_base

rendas_farol_txt = "Cumprido" if rendas_ok else "Não cumprido"
rendas_farol_color = BRAND["ok"] if rendas_ok else BRAND["danger"]
rendas_base_txt = month_label(rendas_last_dt) if rendas_last_dt is not None else "—"


def render_kpi(col, title, value, badge_text=None, badge_color=None, sub_label=None, sub_value=None):
    badge_html = ""
    if badge_text:
        badge_html = f"""
        <div class="badge" style="color:{badge_color}">
          <span class="dot" style="background:{badge_color}"></span>
          <span>{badge_text}</span>
        </div>
        """

    sub_html = ""
    if sub_label is not None or sub_value is not None:
        # sem HTML vindo de variável “solta”; negrito vem do <b> controlado aqui
        _label = (sub_label or "").strip()
        _value = (sub_value or "").strip()
        if _label and _value:
            sub_html = f'<div class="kpi-sub">{_label} <b>{_value}</b></div>'
        elif _label:
            sub_html = f'<div class="kpi-sub">{_label}</div>'
        else:
            sub_html = f'<div class="kpi-sub"><b>{_value}</b></div>'

    html = f"""
    <div class="kpi-card">
      <div class="kpi-label">{title}</div>
      <div class="kpi-value">{value}</div>
      {badge_html}
      {sub_html}
    </div>
    """
    with col:
        st.markdown(html, unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4, gap="small")

render_kpi(
    c1,
    "Saldo • Realizado (último mês com dado)",
    fmt_br(saldo_real_last),
    badge_text=saldo_farol_txt,
    badge_color=saldo_farol_color,
    sub_label="Base:",
    sub_value=base_txt,
)

render_kpi(
    c2,
    "Saldo • Gap vs Orçado (mesmo mês)",
    fmt_br(saldo_gap),
    sub_label="Comparação no mês-base do realizado.",
    sub_value="",
)

render_kpi(
    c3,
    "Rendas • Orçado (acumulado no recorte)",
    fmt_br(rendas_orc),
    sub_label="Base do farol:" if rendas_last_dt is not None else "Sem base de realizado ainda.",
    sub_value=rendas_base_txt if rendas_last_dt is not None else "",
)

render_kpi(
    c4,
    "Rendas • Realizado (acumulado no recorte)",
    fmt_br(rendas_real),
    badge_text=rendas_farol_txt if rendas_last_dt is not None else None,
    badge_color=rendas_farol_color if rendas_last_dt is not None else None,
)

st.markdown("<hr/>", unsafe_allow_html=True)


# ==========================
# Charts
# ==========================
if prod_sel:
    shown = " • ".join([p.split(" - ", 1)[0] for p in prod_sel[:4]])
    if len(prod_sel) > 4:
        shown += f" • +{len(prod_sel) - 4}"
else:
    shown = "Todos"

st.markdown(
    f'<div class="pill"><span style="opacity:.75">Produtos:</span> <b>{shown}</b></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Evolução • Saldo da Carteira</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
fig_saldo = make_evolucao_figure(df_view, "Saldo", "Saldo (Orçado x Realizado)", prod_label)
st.plotly_chart(fig_saldo, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">Evolução • Rendas da Carteira</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
fig_rendas = make_evolucao_figure(df_view, "Rendas", "Rendas (Orçado x Realizado)", prod_label)
st.plotly_chart(fig_rendas, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ==========================
# Representatividade (Top 5 + Outros) - Rendas
# ==========================
st.markdown('<div class="section-title">Representatividade • Produtos (Rendas)</div>', unsafe_allow_html=True)

# representatividade ignora filtro de produto (usa df_period)
rep = top5_representatividade_rendas(df_period)

if rep.empty:
    st.info("Sem dados suficientes de Rendas para calcular representatividade neste recorte.")
else:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
    st.markdown(
        f'<div class="pill"><span style="opacity:.8">Base:</span> <b>Rendas ({metric_lbl})</b> <span style="opacity:.6">• Top 5 + Outros</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(representatividade_figure(rep), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Top 5 + Outros. TOTAL 18202 é excluído para evitar distorção.")
