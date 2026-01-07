# app.py
import re
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==========================================================
# Config
# ==========================================================
st.set_page_config(page_title="Itacibá • Carteira de Crédito", layout="wide")

BRAND = {
    "bg": "#070A16",
    "bg2": "#050816",
    "border": "rgba(242,246,255,0.10)",
    "ink": "rgba(242,246,255,0.95)",
    "muted": "rgba(242,246,255,0.70)",
    "muted2": "rgba(242,246,255,0.55)",
    "blue": "#2B59FF",
    "green": "#00AB16",
    "danger": "#FF3B6B",
}

PT_MONTH = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}
PT_MONTH_INV = {v: k for k, v in PT_MONTH.items()}


# ==========================================================
# CSS (tema dark + sidebar mais estilizada)
# ==========================================================
def inject_css() -> None:
    st.markdown(
        f"""
<style>
/* Fundo do app */
div[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 700px at 20% 10%, rgba(30,10,232,0.18), transparent 60%),
    radial-gradient(900px 600px at 85% 0%, rgba(0,171,22,0.14), transparent 55%),
    linear-gradient(180deg, {BRAND["bg"]}, {BRAND["bg2"]}) !important;
  color: {BRAND["ink"]} !important;
}}
header[data-testid="stHeader"] {{
  background: transparent !important;
}}
.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)) !important;
  border-right: 1px solid {BRAND["border"]} !important;
}}
section[data-testid="stSidebar"] > div {{
  padding-top: 1.2rem;
}}

/* Blocos da sidebar */
.sb-card {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.20);
  margin-bottom: 12px;
}}
.sb-title {{
  font-weight: 900;
  letter-spacing: -0.2px;
  margin-bottom: 6px;
  opacity: .95;
}}
.sb-sub {{
  font-size: 12px;
  opacity: .70;
  margin-top: -2px;
  margin-bottom: 10px;
}}

/* Inputs (BaseWeb) */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {{
  background: rgba(0,0,0,0.18) !important;
  border: 1px solid {BRAND["border"]} !important;
  color: {BRAND["ink"]} !important;
  border-radius: 12px !important;
}}
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] span {{
  color: {BRAND["muted"]} !important;
}}

/* File uploader (caixa) */
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {{
  background: rgba(0,0,0,0.16) !important;
  border: 1px dashed rgba(242,246,255,0.22) !important;
  border-radius: 14px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * {{
  color: {BRAND["muted"]} !important;
}}

/* Radio como pills (melhor contraste) */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
  background: rgba(0,0,0,0.14) !important;
  border: 1px solid {BRAND["border"]} !important;
  border-radius: 999px !important;
  padding: 6px 10px !important;
  margin-right: 8px !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
  border-color: rgba(242,246,255,0.22) !important;
}}

/* MultiSelect tags (chips) */
section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {{
  background: rgba(43, 89, 255, 0.18) !important;
  border: 1px solid rgba(43, 89, 255, 0.35) !important;
  color: {BRAND["ink"]} !important;
}}
section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] svg {{
  color: {BRAND["ink"]} !important;
}}

/* Header */
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
  width: 10px; height: 10px;
  border-radius: 999px;
  display:inline-block;
}}

/* Cards e seções */
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
  min-height: 120px;
}}
.kpi-label {{
  font-size: 12px;
  opacity: .78;
  font-weight: 800;
}}
.kpi-value {{
  font-size: 20px;
  font-weight: 1000;
  margin: 6px 0 8px 0;
  letter-spacing: -0.3px;
}}
.kpi-sub {{
  font-size: 12px;
  opacity: .68;
  margin-top: 6px;
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


# ==========================================================
# Helpers
# ==========================================================
def fmt_br(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def parse_ptbr_number(x) -> float:
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "-"):
        return np.nan
    s = s.replace("\xa0", " ").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_month_label(x: str) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("nan", "none", ""):
        return None
    s = s.split("/")[0].strip()
    s = re.sub(r"[^a-zç]", "", s)
    s = s[:3]
    return s if s in PT_MONTH else None


def type_from_code(code: str) -> str:
    c = str(code)
    if c.startswith("18201"):
        return "saldo"
    if c.startswith("18202"):
        return "rendas"
    return "outros"


def apply_dark_plotly(fig: go.Figure) -> go.Figure:
    # IMPORTANTE: sem recursão aqui
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=BRAND["ink"]),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(
            font=dict(color=BRAND["ink"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,12,24,0.95)",
            bordercolor="rgba(242,246,255,0.12)",
            font=dict(color=BRAND["ink"]),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(242,246,255,0.06)",
        zeroline=False,
        tickfont=dict(color=BRAND["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(242,246,255,0.06)",
        zeroline=False,
        tickfont=dict(color=BRAND["muted"]),
    )
    return fig


# ==========================================================
# Parser (modelo do seu CSV: meses em blocos de 4 colunas)
# ==========================================================
def find_year_row(raw: pd.DataFrame, year: int) -> int | None:
    y = str(year)
    max_scan = min(len(raw), 3000)
    for i in range(max_scan):
        row = raw.iloc[i].astype(str)
        if row.str.contains(y, case=False, na=False).any():
            return i
    return None


def is_likely_header_row(val: str) -> bool:
    s = str(val).strip().lower()
    if s in ("nan", "none", ""):
        return True
    # muitas vezes vem "cód" / "descrição" / "produto"
    if any(k in s for k in ["cód", "cod", "descr", "produto", "carteira", "itaciba"]):
        return True
    # se não tem dígito nenhum, costuma ser header
    if not any(ch.isdigit() for ch in s):
        return True
    return False


def detect_month_groups_from_header_rows(dfmat: pd.DataFrame) -> dict[str, list[str]]:
    """
    Agrupa colunas por mês usando a primeira linha do bloco (header impresso)
    Ex: colunas Unnamed:2..Unnamed:49 e dfmat.iloc[0, col] tem "Jan/2025", etc.
    """
    cols = list(dfmat.columns)
    groups: dict[str, list[str]] = {}

    if dfmat.empty or len(cols) < 3:
        return groups

    header0 = dfmat.iloc[0].astype(str).tolist() if len(dfmat) > 0 else []
    for j in range(2, len(cols)):
        mon = normalize_month_label(header0[j]) if j < len(header0) else None
        if not mon:
            continue
        groups.setdefault(mon, []).append(cols[j])

    return groups


def pick_orcado_realizado_cols(
    dfmat: pd.DataFrame,
    month_cols: list[str],
    has_realizado: bool
) -> tuple[str | None, str | None]:
    """
    Dentro do mês (ex: 4 colunas), tenta achar qual é Orçado e qual é Realizado
    usando a segunda linha (header impresso de subcolunas).
    """
    if not month_cols:
        return None, None

    if len(dfmat) < 2:
        # sem subheader, assume primeira = orçado e segunda = realizado (se houver)
        col_orc = month_cols[0]
        col_real = month_cols[1] if (has_realizado and len(month_cols) > 1) else None
        return col_orc, col_real

    sub = dfmat.iloc[1].astype(str).to_dict()

    def score(col: str) -> str:
        s = str(sub.get(col, "")).lower()
        return s

    # orçado
    col_orc = None
    for c in month_cols:
        s = score(c)
        if "orc" in s or "orç" in s:
            col_orc = c
            break
    if col_orc is None:
        col_orc = month_cols[0]

    # realizado (se existir)
    col_real = None
    if has_realizado:
        for c in month_cols:
            s = score(c)
            if "real" in s:
                col_real = c
                break
        if col_real is None and len(month_cols) > 1:
            col_real = month_cols[1]

    return col_orc, col_real


def parse_block(dfmat: pd.DataFrame, year: int, has_realizado: bool) -> pd.DataFrame:
    if dfmat.empty:
        return pd.DataFrame(
            columns=["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"]
        )

    code_col = dfmat.columns[0]
    desc_col = dfmat.columns[1]

    groups = detect_month_groups_from_header_rows(dfmat)

    # se não conseguiu detectar por header impresso, tenta fallback simples (não recomendado)
    if not groups:
        return pd.DataFrame(
            columns=["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"]
        )

    # decide quantas linhas são header: normalmente 2 (mês + subheader)
    header_rows = 2
    # se a linha 0 já parece dado (código numérico), reduz header
    if len(dfmat) > 0 and not is_likely_header_row(dfmat.iloc[0][code_col]):
        header_rows = 0
    elif len(dfmat) > 1 and not is_likely_header_row(dfmat.iloc[1][code_col]):
        header_rows = 1

    data_rows = dfmat.iloc[header_rows:].copy()

    recs: list[dict] = []

    for _, r in data_rows.iterrows():
        code = str(r.get(code_col, "")).strip()
        if code.lower() in ("nan", "none", ""):
            continue

        desc = str(r.get(desc_col, "")).strip()
        desc = re.sub(r"\s+", " ", desc).strip()

        for mon, cols_in_month in groups.items():
            col_orc, col_real = pick_orcado_realizado_cols(dfmat, cols_in_month, has_realizado)

            if col_orc is None:
                continue

            orc = parse_ptbr_number(r.get(col_orc))
            real = parse_ptbr_number(r.get(col_real)) if (has_realizado and col_real is not None) else np.nan

            # remove linha completamente vazia
            if (isinstance(orc, float) and np.isnan(orc)) and (isinstance(real, float) and np.isnan(real)):
                continue

            dt_ = datetime(year, PT_MONTH[mon], 1)

            recs.append(
                {
                    "data": pd.to_datetime(dt_),
                    "ano": year,
                    "mes": PT_MONTH[mon],
                    "produto_cod": code,
                    "produto": desc,
                    "tipo": type_from_code(code),
                    "orcado": orc,
                    "realizado": real,
                }
            )

    return pd.DataFrame(recs)


@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="latin1")

    # tenta achar onde começa 2025/2026 no arquivo
    i25 = find_year_row(raw, 2025)
    i26 = find_year_row(raw, 2026)

    if i25 is None:
        i25 = 0

    if i26 is None:
        # não achou 2026 como bloco separado; tenta mesmo assim parsear 2025 e 2026 do mesmo bloco:
        block = raw.iloc[i25:].copy()

        df_2025 = parse_block(block, 2025, has_realizado=True)
        df_2026 = parse_block(block, 2026, has_realizado=False)
        df = pd.concat([df_2025, df_2026], ignore_index=True)
    else:
        # split por blocos
        if i26 < i25:
            i25 = 0

        block25 = raw.iloc[i25:i26].copy()
        block26 = raw.iloc[i26:].copy()

        df_2025 = parse_block(block25, 2025, has_realizado=True)
        df_2026 = parse_block(block26, 2026, has_realizado=False)
        df = pd.concat([df_2025, df_2026], ignore_index=True)

    cols_final = ["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"]
    for c in cols_final:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols_final].copy()

    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    df = df.dropna(subset=["data", "ano", "mes"]).copy()
    df = df.sort_values(["data", "produto_cod"]).reset_index(drop=True)

    return df


# ==========================================================
# Timeseries + Figuras
# ==========================================================
def build_timeseries(df_any: pd.DataFrame, tipo: str) -> pd.DataFrame:
    x = df_any[df_any["tipo"] == tipo].copy()
    if x.empty:
        return pd.DataFrame(columns=["data", "orcado", "realizado"])

    g = (
        x.groupby("data", as_index=False)[["orcado", "realizado"]]
        .sum(numeric_only=True)
        .sort_values("data")
    )
    return g


def fig_orcado_bar_real_line(ts: pd.DataFrame, title: str, legend_suffix: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=ts["data"],
            y=ts["orcado"],
            name=f"Orçado • {legend_suffix}",
            marker=dict(color=BRAND["blue"]),
            hovertemplate="<b>%{x|%b/%Y}</b><br>Orçado: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ts["data"],
            y=ts["realizado"],
            name=f"Realizado • {legend_suffix}",
            mode="lines+markers",
            line=dict(color=BRAND["green"], width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x|%b/%Y}</b><br>Realizado: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color=BRAND["ink"])),
        barmode="group",
        xaxis=dict(title=""),
        yaxis=dict(title="Valor"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_dark_plotly(fig)


def top5_representatividade_rendas(df_period: pd.DataFrame) -> pd.DataFrame:
    x = df_period[df_period["tipo"] == "rendas"].copy()
    if x.empty:
        return pd.DataFrame()

    x["produto_cod"] = x["produto_cod"].astype(str)
    x = x[x["produto_cod"].str.startswith("18202")]
    x = x[x["produto_cod"] != "18202"]  # exclui total

    if x.empty:
        return pd.DataFrame()

    has_real = x["realizado"].notna().any()
    metric = "realizado" if has_real else "orcado"

    g = (
        x.groupby(["produto_cod", "produto"], as_index=False)[metric]
        .sum(numeric_only=True)
        .rename(columns={metric: "valor"})
        .sort_values("valor", ascending=False)
    )

    total = float(g["valor"].sum()) if len(g) else 0.0
    if total <= 0:
        return pd.DataFrame()

    g["share"] = g["valor"] / total
    g["metric"] = metric

    top = g.head(5).copy()
    rest = g.iloc[5:].copy()

    if len(rest):
        other_val = float(rest["valor"].sum())
        other_share = other_val / total
        outros = pd.DataFrame([{
            "produto_cod": "OUTROS",
            "produto": "Outros",
            "valor": other_val,
            "share": other_share,
            "metric": metric,
        }])
        out = pd.concat([top, outros], ignore_index=True)
    else:
        out = top

    return out.reset_index(drop=True)


def representatividade_figure(rep: pd.DataFrame) -> go.Figure:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"

    rep2 = rep.copy()
    rep2["label_y"] = rep2.apply(
        lambda r: f"{r['produto_cod']} - {str(r['produto'])}".strip() if r["produto"] != "Outros" else "Outros",
        axis=1,
    )

    rep2["is_outros"] = (rep2["produto"] == "Outros").astype(int)
    rep2 = rep2.sort_values(["is_outros", "valor"], ascending=[True, False]).drop(columns=["is_outros"])

    rep_plot = rep2.iloc[::-1].copy()
    rep_plot["pct"] = (rep_plot["share"] * 100).round(1)
    rep_plot["txt"] = rep_plot.apply(lambda r: f"{r['pct']:.1f}%", axis=1)

    colors = []
    for p in rep_plot["produto"].tolist():
        if p == "Outros":
            colors.append("rgba(242,246,255,0.18)")
        else:
            colors.append(BRAND["blue"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=rep_plot["label_y"],
            x=rep_plot["valor"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=rep_plot["txt"],
            textposition="outside",
            textfont=dict(color=BRAND["ink"], size=12),
            customdata=rep_plot["pct"],
            hovertemplate="<b>%{y}</b><br>"
                          f"Rendas ({metric_lbl}): " + "%{x:,.2f}<br>"
                          "Share: %{customdata:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Rendas • Top 5 + Outros", x=0.02, xanchor="left", font=dict(size=16, color=BRAND["ink"])),
        height=420,
        showlegend=False,
    )

    xmax = float(rep_plot["valor"].max()) if len(rep_plot) else 0
    fig.update_xaxes(range=[0, xmax * 1.18 if xmax else 1])

    return apply_dark_plotly(fig)


# ==========================================================
# UI helpers
# ==========================================================
def kpi_card(label: str, value: str, badge: tuple[str, str] | None = None, sub: str | None = None) -> None:
    badge_html = ""
    if badge:
        dot_color, badge_txt = badge
        badge_html = f"""
        <div class="badge">
          <span class="dot" style="background:{dot_color}"></span>
          <span>{badge_txt}</span>
        </div>
        """
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""

    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {badge_html}
  {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# App
# ==========================================================
inject_css()

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

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="sb-card">'
        '<div class="sb-title">Dados</div>'
        '<div class="sb-sub">Anexe o relatório (CSV exportado)</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader(" ", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Envie o CSV do relatório para carregar os dados.")
        st.stop()

    df = load_report(uploaded.getvalue())

    # Produtos
    produtos = df[["produto_cod", "produto"]].drop_duplicates().sort_values(["produto_cod"])
    produtos["label"] = produtos["produto_cod"].astype(str) + " - " + produtos["produto"].astype(str)
    prod_labels = produtos["label"].tolist()

    default_sel = []
    for cod in ["18201", "18202"]:
        m = produtos[produtos["produto_cod"].astype(str) == cod]
        if not m.empty:
            default_sel.append(m["label"].iloc[0])

    st.markdown(
        '<div class="sb-card">'
        '<div class="sb-title">Filtros</div>',
        unsafe_allow_html=True
    )

    periodo = st.radio("Período", ["Total", "Ano", "Mês"], horizontal=True)

    anos = sorted([int(a) for a in df["ano"].dropna().unique().tolist()])
    ano_sel = None
    mes_sel = None

    if periodo in ["Ano", "Mês"]:
        ano_sel = st.selectbox("Ano", anos, index=len(anos) - 1 if len(anos) else 0)

    if periodo == "Mês":
        meses_no_ano = sorted([int(m) for m in df[df["ano"] == ano_sel]["mes"].dropna().unique().tolist()])
        mes_sel = st.selectbox(
            "Mês",
            meses_no_ano,
            format_func=lambda m: PT_MONTH_INV.get(int(m), str(m)).upper(),
        )

    prod_sel = st.multiselect("Produto (multi)", prod_labels, default=default_sel)

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Aplica filtros (df_period = só período; df_view = período+produto)
# ==========================================================
df_period = df.copy()

if periodo == "Ano" and ano_sel is not None:
    df_period = df_period[df_period["ano"] == ano_sel]
elif periodo == "Mês" and ano_sel is not None and mes_sel is not None:
    df_period = df_period[(df_period["ano"] == ano_sel) & (df_period["mes"] == mes_sel)]

df_view = df_period.copy()

if prod_sel:
    cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
    df_view = df_view[df_view["produto_cod"].astype(str).isin(cod_sel)]

prod_label = "Todos" if not prod_sel else (f"Seleção ({len(prod_sel)})" if len(prod_sel) > 1 else "Seleção (1)")

# ==========================================================
# KPIs
# ==========================================================
st.markdown('<p class="section-title">KPIs</p>', unsafe_allow_html=True)

saldo_all = df_view[df_view["tipo"] == "saldo"].copy()
saldo_real = saldo_all[saldo_all["realizado"].notna()].copy()

saldo_last_dt = saldo_real["data"].max() if not saldo_real.empty else None

saldo_last_real = np.nan
saldo_last_orc = np.nan
saldo_gap = np.nan

if saldo_last_dt is not None:
    s_last = saldo_all[saldo_all["data"] == saldo_last_dt]
    saldo_last_orc = float(s_last["orcado"].sum()) if not s_last.empty else np.nan
    saldo_last_real = float(s_last["realizado"].sum()) if not s_last.empty else np.nan
    if not (np.isnan(saldo_last_orc) or np.isnan(saldo_last_real)):
        saldo_gap = saldo_last_real - saldo_last_orc

rendas_all = df_view[df_view["tipo"] == "rendas"].copy()
rendas_orc_acc = float(rendas_all["orcado"].sum()) if not rendas_all.empty else np.nan
rendas_real_acc = float(rendas_all["realizado"].sum()) if not rendas_all.empty else np.nan

base_txt = saldo_last_dt.strftime("%b/%Y") if saldo_last_dt is not None else "—"

c1, c2, c3, c4 = st.columns(4)

with c1:
    ok = (not np.isnan(saldo_last_real)) and (saldo_last_real >= saldo_last_orc) if not np.isnan(saldo_last_orc) else False
    badge = (BRAND["green"], "Cumprido") if ok else (BRAND["danger"], "Não cumprido")
    kpi_card("Saldo • Realizado (último mês com dado)", fmt_br(saldo_last_real), badge=badge, sub=f"Base: {base_txt}")

with c2:
    kpi_card("Saldo • Gap vs Orçado (mesmo mês)", fmt_br(saldo_gap), sub="Comparação no mês-base do realizado.")

with c3:
    kpi_card("Rendas • Orçado (acumulado no recorte)", fmt_br(rendas_orc_acc), sub=f"Recorte: {periodo}")

with c4:
    ok2 = (not np.isnan(rendas_real_acc)) and (rendas_real_acc >= rendas_orc_acc) if not np.isnan(rendas_orc_acc) else False
    badge2 = (BRAND["green"], "Cumprido") if ok2 else (BRAND["danger"], "Não cumprido")
    kpi_card("Rendas • Realizado (acumulado no recorte)", fmt_br(rendas_real_acc), badge=badge2, sub=f"Recorte: {periodo}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ==========================================================
# Gráficos (Total deve trazer 2025 + 2026)
# ==========================================================
st.markdown('<p class="section-title">Evolução • Saldo da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Legenda:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

ts_saldo = build_timeseries(df_view, "saldo")
if ts_saldo.empty:
    st.info("Sem dados de Saldo para o recorte atual.")
else:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_orcado_bar_real_line(ts_saldo, "Saldo (Orçado x Realizado)", prod_label), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

st.markdown('<p class="section-title">Evolução • Rendas da Carteira</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pill"><span style="opacity:.8">Legenda:</span> <b>{prod_label}</b></div>', unsafe_allow_html=True)

ts_rendas = build_timeseries(df_view, "rendas")
if ts_rendas.empty:
    st.info("Sem dados de Rendas para o recorte atual.")
else:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_orcado_bar_real_line(ts_rendas, "Rendas (Orçado x Realizado)", prod_label), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

st.markdown('<p class="section-title">Representatividade • Produtos (Rendas)</p>', unsafe_allow_html=True)

rep = top5_representatividade_rendas(df_period)
if rep.empty:
    st.info("Sem dados suficientes de Rendas (18202*) para calcular representatividade neste recorte.")
else:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
    st.markdown(
        f'<div class="pill"><span style="opacity:.8">Base:</span> <b>Rendas ({metric_lbl})</b> '
        f'<span style="opacity:.6">• Top 5 + Outros</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(representatividade_figure(rep), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Top 5 + Outros. TOTAL 18202 é excluído para evitar distorção.")
