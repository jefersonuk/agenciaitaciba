import re
from datetime import datetime

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
    page_title="Itacibá • Orçado x Realizado",
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
  padding-top: 1.05rem;
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

/* Tabs: allow horizontal scroll when many tabs */
div[data-baseweb="tab-list"] {{
  gap: 10px !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  white-space: nowrap !important;
  padding-bottom: 6px !important;
}}
div[data-baseweb="tab-list"] > div {{
  flex: 0 0 auto !important;
}}
div[data-baseweb="tab-list"]::-webkit-scrollbar {{
  height: 8px;
}}
div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
  background: rgba(255,255,255,0.18);
  border-radius: 999px;
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
    except Exception:
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

def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
<div class="header-wrap">
  <div>
    <div class="header-title">{title}</div>
    <div class="header-sub">{subtitle}</div>
  </div>
  <div class="legend-pill">
    <span><span class="dot" style="background:{BRAND["blue"]}"></span> Orçado</span>
    <span><span class="dot" style="background:{BRAND["green"]}"></span> Realizado</span>
  </div>

""",
        unsafe_allow_html=True,
    )

# ==========================
# Parser matricial (2020 a 2026) - robusto p/ Carteira e Resultado
# ==========================
def type_from_code_carteira(code_str: str) -> str:
    if str(code_str).startswith("18201"):
        return "Saldo"
    if str(code_str).startswith("18202"):
        return "Rendas"
    return "Outro"

def type_from_code_dre(_: str) -> str:
    return "DRE"

@st.cache_data(show_spinner=False)
def load_matricial(file_bytes: bytes, kind: str) -> pd.DataFrame:
    """
    kind:
      - "carteira": define tipo por prefixos 18201/18202
      - "dre": tipo único (DRE)
    """
    from io import BytesIO
    raw = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="latin1")

    first_col = raw.columns[0]

    # Blocos do relatório começam em "Filtro selecionado" (um por ano)
    starts = raw.index[raw[first_col].astype(str).str.strip().eq("Filtro selecionado")].tolist()

    # Fallback: alguns exports antigos vêm sem "Filtro selecionado"
    if not starts:
        yr_rows = raw.index[raw[first_col].astype(str).str.strip().eq("Ano referência")].tolist()
        if not yr_rows:
            raise ValueError("Não encontrei blocos do relatório (linha 'Filtro selecionado').")
        starts = [max(0, i - 3) for i in yr_rows]

    starts = sorted(list(dict.fromkeys(starts))) + [len(raw)]  # sentinela
    recs = []

    type_func = type_from_code_carteira if kind == "carteira" else type_from_code_dre

    for bi in range(len(starts) - 1):
        s, e = starts[bi], starts[bi + 1]
        blk = raw.iloc[s:e]

        # Ano do bloco
        year = None
        yr_rows = blk.index[blk[first_col].astype(str).str.strip().eq("Ano referência")].tolist()
        if yr_rows:
            vals = blk.loc[yr_rows[0]].values
            for v in vals:
                vv = str(v).strip()
                if re.fullmatch(r"\d{4}", vv):
                    year = int(vv)
                    break
        if year is None:
            continue

        # Linha de cabeçalho "Cód"
        cod_rows = blk.index[blk[first_col].astype(str).str.strip().eq("Cód")].tolist()
        if not cod_rows:
            cod_rows = blk.index[blk[first_col].astype(str).str.strip().str.lower().isin(["cod", "código", "codigo"])].tolist()
            if not cod_rows:
                continue
        idx_cod = cod_rows[0]

        row_cod = blk.loc[idx_cod]
        months_in_cod = [(col, str(row_cod[col]).strip()) for col in blk.columns if str(row_cod[col]).strip() in PT_MONTH]

        # --- Caso A: colunas são os meses (ex.: 2026)
        if len(months_in_cod) >= 10:
            desc_col = blk.columns[1] if len(blk.columns) > 1 else None
            month_cols = sorted([(lab, col) for col, lab in months_in_cod], key=lambda x: PT_MONTH[x[0]])

            for ridx in blk.index[blk.index > idx_cod]:
                code_raw = str(blk.loc[ridx, first_col]).strip()
                code = re.sub(r"\D", "", code_raw)
                if code == "":
                    continue

                desc = str(blk.loc[ridx, desc_col]).strip() if desc_col else ""
                desc = re.sub(r"\s+", " ", desc).strip()

                for mon_label, col in month_cols:
                    orc = parse_ptbr_number(blk.loc[ridx, col])
                    dt_ = datetime(year, PT_MONTH[mon_label], 1)
                    recs.append(
                        {
                            "data": pd.to_datetime(dt_),
                            "ano": year,
                            "mes": PT_MONTH[mon_label],
                            "produto_cod": code,
                            "produto": desc,
                            "tipo": type_func(code),
                            "orcado": orc,
                            "realizado": np.nan,
                        }
                    )

        # --- Caso B: linha anterior tem Jan/Fev/... e cada mês tem 4 colunas (Orçado/Realizado/Var/Var%)
        else:
            idx_month = idx_cod - 1
            if idx_month not in blk.index:
                continue
            row_month = blk.loc[idx_month]

            month_start_cols = []
            for c in blk.columns:
                lab = str(row_month[c]).strip()
                if lab in PT_MONTH:
                    month_start_cols.append(c)

            blocks = []
            blk_cols = list(blk.columns)
            for c in month_start_cols:
                j = blk_cols.index(c)
                block_cols = blk_cols[j : j + 4]  # Orçado, Realizado, Var (R$), Var (%)
                mon = str(row_month[c]).strip()
                if len(block_cols) >= 2:
                    blocks.append((mon, block_cols))

            desc_col = blk.columns[1] if len(blk.columns) > 1 else None

            for ridx in blk.index[blk.index > idx_cod]:
                code_raw = str(blk.loc[ridx, first_col]).strip()
                code = re.sub(r"\D", "", code_raw)
                if code == "":
                    continue

                desc = str(blk.loc[ridx, desc_col]).strip() if desc_col else ""
                desc = re.sub(r"\s+", " ", desc).strip()

                for mon, cols in blocks:
                    orc = parse_ptbr_number(blk.loc[ridx, cols[0]])
                    rea = parse_ptbr_number(blk.loc[ridx, cols[1]])
                    dt_ = datetime(year, PT_MONTH[mon], 1)
                    recs.append(
                        {
                            "data": pd.to_datetime(dt_),
                            "ano": year,
                            "mes": PT_MONTH[mon],
                            "produto_cod": code,
                            "produto": desc,
                            "tipo": type_func(code),
                            "orcado": orc,
                            "realizado": rea,
                        }
                    )

    df = pd.DataFrame(recs)

    if df.empty:
        return pd.DataFrame(columns=["data","ano","mes","produto_cod","produto","tipo","orcado","realizado"])

    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)

    df = df[~(df["orcado"].isna() & df["realizado"].isna())].copy()
    df.sort_values(["data", "produto_cod"], inplace=True)
    return df

# ==========================
# KPI helpers
# ==========================
def last_real_month(d: pd.DataFrame) -> pd.Timestamp | None:
    if d.empty:
        return None
    g = d.groupby("data", as_index=False).agg(
        real=("realizado", lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan)
    )
    mask = g["real"].notna() & (g["real"].abs() > 0)
    if not mask.any():
        return None
    return g.loc[mask, "data"].max()

def value_at_month(d: pd.DataFrame, dt_: pd.Timestamp | None, col: str) -> float:
    if d.empty or dt_ is None:
        return float("nan")
    g = d.groupby("data", as_index=False).agg(
        val=(col, lambda s: float(np.nansum(s.values)) if s.notna().any() else np.nan)
    )
    row = g[g["data"] == dt_]
    return float(row["val"].iloc[0]) if not row.empty else float("nan")

def acumulado(d: pd.DataFrame, col: str, upto_dt: pd.Timestamp | None = None) -> float:
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

# ==========================
# Charts
# ==========================
def make_evolucao_figure(df_view: pd.DataFrame, title: str, prod_label: str) -> go.Figure:
    d = df_view.copy()
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

def top5_representatividade(df_base: pd.DataFrame, exclude_codes: list[str] | None = None) -> pd.DataFrame:
    if df_base.empty:
        return pd.DataFrame()

    x = df_base.copy()
    x["produto_cod"] = x["produto_cod"].astype(str)

    if exclude_codes:
        x = x[~x["produto_cod"].isin([str(c) for c in exclude_codes])]

    if x.empty:
        return pd.DataFrame()

    has_real = x["realizado"].notna().any() and float(np.nansum(np.abs(x["realizado"].values))) > 0
    metric = "realizado" if has_real else "orcado"

    g = (
        x.groupby(["produto_cod", "produto"], as_index=False)
        .agg(valor=(metric, lambda s: float(np.nansum(s.values))))
    )
    g = g[g["valor"].abs() > 0].copy()
    if g.empty:
        return pd.DataFrame()

    g["abs"] = g["valor"].abs()
    g = g.sort_values("abs", ascending=False).drop(columns=["abs"])

    top5 = g.head(5).copy()

    rest = float(g.iloc[5:]["valor"].abs().sum()) if len(g) > 5 else 0.0
    if rest > 0:
        top5 = pd.concat(
            [top5, pd.DataFrame([{"produto_cod": "OUTROS", "produto": "Outros", "valor": rest}])],
            ignore_index=True,
        )

    total_abs = float(top5.apply(lambda r: abs(r["valor"]) if r["produto"] != "Outros" else float(r["valor"]), axis=1).sum())
    if total_abs <= 0:
        top5["share"] = 0.0
    else:
        top5["share"] = top5.apply(lambda r: (abs(r["valor"]) if r["produto"] != "Outros" else float(r["valor"])) / total_abs, axis=1)

    top5["metric"] = metric
    return top5

def representatividade_figure(rep: pd.DataFrame) -> go.Figure:
    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"

    rep2 = rep.copy()
    rep2["valor_plot"] = rep2["valor"].abs()
    rep2["is_outros"] = (rep2["produto"] == "Outros").astype(int)
    rep2 = rep2.sort_values(["is_outros", "valor_plot"], ascending=[True, False])

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
            x=rep_plot["valor_plot"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=rep_plot["label"],
            textposition="outside",
            textfont=dict(color=BRAND["muted"], size=12),
            customdata=np.c_[rep_plot["pct"].values, rep_plot["valor"].values],
            hovertemplate="<b>%{y}</b><br>"
                          f"Valor ({metric_lbl}): " + "%{customdata[1]:,.2f}<br>"
                          "Share: %{customdata[0]:.1f}%<extra></extra>",
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

    xmax = float(rep_plot["valor_plot"].max()) if len(rep_plot) else 0
    fig.update_xaxes(range=[0, xmax * 1.18 if xmax > 0 else 1])

    return fig

# ==========================
# KPI Card rendering
# ==========================
from textwrap import dedent as _dedent

def render_kpi(col, title, value, badge_text=None, badge_color=None, sub_label=None, sub_value=None):
    badge_html = ""
    if badge_text and badge_color:
        badge_html = _dedent(f"""
        <div class="badge" style="color:{badge_color}">
          <span class="dot" style="background:{badge_color}"></span>
          <span>{badge_text}</span>
        </div>
        """).strip()

    sub_html = ""
    if sub_label is not None:
        if sub_value is None:
            sub_html = f'<div class="kpi-sub">{sub_label}</div>'
        else:
            sub_html = f'<div class="kpi-sub">{sub_label} <b>{sub_value}</b></div>'

    html = _dedent(f"""
    <div class="kpi-card">
      <div class="kpi-label">{title}</div>
      <div class="kpi-value">{value}</div>
      {badge_html}
      {sub_html}
    </div>
    """).strip()

    with col:
        st.markdown(html, unsafe_allow_html=True)

# ==========================
# UI: Upload + mapeamento de arquivos por área
# ==========================
st.sidebar.markdown("## Dados")
uploads = st.sidebar.file_uploader(
    "Anexe 1 ou 2 relatórios (CSV exportado)",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploads:
    st.sidebar.info("Envie o(s) CSV(s) do relatório para carregar os dados.")
    st.stop()

files_map = {f.name: f.getvalue() for f in uploads}
names = list(files_map.keys())

def _default_pick(contains: list[str]) -> str | None:
    for n in names:
        ln = n.lower()
        if any(c in ln for c in contains):
            return n
    return names[0] if names else None

st.sidebar.markdown("## Arquivos por área")
file_carteira_name = st.sidebar.selectbox(
    "Arquivo • Carteira",
    options=names,
    index=names.index(_default_pick(["carteira", "credito"])) if _default_pick(["carteira", "credito"]) in names else 0,
)
file_resultado_name = st.sidebar.selectbox(
    "Arquivo • Resultado",
    options=names,
    index=names.index(_default_pick(["resultado", "dre", "demonstracao"])) if _default_pick(["resultado", "dre", "demonstracao"]) in names else min(1, len(names)-1),
)

df_carteira = load_matricial(files_map[file_carteira_name], kind="carteira") if file_carteira_name else pd.DataFrame()
df_resultado = load_matricial(files_map[file_resultado_name], kind="dre") if file_resultado_name else pd.DataFrame()

# ==========================
# Filters (por aba, no corpo)
# ==========================
def period_controls(df: pd.DataFrame, key: str):
    cols = st.columns([1.1, 1.0, 1.0, 3.2], gap="small")
    with cols[0]:
        periodo = st.radio("Período", ["Total", "Ano", "Mês"], horizontal=True, key=f"{key}_periodo")
    anos = sorted(df["ano"].unique().tolist()) if not df.empty else []
    ano_sel = None
    mes_sel = None
    with cols[1]:
        if periodo in ["Ano", "Mês"]:
            if anos:
                ano_sel = st.selectbox("Ano", anos, index=len(anos) - 1, key=f"{key}_ano")
            else:
                st.selectbox("Ano", [], key=f"{key}_ano_disabled")
        else:
            st.markdown("")
    with cols[2]:
        if periodo == "Mês" and anos and ano_sel is not None:
            meses_no_ano = sorted(df[df["ano"] == ano_sel]["mes"].unique().tolist())
            mes_sel = st.selectbox(
                "Mês",
                meses_no_ano,
                format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m)),
                key=f"{key}_mes",
            )
        else:
            st.markdown("")
    with cols[3]:
        st.markdown("")  # reservado (produto entra depois, por aba)
    return periodo, ano_sel, mes_sel, cols

def apply_period(df: pd.DataFrame, periodo: str, ano_sel: int | None, mes_sel: int | None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if periodo == "Ano" and ano_sel is not None:
        out = out[out["ano"] == ano_sel]
    elif periodo == "Mês" and (ano_sel is not None) and (mes_sel is not None):
        out = out[(out["ano"] == ano_sel) & (out["mes"] == mes_sel)]
    return out

def product_controls(df_area: pd.DataFrame, default_codes: list[str], key: str, col_container):
    produtos = (
        df_area[["produto_cod", "produto"]]
        .drop_duplicates()
        .sort_values(["produto_cod"])
        .copy()
    )
    produtos["label"] = produtos["produto_cod"] + " - " + produtos["produto"]
    labels = produtos["label"].tolist()

    default_sel = []
    for cod in default_codes:
        match = produtos[produtos["produto_cod"] == str(cod)]
        if not match.empty:
            default_sel.append(match["label"].iloc[0])

    with col_container:
        prod_sel = st.multiselect("Produto (multi)", labels, default=default_sel, key=f"{key}_prod")
    return prod_sel, produtos

def prod_label_from_selection(prod_sel: list[str]) -> str:
    if not prod_sel:
        return "Todos"
    if len(prod_sel) == 1:
        return prod_sel[0].split(" - ", 1)[1][:28]
    return f"Seleção ({len(prod_sel)})"

def prod_shown_pill(prod_sel: list[str]) -> str:
    if prod_sel:
        shown = " • ".join([p.split(" - ", 1)[0] for p in prod_sel[:4]])
        if len(prod_sel) > 4:
            shown += f" • +{len(prod_sel) - 4}"
    else:
        shown = "Todos"
    st.markdown(f'<div class="pill"><span style="opacity:.75">Produtos:</span> <b>{shown}</b></div>', unsafe_allow_html=True)

# ==========================
# Renderers por aba
# ==========================
def render_carteira_tab():
    if df_carteira.empty:
        st.info("Sem dados carregados para Carteira.")
        return

    render_header(
        "Itacibá • Carteira de Crédito",
        "Orçado x Realizado (2020 a 2026) • filtros por período e produto",
    )

    periodo, ano_sel, mes_sel, cols = period_controls(df_carteira, key="t1")

    prod_sel, _ = product_controls(
        df_carteira,
        default_codes=["18201", "18202"],
        key="t1",
        col_container=cols[3],
    )

    df_period = apply_period(df_carteira, periodo, ano_sel, mes_sel)
    df_view = df_period.copy()
    if prod_sel:
        cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
        df_view = df_view[df_view["produto_cod"].isin(cod_sel)]

    prod_label = prod_label_from_selection(prod_sel)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    saldo = df_view[df_view["tipo"] == "Saldo"].copy()
    rendas = df_view[df_view["tipo"] == "Rendas"].copy()

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

    c1, c2, c3, c4 = st.columns(4, gap="small")

    render_kpi(
        c1,
        "Saldo • Realizado (último mês com dado)",
        fmt_br(saldo_real_last),
        badge_text=saldo_farol_txt if saldo_last_dt is not None else None,
        badge_color=saldo_farol_color if saldo_last_dt is not None else None,
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
        sub_label=None,
        sub_value=None,
    )

    prod_shown_pill(prod_sel)

    # ✅ Ajuste 1: Aba 1 sempre mostra os DOIS gráficos (Saldo e Rendas)
    st.markdown('<div class="section-title">Evolução • Saldo da Carteira</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(make_evolucao_figure(saldo, "Saldo (Orçado x Realizado)", prod_label), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Evolução • Rendas da Carteira</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(make_evolucao_figure(rendas, "Rendas (Orçado x Realizado)", prod_label), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Representatividade • Produtos (Rendas)</div>', unsafe_allow_html=True)
    rep_base = df_period[df_period["tipo"] == "Rendas"].copy()
    rep_base = rep_base[rep_base["produto_cod"].str.startswith("18202")]
    rep = top5_representatividade(rep_base, exclude_codes=["18202"])
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

def render_resultado_area_tab(
    tab_key: str,
    title: str,
    prefix_code: str,
    exclude_total: list[str],
    subtitle: str = "Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
):
    if df_resultado.empty:
        st.info("Envie o CSV de Resultado (DRE) para habilitar esta aba.")
        return

    df_area = df_resultado[df_resultado["produto_cod"].str.startswith(str(prefix_code))].copy()
    if df_area.empty:
        st.warning(f"Não encontrei linhas no DRE para o código/prefixo {prefix_code}.")
        return

    render_header(title, subtitle)

    periodo, ano_sel, mes_sel, cols = period_controls(df_area, key=tab_key)

    prod_sel, _ = product_controls(
        df_area,
        default_codes=[prefix_code],
        key=tab_key,
        col_container=cols[3],
    )

    df_period = apply_period(df_area, periodo, ano_sel, mes_sel)
    df_view = df_period.copy()
    if prod_sel:
        cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
        df_view = df_view[df_view["produto_cod"].isin(cod_sel)]

    prod_label = prod_label_from_selection(prod_sel)

    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    last_dt = last_real_month(df_view)
    real_last = value_at_month(df_view, last_dt, "realizado")
    orc_last = value_at_month(df_view, last_dt, "orcado")
    gap = real_last - orc_last

    ok = (
        (last_dt is not None)
        and (not np.isnan(real_last))
        and (not np.isnan(orc_last))
        and (real_last >= orc_last)
    )
    farol_txt = "Cumprido" if ok else "Não cumprido"
    farol_color = BRAND["ok"] if ok else BRAND["danger"]
    base_txt = month_label(last_dt)

    orc_acum = acumulado(df_view, "orcado", upto_dt=None)
    real_acum = acumulado(df_view, "realizado", upto_dt=None)

    orc_base = acumulado(df_view, "orcado", upto_dt=last_dt) if last_dt is not None else np.nan
    real_base = acumulado(df_view, "realizado", upto_dt=last_dt) if last_dt is not None else np.nan
    ok_acum = False
    if last_dt is not None and (not np.isnan(orc_base)) and (not np.isnan(real_base)):
        ok_acum = real_base >= orc_base

    c1, c2, c3, c4 = st.columns(4, gap="small")

    render_kpi(
        c1,
        "Realizado (último mês com dado)",
        fmt_br(real_last),
        badge_text=farol_txt if last_dt is not None else None,
        badge_color=farol_color if last_dt is not None else None,
        sub_label="Base:",
        sub_value=base_txt,
    )

    render_kpi(
        c2,
        "Gap vs Orçado (no mês-base)",
        fmt_br(gap),
        sub_label="Comparação no mês-base do realizado.",
        sub_value="",
    )

    render_kpi(
        c3,
        "Orçado (acumulado no recorte)",
        fmt_br(orc_acum),
        sub_label="Base do farol:" if last_dt is not None else "Sem base de realizado ainda.",
        sub_value=base_txt if last_dt is not None else "",
    )

    render_kpi(
        c4,
        "Realizado (acumulado no recorte)",
        fmt_br(real_acum),
        badge_text=("Cumprido" if ok_acum else "Não cumprido") if last_dt is not None else None,
        badge_color=(BRAND["ok"] if ok_acum else BRAND["danger"]) if last_dt is not None else None,
        sub_label=None,
        sub_value=None,
    )

    prod_shown_pill(prod_sel)
    st.markdown('<div class="section-title">Evolução</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(make_evolucao_figure(df_view, f"{title} (Orçado x Realizado)", prod_label), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Representatividade • Produtos</div>', unsafe_allow_html=True)

    uniq_codes = df_period[["produto_cod"]].drop_duplicates().shape[0]
    if uniq_codes <= 1:
        st.info("Esta linha é um total (sem abertura por produtos).")
        return

    rep = top5_representatividade(df_period, exclude_codes=exclude_total)
    if rep.empty:
        st.info("Sem dados suficientes para calcular representatividade neste recorte.")
    else:
        metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
        st.markdown(
            f'<div class="pill"><span style="opacity:.8">Base:</span> <b>{title} ({metric_lbl})</b> <span style="opacity:.6">• Top 5 + Outros</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(representatividade_figure(rep), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if exclude_total:
            st.caption(f"Top 5 + Outros. Total(is) excluído(s): {', '.join(map(str, exclude_total))}")

# ==========================
# Tabs
# ==========================
tabs = st.tabs([
    "Aba 1 • Carteira de Crédito",
    "Aba 2 • Receita da prestação de serviços",
    "Aba 3 • Receitas de tesouraria",
    "Aba 4 • Receitas de participação das controladas",
    "Aba 5 • Despesas de pessoal",
    "Aba 6 • Outras despesas administrativas",
    "Aba 7 • Despesas de PDD",
    "Aba 8 • Lucro líquido",
])

with tabs[0]:
    render_carteira_tab()

with tabs[1]:
    render_resultado_area_tab(
        tab_key="t2",
        title="Itacibá • Receita da prestação de serviços",
        prefix_code="2820401",
        exclude_total=["2820401"],
    )

with tabs[2]:
    render_resultado_area_tab(
        tab_key="t3",
        title="Itacibá • Receitas de tesouraria",
        prefix_code="2820101",
        exclude_total=["2820101"],
    )

with tabs[3]:
    # ✅ Ajuste 2: Controladas = prefixo 2820405
    render_resultado_area_tab(
        tab_key="t4",
        title="Itacibá • Receitas de participação das controladas",
        prefix_code="2820405",
        exclude_total=["2820405"],
    )

with tabs[4]:
    render_resultado_area_tab(
        tab_key="t5",
        title="Itacibá • Despesas de pessoal",
        prefix_code="2820402",
        exclude_total=["2820402"],
    )

with tabs[5]:
    render_resultado_area_tab(
        tab_key="t6",
        title="Itacibá • Outras despesas administrativas",
        prefix_code="2820403",
        exclude_total=["2820403", "282040320"],
    )

with tabs[6]:
    # ✅ Ajuste 3: Aba 7 = PDD (prefixo 2820204)
    render_resultado_area_tab(
        tab_key="t7",
        title="Itacibá • Despesas de PDD",
        prefix_code="2820204",
        exclude_total=["2820204"],
    )

with tabs[7]:
    # ✅ Ajuste 3: Aba 8 = Lucro líquido (prefixo 28214)
    render_resultado_area_tab(
        tab_key="t8",
        title="Itacibá • Lucro líquido",
        prefix_code="28214",
        exclude_total=["28214"],
    )
