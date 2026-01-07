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
    "blue": "#1E0AE8",
    "green": "#00AB16",
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
    page_title="Itacibá • Dashboards",
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
# Parser universal (aceita CSV antigo 2025/2026 e CSV novo 2020-2026)
# ==========================
@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="latin1")
    first_col = raw.columns[0]

    # Detecta formato novo (blocos por ano)
    has_blocks = raw[first_col].astype(str).str.strip().eq("Filtro selecionado").any()

    # --------------------------
    # Formato novo: 2020 a 2026
    # --------------------------
    if has_blocks:
        starts = raw.index[raw[first_col].astype(str).str.strip().eq("Filtro selecionado")].tolist()
        starts = starts + [len(raw)]
        recs = []

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

            # Linha "Cód"
            cod_rows = blk.index[blk[first_col].astype(str).str.strip().eq("Cód")].tolist()
            if not cod_rows:
                continue
            idx_cod = cod_rows[0]

            row_cod = blk.loc[idx_cod]
            months_in_cod = [(col, str(row_cod[col]).strip()) for col in blk.columns if str(row_cod[col]).strip() in PT_MONTH]

            # Caso A: estilo 2026 (Cód | Desc | Jan | Fev | ...)
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
                    desc = re.sub(rf"^{re.escape(code)}\s*-\s*", "", desc).strip()

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
                                "orcado": orc,
                                "realizado": np.nan,
                            }
                        )

            # Caso B: 2020-2025 (linha anterior tem meses; cada mês 4 colunas)
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
                    block_cols = blk_cols[j: j + 4]  # Orçado, Realizado, Var(R$), Var(%)
                    mon = str(row_month[c]).strip()
                    if len(block_cols) >= 2:
                        blocks.append((mon, block_cols))

                desc_col = blk.columns[1]

                for ridx in blk.index[blk.index > idx_cod]:
                    code_raw = str(blk.loc[ridx, first_col]).strip()
                    code = re.sub(r"\D", "", code_raw)
                    if code == "":
                        continue

                    desc = str(blk.loc[ridx, desc_col]).strip()
                    desc = re.sub(r"\s+", " ", desc).strip()
                    desc = re.sub(rf"^{re.escape(code)}\s*-\s*", "", desc).strip()

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
                                "orcado": orc,
                                "realizado": rea,
                            }
                        )

        df = pd.DataFrame(recs)
        if df.empty:
            return df

        df["produto_cod"] = df["produto_cod"].astype(str)
        df["produto"] = df["produto"].astype(str)
        df = df[~(df["orcado"].isna() & df["realizado"].isna())].copy()
        df.sort_values(["data", "produto_cod"], inplace=True)
        return df

    # --------------------------
    # Formato antigo: 2025 + 2026 (matricial legado)
    # --------------------------
    # 2025: meses na linha 0 (Jan/Fev/...); dados em linhas 2:64; blocos de 4 colunas por mês (Orçado/Realizado/Var/Var%)
    col_code_2025 = raw.columns[0]
    col_desc_2025 = raw.columns[1] if len(raw.columns) > 1 else raw.columns[0]

    month_start_cols = []
    # tenta achar os meses na linha 0
    for c in raw.columns[2:]:
        if str(raw.loc[0, c]).strip() in PT_MONTH:
            month_start_cols.append(c)

    blocks_2025 = []
    for start in month_start_cols:
        idx = raw.columns.get_loc(start)
        block_cols = raw.columns[idx: idx + 4]
        mon = str(raw.loc[0, start]).strip()
        blocks_2025.append((mon, block_cols.tolist()))

    recs = []

    # linhas de dados típicas
    data_rows_2025 = raw.iloc[2:64].copy()

    for _, r in data_rows_2025.iterrows():
        code = str(r[col_code_2025]).strip()
        code_digits = re.sub(r"\D", "", code)
        if code_digits == "":
            continue

        desc = str(r[col_desc_2025]).strip()
        desc = re.sub(r"\s+", " ", desc).strip()
        desc = re.sub(rf"^{re.escape(code_digits)}\s*-\s*", "", desc).strip()

        for mon, cols in blocks_2025:
            orc = parse_ptbr_number(r[cols[0]]) if len(cols) > 0 else np.nan
            rea = parse_ptbr_number(r[cols[1]]) if len(cols) > 1 else np.nan
            dt_ = datetime(2025, PT_MONTH.get(mon, 1), 1)

            recs.append(
                {
                    "data": pd.to_datetime(dt_),
                    "ano": 2025,
                    "mes": PT_MONTH.get(mon, 1),
                    "produto_cod": code_digits,
                    "produto": desc,
                    "orcado": orc,
                    "realizado": rea,
                }
            )

    # 2026: procura um cabeçalho "Cód" em alguma coluna da linha 0
    code_col_2026 = None
    desc_col_2026 = None
    for c in raw.columns:
        if str(raw.loc[0, c]).strip() == "Cód":
            code_col_2026 = c
            idxc = raw.columns.get_loc(c)
            if idxc + 1 < len(raw.columns):
                desc_col_2026 = raw.columns[idxc + 1]
            break

    if code_col_2026 is not None and desc_col_2026 is not None:
        idx_code = raw.columns.get_loc(code_col_2026)
        month_cols_2026 = []
        for c in raw.columns[idx_code + 1:]:
            lab = str(raw.loc[0, c]).strip()
            if lab in PT_MONTH:
                month_cols_2026.append((lab, c))

        data_rows_2026 = raw.iloc[1:64].copy()

        for _, r in data_rows_2026.iterrows():
            code = str(r[code_col_2026]).strip()
            code_digits = re.sub(r"\D", "", code)
            if code_digits == "":
                continue

            desc = str(r[desc_col_2026]).strip()
            desc = re.sub(r"\s+", " ", desc).strip()
            desc = re.sub(rf"^{re.escape(code_digits)}\s*-\s*", "", desc).strip()

            for mon, c in month_cols_2026:
                orc = parse_ptbr_number(r[c])
                dt_ = datetime(2026, PT_MONTH[mon], 1)
                recs.append(
                    {
                        "data": pd.to_datetime(dt_),
                        "ano": 2026,
                        "mes": PT_MONTH[mon],
                        "produto_cod": code_digits,
                        "produto": desc,
                        "orcado": orc,
                        "realizado": np.nan,
                    }
                )

    df = pd.DataFrame(recs)
    if df.empty:
        return df

    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df = df[~(df["orcado"].isna() & df["realizado"].isna())].copy()
    df.sort_values(["data", "produto_cod"], inplace=True)
    return df


# ==========================
# KPI helpers
# ==========================
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


# ==========================
# Charts
# ==========================
def make_evolucao_figure(df_view: pd.DataFrame, title: str, prod_label: str) -> go.Figure:
    if df_view.empty:
        fig = go.Figure()
        fig.update_layout(**make_plotly_layout_base(), height=360)
        fig.add_annotation(
            text="Sem dados para este recorte/produtos.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=BRAND["muted"], size=14),
        )
        return fig

    agg = df_view.groupby("data", as_index=False).agg(
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


def top5_representatividade(df_base: pd.DataFrame, exclude_exact=None) -> pd.DataFrame:
    if df_base.empty:
        return pd.DataFrame()

    x = df_base.copy()
    x["produto_cod"] = x["produto_cod"].astype(str)

    if exclude_exact:
        x = x[~x["produto_cod"].isin(set(map(str, exclude_exact)))]

    if x.empty:
        return pd.DataFrame()

    has_real = x["realizado"].notna().any() and float(np.nansum(x["realizado"].values)) > 0
    metric = "realizado" if has_real else "orcado"

    g = (
        x.groupby(["produto_cod", "produto"], as_index=False)
        .agg(valor=(metric, lambda s: float(np.nansum(s.values))))
    )
    g = g[g["valor"] != 0].sort_values("valor", ascending=False)

    if g.empty:
        return pd.DataFrame()

    top5 = g.head(5).copy()
    rest = float(g.iloc[5:]["valor"].sum()) if len(g) > 5 else 0.0
    if rest != 0:
        top5 = pd.concat(
            [top5, pd.DataFrame([{"produto_cod": "OUTROS", "produto": "Outros", "valor": rest}])],
            ignore_index=True,
        )

    total = float(top5["valor"].sum())
    top5["share"] = top5["valor"] / total if total != 0 else 0.0
    top5["metric"] = metric
    return top5


def representatividade_figure(rep: pd.DataFrame, title: str) -> go.Figure:
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
                          f"{title} ({metric_lbl}): " + "%{x:,.2f}<br>"
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
# UI helpers
# ==========================
from textwrap import dedent


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
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi(col, title, value, badge_text=None, badge_color=None, sub_label=None, sub_value=None):
    badge_html = ""
    if badge_text and badge_color:
        badge_html = dedent(f"""
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

    html = dedent(f"""
    <div class="kpi-card">
      <div class="kpi-label">{title}</div>
      <div class="kpi-value">{value}</div>
      {badge_html}
      {sub_html}
    </div>
    """).strip()

    with col:
        st.markdown(html, unsafe_allow_html=True)


def render_pill_produtos(prod_sel):
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


def build_labels(df_scope: pd.DataFrame) -> pd.DataFrame:
    produtos = df_scope[["produto_cod", "produto"]].drop_duplicates().sort_values("produto_cod")
    produtos["label"] = produtos["produto_cod"].astype(str) + " - " + produtos["produto"].astype(str)
    return produtos


def default_labels_for_codes(produtos_df: pd.DataFrame, codes):
    out = []
    for c in codes:
        m = produtos_df[produtos_df["produto_cod"].astype(str) == str(c)]
        if not m.empty:
            out.append(m["label"].iloc[0])
    return out


def apply_period_filter(df_in: pd.DataFrame, periodo: str, ano_sel, mes_sel):
    df_out = df_in.copy()
    if periodo == "Ano" and ano_sel is not None:
        df_out = df_out[df_out["ano"] == ano_sel]
    elif periodo == "Mês" and ano_sel is not None and mes_sel is not None:
        df_out = df_out[(df_out["ano"] == ano_sel) & (df_out["mes"] == mes_sel)]
    return df_out


def render_filters_in_tab(df_scope: pd.DataFrame, key_prefix: str, default_sel):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 2.4], gap="small")

    with c1:
        periodo = st.radio(
            "Período",
            ["Total", "Ano", "Mês"],
            horizontal=True,
            key=f"{key_prefix}_periodo",
        )

    anos = sorted(df_scope["ano"].unique().tolist()) if not df_scope.empty else []
    ano_sel = None
    mes_sel = None

    with c2:
        if periodo in ["Ano", "Mês"] and anos:
            ano_sel = st.selectbox("Ano", anos, index=len(anos) - 1, key=f"{key_prefix}_ano")
        else:
            st.caption("Ano")

    with c3:
        if periodo == "Mês" and ano_sel is not None:
            meses_no_ano = sorted(df_scope[df_scope["ano"] == ano_sel]["mes"].unique().tolist())
            mes_sel = st.selectbox(
                "Mês",
                meses_no_ano,
                format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m)),
                key=f"{key_prefix}_mes",
            )
        else:
            st.caption("Mês")

    produtos = build_labels(df_scope)
    prod_labels = produtos["label"].tolist()
    default_ok = [x for x in default_sel if x in prod_labels]

    with c4:
        prod_sel = st.multiselect(
            "Produto (multi)",
            prod_labels,
            default=default_ok,
            key=f"{key_prefix}_prod",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    df_period = apply_period_filter(df_scope, periodo, ano_sel, mes_sel)
    df_view = df_period.copy()

    if prod_sel:
        cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
        df_view = df_view[df_view["produto_cod"].astype(str).isin(cod_sel)]

    if not prod_sel:
        prod_label = "Todos"
    else:
        if len(prod_sel) == 1:
            prod_label = prod_sel[0].split(" - ", 1)[1][:28]
        else:
            prod_label = f"Seleção ({len(prod_sel)})"

    return df_period, df_view, prod_sel, prod_label


def render_kpis_and_charts(
    df_scope: pd.DataFrame,
    df_period: pd.DataFrame,
    df_view: pd.DataFrame,
    prod_sel,
    prod_label: str,
    title: str,
    is_expense: bool,
    rep_exclude_exact=None,
):
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    base_dt = last_real_month(df_view)
    real_last = value_at_month(df_view, base_dt, "realizado")
    orc_last = value_at_month(df_view, base_dt, "orcado")
    gap = real_last - orc_last

    orc_acc = acumulado(df_view, "orcado", upto_dt=None)
    real_acc = acumulado(df_view, "realizado", upto_dt=None)

    orc_base = acumulado(df_view, "orcado", upto_dt=base_dt) if base_dt is not None else np.nan
    real_base = acumulado(df_view, "realizado", upto_dt=base_dt) if base_dt is not None else np.nan

    ok = False
    if base_dt is not None and (not np.isnan(orc_base)) and (not np.isnan(real_base)):
        ok = (real_base <= orc_base) if is_expense else (real_base >= orc_base)

    badge_txt = None
    badge_color = None
    if base_dt is not None:
        badge_txt = ("Dentro do orçamento" if ok else "Fora do orçamento") if is_expense else ("Cumprido" if ok else "Não cumprido")
        badge_color = BRAND["ok"] if ok else BRAND["danger"]

    base_txt = month_label(base_dt) if base_dt is not None else "—"

    c1, c2, c3, c4 = st.columns(4, gap="small")

    render_kpi(
        c1,
        "Realizado (último mês com dado)",
        fmt_br(real_last),
        badge_text=badge_txt,
        badge_color=badge_color,
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
        fmt_br(orc_acc),
        sub_label="",
        sub_value="",
    )

    render_kpi(
        c4,
        "Realizado (acumulado no recorte)",
        fmt_br(real_acc),
        badge_text=badge_txt if base_dt is not None else None,
        badge_color=badge_color if base_dt is not None else None,
    )

    render_pill_produtos(prod_sel)

    st.markdown('<div class="section-title">Evolução</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig = make_evolucao_figure(df_view, f"{title} (Orçado x Realizado)", prod_label)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    rep = top5_representatividade(df_period, exclude_exact=rep_exclude_exact)
    if not rep.empty:
        metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"
        st.markdown('<div class="section-title">Representatividade • Produtos</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pill"><span style="opacity:.8">Base:</span> <b>{title} ({metric_lbl})</b> <span style="opacity:.6">• Top 5 + Outros</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(representatividade_figure(rep, title=title), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_tab(
    df_source: pd.DataFrame,
    key_prefix: str,
    title: str,
    subtitle: str,
    default_codes,
    is_expense: bool,
    filter_fn,
    rep_exclude_exact=None,
):
    if df_source.empty:
        st.info("Sem dados no arquivo carregado.")
        return

    df_scope = df_source[filter_fn(df_source)].copy()
    if df_scope.empty:
        st.info("Este arquivo não contém dados para esta aba.")
        return

    produtos = build_labels(df_scope)
    default_sel = default_labels_for_codes(produtos, default_codes)

    render_header(title, subtitle)

    df_period, df_view, prod_sel, prod_label = render_filters_in_tab(df_scope, key_prefix, default_sel)

    render_kpis_and_charts(
        df_scope=df_scope,
        df_period=df_period,
        df_view=df_view,
        prod_sel=prod_sel,
        prod_label=prod_label,
        title=title,
        is_expense=is_expense,
        rep_exclude_exact=rep_exclude_exact,
    )


# ==========================
# Sidebar: upload (1 ou 2 arquivos)
# ==========================
st.sidebar.markdown("## Dados")
uploads = st.sidebar.file_uploader(
    "Anexe 1 ou 2 relatórios (CSV exportado)",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploads:
    st.sidebar.info("Envie pelo menos 1 CSV para carregar os dados.")
    st.stop()

files_map = {u.name: u.getvalue() for u in uploads}
names = list(files_map.keys())

if len(names) == 1:
    file_carteira = names[0]
    file_resultado = names[0]
else:
    st.sidebar.markdown("## Arquivos por área")
    file_carteira = st.sidebar.selectbox("Arquivo • Carteira", names, index=0)
    file_resultado = st.sidebar.selectbox("Arquivo • Resultado", names, index=1)

try:
    df_carteira = load_report(files_map[file_carteira])
except Exception as e:
    st.sidebar.error("Falha ao ler o arquivo de Carteira. Verifique o CSV.")
    raise

try:
    df_resultado = load_report(files_map[file_resultado])
except Exception as e:
    st.sidebar.error("Falha ao ler o arquivo de Resultado. Verifique o CSV.")
    raise


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


# ==========================
# Aba 1: Carteira (18201/18202)
# ==========================
with tabs[0]:
    render_tab(
        df_source=df_carteira,
        key_prefix="tab1_carteira",
        title="Itacibá • Carteira de Crédito",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=["18201", "18202"],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("18201") | d["produto_cod"].astype(str).str.startswith("18202"),
        rep_exclude_exact=["18202"],  # remove total 18202 para não distorcer
    )


# ==========================
# Aba 2: Receita da prestação de serviços (2820401*)
# ==========================
with tabs[1]:
    prefix = "2820401"
    render_tab(
        df_source=df_resultado,
        key_prefix="tab2_servicos",
        title="Itacibá • Receita da prestação de serviços",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=[prefix],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefix),
        rep_exclude_exact=[prefix],  # remove total se existir
    )


# ==========================
# Aba 3: Receitas de tesouraria (2820101*)
# ==========================
with tabs[2]:
    prefix = "2820101"
    render_tab(
        df_source=df_resultado,
        key_prefix="tab3_tesouraria",
        title="Itacibá • Receitas de tesouraria",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=[prefix],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefix),
        rep_exclude_exact=[prefix],
    )


# ==========================
# Aba 4: Receitas de participação das controladas (282040501*)
# ==========================
with tabs[3]:
    prefix = "282040501"
    render_tab(
        df_source=df_resultado,
        key_prefix="tab4_controladas",
        title="Itacibá • Receitas de participação das controladas",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=[prefix],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefix),
        rep_exclude_exact=[prefix],
    )


# ==========================
# Aba 5: Despesas de pessoal (2820402*)
# ==========================
with tabs[4]:
    prefix = "2820402"
    render_tab(
        df_source=df_resultado,
        key_prefix="tab5_pessoal",
        title="Itacibá • Despesas de pessoal",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=[prefix],
        is_expense=True,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefix),
        rep_exclude_exact=[prefix],
    )


# ==========================
# Aba 6: Outras despesas administrativas (2820403*)
# ==========================
with tabs[5]:
    prefix = "2820403"
    render_tab(
        df_source=df_resultado,
        key_prefix="tab6_admin",
        title="Itacibá • Outras despesas administrativas",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=[prefix],
        is_expense=True,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefix),
        rep_exclude_exact=[prefix],
    )


# ==========================
# Aba 7: Despesas de PDD (282020404* + ajustes)
# ==========================
with tabs[6]:
    prefixes = ("282020404", "282020405", "282020406", "282020407")
    render_tab(
        df_source=df_resultado,
        key_prefix="tab7_pdd",
        title="Itacibá • Despesas de PDD",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=["282020404"],
        is_expense=True,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith(prefixes),
        rep_exclude_exact=[],
    )


# ==========================
# Aba 8: Lucro líquido (28210 / 28212 / 28214)
# ==========================
with tabs[7]:
    lucro_codes = {"28210", "28212", "28214"}
    render_tab(
        df_source=df_resultado,
        key_prefix="tab8_lucro",
        title="Itacibá • Lucro líquido",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_codes=["28210"],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).isin(lucro_codes),
        rep_exclude_exact=[],
    )
