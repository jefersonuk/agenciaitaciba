import re
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ==========================================================
# Config
# ==========================================================
st.set_page_config(
    page_title="Itacibá • Carteira de Crédito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Identidade (Banestes)
BRAND = {
    "blue": "#1E0AE8",     # Pantone 2728C
    "green": "#00AB16",    # Pantone 2423C
    "bg": "#070B18",
    "bg2": "#050714",
    "card": "rgba(255,255,255,0.06)",
    "card2": "rgba(255,255,255,0.04)",
    "border": "rgba(255,255,255,0.10)",
    "grid": "rgba(255,255,255,0.06)",
    "ink": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.72)",
    "muted2": "rgba(255,255,255,0.55)",
    "danger": "#FF4D6D",
    "ok": "#27D17F",
}

PT_MONTH = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

MONTH_NUM_TO_LABEL = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}


def inject_css() -> None:
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"] {{

  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
}}

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

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
  border-right: 1px solid {BRAND["border"]} !important;
}}

.block-container {{
  padding-top: 1.1rem;
  padding-bottom: 2.2rem;
  max-width: 1200px;
}}

h1, h2, h3 {{
  font-family: Poppins, Inter, sans-serif;
  letter-spacing: -0.3px;
}}

.small-muted {{
  opacity: .75;
  font-size: 12px;
}}

.header-wrap {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border-radius: 18px;
  padding: 18px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 14px 40px rgba(0,0,0,0.35);
}}

.header-title {{
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.25px;
  line-height: 1.1;
}}

.header-sub {{
  font-size: 13px;
  opacity: .74;
  margin-top: 4px;
}}

.legend-pill {{
  border: 1px solid {BRAND["border"]};
  background: rgba(255,255,255,0.05);
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 13px;
  display: inline-flex;
  gap: 12px;
  align-items:center;
}}

.dot {{
  width: 10px; height: 10px;
  border-radius: 999px;
  display:inline-block;
}}

.section-title {{
  font-size: 18px;
  font-weight: 800;
  margin: 16px 0 10px 0;
  letter-spacing: -0.2px;
}}

.kpi-card {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 12px 35px rgba(0,0,0,0.35);
  min-height: 110px;
}}

.kpi-label {{
  font-size: 12px;
  opacity: .80;
  font-weight: 700;
}}

.kpi-value {{
  font-family: Poppins, Inter, sans-serif;
  font-size: 22px;
  font-weight: 800;
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
  font-weight: 800;
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
  box-shadow: 0 12px 35px rgba(0,0,0,0.35);
}}

hr {{
  border: none;
  border-top: 1px solid {BRAND["border"]};
  margin: 14px 0;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def make_plotly_template() -> go.layout.Template:
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=BRAND["ink"]),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=BRAND["muted"]),
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=BRAND["grid"],
                zeroline=False,
                tickfont=dict(color=BRAND["muted"]),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=BRAND["grid"],
                zeroline=False,
                tickfont=dict(color=BRAND["muted"]),
            ),
        )
    )


px.defaults.template = make_plotly_template()


# ==========================================================
# Helpers
# ==========================================================
def fmt_br(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


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


def parse_ptbr_number(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        # mantém floats; converte 0 como 0
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "-", "—"):
        return np.nan
    # remove % e espaços
    s = s.replace("%", "").strip()
    # pt-br: 1.234.567,89
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def type_from_code(code: str) -> str:
    c = str(code)
    if c.startswith("18201"):
        return "saldo"
    if c.startswith("18202"):
        return "rendas"
    return "outros"


@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê o CSV matricial (anos em blocos por colunas).
    Espera:
    - Linha 0: meses ('jan','fev',...)
    - Linha 1: rótulos de colunas (Orçado/Realizado/Var...)
    - A partir da linha 2: dados
    """
    bio = BytesIO(file_bytes)

    # tenta ler como ; (padrão export) / fallback
    try:
        raw = pd.read_csv(bio, sep=";", engine="python", encoding="latin1")
    except Exception:
        bio.seek(0)
        raw = pd.read_csv(bio, sep=None, engine="python", encoding="latin1")

    cols = [str(c) for c in raw.columns]

    # anos = colunas que são números (ex: '2025','2026','2024'...)
    year_cols = []
    for c in cols:
        if re.fullmatch(r"\d{4}", str(c).strip()):
            year_cols.append(str(c).strip())

    if not year_cols:
        raise ValueError("Não encontrei colunas de ano (ex: 2025/2026) no CSV.")

    # ordena por posição no arquivo (ordem visual)
    year_cols = sorted(year_cols, key=lambda y: cols.index(y))

    records = []

    for i, year_str in enumerate(year_cols):
        y = int(year_str)
        start = cols.index(year_str)
        end = cols.index(year_cols[i + 1]) if i + 1 < len(year_cols) else len(cols)

        code_col = cols[start]
        desc_col = cols[start + 1] if (start + 1) < end else None

        if desc_col is None:
            continue

        # colunas de meses começam em start+2, em blocos de 4
        month_blocks = []
        j = start + 2
        while j + 1 < end:
            mon_lbl = normalize_month_label(raw.iloc[0, j])  # linha 0
            if mon_lbl is None:
                j += 1
                continue
            orc_col = cols[j]
            rea_col = cols[j + 1]  # costuma ser Realizado
            month_blocks.append((mon_lbl, orc_col, rea_col))
            j += 4  # pula Var(R$) e Var(%)

        data_rows = raw.iloc[2:, start:end]

        for _, r in data_rows.iterrows():
            code = str(r[code_col]).strip()
            if code.lower() in ("nan", "none") or code == "":
                continue

            desc = str(r[desc_col]).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            for mon_lbl, orc_col, rea_col in month_blocks:
                dt_ = datetime(y, PT_MONTH[mon_lbl], 1)
                orc = parse_ptbr_number(r.get(orc_col))
                rea = parse_ptbr_number(r.get(rea_col))

                records.append(
                    {
                        "data": pd.to_datetime(dt_),
                        "ano": y,
                        "mes": PT_MONTH[mon_lbl],
                        "produto_cod": str(code),
                        "produto": desc,
                        "tipo": type_from_code(code),
                        "orcado": orc,
                        "realizado": rea,
                    }
                )

    df = pd.DataFrame(records)

    # limpeza final
    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)

    # remove linhas totalmente vazias (sem orçado e sem realizado)
    df = df[~(df["orcado"].isna() & df["realizado"].isna())].copy()

    df.sort_values(["data", "produto_cod"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def last_nonzero_date(s: pd.Series, dates: pd.Series) -> pd.Timestamp | None:
    mask = s.notna() & (s.abs() > 0)
    if not mask.any():
        return None
    return dates.loc[mask].max()


def build_timeseries(df_view: pd.DataFrame, tipo: str) -> pd.DataFrame:
    x = df_view[df_view["tipo"] == tipo].copy()
    if x.empty:
        return pd.DataFrame(columns=["data", "orcado", "realizado"])

    ts = (
        x.groupby("data", as_index=False)[["orcado", "realizado"]]
        .sum(min_count=1)
        .sort_values("data")
    )

    # quebra linha do realizado após último mês com dado concreto (não desce para zero)
    last_dt = last_nonzero_date(ts["realizado"], ts["data"])
    if last_dt is not None:
        ts.loc[ts["data"] > last_dt, "realizado"] = np.nan
        # se houver mês "zerado" depois (ex: Dez/2025 = 0), também vira NaN
        ts.loc[(ts["data"] > last_dt) & (ts["realizado"] == 0), "realizado"] = np.nan

    return ts


def fig_orcado_bar_real_line(ts: pd.DataFrame, title: str, prod_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=ts["data"],
            y=ts["orcado"],
            name=f"Orçado • {prod_label}",
            marker=dict(
                color=BRAND["blue"],
                opacity=0.75,
                line=dict(width=0),
            ),
            hovertemplate="%{x|%b/%Y}<br><b>Orçado</b>: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ts["data"],
            y=ts["realizado"],
            name=f"Realizado • {prod_label}",
            mode="lines+markers",
            line=dict(color=BRAND["green"], width=3),
            marker=dict(size=6),
            connectgaps=False,
            hovertemplate="%{x|%b/%Y}<br><b>Realizado</b>: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, family="Poppins")),
        bargap=0.18,
        height=420,
    )

    fig.update_xaxes(
        tickformat="%b\n%Y",
        showgrid=False,
    )
    fig.update_yaxes(
        title="Valor",
        titlefont=dict(color=BRAND["muted"]),
        showgrid=True,
        gridcolor=BRAND["grid"],
    )

    return fig


def farol_badge(ok: bool) -> str:
    color = BRAND["ok"] if ok else BRAND["danger"]
    label = "Cumprido" if ok else "Não cumprido"
    return f"""
<div class="badge">
  <span class="dot" style="background:{color}"></span>
  <span style="color:{color}">{label}</span>
</div>
"""


def compute_kpis(df_view: pd.DataFrame, df_period: pd.DataFrame) -> dict:
    out = {
        "saldo_real_last": np.nan,
        "saldo_orc_same": np.nan,
        "saldo_gap": np.nan,
       
