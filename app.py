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


inject_css()
px.defaults.template = make_plotly_template()

st.markdown(
    f"""
<style>
/* =========================================================
   SIDEBAR — visual premium (cards + contraste)
   ========================================================= */

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)) !important;
  border-right: 1px solid {BRAND["border"]} !important;
}}

section[data-testid="stSidebar"] * {{
  color: {BRAND["ink"]};
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {{
  color: {BRAND["muted"]} !important;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
  color: {BRAND["ink"]} !important;
  letter-spacing: -0.2px;
}}

section[data-testid="stSidebar"] .sb-title {{
  font-size: 14px;
  font-weight: 900;
  color: {BRAND["ink"]};
  margin: 0 0 10px 0;
}}

section[data-testid="stSidebar"] .sb-card {{
  border: 1px solid {BRAND["border"]};
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border-radius: 16px;
  padding: 12px 12px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
  margin-bottom: 12px;
}}

section[data-testid="stSidebar"] .sb-card .sb-card-title {{
  font-size: 12px;
  font-weight: 900;
  letter-spacing: -0.2px;
  color: {BRAND["ink"]};
  opacity: .95;
  margin-bottom: 10px;
}}

section[data-testid="stSidebar"] .sb-helper {{
  font-size: 11px;
  color: {BRAND["muted"]};
  opacity: .9;
  margin-top: 6px;
}}

/* Uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {{
  border-radius: 14px !important;
  border: 1px dashed rgba(242,246,255,0.22) !important;
  background: rgba(0,0,0,0.18) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
  border-radius: 12px !important;
  border: 1px solid {BRAND["border"]} !important;
  background: rgba(255,255,255,0.06) !important;
  color: {BRAND["ink"]} !important;
}}

/* Inputs (Selectbox / Multiselect) baseweb */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  border-radius: 14px !important;
  border: 1px solid {BRAND["border"]} !important;
  background: rgba(0,0,0,0.20) !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
  color: {BRAND["ink"]} !important;
}}

/* Tags do multiselect (chips) */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
  border-radius: 999px !important;
  background: rgba(43,89,255,0.25) !important;
  border: 1px solid rgba(43,89,255,0.35) !important;
}}
section[data-testid="stSidebar"] span[data-baseweb="tag"] * {{
  color: {BRAND["ink"]} !important;
  font-weight: 800 !important;
}}

/* Radio “Período” */
section[data-testid="stSidebar"] [role="radiogroup"] {{
  background: rgba(0,0,0,0.18);
  border: 1px solid {BRAND["border"]};
  border-radius: 14px;
  padding: 10px 10px;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  font-weight: 800;
  color: {BRAND["ink"]} !important;
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
def apply_dark_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=BRAND["muted"], size=12),
        ),
        font=dict(color=BRAND["ink"]),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.update_xaxes(
        color=BRAND["muted"],
        gridcolor=BRAND["grid"],
        zerolinecolor=BRAND["grid"],
        tickfont=dict(color=BRAND["muted"]),
        titlefont=dict(color=BRAND["muted"]),
    )
    fig.update_yaxes(
        color=BRAND["muted"],
        gridcolor=BRAND["grid"],
        zerolinecolor=BRAND["grid"],
        tickfont=dict(color=BRAND["muted"]),
        titlefont=dict(color=BRAND["muted"]),
    )
    return fig   # <- TEM que ser isso (não pode chamar a função de novo)




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
    from io import BytesIO

    bio = BytesIO(file_bytes)

    # 1) tenta ler normalmente (detecção automática)
    raw = pd.read_csv(bio, header=None, dtype=str, encoding="latin1", sep=None, engine="python")

    # 2) se veio “tudo em 1 coluna”, tenta resolver (delimitador ; ou ,)
    if raw.shape[1] == 1:
        s = raw.iloc[:, 0].astype(str)
        # escolhe o delimitador mais provável olhando a 1ª linha
        first = s.iloc[0]
        cand = ";" if first.count(";") >= first.count(",") else ","
        raw = s.str.split(cand, expand=True)

    # remove linhas/colunas 100% vazias
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

    cols_final = ["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"]
    if raw.empty or raw.shape[1] < 3:
        return pd.DataFrame(columns=cols_final)

    # ---------------- helpers ----------------
    def find_year_start(dfmat: pd.DataFrame, year: int) -> int | None:
        y = str(year)
        for i in range(min(len(dfmat), 200)):
            row_txt = " ".join(dfmat.iloc[i].astype(str).tolist()).lower()
            if y in row_txt:
                return i
        return None

    def find_header_row_with_months(dfmat: pd.DataFrame) -> int | None:
        # procura, nas primeiras 30 linhas do bloco, a linha com mais meses detectados
        best_i, best_score = None, 0
        lim = min(len(dfmat), 30)
        for i in range(lim):
            cells = dfmat.iloc[i].tolist()
            score = sum(1 for c in cells if normalize_month_label(c) is not None)
            if score > best_score:
                best_score = score
                best_i = i
        return best_i if (best_i is not None and best_score >= 3) else None

    def parse_part(dfmat: pd.DataFrame, year: int, has_realizado: bool) -> pd.DataFrame:
        if dfmat.empty:
            return pd.DataFrame(columns=cols_final)

        h = find_header_row_with_months(dfmat)
        if h is None:
            return pd.DataFrame(columns=cols_final)

        header = dfmat.iloc[h].tolist()

        # mapeia meses -> colunas
        month_cols = []
        for j, cell in enumerate(header):
            mon = normalize_month_label(cell)
            if mon is None:
                continue
            orc_col = j + 1
            rea_col = j + 2 if has_realizado else None
            if orc_col < dfmat.shape[1]:
                month_cols.append((mon, orc_col, rea_col))

        if not month_cols:
            return pd.DataFrame(columns=cols_final)

        data_rows = dfmat.iloc[h + 1 :].copy()

        recs = []
        for _, r in data_rows.iterrows():
            code = str(r.iloc[0]).strip()
            if code.lower() in ("nan", "none", ""):
                continue

            desc = str(r.iloc[1]).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            for mon, orc_col, rea_col in month_cols:
                orc = parse_ptbr_number(r.iloc[orc_col]) if orc_col is not None else np.nan
                rea = (
                    parse_ptbr_number(r.iloc[rea_col])
                    if (has_realizado and rea_col is not None and rea_col < dfmat.shape[1])
                    else np.nan
                )

                # não cria linha toda vazia
                if (orc is None or (isinstance(orc, float) and np.isnan(orc))) and (rea is None or (isinstance(rea, float) and np.isnan(rea))):
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
                        "realizado": rea,
                    }
                )

        out = pd.DataFrame(recs)
        return out if not out.empty else pd.DataFrame(columns=cols_final)

    # ---------------- split 2025 / 2026 ----------------
    i25 = find_year_start(raw, 2025)
    i26 = find_year_start(raw, 2026)

    # se não achar “2025” explícito, assume tudo como bloco 2025
    if i25 is None:
        i25 = 0

    if i26 is None:
        block25 = raw.iloc[i25:].copy()
        df_2025 = parse_part(block25, 2025, has_realizado=True)
        df = df_2025
    else:
        block25 = raw.iloc[i25:i26].copy()
        block26 = raw.iloc[i26:].copy()

        df_2025 = parse_part(block25, 2025, has_realizado=True)
        df_2026 = parse_part(block26, 2026, has_realizado=False)
        df = pd.concat([df_2025, df_2026], ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=cols_final)

    # limpeza final
    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df = df.sort_values(["data", "produto_cod"]).reset_index(drop=True)

    return df


    # --- separa 2025 e 2026 pelo marcador ---
    idx_2026 = find_year_row(raw, 2026)

    if idx_2026 is None:
        # não achou o bloco 2026 no CSV -> retorna só 2025
        df_2025 = parse_part(raw, 2025, has_realizado=True)
        df = df_2025
    else:
        raw_2025 = raw.iloc[:idx_2026].copy()
        raw_2026 = raw.iloc[idx_2026:].copy()

        # às vezes a linha do "2026" é só um título. Se tiver pouca info, pula uma linha.
        # (heurística: se a "linha 0" do bloco 2026 não contém meses, tenta usar a próxima)
        header_2026 = raw_2026.iloc[0].tolist()
        has_months_2026 = any(normalize_month_label(x) is not None for x in header_2026)
        if not has_months_2026 and len(raw_2026) > 1:
            raw_2026 = raw_2026.iloc[1:].copy()

        df_2025 = parse_part(raw_2025, 2025, has_realizado=True)
        df_2026 = parse_part(raw_2026, 2026, has_realizado=False)

        df = pd.concat([df_2025, df_2026], ignore_index=True)

    # limpeza final
    if df.empty:
        return pd.DataFrame(columns=["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"])

    df["produto_cod"] = df["produto_cod"].astype(str)
    df["produto"] = df["produto"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df = df.sort_values(["data", "produto_cod"]).reset_index(drop=True)

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

    # garante tipos
    x["data"] = pd.to_datetime(x["data"], errors="coerce")

    # agrega por mês
    ts = (
        x.groupby("data", as_index=False)[["orcado", "realizado"]]
        .sum(min_count=1)
        .sort_values("data")
    )

    # acha o último mês com realizado "concreto" (não nulo e != 0)
    mask_real = ts["realizado"].notna() & (ts["realizado"] != 0)
    last_real_dt = ts.loc[mask_real, "data"].max()

    # IMPORTANTÍSSIMO: não corta a série (pra não perder 2026),
    # só "remove" realizado após o último mês com dado
    if pd.notna(last_real_dt):
        ts.loc[ts["data"] > last_real_dt, "realizado"] = np.nan

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

    return apply_dark_plotly(fig)



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
    """
    KPIs:
    - Saldo Realizado (último mês com dado) + farol vs Orçado (mesmo mês)
    - Gap Saldo (Real - Orc) no mês-base
    - Rendas Orçado acumulado (recorte)
    - Rendas Realizado acumulado (recorte) + farol vs orçado acumulado
    """
    out = {
        "saldo_real_last": np.nan,
        "saldo_orc_same": np.nan,
        "saldo_gap": np.nan,
        "saldo_base_dt": None,
        "saldo_ok": None,
        "rendas_orc_sum": np.nan,
        "rendas_real_sum": np.nan,
        "rendas_ok": None,
        "rendas_base_txt": None,
    }

    # SALDO: usa df_view (com filtro de produto) mas somente tipo saldo
    saldo = df_view[df_view["tipo"] == "saldo"].copy()
    if not saldo.empty:
        by_m = (
            saldo.groupby("data", as_index=False)[["orcado", "realizado"]]
            .sum(min_count=1)
            .sort_values("data")
        )
        last_dt = last_nonzero_date(by_m["realizado"], by_m["data"])
        if last_dt is not None:
            row = by_m[by_m["data"] == last_dt].iloc[0]
            out["saldo_real_last"] = float(row["realizado"]) if pd.notna(row["realizado"]) else np.nan
            out["saldo_orc_same"] = float(row["orcado"]) if pd.notna(row["orcado"]) else np.nan
            out["saldo_gap"] = out["saldo_real_last"] - out["saldo_orc_same"]
            out["saldo_base_dt"] = last_dt
            out["saldo_ok"] = (pd.notna(out["saldo_real_last"]) and pd.notna(out["saldo_orc_same"]) and out["saldo_real_last"] >= out["saldo_orc_same"])

    # RENDAS: acumulado no recorte (df_view, tipo rendas)
    rendas = df_view[df_view["tipo"] == "rendas"].copy()
    if not rendas.empty:
        out["rendas_orc_sum"] = float(rendas["orcado"].sum(skipna=True))
        out["rendas_real_sum"] = float(rendas["realizado"].sum(skipna=True))
        out["rendas_ok"] = (out["rendas_real_sum"] >= out["rendas_orc_sum"]) if (pd.notna(out["rendas_real_sum"]) and pd.notna(out["rendas_orc_sum"])) else None

        # texto base: último mês com realizado no recorte (para contexto)
        ts_r = build_timeseries(df_view, "rendas")
        last_dt_r = last_nonzero_date(ts_r["realizado"], ts_r["data"]) if not ts_r.empty else None
        if last_dt_r is not None:
            out["rendas_base_txt"] = f"{MONTH_NUM_TO_LABEL[int(last_dt_r.month)]}/{int(last_dt_r.year)}"

    return out


def top5_representatividade_rendas(df_period: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula representatividade por produtos de RENDAS (Top5 + Outros).
    - aplica SOMENTE o período (df_period)
    - ignora filtro de produto para mostrar composição do resultado de crédito
    - exclui TOTAL 18202 (para não distorcer)
    - base: Realizado se houver, senão Orçado
    """
    x = df_period.copy()
    if x.empty:
        return pd.DataFrame()

    x["produto_cod"] = x["produto_cod"].astype(str)
    x = x[x["produto_cod"].str.startswith("18202")]
    x = x[x["produto_cod"] != "18202"]  # exclui total

    if x.empty:
        return pd.DataFrame()

    # define base métrica
    has_real = (x["realizado"].notna() & (x["realizado"].abs() > 0)).any()
    metric = "realizado" if has_real else "orcado"

    agg = (
        x.groupby(["produto_cod", "produto"], as_index=False)[metric]
        .sum(min_count=1)
        .rename(columns={metric: "valor"})
    )

    agg = agg[agg["valor"].notna()]
    agg = agg[agg["valor"] > 0]
    if agg.empty:
        return pd.DataFrame()

    agg["label"] = agg["produto_cod"] + " - " + agg["produto"]
    total = float(agg["valor"].sum())

    agg = agg.sort_values("valor", ascending=False)

    top = agg.head(5).copy()
    rest = agg.iloc[5:].copy()

    rows = []
    for _, r in top.iterrows():
        rows.append({"produto": r["label"], "valor": float(r["valor"]), "share": float(r["valor"] / total), "metric": metric})

    if not rest.empty:
        v = float(rest["valor"].sum())
        rows.append({"produto": "Outros", "valor": v, "share": float(v / total), "metric": metric})

    rep = pd.DataFrame(rows)

    # ordenação: maiores primeiro, e "Outros" por último
    rep["is_outros"] = (rep["produto"] == "Outros").astype(int)
    rep = rep.sort_values(["is_outros", "valor"], ascending=[True, False]).drop(columns=["is_outros"])

    return rep


def representatividade_figure(rep: pd.DataFrame) -> go.Figure:
    """
    Visual moderno:
    - barras horizontais com destaque gradual (Top)
    - "Outros" neutro
    - labels com % e valor abreviado
    """
    def human(v: float) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        av = abs(v)
        if av >= 1_000_000:
            return f"{v/1_000_000:.1f}M".replace(".", ",")
        if av >= 1_000:
            return f"{v/1_000:.0f}k".replace(".", ",")
        return f"{v:.0f}".replace(".", ",")

    metric_lbl = "Realizado" if rep["metric"].iloc[0] == "realizado" else "Orçado"

    rep_plot = rep.iloc[::-1].copy()
    rep_plot["pct"] = (rep_plot["share"] * 100).round(1)
    rep_plot["label"] = rep_plot.apply(lambda r: f"{r['pct']:.1f}% • {human(float(r['valor']))}", axis=1)

    # paleta leve (menos pesado): degrade do azul para roxo, com "Outros" neutro
    grad = ["#2B59FF", "#3550FF", "#3D47FF", "#453DFF", "#4D33FF", "#5B2BD6"]
    colors = []
    k = 0
    for p in rep_plot["produto"].tolist():
        if p == "Outros":
            colors.append("rgba(255,255,255,0.16)")
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
                          + f"Rendas ({metric_lbl}): "
                          + "%{x:,.2f}<br>"
                          + "Share: %{customdata:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=14, r=18, t=10, b=14),
        showlegend=False,
        xaxis=dict(
            title="",
            gridcolor=BRAND["grid"],
            zeroline=False,
            tickfont=dict(color=BRAND["muted"]),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=BRAND["ink"]),
        ),
    )

    xmax = float(rep_plot["valor"].max()) if len(rep_plot) else 0
    fig.update_xaxes(range=[0, xmax * 1.18 if xmax > 0 else 1])

    return apply_dark_plotly(fig)



# ==========================================================
# Sidebar: upload + filtros
# ==========================================================
st.sidebar.markdown("## Dados")
uploaded = st.sidebar.file_uploader("Anexe o relatório (CSV exportado)", type=["csv"])

if uploaded is None:
    st.sidebar.info("Envie o CSV do relatório para carregar os dados.")
    st.stop()

df = load_report(uploaded.getvalue())

# lista de produtos
produtos = (
    df[["produto_cod", "produto"]]
    .drop_duplicates()
    .sort_values(["produto_cod"])
)
produtos["label"] = produtos["produto_cod"] + " - " + produtos["produto"]
prod_labels = produtos["label"].tolist()

# defaults: 18201 e 18202 se existirem
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
        format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m))
    )

prod_sel = st.sidebar.multiselect("Produto (multi)", prod_labels, default=default_sel)

# ==========================================================
# df_period: aplica SOMENTE o filtro de período (sem produto)
# ==========================================================
df_period = df.copy()
if periodo == "Ano":
    df_period = df_period[df_period["ano"] == ano_sel]
elif periodo == "Mês":
    df_period = df_period[(df_period["ano"] == ano_sel) & (df_period["mes"] == mes_sel)]

# ==========================================================
# df_view: aplica período + produto
# ==========================================================
df_view = df_period.copy()
if prod_sel:
    cod_sel = [p.split(" - ")[0].strip() for p in prod_sel]
    df_view = df_view[df_view["produto_cod"].isin(cod_sel)]

# label para legenda
if not prod_sel:
    prod_label = "Todos"
else:
    if len(prod_sel) == 1:
        prod_label = prod_sel[0].split(" - ", 1)[1][:28]
    else:
        prod_label = f"Seleção ({len(prod_sel)})"


# ==========================================================
# Header
# ==========================================================
st.markdown(
    f"""
<div class="header-wrap">
  <div>
    <div class="header-title">Itacibá • Carteira de Crédito</div>
    <div class="header-sub">Orçado x Realizado • filtros por período e produto</div>
  </div>
  <div class="legend-pill">
    <span><span class="dot" style="background:{BRAND["blue"]}"></span> Orçado</span>
    <span><span class="dot" style="background:{BRAND["green"]}"></span> Realizado</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

# ==========================================================
# KPIs (em blocos, 4 colunas)
# ==========================================================
st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
k = compute_kpis(df_view=df_view, df_period=df_period)

base_saldo_txt = "—"
if k["saldo_base_dt"] is not None:
    base_saldo_txt = f"{MONTH_NUM_TO_LABEL[int(k['saldo_base_dt'].month)]}/{int(k['saldo_base_dt'].year)}"

# faróis
saldo_farol_html = ""
if k["saldo_ok"] is not None:
    saldo_farol_html = farol_badge(bool(k["saldo_ok"]))

rendas_farol_html = ""
if k["rendas_ok"] is not None:
    rendas_farol_html = farol_badge(bool(k["rendas_ok"]))

c1, c2, c3, c4 = st.columns(4, gap="large")

with c1:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Saldo • Realizado (último mês com dado)</div>
  <div class="kpi-value">{fmt_br(k["saldo_real_last"])}</div>
  {saldo_farol_html}
  <div class="kpi-sub">Base: {base_saldo_txt}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Saldo • Gap vs Orçado (mesmo mês)</div>
  <div class="kpi-value">{fmt_br(k["saldo_gap"])}</div>
  <div class="kpi-sub">Comparação no mês-base do realizado.</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Rendas • Orçado (acumulado no recorte)</div>
  <div class="kpi-value">{fmt_br(k["rendas_orc_sum"])}</div>
  <div class="kpi-sub">Base do farol: {k["rendas_base_txt"] or "—"}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Rendas • Realizado (acumulado no recorte)</div>
  <div class="kpi-value">{fmt_br(k["rendas_real_sum"])}</div>
  {rendas_farol_html}
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ==========================================================
# Gráficos: Saldo / Rendas (um embaixo do outro)
# ==========================================================
st.markdown('<div class="section-title">Evolução • Saldo da Carteira</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="pill"><span style="opacity:.8">Legenda:</span> <b>{prod_label}</b></div>',
    unsafe_allow_html=True,
)
with st.expander("DEBUG (remover depois)"):
    st.write("Anos no df:", sorted(df["ano"].dropna().unique().tolist()))
    st.write("Anos no df_view:", sorted(df_view["ano"].dropna().unique().tolist()))
    st.write("Linhas 2026 no df_view:", int((df_view["ano"] == 2026).sum()))
    st.dataframe(df_view[df_view["ano"] == 2026].head(10))
ts_saldo = build_timeseries(df_view, "saldo")
if ts_saldo.empty:
    st.info("Sem dados de Saldo para o recorte atual.")
else:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(
        fig_orcado_bar_real_line(ts_saldo, "Saldo (Orçado x Realizado)", prod_label),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Evolução • Rendas da Carteira</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="pill"><span style="opacity:.8">Legenda:</span> <b>{prod_label}</b></div>',
    unsafe_allow_html=True,
)

ts_rendas = build_timeseries(df_view, "rendas")
if ts_rendas.empty:
    st.info("Sem dados de Rendas para o recorte atual.")
else:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(
        fig_orcado_bar_real_line(ts_rendas, "Rendas (Orçado x Realizado)", prod_label),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ==========================================================
# Representatividade (Top 5 + Outros) - Rendas
# ==========================================================
st.markdown('<div class="section-title">Representatividade • Produtos (Rendas)</div>', unsafe_allow_html=True)

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
