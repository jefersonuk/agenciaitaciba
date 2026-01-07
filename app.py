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
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "-", "—"):
        return np.nan
    s = s.replace("%", "").strip()
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


def _base_year_from_col(col: str) -> int | None:
    """
    Detecta ano em colunas como:
    '2025', '2026', '2025.1' (quando pandas mangla duplicadas), ' 2026 ' etc.
    """
    if col is None:
        return None
    s = str(col).strip()
    m = re.match(r"^(\d{4})(?:\.\d+)?$", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê o CSV matricial (anos em blocos por colunas).
    Espera:
    - Linha 0: meses ('jan','fev',...)
    - Linha 1: rótulos (Orçado/Realizado/Var...)
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

    # acha inícios de blocos por ano, preservando ordem no arquivo
    year_first_idx: dict[int, int] = {}
    year_order: list[tuple[int, int]] = []
    for idx, c in enumerate(cols):
        y = _base_year_from_col(c)
        if y is None:
            continue
        if y not in year_first_idx:
            year_first_idx[y] = idx
            year_order.append((y, idx))

    if not year_order:
        raise ValueError("Não encontrei colunas de ano (ex: 2025/2026) no CSV.")

    year_order.sort(key=lambda t: t[1])

    records: list[dict] = []

    for i, (y, start) in enumerate(year_order):
        end = year_order[i + 1][1] if i + 1 < len(year_order) else len(cols)

        # colunas base do bloco
        code_col = cols[start]
        desc_col = cols[start + 1] if (start + 1) < end else None
        if desc_col is None:
            continue

        # Monta mapa de meses lendo:
        # - linha 0: mês
        # - linha 1: tipo da coluna (orçado/realizado)
        month_map: dict[str, dict[str, str]] = {}
        for j in range(start + 2, end):
            mon_lbl = normalize_month_label(raw.iloc[0, j])  # linha 0
            if mon_lbl is None:
                continue

            hdr = str(raw.iloc[1, j]).strip().lower()  # linha 1
            if any(k in hdr for k in ["orçado", "orcado", "orç", "orc"]):
                month_map.setdefault(mon_lbl, {})["orcado"] = cols[j]
            elif "real" in hdr:
                month_map.setdefault(mon_lbl, {})["realizado"] = cols[j]
            else:
                # ignora variações (var r$, var %, etc.)
                continue

        # se não achou nada (export diferente), faz fallback simples:
        # tenta inferir por padrão de 4 colunas
        if not month_map:
            j = start + 2
            while j < end:
                mon_lbl = normalize_month_label(raw.iloc[0, j])
                if mon_lbl is None:
                    j += 1
                    continue
                orc_col = cols[j]
                rea_col = cols[j + 1] if (j + 1) < end else None
                month_map.setdefault(mon_lbl, {})["orcado"] = orc_col
                if rea_col is not None:
                    month_map.setdefault(mon_lbl, {})["realizado"] = rea_col
                j += 4

        # ordena meses
        month_blocks: list[tuple[str, str, str | None]] = []
        for mon_lbl in sorted(month_map.keys(), key=lambda m: PT_MONTH[m]):
            orc_col = month_map[mon_lbl].get("orcado")
            rea_col = month_map[mon_lbl].get("realizado")
            if not orc_col:
                continue
            month_blocks.append((mon_lbl, orc_col, rea_col))

        data_rows = raw.iloc[2:, start:end]

        for _, r in data_rows.iterrows():
            code = str(r.get(code_col, "")).strip()
            if code.lower() in ("nan", "none") or code == "":
                continue

            desc = str(r.get(desc_col, "")).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            for mon_lbl, orc_col, rea_col in month_blocks:
                dt_ = datetime(y, PT_MONTH[mon_lbl], 1)
                orc = parse_ptbr_number(r.get(orc_col))
                rea = parse_ptbr_number(r.get(rea_col)) if rea_col else np.nan

                records.append(
                    {
                        "data": pd.to_datetime(dt_),
                        "ano": int(y),
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
    if df.empty:
        return pd.DataFrame(columns=["data", "ano", "mes", "produto_cod", "produto", "tipo", "orcado", "realizado"])

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

    # SALDO
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
            out["saldo_ok"] = (
                pd.notna(out["saldo_real_last"])
                and pd.notna(out["saldo_orc_same"])
                and out["saldo_real_last"] >= out["saldo_orc_same"]
            )

    # RENDAS
    rendas = df_view[df_view["tipo"] == "rendas"].copy()
    if not rendas.empty:
        out["rendas_orc_sum"] = float(rendas["orcado"].sum(skipna=True))
        out["rendas_real_sum"] = float(rendas["realizado"].sum(skipna=True))
        out["rendas_ok"] = (
            (out["rendas_real_sum"] >= out["rendas_orc_sum"])
            if (pd.notna(out["rendas_real_sum"]) and pd.notna(out["rendas_orc_sum"]))
            else None
        )

        ts_r = build_timeseries(df_view, "rendas")
        last_dt_r = last_nonzero_date(ts_r["realizado"], ts_r["data"]) if not ts_r.empty else None
        if last_dt_r is not None:
            out["rendas_base_txt"] = f"{MONTH_NUM_TO_LABEL[int(last_dt_r.month)]}/{int(last_dt_r.year)}"

    return out


def top5_representatividade_rendas(df_period: pd.DataFrame) -> pd.DataFrame:
    """
    Representatividade por produtos de RENDAS (Top5 + Outros).
    - aplica SOMENTE o período (df_period)
    - ignora filtro de produto para composição da carteira
    - exclui TOTAL 18202
    - base: Realizado se houver, senão Orçado
    """
    x = df_period.copy()
    if x.empty:
        return pd.DataFrame()

    x["produto_cod"] = x["produto_cod"].astype(str)
    x = x[x["produto_cod"].str.startswith("18202")]
    x = x[x["produto_cod"] != "18202"]

    if x.empty:
        return pd.DataFrame()

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
        rows.append(
            {
                "produto": r["label"],
                "valor": float(r["valor"]),
                "share": float(r["valor"] / total),
                "metric": metric,
            }
        )

    if not rest.empty:
        v = float(rest["valor"].sum())
        rows.append({"produto": "Outros", "valor": v, "share": float(v / total), "metric": metric})

    rep = pd.DataFrame(rows)

    rep["is_outros"] = (rep["produto"] == "Outros").astype(int)
    rep = rep.sort_values(["is_outros", "valor"], ascending=[True, False]).drop(columns=["is_outros"])
    return rep


def representatividade_figure(rep: pd.DataFrame) -> go.Figure:
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

    return fig


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
produtos = df[["produto_cod", "produto"]].drop_duplicates().sort_values(["produto_cod"])
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
        format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m)),
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
# KPIs
# ==========================================================
st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)
k = compute_kpis(df_view=df_view, df_period=df_period)

base_saldo_txt = "—"
if k["saldo_base_dt"] is not None:
    base_saldo_txt = f"{MONTH_NUM_TO_LABEL[int(k['saldo_base_dt'].month)]}/{int(k['saldo_base_dt'].year)}"

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
# Gráficos
# ==========================================================
st.markdown('<div class="section-title">Evolução • Saldo da Carteira</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="pill"><span style="opacity:.8">Legenda:</span> <b>{prod_label}</b></div>',
    unsafe_allow_html=True,
)

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
# Representatividade
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
