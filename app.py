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
# Parser do CSV matricial (2020 a 2026)
# ==========================
@st.cache_data(show_spinner=False)
def load_report(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="latin1")
    first_col = raw.columns[0]

    starts = raw.index[raw[first_col].astype(str).str.strip().eq("Filtro selecionado")].tolist()
    if not starts:
        raise ValueError("Não encontrei blocos do relatório (linha 'Filtro selecionado').")

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
                # remove "123 - " no início, se vier embutido
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

    # corta realizado após último mês válido
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


from textwrap import dedent
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


# ==========================
# Generic filters (per tab) - sidebar
# ==========================
def build_labels(df_scope: pd.DataFrame) -> pd.DataFrame:
    produtos = df_scope[["produto_cod", "produto"]].drop_duplicates().sort_values("produto_cod")
    produtos["label"] = produtos["produto_cod"] + " - " + produtos["produto"]
    return produtos


def default_labels_for_codes(produtos_df: pd.DataFrame, codes: list[str]) -> list[str]:
    out = []
    for c in codes:
        m = produtos_df[produtos_df["produto_cod"] == str(c)]
        if not m.empty:
            out.append(m["label"].iloc[0])
    return out


def apply_period_filter(df_in: pd.DataFrame, periodo: str, ano_sel: int | None, mes_sel: int | None) -> pd.DataFrame:
    df_out = df_in.copy()
    if periodo == "Ano" and ano_sel is not None:
        df_out = df_out[df_out["ano"] == ano_sel]
    elif periodo == "Mês" and ano_sel is not None and mes_sel is not None:
        df_out = df_out[(df_out["ano"] == ano_sel) & (df_out["mes"] == mes_sel)]
    return df_out


def render_filters_sidebar(df_scope: pd.DataFrame, key_prefix: str, default_sel: list[str]):
    st.sidebar.markdown("## Filtros")
    periodo = st.sidebar.radio("Período", ["Total", "Ano", "Mês"], horizontal=True, key=f"{key_prefix}_periodo")

    anos = sorted(df_scope["ano"].unique().tolist()) if not df_scope.empty else []
    ano_sel = None
    mes_sel = None

    if periodo in ["Ano", "Mês"] and anos:
        ano_sel = st.sidebar.selectbox("Ano", anos, index=len(anos) - 1, key=f"{key_prefix}_ano")

    if periodo == "Mês" and ano_sel is not None:
        meses_no_ano = sorted(df_scope[df_scope["ano"] == ano_sel]["mes"].unique().tolist())
        mes_sel = st.sidebar.selectbox(
            "Mês",
            meses_no_ano,
            format_func=lambda m: MONTH_NUM_TO_LABEL.get(int(m), str(m)),
            key=f"{key_prefix}_mes",
        )

    produtos = build_labels(df_scope)
    prod_labels = produtos["label"].tolist()
    default_ok = [x for x in default_sel if x in prod_labels]

    prod_sel = st.sidebar.multiselect("Produto (multi)", prod_labels, default=default_ok, key=f"{key_prefix}_prod")

    # df_period: só período; df_view: período + produto
    df_period = apply_period_filter(df_scope, periodo, ano_sel, mes_sel)
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

    return df_period, df_view, prod_sel, prod_label


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


def render_pill_produtos(prod_sel: list[str]):
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


def render_tab_generic(
    df_source: pd.DataFrame,
    key_prefix: str,
    title: str,
    subtitle: str,
    default_code_list: list[str],
    is_expense: bool,
    filter_fn,
):
    df_scope = df_source.copy()
    if df_scope.empty:
        st.info("Sem dados no arquivo carregado.")
        return

    df_scope = df_scope[filter_fn(df_scope)].copy()
    if df_scope.empty:
        st.info("Este arquivo não contém dados para esta aba.")
        return

    produtos = build_labels(df_scope)
    default_sel = default_labels_for_codes(produtos, default_code_list)

    # filtros (sidebar) só dentro da aba
    df_period, df_view, prod_sel, prod_label = render_filters_sidebar(df_scope, key_prefix, default_sel)

    render_header(title, subtitle)

    # KPIs
    st.markdown('<div class="section-title">KPIs</div>', unsafe_allow_html=True)

    base_dt = last_real_month(df_view)
    real_last = value_at_month(df_view, base_dt, "realizado")
    orc_last = value_at_month(df_view, base_dt, "orcado")
    gap = real_last - orc_last

    orc_acc = acumulado(df_view, "orcado", upto_dt=None)
    real_acc = acumulado(df_view, "realizado", upto_dt=None)

    # base de farol: acumulado até o último mês com realizado
    orc_base = acumulado(df_view, "orcado", upto_dt=base_dt) if base_dt is not None else np.nan
    real_base = acumulado(df_view, "realizado", upto_dt=base_dt) if base_dt is not None else np.nan

    ok = False
    if base_dt is not None and (not np.isnan(orc_base)) and (not np.isnan(real_base)):
        # Receita/Lucro: bom >= orçado; Despesa: bom <= orçado
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

    # Charts
    render_pill_produtos(prod_sel)

    st.markdown('<div class="section-title">Evolução</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig = make_evolucao_figure(df_view, f"{title} (Orçado x Realizado)", prod_label)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


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

df_carteira = load_report(files_map[file_carteira])
df_resultado = load_report(files_map[file_resultado])


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
# Aba 1: Carteira (se existir no arquivo)
# ==========================
with tabs[0]:
    # carteira: códigos 18201 / 18202 (se o arquivo tiver)
    def carteira_filter(d: pd.DataFrame):
        c = d["produto_cod"].astype(str)
        return c.str.startswith("18201") | c.str.startswith("18202")

    render_tab_generic(
        df_source=df_carteira,
        key_prefix="tab1_carteira",
        title="Itacibá • Carteira de Crédito",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["18201", "18202"],
        is_expense=False,
        filter_fn=carteira_filter,
    )


# ==========================
# Aba 2: Receita da prestação de serviços (2820401*)
# ==========================
with tabs[1]:
    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab2_servicos",
        title="Itacibá • Receita da prestação de serviços",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["2820401"],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("2820401"),
    )


# ==========================
# Aba 3: Receitas de tesouraria (2820101*)
# ==========================
with tabs[2]:
    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab3_tesouraria",
        title="Itacibá • Receitas de tesouraria",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["2820101"],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("2820101"),
    )


# ==========================
# Aba 4: Receitas de participação das controladas (282040501*)
# ==========================
with tabs[3]:
    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab4_controladas",
        title="Itacibá • Receitas de participação das controladas",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["282040501"],
        is_expense=False,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("282040501"),
    )


# ==========================
# Aba 5: Despesas de pessoal (2820402*)
# ==========================
with tabs[4]:
    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab5_pessoal",
        title="Itacibá • Despesas de pessoal",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["2820402"],
        is_expense=True,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("2820402"),
    )


# ==========================
# Aba 6: Outras despesas administrativas (2820403*)
# ==========================
with tabs[5]:
    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab6_admin",
        title="Itacibá • Outras despesas administrativas",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["2820403"],
        is_expense=True,
        filter_fn=lambda d: d["produto_cod"].astype(str).str.startswith("2820403"),
    )


# ==========================
# Aba 7: Despesas de PDD (foco em despesas de provisão)
#   - inclui: 282020404* (despesas provisão), 282020405*, 282020406*, 282020407* (ajustes)
# ==========================
with tabs[6]:
    def pdd_filter(d: pd.DataFrame):
        c = d["produto_cod"].astype(str)
        return (
            c.str.startswith("282020404")
            | c.str.startswith("282020405")
            | c.str.startswith("282020406")
            | c.str.startswith("282020407")
        )

    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab7_pdd",
        title="Itacibá • Despesas de PDD",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["282020404"],
        is_expense=True,
        filter_fn=pdd_filter,
    )


# ==========================
# Aba 8: Lucro líquido (28210 / 28212 / 28214)
# ==========================
with tabs[7]:
    def lucro_filter(d: pd.DataFrame):
        c = d["produto_cod"].astype(str)
        return c.isin(["28210", "28212", "28214"])

    render_tab_generic(
        df_source=df_resultado,
        key_prefix="tab8_lucro",
        title="Itacibá • Lucro líquido",
        subtitle="Orçado x Realizado • 2020 a 2026 • filtros por período e produto",
        default_code_list=["28210"],
        is_expense=False,
        filter_fn=lucro_filter,
    )
