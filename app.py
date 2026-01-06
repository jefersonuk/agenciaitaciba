import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# BANESTES DESIGN TOKENS (identidade visual)
# ==========================================================
BRAND = {
    "blue": "#1E0AE8",     # Azul (Pantone 2728 C na referência enviada)
    "green": "#00AB16",    # Verde (Pantone 2423 C na referência enviada)
    "ink": "#0B1220",
    "muted": "#64748B",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "border": "rgba(15, 23, 42, 0.08)",
    "grid": "rgba(15, 23, 42, 0.06)",
}

DISCRETE = [BRAND["blue"], BRAND["green"], "#6D5BFF", "#2DD4BF", "#A78BFA", "#60A5FA"]
CONT_SCALE = [[0.0, BRAND["blue"]], [1.0, BRAND["green"]]]

st.set_page_config(
    page_title="Banestes | Itacibá — Carteira de Crédito",
    page_icon="📊",
    layout="wide",
)

# ==========================================================
# UI / CSS
# ==========================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700;800&display=swap');

          :root {{
            --b-blue: {BRAND["blue"]};
            --b-green: {BRAND["green"]};
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

          /* Remove chrome padrão do Streamlit */
          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          .stApp {{
            background: var(--b-bg);
          }}

          section[data-testid="stSidebar"] {{
            background: var(--b-card);
            border-right: 1px solid var(--b-border);
          }}

          .topbar {{
            display:flex; align-items:center; justify-content:space-between; gap:16px;
            padding:14px 16px; background: var(--b-card);
            border: 1px solid var(--b-border); border-radius: 16px;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.04);
            margin-bottom: 14px;
          }}
          .title {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800; letter-spacing: -0.02em;
            font-size: 20px; margin:0; line-height:1.1;
          }}
          .subtitle {{
            margin:2px 0 0 0; color: var(--b-muted); font-size: 13px;
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
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.04);
          }}
          .kpi-label {{
            color: var(--b-muted);
            font-size: 12px;
            margin-bottom: 6px;
          }}
          .kpi-value {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800;
            font-size: 24px;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.05;
          }}
          .section-title {{
            font-family: Poppins, Inter, sans-serif;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 10px 0 8px 0;
          }}
          .pill {{
            display:inline-flex; align-items:center; gap:8px;
            padding: 4px 10px; border-radius: 999px;
            border: 1px solid var(--b-border);
            background: rgba(30, 10, 232, 0.06);
            font-size: 12px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def make_plotly_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["card"],
        font=dict(family="Inter, system-ui, sans-serif", color=BRAND["ink"], size=13),
        colorway=DISCRETE,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor=BRAND["grid"], zeroline=False, showline=False),
        yaxis=dict(gridcolor=BRAND["grid"], zeroline=False, showline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return t

def fmt_br(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

# ==========================================================
# Parsing helpers (BR locale numbers)
# ==========================================================
def br_to_float(x) -> float:
    """Converte string BR (1.234.567,89) -> float. Suporta (123,45) como negativo."""
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

MONTH_MAP = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}

@st.cache_data(show_spinner=False)
def parse_relatorio(uploaded_file) -> pd.DataFrame:
    """
    Parser do layout exportado do relatório (formato 'wide'):
      - 2025: colunas em blocos [Orçado, Realizado] por mês (Jan..Dez)
      - 2026: colunas por mês com apenas Orçado
    Retorna dataset tidy com colunas:
      tipo (Saldo/Rendas), data, ano, mes, produto_cod, produto_desc, produto, orcado, realizado
    """
    raw = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")
    cols = list(raw.columns)

    # ----- 2025 blocks -----
    code_2025 = "2025"
    desc_2025 = "Unnamed: 1"

    blocks_2025 = []
    # no arquivo atual, os blocos começam na coluna index 2 com step 4
    for start in range(2, min(50, len(cols)), 4):
        mlabel = raw.iloc[0, start] if start < len(cols) else None
        if pd.isna(mlabel):
            continue
        mlabel = str(mlabel).strip()
        if mlabel in MONTH_MAP and (start + 1) < len(cols):
            blocks_2025.append((MONTH_MAP[mlabel], cols[start], cols[start+1]))

    # ----- 2026 blocks -----
    code_2026 = "Unnamed: 50"
    desc_2026 = "Unnamed: 51"
    blocks_2026 = []
    # no arquivo atual, meses 2026 em Unnamed: 53..64 (Jan..Dez)
    for idx in range(53, 65):
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

    # ----- rows 2025: começam tipicamente na linha 2 (0=header labels, 1=subheader) -----
    for i in range(2, len(raw)):
        code = raw.at[i, code_2025] if code_2025 in raw.columns else None
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

    # ----- rows 2026: começam tipicamente na linha 1 -----
    for i in range(1, len(raw)):
        code = raw.at[i, code_2026] if code_2026 in raw.columns else None
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
                "realizado": np.nan,  # 2026 no relatório atual é somente orçado
            })

    tidy = pd.DataFrame.from_records(records)
    tidy["produto"] = tidy["produto_cod"] + " - " + tidy["produto_desc"]
    return tidy

def safe_drop_totals(selected_codes: set[str]) -> tuple[set[str], list[str]]:
    """
    Evita dupla contagem: se usuário selecionar TOTAL (18201/18202) e também subprodutos,
    removemos o TOTAL automaticamente e avisamos.
    """
    warnings = []
    out = set(selected_codes)

    for total in ["18201", "18202"]:
        has_total = total in out
        has_children = any((c != total and c.startswith(total)) for c in out)
        if has_total and has_children:
            out.remove(total)
            warnings.append(
                f"Removi automaticamente o TOTAL {total} para evitar dupla contagem (você selecionou subprodutos também)."
            )

    return out, warnings

# ==========================================================
# App
# ==========================================================
inject_css()
px.defaults.template = make_plotly_template()

# Top bar
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

# Optional logo
logo_path = "assets/banestes_logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=180)

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

# Period controls
years = sorted(df["ano"].unique().tolist())

with st.sidebar:
    ano_sel = None
    mes_sel = None
    if periodo in ["Ano", "Mês"]:
        ano_sel = st.selectbox("Ano", years, index=0)
    if periodo == "Mês":
        mes_sel = st.selectbox("Mês", list(range(1, 13)), index=0)

# Produto controls
products = df[["produto_cod", "produto"]].drop_duplicates().sort_values("produto")
default_codes = [c for c in ["18201", "18202"] if c in set(products["produto_cod"])]

with st.sidebar:
    selected = st.multiselect(
        "Produto (multi)",
        options=products["produto_cod"].tolist(),
        default=default_codes,
        format_func=lambda c: products.loc[products["produto_cod"] == c, "produto"].iloc[0],
    )

selected_set, warn_list = safe_drop_totals(set(selected))
for w in warn_list:
    st.warning(w)

dff = df[df["produto_cod"].isin(selected_set)] if selected_set else df.copy()

# Apply time filter
if periodo == "Ano" and ano_sel is not None:
    dff = dff[dff["ano"] == ano_sel]
if periodo == "Mês" and ano_sel is not None and mes_sel is not None:
    dff = dff[(dff["ano"] == ano_sel) & (dff["mes"] == mes_sel)]

# =============================
# KPIs (recorte atual)
# =============================
def kpis(tipo: str) -> tuple[float, float]:
    x = dff[dff["tipo"] == tipo]
    total_orc = float(x["orcado"].sum(skipna=True))
    total_real = float(x["realizado"].sum(skipna=True)) if x["realizado"].notna().any() else np.nan
    return total_orc, total_real

saldo_orc, saldo_real = kpis("Saldo")
renda_orc, renda_real = kpis("Rendas")

st.markdown('<p class="section-title">KPIs do recorte</p>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Orçado</div>
        <p class="kpi-value">{fmt_br(saldo_orc)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Saldo • Realizado</div>
        <p class="kpi-value">{fmt_br(saldo_real)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Rendas • Orçado</div>
        <p class="kpi-value">{fmt_br(renda_orc)}</p>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Rendas • Realizado</div>
        <p class="kpi-value">{fmt_br(renda_real)}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Charts (2 principais)
# =============================
def series_for(tipo: str) -> pd.DataFrame:
    x = dff[dff["tipo"] == tipo].copy()
    if x.empty:
        return x
    g = x.groupby("data", as_index=False).agg(
        orcado=("orcado", "sum"),
        realizado=("realizado", "sum"),
    )
    long = g.melt(id_vars=["data"], value_vars=["orcado", "realizado"], var_name="cenario", value_name="valor")
    long["cenario"] = long["cenario"].map({"orcado": "Orçado", "realizado": "Realizado"})
    return long.sort_values("data")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">Evolução • Saldo da Carteira</p>', unsafe_allow_html=True)
    s = series_for("Saldo")
    if s.empty:
        st.info("Sem dados para Saldo nesse recorte.")
    else:
        fig = px.line(
            s,
            x="data",
            y="valor",
            color="cenario",
            color_discrete_map={"Orçado": BRAND["blue"], "Realizado": BRAND["green"]},
        )
        fig.update_layout(height=380, legend_title_text="")
        fig.update_traces(hovertemplate="<b>%{x|%b/%Y}</b><br>%{legendgroup}: %{y:,.2f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">Evolução • Rendas da Carteira</p>', unsafe_allow_html=True)
    s = series_for("Rendas")
    if s.empty:
        st.info("Sem dados para Rendas nesse recorte.")
    else:
        fig = px.line(
            s,
            x="data",
            y="valor",
            color="cenario",
            color_discrete_map={"Orçado": BRAND["blue"], "Realizado": BRAND["green"]},
        )
        fig.update_layout(height=380, legend_title_text="")
        fig.update_traces(hovertemplate="<b>%{x|%b/%Y}</b><br>%{legendgroup}: %{y:,.2f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

# =============================
# Reconciliação (fidelidade ao relatório)
# =============================
st.divider()
st.markdown('<p class="section-title">Reconciliação (fidelidade ao relatório)</p>', unsafe_allow_html=True)

def check_value(tipo: str, cod: str, dt: pd.Timestamp, col: str, expected: float) -> tuple[bool, float | None]:
    found = df[(df["tipo"] == tipo) & (df["produto_cod"] == cod) & (df["data"] == dt)][col]
    if len(found) != 1 or pd.isna(found.iloc[0]):
        return False, None
    val = float(found.iloc[0])
    ok = abs(val - expected) < 0.005  # tolerância de centavos
    return ok, val

checks = [
    ("Saldo", "18201", pd.Timestamp(2025, 1, 1),  "orcado",    106_033_327.59, "Saldo (18201) Orçado Jan/2025"),
    ("Saldo", "18201", pd.Timestamp(2025, 12, 1), "realizado", 0.00,          "Saldo (18201) Realizado Dez/2025"),
    ("Saldo", "18201", pd.Timestamp(2026, 1, 1),  "orcado",    97_894_933.03, "Saldo (18201) Orçado Jan/2026"),
]

all_ok = True
for tipo, cod, dt, col, expected, label in checks:
    ok, val = check_value(tipo, cod, dt, col, expected)
    all_ok = all_ok and ok
    if ok:
        st.success(f"✅ {label} = {fmt_br(expected)} (bate)")
    else:
        st.error(f"❌ {label} esperado {fmt_br(expected)} | encontrado {fmt_br(val) if val is not None else '—'}")

if all_ok:
    st.success("✅ Reconciliação completa: os controles conferem com o relatório.")
else:
    st.warning("Atenção: houve divergência. Isso pode indicar arquivo diferente, exportação alterada ou layout deslocado.")

with st.expander("Ver amostra do dataset normalizado (tidy)", expanded=False):
    st.dataframe(df.head(120), use_container_width=True, height=320)

# Download do recorte filtrado (para auditoria)
csv_bytes = dff.to_csv(index=False).encode("utf-8")
st.download_button("Baixar recorte filtrado (CSV)", data=csv_bytes, file_name="recorte_filtrado.csv", mime="text/csv")
