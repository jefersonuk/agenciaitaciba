# Banestes • Itacibá — Carteira de Crédito (Dashboard Interativo)

Dashboard interativo (Streamlit + Plotly) com identidade visual Banestes (Azul `#1E0AE8` e Verde `#00AB16`).

## O que tem
- **2 gráficos principais**
  - Evolução do **Saldo da Carteira** (Orçado x Realizado)
  - Evolução das **Rendas da Carteira** (Orçado x Realizado)
- **Filtros**
  - Período: **Total / Ano / Mês**
  - **Produto** (multi-seleção)
- **Reconciliação (fidelidade)**
  - Checks automáticos com “números de controle” do relatório.

> ⚠️ **Dados sensíveis:** por padrão, o repositório ignora `.csv/.xlsx/.xls` (ver `.gitignore`).
> Rode localmente apontando para o arquivo exportado do sistema.

## Como rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como usar
1) Abra o app e **anexe o CSV exportado** do relatório (mesmo layout).
2) Use os filtros de período e produto.
3) Confira a seção **Reconciliação** — ela precisa ficar toda verde (✅).

## Logo (opcional)
Se quiser exibir o logo, coloque o arquivo em:
`assets/banestes_logo.png`
