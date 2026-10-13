import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
from dash.dash_table.Format import Format, Scheme, Symbol

# Paleta Okabe-Ito (segura para os principais tipos de daltonismo)
AZUL, LARANJA, CINZA, TEXTO = "#0072B2", "#E69F00", "#C8C8C8", "#333333"

# ---------------- Dados sintéticos reprodutíveis ----------------
rng = np.random.default_rng(42)
N = 40_000
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
CATS = {"Eletrônicos": (1200, .18), "Móveis": (800, .12), "Vestuário": (150, .35),
        "Alimentos": (60, .22), "Papelaria": (30, .40)}  # (ticket médio, margem)

df = pd.DataFrame({
    "data": pd.to_datetime(rng.choice(pd.date_range("2023-01-01", "2025-12-31"), N)),
    "regiao": rng.choice(REGIOES, N, p=[.08, .22, .10, .42, .18]),
    "categoria": rng.choice(list(CATS), N, p=[.15, .10, .25, .35, .15]),
})
saz = 1 + .25 * np.sin(2 * np.pi * df.data.dt.dayofyear / 365) + .5 * (df.data.dt.month == 11)
df["receita"] = (df.categoria.map({k: v[0] for k, v in CATS.items()})
                 * rng.lognormal(0, .4, N) * saz).round(2)
df["lucro"] = (df.receita * (df.categoria.map({k: v[1] for k, v in CATS.items()})
                             + rng.normal(0, .05, N))).round(2)
df = df.sort_values("data").reset_index(drop=True)
df.insert(0, "pedido", df.index + 100001)

# ---------------- Estado global de seleção ----------------
FILTRO0 = {"inicio": None, "fim": None, "categorias": [], "regioes": [], "rev": 0}

def filtrar(f, periodo=True, cat=True, reg=True):
    """Crossfilter: cada visão ignora a própria seleção."""
    d = df
    if periodo and f["inicio"]:
        d = d[(d.data >= f["inicio"]) & (d.data <= f["fim"])]
    if cat and f["categorias"]:
        d = d[d.categoria.isin(f["categorias"])]
    if reg and f["regioes"]:
        d = d[d.regiao.isin(f["regioes"])]
    return d

def minimalista(fig, altura):
    """Data-ink ratio: sem bordas, legendas ou fundos supérfluos."""
    fig.update_layout(template="simple_white", height=altura,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      font=dict(family="Arial", size=12, color=TEXTO),
                      hoverlabel=dict(bgcolor="white", font_size=12))
    return fig

# ---------------- Visões ----------------
def fig_serie(f):
    d = filtrar(f, periodo=False)
    s = d.set_index("data").resample("W")["receita"].sum().reset_index()
    fig = go.Figure(go.Scatter(
        x=s.data, y=s.receita, mode="lines+markers",
        line=dict(color=AZUL, width=1.5), marker=dict(size=3, color=AZUL),
        hovertemplate="Semana de %{x|%d/%m/%Y}<br>Receita: R$ %{y:,.0f}<extra></extra>"))
    if f["inicio"]:
        fig.add_vrect(x0=f["inicio"], x1=f["fim"], fillcolor=LARANJA, opacity=.15, line_width=0)
    fig.update_layout(dragmode="select", selectdirection="h", uirevision=f["rev"])
    fig.update_yaxes(showline=False, ticks="", tickprefix="R$ ", tickformat="~s",
                     showgrid=True, gridcolor="#EEEEEE")
    fig.update_xaxes(showline=False)
    return minimalista(fig, 260)

def fig_barras(f):
    d = filtrar(f, cat=False)
    g = (d.groupby("categoria")
           .agg(receita=("receita", "sum"), lucro=("lucro", "sum"), pedidos=("pedido", "count"))
           .sort_values("receita"))
    g["margem"] = g.lucro / g.receita
    sel = f["categorias"]
    fig = go.Figure(go.Bar(
        x=g.receita, y=g.index, orientation="h",
        marker_color=[AZUL if (not sel or c in sel) else CINZA for c in g.index],
        text=[f"R$ {v/1e6:,.1f} mi" for v in g.receita], textposition="outside", cliponaxis=False,
        customdata=g[["lucro", "margem", "pedidos"]],
        hovertemplate=("<b>%{y}</b><br>Receita: R$ %{x:,.0f}<br>Lucro: R$ %{customdata[0]:,.0f}"
                       "<br>Margem: %{customdata[1]:.1%}<br>Pedidos: %{customdata[2]:,}<extra></extra>")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(showline=False, ticks="")
    minimalista(fig, 260)
    fig.update_layout(margin=dict(r=80))
    return fig

def fig_heatmap(f):
    d = filtrar(f, cat=False, reg=False)
    p = d.groupby(["regiao", "categoria"]).agg(receita=("receita", "sum"), lucro=("lucro", "sum"))
    m = (p.lucro / p.receita).unstack().reindex(index=REGIOES[::-1], columns=list(CATS))
    rec = p.receita.unstack().reindex(index=m.index, columns=m.columns)
    fig = go.Figure(go.Heatmap(
        z=m.values, x=list(m.columns), y=list(m.index), colorscale="Cividis", showscale=False,
        text=[[f"{v:.0%}" for v in linha] for linha in m.values], texttemplate="%{text}",
        customdata=rec.values, xgap=2, ygap=2,
        hovertemplate="<b>%{y} · %{x}</b><br>Margem: %{z:.1%}<br>Receita: R$ %{customdata:,.0f}<extra></extra>"))
    regs, cats = f["regioes"], f["categorias"]
    if regs or cats:  # destaque redundante (borda), não só cor
        for j, r in enumerate(m.index):
            for i, c in enumerate(m.columns):
                if (not regs or r in regs) and (not cats or c in cats):
                    fig.add_shape(type="rect", x0=i - .5, x1=i + .5, y0=j - .5, y1=j + .5,
                                  line=dict(color=LARANJA, width=3))
    fig.update_xaxes(side="top", showline=False, ticks="")
    fig.update_yaxes(showline=False, ticks="")
    return minimalista(fig, 260)

def kpi(rotulo, valor):
    return html.Div([html.Div(rotulo, style={"fontSize": 12, "color": "#666"}),
                     html.Div(valor, style={"fontSize": 22, "fontWeight": "bold"})],
                    style={"background": "white", "padding": "10px 16px", "borderRadius": "6px", "flex": 1})

# ---------------- Layout ----------------
app = Dash(__name__, title="Painel Gerencial de Vendas")
server = app.server  # usado pelo gunicorn na publicação

CARD = {"background": "white", "padding": "12px 16px", "borderRadius": "6px"}
GCONF = {"displayModeBar": False}
MOEDA = Format(precision=2, scheme=Scheme.fixed).group(True).symbol(Symbol.yes).symbol_prefix("R$ ")

app.layout = html.Div(style={"fontFamily": "Arial", "color": TEXTO, "background": "#F7F7F7",
                             "padding": "16px", "maxWidth": "1300px", "margin": "auto"}, children=[
    dcc.Store(id="filtros", data=FILTRO0),
    html.Div([html.H2("Painel Gerencial de Vendas 2023–2025", style={"margin": 0}),
              html.Button("Limpar filtros", id="limpar", n_clicks=0)],
             style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    html.P(id="resumo", style={"color": "#666"}),
    html.Div(id="kpis", style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
    html.Div([html.H4("1 · Receita semanal: arraste na horizontal para selecionar um período "
                      "(duplo clique limpa)"),
              dcc.Graph(id="ts", config=GCONF)], style=CARD),
    html.Div([
        html.Div([html.H4("2 · Receita por categoria: clique para filtrar"),
                  dcc.Graph(id="barras", config=GCONF)], style={**CARD, "flex": 1}),
        html.Div([html.H4("3 · Margem por região × categoria: clique numa célula"),
                  dcc.Graph(id="heat", config=GCONF)], style={**CARD, "flex": 1}),
    ], style={"display": "flex", "gap": "12px", "margin": "12px 0", "flexWrap": "wrap"}),
    html.Div([html.H4("4 · Detalhes sob demanda: pedidos do recorte atual (maiores receitas)"),
              dash_table.DataTable(
                  id="tabela", page_size=10, sort_action="native", filter_action="native",
                  style_as_list_view=True,
                  columns=[{"name": "Pedido", "id": "pedido"}, {"name": "Data", "id": "data"},
                           {"name": "Região", "id": "regiao"}, {"name": "Categoria", "id": "categoria"},
                           {"name": "Receita", "id": "receita", "type": "numeric", "format": MOEDA},
                           {"name": "Lucro", "id": "lucro", "type": "numeric", "format": MOEDA}],
                  style_header={"fontWeight": "bold", "background": "white", "borderBottom": "1px solid #999"},
                  style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "6px", "textAlign": "left"})],
             style=CARD),
])

# ---------------- Brushing & Linking ----------------
@app.callback(Output("filtros", "data"), Output("barras", "clickData"), Output("heat", "clickData"),
              Input("ts", "selectedData"), Input("barras", "clickData"),
              Input("heat", "clickData"), Input("limpar", "n_clicks"),
              State("filtros", "data"), prevent_initial_call=True)
def atualizar_filtros(sel, clk_bar, clk_heat, _, f):
    f, gat = dict(f), ctx.triggered_id
    if gat == "limpar":
        return {**FILTRO0, "rev": f["rev"] + 1}, None, None
    if gat == "ts":
        if sel and sel.get("range"):
            x0, x1 = sel["range"]["x"]
            f["inicio"], f["fim"] = str(x0)[:10], str(x1)[:10]
        else:
            f["inicio"] = f["fim"] = None
    elif gat == "barras" and clk_bar:
        f["categorias"] = sorted(set(f["categorias"]) ^ {clk_bar["points"][0]["y"]})
    elif gat == "heat" and clk_heat:
        p = clk_heat["points"][0]
        mesmo = f["regioes"] == [p["y"]] and f["categorias"] == [p["x"]]
        f["regioes"], f["categorias"] = ([], []) if mesmo else ([p["y"]], [p["x"]])
    return f, None, None

@app.callback(Output("ts", "figure"), Output("barras", "figure"), Output("heat", "figure"),
              Output("tabela", "data"), Output("kpis", "children"), Output("resumo", "children"),
              Input("filtros", "data"))
def renderizar(f):
    d = filtrar(f)
    rec, luc = d.receita.sum(), d.lucro.sum()
    kpis = [kpi("Receita", f"R$ {rec/1e6:,.2f} mi"), kpi("Lucro", f"R$ {luc/1e6:,.2f} mi"),
            kpi("Margem", f"{luc/rec:.1%}" if rec else "–"), kpi("Pedidos", f"{len(d):,}")]
    partes = []
    if f["inicio"]:
        partes.append(f"{f['inicio']} a {f['fim']}")
    partes += [", ".join(f[k]) for k in ("categorias", "regioes") if f[k]]
    resumo = "Filtros ativos: " + (" · ".join(partes) if partes else "nenhum (visão geral)")
    tabela = (d.nlargest(500, "receita")
                .assign(data=lambda x: x.data.dt.strftime("%d/%m/%Y")).to_dict("records"))
    return fig_serie(f), fig_barras(f), fig_heatmap(f), tabela, kpis, resumo

if __name__ == "__main__":
    app.run(debug=True)
