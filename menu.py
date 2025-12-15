import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Transferências em Grafos", layout="wide")

logo = Image.open("assets/logo.jfif")
st.sidebar.image(logo, width=160)


# =========================
# Carregamento e pré-processamento
# =========================
@st.cache_data
def load_csv(uploaded_file=None, path=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if path:
        return pd.read_csv(path)
    return None

@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = [
        "Team_from", "Team_to", "Name", "Season",
        "Transfer_fee", "League_from", "League_to"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no CSV: {missing}")

    df["Transfer_fee"] = pd.to_numeric(df["Transfer_fee"], errors="coerce").fillna(0.0)

    for c in ["Team_from", "Team_to", "Name", "Season", "League_from", "League_to"]:
        df[c] = df[c].astype(str).str.strip()

    return df

@st.cache_resource
def build_graphs(df: pd.DataFrame):
    # MultiDiGraph: 1 aresta por transferência
    Gm = nx.MultiDiGraph()
    for _, r in df.iterrows():
        Gm.add_edge(
            r["Team_from"], r["Team_to"],
            player=r["Name"],
            season=r["Season"],
            fee=float(r["Transfer_fee"]),
            league_from=r["League_from"],
            league_to=r["League_to"]
        )

    # DiGraph agregado por par (u,v) - útil para grau/hubs/conectividade
    G = nx.DiGraph()
    for u, v, data in Gm.edges(data=True):
        fee = data.get("fee", 0.0)
        if G.has_edge(u, v):
            G[u][v]["count"] += 1
            G[u][v]["total_fee"] += fee
            G[u][v]["max_fee"] = max(G[u][v]["max_fee"], fee)
        else:
            G.add_edge(u, v, count=1, total_fee=fee, max_fee=fee)

    return Gm, G

# =========================
# Utilitários
# =========================
def get_giant_wcc_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    # maior componente fracamente conectada (equivalente a "componente gigante" no relatório)
    comps = list(nx.weakly_connected_components(G))
    if not comps:
        return G
    giant = max(comps, key=len)
    return G.subgraph(giant).copy()

def plot_degree_histogram(G: nx.DiGraph):
    Gu = G.to_undirected()
    degs = [d for _, d in Gu.degree()]
    fig = plt.figure()
    plt.hist(degs, bins=30)
    plt.title("Distribuição de grau (vizinhos únicos)")
    plt.xlabel("Grau (número de vizinhos)")
    plt.ylabel("Frequência (clubes)")
    plt.tight_layout()
    st.pyplot(fig)

def top_hubs(G: nx.DiGraph, k=15):
    Gu = G.to_undirected()
    return sorted(Gu.degree(), key=lambda x: x[1], reverse=True)[:k]

def draw_sample_graph(Gm: nx.MultiDiGraph, n_edges=250, seed=42, show_labels=False):
    import random, math
    random.seed(seed)

    edges = list(Gm.edges(keys=True, data=True))
    if not edges:
        st.warning("Grafo vazio.")
        return

    sample = edges if len(edges) <= n_edges else random.sample(edges, n_edges)

    # agrega por par (u,v) para reduzir poluição visual
    H = nx.DiGraph()
    for u, v, k, data in sample:
        fee = float(data.get("fee", 0.0))
        if H.has_edge(u, v):
            H[u][v]["fee_sum"] += fee
            H[u][v]["count"] += 1
        else:
            H.add_edge(u, v, fee_sum=fee, count=1)

    # layout
    pos = nx.spring_layout(H, seed=seed, k=1.2)

    # largura com log e normalização por max
    fee_vals = [d.get("fee_sum", 0.0) for _, _, d in H.edges(data=True)]
    max_fee = max(fee_vals) if fee_vals else 1.0

    widths = []
    for _, _, d in H.edges(data=True):
        w = d.get("fee_sum", 0.0)
        # log para comprimir valores muito grandes
        w_log = math.log1p(w)
        max_log = math.log1p(max_fee)
        widths.append(0.5 + 4.0 * (w_log / max_log if max_log > 0 else 0))

    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()
    ax.set_facecolor("white")  # evita “fundo preto”
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(H, pos, node_size=60, alpha=0.9)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.25, arrows=True)

    if show_labels:
        nx.draw_networkx_labels(H, pos, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)


def avg_shortest_path_length_giant(G: nx.DiGraph):
    Gu = get_giant_wcc_subgraph(G).to_undirected()
    if Gu.number_of_nodes() < 2:
        return None
    try:
        return nx.average_shortest_path_length(Gu)
    except Exception:
        return None

def shortest_path_between(G: nx.DiGraph, a: str, b: str):
    Gu = get_giant_wcc_subgraph(G).to_undirected()
    return nx.shortest_path(Gu, a, b)

def cycles_ranking(G: nx.DiGraph, limit_cycles=2000, node_limit=800):
    # aproximação prática: pega ciclos simples num subgrafo menor (pra não travar)
    H = get_giant_wcc_subgraph(G)
    nodes = list(H.nodes())
    if len(nodes) > node_limit:
        H = H.subgraph(nodes[:node_limit]).copy()

    # usa ciclo simples em digrafo pode ser pesado; converte pra não-dir para ranking estrutural
    Hu = H.to_undirected()
    count = {}
    cycles_found = 0
    try:
        for cyc in nx.cycle_basis(Hu):
            cycles_found += 1
            for v in cyc:
                count[v] = count.get(v, 0) + 1
            if cycles_found >= limit_cycles:
                break
    except Exception:
        pass

    ranking = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return cycles_found, ranking

def build_league_graph(df: pd.DataFrame):
    # vértices = ligas, aresta = transferência entre ligas
    L = nx.DiGraph()
    for _, r in df.iterrows():
        a = r["League_from"]
        b = r["League_to"]
        if L.has_edge(a, b):
            L[a][b]["count"] += 1
            L[a][b]["total_fee"] += float(r["Transfer_fee"])
        else:
            L.add_edge(a, b, count=1, total_fee=float(r["Transfer_fee"]))
    return L

def top_league_pairs(L: nx.DiGraph, k=10):
    rows = []
    for u, v, d in L.edges(data=True):
        rows.append({"league_from": u, "league_to": v, "count": d["count"], "total_fee": d["total_fee"]})
    edf = pd.DataFrame(rows).sort_values("count", ascending=False).head(k)
    return edf

def top_transfers(df: pd.DataFrame, k=10):
    cols = ["Name", "Team_from", "Team_to", "Season", "Transfer_fee"]
    return df[cols].sort_values("Transfer_fee", ascending=False).head(k)

def clique_analysis(G: nx.DiGraph):
    # maior clique na maior componente (não-direcionado)
    Hu = get_giant_wcc_subgraph(G).to_undirected()

    # maior clique (pode ser pesado, mas com 600 nós costuma ir)
    max_clique = nx.algorithms.clique.find_cliques(Hu)
    best = []
    for c in max_clique:
        if len(c) > len(best):
            best = c
    # triângulos por nó
    tri = nx.triangles(Hu)
    tri_sorted = sorted(tri.items(), key=lambda x: x[1], reverse=True)

    return Hu, best, tri_sorted

def spanning_tree_giant(G: nx.DiGraph):
    Hu = get_giant_wcc_subgraph(G).to_undirected()
    # árvore de abrangência: BFS tree a partir de um nó qualquer
    if Hu.number_of_nodes() == 0:
        return None, None
    root = next(iter(Hu.nodes()))
    T = nx.bfs_tree(Hu, root).to_undirected()
    return Hu, T

def get_wcc_sorted(G: nx.DiGraph):
    """Componentes fracamente conectadas ordenadas por tamanho (desc)."""
    comps = list(nx.weakly_connected_components(G))
    comps = sorted(comps, key=len, reverse=True)
    return comps

def draw_component_graph(G: nx.DiGraph, nodes, seed=42, show_labels=False):
    H = G.subgraph(nodes).to_undirected()

    if H.number_of_nodes() == 0:
        st.warning("Componente vazia.")
        return

    # layout (requer scipy; se não tiver, use kamada_kawai)
    try:
        pos = nx.spring_layout(H, seed=seed, k=1.5)
    except Exception:
        pos = nx.kamada_kawai_layout(H)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(H, pos, node_size=60, alpha=0.9)
    nx.draw_networkx_edges(H, pos, alpha=0.18)

    if show_labels:
        nx.draw_networkx_labels(H, pos, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)


# =========================
# UI
# =========================
st.sidebar.title("Menu")

menu = st.sidebar.selectbox(
    "Escolha a análise",
    [
        "Carregar dados",
        "Grafos gerados",
        "Conectividade",
        "Graus",
        "Hubs",
        "Caminhos mínimos",
        "Ciclos",
        "Arestas de maior peso",
        "Árvore de abrangência",
        "Cortes de arestas"
    ],
)

with st.sidebar.expander("Entrada de dados", expanded=True):
    uploaded = st.file_uploader("Envie o CSV", type=["csv"])
    path = st.text_input("Ou caminho local do CSV", value="top250-00-19.csv")

df = load_csv(uploaded_file=uploaded, path=path)
if df is None:
    st.info("Envie o CSV ou informe o caminho para começar.")
    st.stop()

try:
    df = preprocess(df)
except Exception as e:
    st.error(str(e))
    st.stop()

Gm, G = build_graphs(df)

st.title("Uma análise do mercado do futebol com grafos")

# =========================
# Telas
# =========================
if menu == "Carregar dados":
    st.subheader("Amostra do dataset")
    st.dataframe(df.head(30), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Transferências (linhas)", f"{len(df)}")
    c2.metric("Clubes (vértices)", f"{G.number_of_nodes()}")
    c3.metric("Arestas agregadas (pares)", f"{G.number_of_edges()}")

elif menu == "Grafos gerados":
    st.subheader("Visualização do grafo")
    st.write(
        "Abaixo, uma visualização por amostragem. "
        "A espessura das arestas representa o peso (valor da transferência)."
    )

    n_edges = st.slider(
        "Qtd. de arestas na amostra",
        min_value=50,
        max_value=800,
        value=250,
        step=50
    )

    show_labels = st.checkbox(
        "Mostrar nomes dos clubes (pode poluir)",
        value=False
    )

    draw_sample_graph(
        Gm,
        n_edges=n_edges,
        show_labels=show_labels
    )


elif menu == "Conectividade":
    st.subheader("Análise da conectividade (componentes conexas)")

    comps = get_wcc_sorted(G)
    sizes = [len(c) for c in comps]

    c1, c2 = st.columns(2)
    c1.metric("Componentes (WCC)", f"{len(comps)}")
    c2.metric("Tamanho da componente gigante", f"{sizes[0] if sizes else 0}")

    st.write("Top 10 tamanhos de componentes (WCC)")
    st.dataframe(pd.DataFrame(sizes[:10], columns=["tamanho"]), use_container_width=True)

    st.divider()
    st.subheader("Visualizar uma componente")

    if not comps:
        st.info("Não há componentes para exibir.")
    else:
        total_nodes = G.number_of_nodes()
        options = []
        for i, comp in enumerate(comps[:50]):
            pct = 100 * len(comp) / total_nodes if total_nodes else 0
            options.append(f"Componente {i} | nós={len(comp)} | {pct:.1f}% do grafo")

        chosen = st.selectbox("Escolha a componente", options, index=0)
        idx = int(chosen.split("|")[0].replace("Componente", "").strip())
        nodes = comps[idx]

        st.write(f"Exibindo componente {idx} com {len(nodes)} nós.")

        # checkbox só se <= 300 nós
        labels_ok = len(nodes) <= 300
        show_labels = st.checkbox(
            "Mostrar labels (somente se <= 300 nós)",
            value=False,
            disabled=not labels_ok
        )

        if not labels_ok:
            st.info("Labels desativados porque a componente tem mais de 300 nós (fica ilegível).")

        draw_component_graph(G, nodes, show_labels=show_labels)



elif menu == "Graus":
    st.subheader("Análise de graus dos vértices")

    st.write(
        "O grau de um vértice representa o número de clubes distintos "
        "com os quais um clube realizou transferências."
    )

    # ===== visão global =====
    st.markdown("### Distribuição de graus")
    plot_degree_histogram(G)

    hubs = top_hubs(G, k=10)
    st.write("Top 10 clubes por grau (grafo não direcionado):")
    st.dataframe(
        pd.DataFrame(hubs, columns=["clube", "grau"]),
        use_container_width=True
    )

    st.divider()

    # ===== análise individual =====
    st.markdown("### Grau de um clube específico")

    Gu = G.to_undirected()
    clubs = sorted(list(Gu.nodes()))

    club = st.selectbox(
        "Selecione um clube",
        clubs
    )

    if club:
        grau = Gu.degree(club)
        grau_in = G.in_degree(club)
        grau_out = G.out_degree(club)

        c1, c2, c3 = st.columns(3)
        c1.metric("Grau total", grau)
        c2.metric("Grau de entrada", grau_in)
        c3.metric("Grau de saída", grau_out)

        # interpretação automática (estilo relatório)
        if grau <= 2:
            st.info(
                "Interpretação: clube periférico, com poucas conexões "
                "no mercado de transferências."
            )
        elif grau <= 10:
            st.info(
                "Interpretação: clube moderadamente conectado, "
                "participa de algumas rotas do mercado."
            )
        else:
            st.success(
                "Interpretação: clube central (hub), "
                "com muitas conexões e papel estrutural relevante na rede."
            )


elif menu == "Hubs":
    st.subheader("Hubs do grafo")
    k = st.slider("Qtd. de hubs (top K)", 5, 30, 15)
    hubs = top_hubs(G, k=k)
    st.dataframe(pd.DataFrame(hubs, columns=["clube", "grau"]), use_container_width=True)

    # subgrafo dos hubs
    st.write("Subgrafo induzido pelos hubs (visualização)")
    hub_nodes = [h[0] for h in hubs]
    H = G.to_undirected().subgraph(hub_nodes).copy()
    pos = nx.spring_layout(H, seed=42)

    fig = plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(H, pos, node_size=500)
    nx.draw_networkx_edges(H, pos, alpha=0.6)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Caminhos mínimos":
    st.subheader("Caminhos mínimos (na maior componente)")
    L = avg_shortest_path_length_giant(G)
    if L is not None:
        st.metric("Comprimento médio do menor caminho", f"{L:.2f}")
    else:
        st.warning("Não consegui calcular a média (grafo pequeno/desconectado ou limitação de cálculo).")

    Hu = get_giant_wcc_subgraph(G).to_undirected()
    clubs = sorted(list(Hu.nodes()))
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs)-1))
    with col3:
        go = st.button("Calcular")

    if go:
        try:
            path = nx.shortest_path(Hu, a, b)
            st.success(f"Comprimento: {len(path)-1}")
            st.write(" -> ".join(path))
        except nx.NetworkXNoPath:
            st.warning("Sem caminho entre esses clubes dentro da componente gigante (isso é raro).")

elif menu == "Ciclos":
    st.subheader("Ciclos")
    st.write("Ranking aproximado de clubes que mais aparecem em ciclos (baseado em ciclos no subgrafo da componente gigante).")
    cycles_found, ranking = cycles_ranking(G)
    st.metric("Ciclos analisados (aprox.)", f"{cycles_found}")

    topn = st.slider("Top N", 5, 50, 15)
    st.dataframe(pd.DataFrame(ranking[:topn], columns=["clube", "participações_em_ciclos"]), use_container_width=True)

elif menu == "Arestas de maior peso":
    st.subheader("Arestas de maior peso (transferências mais caras)")
    k = st.slider("Top K", 5, 30, 10)
    st.dataframe(top_transfers(df, k=k), use_container_width=True)

elif menu == "Árvore de abrangência":
    st.subheader("Árvore de abrangência (na maior componente)")
    Hu, T = spanning_tree_giant(G)
    if T is None:
        st.warning("Não foi possível construir a árvore.")
    else:
        st.metric("Vértices na componente gigante", f"{Hu.number_of_nodes()}")
        st.metric("Arestas no grafo original (gigante)", f"{Hu.number_of_edges()}")
        st.metric("Arestas na árvore (n-1)", f"{T.number_of_edges()}")

        st.write("Visualização: amostra da árvore (pode ficar grande)")
        sample_n = st.slider("Qtd. de nós para visualizar", 50, 400, 120, step=10)
        nodes = list(T.nodes())[:sample_n]
        Ts = T.subgraph(nodes).copy()
        pos = nx.spring_layout(Ts, seed=42)

        fig = plt.figure(figsize=(9, 6))
        nx.draw_networkx_nodes(Ts, pos, node_size=80)
        nx.draw_networkx_edges(Ts, pos, alpha=0.6)
        nx.draw_networkx_labels(Ts, pos, font_size=6)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

elif menu == "Cortes de arestas":
    st.subheader("Cortes de arestas (corte mínimo)")

    st.write(
        "Aqui modelamos o mercado como **grafo não direcionado**: clubes = vértices, "
        "transferências = arestas. O corte mínimo de arestas entre dois clubes indica "
        "quantas arestas precisam ser removidas para desconectá-los."
    )

    # recomendação: usar a componente gigante para evitar 'sem caminho'
    Hu = get_giant_wcc_subgraph(G).to_undirected()

    clubs = sorted(list(Hu.nodes()))
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs)-1))

    show_cut_edges = st.checkbox("Mostrar as arestas do corte mínimo (pode ser pesado)", value=False)

    if st.button("Calcular corte mínimo"):
        if a == b:
            st.warning("Escolha dois clubes diferentes.")
        else:
            try:
                # tamanho do corte mínimo
                k = nx.edge_connectivity(Hu, a, b)
                st.success(f"Tamanho do corte mínimo de arestas entre **{a}** e **{b}**: **{k}**")

                # interpretação rápida (igual a narrativa do relatório)
                if k <= 2:
                    st.info(
                        "Interpretação: conectividade frágil — poucos caminhos independentes ligam os clubes."
                    )
                elif k <= 10:
                    st.info(
                        "Interpretação: conectividade moderada — existem caminhos alternativos, mas ainda limitados."
                    )
                else:
                    st.info(
                        "Interpretação: conectividade alta — rede redundante entre esses clubes (região central/hubs)."
                    )

                # opcional: listar as arestas do corte
                if show_cut_edges:
                    cut = nx.minimum_edge_cut(Hu, a, b)
                    cut_list = [(u, v) for (u, v) in cut]
                    st.write(f"Arestas no corte mínimo (total: {len(cut_list)}):")
                    st.dataframe(pd.DataFrame(cut_list, columns=["u", "v"]), use_container_width=True)

            except nx.NetworkXError as e:
                st.error(f"Erro: {e}")
            except Exception as e:
                st.error(
                    "Não consegui calcular o corte mínimo. "
                    "Se o grafo for grande, isso pode ser pesado. "
                    f"Detalhe: {e}"
                )

