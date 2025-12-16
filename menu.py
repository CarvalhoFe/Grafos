import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import random
import math

# =========================
# Config
# =========================
st.set_page_config(page_title="Transferências em Grafos", layout="wide")

# Logo (opcional)
try:
    logo = Image.open("assets/logo.jfif")
    st.sidebar.image(logo, width=160)
except Exception:
    st.sidebar.warning("Logo não encontrada em assets/logo.jfif")

st.sidebar.title("Menu")

# =========================
# Carregamento e pré-processamento
# =========================
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

    # DiGraph agregado por par (u,v)
    G = nx.DiGraph()
    for u, v, data in Gm.edges(data=True):
        fee = float(data.get("fee", 0.0))
        if G.has_edge(u, v):
            G[u][v]["count"] += 1
            G[u][v]["total_fee"] += fee
            G[u][v]["max_fee"] = max(G[u][v]["max_fee"], fee)
        else:
            G.add_edge(u, v, count=1, total_fee=fee, max_fee=fee)

    return Gm, G

# =========================
# Utilitários (SEM cache com grafos!)
# =========================
def get_giant_wcc_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    comps = list(nx.weakly_connected_components(G))
    if not comps:
        return G
    giant = max(comps, key=len)
    return G.subgraph(giant).copy()

def giant_undirected_graph(G: nx.DiGraph) -> nx.Graph:
    """Componente gigante, como grafo não direcionado."""
    return get_giant_wcc_subgraph(G).to_undirected()

def plot_degree_histogram(G: nx.DiGraph):
    Gu = G.to_undirected()
    degs = [d for _, d in Gu.degree()]

    fig = plt.figure()
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

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
    random.seed(seed)

    edges = list(Gm.edges(keys=True, data=True))
    if not edges:
        st.warning("Grafo vazio.")
        return

    sample = edges if len(edges) <= n_edges else random.sample(edges, n_edges)

    # agrega por par (u,v)
    H = nx.DiGraph()
    for u, v, k, data in sample:
        fee = float(data.get("fee", 0.0))
        if H.has_edge(u, v):
            H[u][v]["fee_sum"] += fee
            H[u][v]["count"] += 1
        else:
            H.add_edge(u, v, fee_sum=fee, count=1)

    # layout (spring_layout pode depender de scipy em alguns ambientes;
    # aqui mantemos try/except para cair em kamada_kawai se necessário)
    try:
        pos = nx.spring_layout(H, seed=seed, k=1.2)
    except Exception:
        pos = nx.kamada_kawai_layout(H)

    fee_vals = [d.get("fee_sum", 0.0) for _, _, d in H.edges(data=True)]
    max_fee = max(fee_vals) if fee_vals else 1.0

    widths = []
    max_log = math.log1p(max_fee)
    for _, _, d in H.edges(data=True):
        w = d.get("fee_sum", 0.0)
        w_log = math.log1p(w)
        widths.append(0.5 + 4.0 * (w_log / max_log if max_log > 0 else 0))

    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(H, pos, node_size=60, alpha=0.9)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.25, arrows=True)

    if show_labels:
        nx.draw_networkx_labels(H, pos, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

def cycles_ranking(G: nx.DiGraph, limit_cycles=2000, node_limit=800):
    H = get_giant_wcc_subgraph(G)
    nodes = list(H.nodes())
    if len(nodes) > node_limit:
        H = H.subgraph(nodes[:node_limit]).copy()

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

def top_transfers(df: pd.DataFrame, k=10):
    cols = ["Name", "Team_from", "Team_to", "Season", "Transfer_fee"]
    return df[cols].sort_values("Transfer_fee", ascending=False).head(k)

def spanning_tree_giant(Hu: nx.Graph) -> nx.Graph | None:
    if Hu.number_of_nodes() == 0:
        return None
    root = next(iter(Hu.nodes()))
    return nx.bfs_tree(Hu, root).to_undirected()

def get_wcc_sorted(G: nx.DiGraph):
    comps = list(nx.weakly_connected_components(G))
    return sorted(comps, key=len, reverse=True)

def draw_component_graph(G: nx.DiGraph, nodes, seed=42, show_labels=False):
    H = G.subgraph(nodes).to_undirected()
    if H.number_of_nodes() == 0:
        st.warning("Componente vazia.")
        return

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

def graph_diameter_and_path(Hu: nx.Graph):
    lengths = dict(nx.all_pairs_shortest_path_length(Hu))
    diameter = 0
    pair = (None, None)

    for u, dist_dict in lengths.items():
        for v, d in dist_dict.items():
            if d > diameter:
                diameter = d
                pair = (u, v)

    path = nx.shortest_path(Hu, pair[0], pair[1])
    return diameter, pair, path

def avg_distance_between_sets(Hu: nx.Graph, A, B):
    dists = []
    for a in A:
        for b in B:
            if a == b:
                continue
            try:
                d = nx.shortest_path_length(Hu, a, b)
                dists.append(d)
            except nx.NetworkXNoPath:
                pass
    return sum(dists) / len(dists) if dists else None

def clique_analysis(G: nx.DiGraph):
    Hu = giant_undirected_graph(G)

    # maior clique
    best = []
    for c in nx.algorithms.clique.find_cliques(Hu):
        if len(c) > len(best):
            best = c

    tri = nx.triangles(Hu)
    tri_sorted = sorted(tri.items(), key=lambda x: x[1], reverse=True)

    return Hu, best, tri_sorted

def center_nodes_and_eccentricity(Hu: nx.Graph):
    ecc = nx.eccentricity(Hu)
    min_ecc = min(ecc.values()) if ecc else None
    centers = [v for v, e in ecc.items() if e == min_ecc] if min_ecc is not None else []
    return centers, ecc, min_ecc

def central_tree_from_root(Hu: nx.Graph, root: str) -> nx.Graph:
    # árvore BFS a partir de um "centro" (ou do nó escolhido)
    return nx.bfs_tree(Hu, root).to_undirected()

def draw_ego_graph(G: nx.DiGraph, node: str, seed=42):
    """
    Desenha o subgrafo induzido pelo nó e seus vizinhos (ego graph).
    """
    Gu = G.to_undirected()

    if node not in Gu:
        st.warning("Clube não encontrado no grafo.")
        return

    # nó + vizinhos
    neighbors = list(Gu.neighbors(node))
    nodes = [node] + neighbors

    H = Gu.subgraph(nodes).copy()

    try:
        pos = nx.spring_layout(H, seed=seed, k=0.9)
    except Exception:
        pos = nx.kamada_kawai_layout(H)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # desenha vizinhos
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=neighbors,
        node_size=300,
        alpha=0.8
    )

    # desenha nó central em destaque
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=[node],
        node_size=800,
        node_color="orange",
        alpha=0.95
    )

    nx.draw_networkx_edges(H, pos, alpha=0.5)

    nx.draw_networkx_labels(H, pos, font_size=8)

    plt.title(f"Vizinhança imediata do clube: {node}")
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)


# =========================
# Dataset fixo
# =========================
CSV_PATH = "data/top250-00-19.csv"

try:
    df = preprocess(pd.read_csv(CSV_PATH))
except Exception as e:
    st.error(f"Erro ao carregar CSV fixo em {CSV_PATH}: {e}")
    st.stop()

Gm, G = build_graphs(df)

# =========================
# UI
# =========================
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
        "Cliques",
        "Arestas de maior peso",
        "Árvore de abrangência",
        "Árvore central",
        "Cortes de arestas",
    ],
)

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

    n_edges = st.slider("Qtd. de arestas na amostra", 50, 800, 250, step=50)
    show_labels = st.checkbox("Mostrar nomes dos clubes (pode poluir)", value=False)

    draw_sample_graph(Gm, n_edges=n_edges, show_labels=show_labels)

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

    st.markdown("### Distribuição de graus")
    plot_degree_histogram(G)

    hubs = top_hubs(G, k=10)
    st.write("Top 10 clubes por grau (grafo não direcionado):")
    st.dataframe(pd.DataFrame(hubs, columns=["clube", "grau"]), use_container_width=True)

    st.divider()
    st.markdown("### Grau de um clube específico")

    Gu = G.to_undirected()
    clubs = sorted(list(Gu.nodes()))
    club = st.selectbox("Selecione um clube", clubs)

    if club:
        grau = Gu.degree(club)
        grau_in = G.in_degree(club)
        grau_out = G.out_degree(club)

        c1, c2, c3 = st.columns(3)
        c1.metric("Grau total", grau)
        c2.metric("Grau de entrada", grau_in)
        c3.metric("Grau de saída", grau_out)

        if grau <= 2:
            st.info("Interpretação: clube periférico, com poucas conexões no mercado de transferências.")
        elif grau <= 10:
            st.info("Interpretação: clube moderadamente conectado, participa de algumas rotas do mercado.")
        else:
            st.success("Interpretação: clube central (hub), com muitas conexões e papel estrutural relevante na rede.")
        
        st.divider()
        st.markdown("### Vizinhança imediata do clube")
        st.write(
            "O grafo abaixo representa o **subgrafo induzido** pelo clube selecionado "
            "e seus vizinhos diretos, ilustrando visualmente o conceito de grau."
        )

    draw_ego_graph(G, club)


elif menu == "Hubs":
    st.subheader("Hubs do grafo")

    k = st.slider("Qtd. de hubs (top K)", 5, 30, 15)
    hubs = top_hubs(G, k=k)
    st.dataframe(pd.DataFrame(hubs, columns=["clube", "grau"]), use_container_width=True)

    st.write("Subgrafo induzido pelos hubs (visualização)")
    hub_nodes = [h[0] for h in hubs]
    H = G.to_undirected().subgraph(hub_nodes).copy()

    try:
        pos = nx.spring_layout(H, seed=42)
    except Exception:
        pos = nx.kamada_kawai_layout(H)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(H, pos, node_size=500)
    nx.draw_networkx_edges(H, pos, alpha=0.6)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Caminhos mínimos":
    st.subheader("Caminhos mínimos e maior distância")

    Hu = giant_undirected_graph(G)

    st.write(
        f"A análise é feita na **componente gigante**: "
        f"**{Hu.number_of_nodes()}** clubes e **{Hu.number_of_edges()}** arestas."
    )

    # comprimento médio
    try:
        avg_len = nx.average_shortest_path_length(Hu)
        st.metric("Comprimento médio do menor caminho", f"{avg_len:.2f}")
    except Exception:
        st.warning("Não foi possível calcular o comprimento médio.")

    st.divider()

    # menor caminho entre dois clubes
    st.markdown("### Menor caminho entre dois clubes")

    clubs = sorted(list(Hu.nodes()))
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs) - 1))
    with col3:
        calc = st.button("Calcular menor caminho")

    if calc:
        try:
            path = nx.shortest_path(Hu, a, b)
            st.success(f"Comprimento do caminho: {len(path) - 1}")
            st.write(" → ".join(path))
        except nx.NetworkXNoPath:
            st.warning("Não existe caminho entre esses clubes.")

    st.divider()

    # diâmetro
    st.markdown("### Maior caminho da rede (diâmetro)")

    if st.button("Calcular maior caminho (diâmetro)"):
        with st.spinner("Calculando diâmetro..."):
            diameter, (u, v), path = graph_diameter_and_path(Hu)

        st.success(f"Diâmetro da rede: {diameter}")
        st.write(f"Par mais distante: **{u}** ↔ **{v}**")
        st.write("Caminho correspondente:")
        st.write(" → ".join(path))

        st.info(
            "Interpretação: o diâmetro representa a maior distância mínima "
            "entre dois clubes na rede."
        )

    # papel estrutural dos hubs
    st.divider()
    st.markdown("### Papel estrutural dos hubs")

    st.write(
        "Comparação entre distâncias envolvendo clubes periféricos (baixo grau) "
        "e clubes centrais (alto grau), sugerindo hubs como 'atalhos' estruturais."
    )

    degrees = dict(Hu.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])

    k_group = st.slider("Quantidade de clubes por grupo (k)", 5, 30, 15)
    perifericos = [n for n, _ in sorted_nodes[:k_group]]
    hubs_sel = [n for n, _ in sorted_nodes[-k_group:]]

    if st.button("Calcular distâncias médias (periferia x hubs)"):
        with st.spinner("Calculando distâncias..."):
            d_ph = avg_distance_between_sets(Hu, perifericos, hubs_sel)
            d_pp = avg_distance_between_sets(Hu, perifericos, perifericos)

        c1, c2 = st.columns(2)
        c1.metric("Distância média periférico → hub", f"{d_ph:.2f}" if d_ph is not None else "N/A")
        c2.metric("Distância média periférico → periférico", f"{d_pp:.2f}" if d_pp is not None else "N/A")

        st.info(
            "Interpretação: se periférico→hub < periférico→periférico, "
            "isso sugere que hubs reduzem distâncias na rede."
        )

elif menu == "Ciclos":
    st.subheader("Ciclos")

    st.write(
        "Ranking aproximado de clubes que mais aparecem em ciclos "
        "(baseado em ciclos no subgrafo da componente gigante)."
    )

    cycles_found, ranking = cycles_ranking(G)
    st.metric("Ciclos analisados (aprox.)", f"{cycles_found}")

    topn = st.slider("Top N", 5, 50, 15)
    st.dataframe(
        pd.DataFrame(ranking[:topn], columns=["clube", "participações_em_ciclos"]),
        use_container_width=True
    )

elif menu == "Cliques":
    st.subheader("Cliques e triângulos no grafo")

    st.write(
        "A análise de cliques é feita na **componente gigante** do grafo "
        "tratado como **não direcionado**. "
        "Cliques representam subconjuntos totalmente conectados."
    )

    Hu, max_clique, tri_sorted = clique_analysis(G)

    st.markdown("### Maior clique")
    if max_clique:
        st.metric("Tamanho da maior clique", len(max_clique))
        st.write("Clubes na maior clique:")
        st.write(", ".join(sorted(max_clique)))
    else:
        st.warning("Não foi possível identificar uma clique.")

    st.info(
        "Interpretação: uma clique grande sugere um submercado altamente "
        "interconectado (muitas relações redundantes entre os clubes)."
    )

    st.divider()
    st.markdown("### Participação em triângulos")
    topn = st.slider("Top N clubes por triângulos", 5, 50, 15)
    tri_df = pd.DataFrame(tri_sorted[:topn], columns=["clube", "número de triângulos"])
    st.dataframe(tri_df, use_container_width=True)

    if len(max_clique) <= 25 and len(max_clique) > 0:
        st.divider()
        st.markdown("### Visualização da maior clique")
        Hc = Hu.subgraph(max_clique).copy()
        try:
            pos = nx.spring_layout(Hc, seed=42)
        except Exception:
            pos = nx.kamada_kawai_layout(Hc)

        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        nx.draw_networkx_nodes(Hc, pos, node_size=600)
        nx.draw_networkx_edges(Hc, pos, alpha=0.7)
        nx.draw_networkx_labels(Hc, pos, font_size=9)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
    elif len(max_clique) > 25:
        st.info("A visualização da maior clique foi omitida (muitos nós, ficaria ilegível).")

elif menu == "Arestas de maior peso":
    st.subheader("Transferências mais caras (top K)")
    k = st.slider("Top K", 5, 30, 10)
    st.dataframe(top_transfers(df, k=k), use_container_width=True)

elif menu == "Árvore de abrangência":
    st.subheader("Árvore de abrangência (esqueleto mínimo)")

    Hu = giant_undirected_graph(G)
    T = spanning_tree_giant(Hu)

    if T is None:
        st.warning("Não foi possível construir a árvore.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nós (componente gigante)", f"{Hu.number_of_nodes()}")
        c2.metric("Arestas (grafo original)", f"{Hu.number_of_edges()}")
        c3.metric("Arestas na árvore (n-1)", f"{T.number_of_edges()}")

        st.info(
            "Interpretação (relatório): a árvore de abrangência mostra o **mínimo necessário** "
            "para manter a conectividade. O fato do grafo original ter muito mais arestas "
            "indica **redundância estrutural**."
        )

        st.divider()
        st.write("Visualização: amostra da árvore (para não ficar ilegível)")
        sample_n = st.slider("Qtd. de nós para visualizar", 50, 600, 160, step=10)

        nodes = list(T.nodes())[:sample_n]
        Ts = T.subgraph(nodes).copy()

        try:
            pos = nx.spring_layout(Ts, seed=42)
        except Exception:
            pos = nx.kamada_kawai_layout(Ts)

        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        nx.draw_networkx_nodes(Ts, pos, node_size=70)
        nx.draw_networkx_edges(Ts, pos, alpha=0.6)

        show_labels = st.checkbox("Mostrar labels (pode poluir)", value=False)
        if show_labels:
            nx.draw_networkx_labels(Ts, pos, font_size=6)

        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

elif menu == "Árvore central":
    st.subheader("Árvore central (BFS a partir do centro)")

    Hu = giant_undirected_graph(G)

    st.write(
        "Aqui calculamos o(s) **centro(s)** do grafo (menor excentricidade) e construímos "
        "uma árvore BFS a partir de um centro. Isso ajuda a visualizar como um 'núcleo' "
        "espalha conectividade pela rede."
    )

    if Hu.number_of_nodes() == 0:
        st.warning("Grafo vazio.")
    else:
        with st.spinner("Calculando centro(s) e excentricidade..."):
            centers, ecc, min_ecc = center_nodes_and_eccentricity(Hu)

        if not centers:
            st.warning("Não foi possível calcular o centro.")
        else:
            st.metric("Excentricidade mínima (raio do grafo)", f"{min_ecc}")

            center_choice = st.selectbox(
                "Escolha um centro para enraizar a árvore",
                centers
            )

            Tcentral = central_tree_from_root(Hu, center_choice)

            st.success(f"Árvore central gerada a partir do centro: **{center_choice}**")
            st.info(
                "Interpretação: centros tendem a ter **menor distância máxima** até os demais clubes. "
                "Isso está alinhado com a ideia de um **núcleo** bem conectado."
            )

            st.divider()
            st.write("Visualização: amostra da árvore central")
            sample_n = st.slider("Qtd. de nós para visualizar (árvore central)", 50, 450, 180, step=10)

            nodes = list(Tcentral.nodes())[:sample_n]
            Ts = Tcentral.subgraph(nodes).copy()

            try:
                pos = nx.spring_layout(Ts, seed=42)
            except Exception:
                pos = nx.kamada_kawai_layout(Ts)

            fig = plt.figure(figsize=(10, 7))
            ax = plt.gca()
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            nx.draw_networkx_nodes(Ts, pos, node_size=70)
            nx.draw_networkx_edges(Ts, pos, alpha=0.6)

            show_labels = st.checkbox("Mostrar labels (árvore central)", value=False)
            if show_labels:
                nx.draw_networkx_labels(Ts, pos, font_size=6)

            plt.axis("off")
            plt.tight_layout()
            st.pyplot(fig)

elif menu == "Cortes de arestas":
    st.subheader("Cortes de arestas (corte mínimo)")

    st.write(
        "O corte mínimo de arestas entre dois clubes indica **quantas arestas precisam ser removidas** "
        "para desconectá-los. No relatório, isso diferencia **periferia frágil** vs **núcleo robusto**."
    )

    Hu = giant_undirected_graph(G)

    clubs = sorted(list(Hu.nodes()))
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs) - 1))

    show_cut_edges = st.checkbox("Mostrar arestas do corte mínimo (pode ser pesado)", value=False)

    if st.button("Calcular corte mínimo"):
        if a == b:
            st.warning("Escolha dois clubes diferentes.")
        else:
            try:
                kcut = nx.edge_connectivity(Hu, a, b)
                st.success(f"Tamanho do corte mínimo entre **{a}** e **{b}**: **{kcut}**")

                if kcut <= 2:
                    st.info("Interpretação: conectividade frágil — depende de poucas arestas críticas.")
                elif kcut <= 10:
                    st.info("Interpretação: conectividade moderada — existem caminhos alternativos, mas limitados.")
                else:
                    st.info("Interpretação: conectividade alta — região densa e redundante (típico de hubs).")

                if show_cut_edges:
                    with st.spinner("Calculando mínimo corte (arestas)..."):
                        cut = nx.minimum_edge_cut(Hu, a, b)
                    cut_list = [(u, v) for (u, v) in cut]
                    st.write(f"Arestas no corte mínimo (total: {len(cut_list)}):")
                    st.dataframe(pd.DataFrame(cut_list, columns=["u", "v"]), use_container_width=True)

            except nx.NetworkXError as e:
                st.error(f"Erro: {e}")
            except Exception as e:
                st.error(
                    "Não consegui calcular o corte mínimo (pode ser pesado em alguns grafos). "
                    f"Detalhe: {e}"
                )
