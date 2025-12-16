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
# Grafo de ligas
# =========================
@st.cache_resource
def build_league_graph(df: pd.DataFrame):
    Glm = nx.MultiDiGraph()
    for _, r in df.iterrows():
        Glm.add_edge(
            r["League_from"], r["League_to"],
            player=r["Name"],
            season=r["Season"],
            fee=float(r["Transfer_fee"]),
        )

    Gl = nx.DiGraph()
    for u, v, data in Glm.edges(data=True):
        fee = float(data.get("fee", 0.0))
        if Gl.has_edge(u, v):
            Gl[u][v]["count"] += 1
            Gl[u][v]["total_fee"] += fee
            Gl[u][v]["max_fee"] = max(Gl[u][v]["max_fee"], fee)
        else:
            Gl.add_edge(u, v, count=1, total_fee=fee, max_fee=fee)

    return Glm, Gl


def top_league_hubs(Gl: nx.DiGraph, k=10):
    Gu = Gl.to_undirected()
    return sorted(Gu.degree(), key=lambda x: x[1], reverse=True)[:k]


# =========================
# Utilitários
# =========================
def get_giant_wcc_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    comps = list(nx.weakly_connected_components(G))
    if not comps:
        return G
    giant = max(comps, key=len)
    return G.subgraph(giant).copy()

def giant_undirected_graph(G: nx.DiGraph) -> nx.Graph:
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

    H = nx.DiGraph()
    for u, v, k, data in sample:
        fee = float(data.get("fee", 0.0))
        if H.has_edge(u, v):
            H[u][v]["fee_sum"] += fee
            H[u][v]["count"] += 1
        else:
            H.add_edge(u, v, fee_sum=fee, count=1)

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

def clique_analysis(G: nx.DiGraph):
    Hu = giant_undirected_graph(G)

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
    return nx.bfs_tree(Hu, root).to_undirected()

def draw_edge_cut_zoom(Hu: nx.Graph, a: str, b: str, cut_edges, hops=2, seed=42):
    def k_hop_nodes(G, src, k):
        visited = {src}
        frontier = {src}
        for _ in range(k):
            nxt = set()
            for x in frontier:
                nxt |= set(G.neighbors(x))
            nxt -= visited
            visited |= nxt
            frontier = nxt
        return visited

    A = k_hop_nodes(Hu, a, hops)
    B = k_hop_nodes(Hu, b, hops)
    nodes = A | B
    Z = Hu.subgraph(nodes).copy()

    cut_norm = set()
    for (u, v) in cut_edges:
        cut_norm.add((u, v))
        cut_norm.add((v, u))

    red_edges = []
    gray_edges = []
    for u, v in Z.edges():
        if (u, v) in cut_norm:
            red_edges.append((u, v))
        else:
            gray_edges.append((u, v))

    try:
        pos = nx.spring_layout(Z, seed=seed, k=1.0)
    except Exception:
        pos = nx.kamada_kawai_layout(Z)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(Z, pos, node_size=80, alpha=0.85)
    nx.draw_networkx_nodes(Z, pos, nodelist=[a], node_size=450, node_color="orange")
    nx.draw_networkx_nodes(Z, pos, nodelist=[b], node_size=450, node_color="orange")

    nx.draw_networkx_edges(Z, pos, edgelist=gray_edges, alpha=0.20, width=1.0)
    nx.draw_networkx_edges(Z, pos, edgelist=red_edges, alpha=0.95, width=2.5, edge_color="red")

    nx.draw_networkx_labels(Z, pos, labels={a: a, b: b}, font_size=9)

    plt.title(f"Zoom do corte mínimo entre {a} e {b} | hops={hops} | nós={Z.number_of_nodes()}")
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

def all_pairs_shortest_lengths(Hu: nx.Graph):
    lengths = dict(nx.all_pairs_shortest_path_length(Hu))
    pairs = {}
    nodes = list(Hu.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    for u, dist_dict in lengths.items():
        iu = idx[u]
        for v, d in dist_dict.items():
            iv = idx[v]
            if iu < iv:
                pairs[(u, v)] = d
    return pairs

def top_k_extremes_shortest_paths(Hu: nx.Graph, k=5, min_dist=1):
    pairs = all_pairs_shortest_lengths(Hu)
    if not pairs:
        return [], [], 0

    diameter = max(pairs.values())

    max_pairs = [(uv, d) for uv, d in pairs.items() if d == diameter]
    max_pairs = max_pairs[:k]

    max_items = []
    for (u, v), d in max_pairs:
        path = nx.shortest_path(Hu, u, v)
        max_items.append({"u": u, "v": v, "dist": d, "path": path})

    min_candidates = [(uv, d) for uv, d in pairs.items() if d >= min_dist]
    if not min_candidates:
        return max_items, [], diameter

    best_min = min(d for _, d in min_candidates)
    min_pairs = [(uv, d) for uv, d in min_candidates if d == best_min]
    min_pairs = min_pairs[:k]

    min_items = []
    for (u, v), d in min_pairs:
        path = nx.shortest_path(Hu, u, v)
        min_items.append({"u": u, "v": v, "dist": d, "path": path})

    return max_items, min_items, diameter


# =========================
# NOVO: subgrafo por jogador
# =========================
def season_key(s: str):
    """
    Tenta ordenar temporadas de forma estável.
    Funciona bem para formatos tipo "2008", "2008/09", "08/09", etc.
    """
    s = str(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    if len(digits) >= 2:
        return int(digits[:2])
    return 0

def build_player_subgraph(df: pd.DataFrame, player_name: str):
    dpf = df[df["Name"] == player_name].copy()
    if dpf.empty:
        return None, dpf

    # ordena por temporada (aprox.)
    dpf["__season_key"] = dpf["Season"].apply(season_key)
    dpf = dpf.sort_values(["__season_key", "Season"]).drop(columns=["__season_key"])

    # grafo do jogador (agregado por par de clubes)
    Hp = nx.DiGraph()
    for _, r in dpf.iterrows():
        u = r["Team_from"]
        v = r["Team_to"]
        fee = float(r["Transfer_fee"])
        season = r["Season"]

        if Hp.has_edge(u, v):
            Hp[u][v]["count"] += 1
            Hp[u][v]["total_fee"] += fee
            Hp[u][v]["seasons"].append(season)
        else:
            Hp.add_edge(u, v, count=1, total_fee=fee, seasons=[season])

    return Hp, dpf

def draw_player_graph(Hp: nx.DiGraph, show_labels=True, seed=42):
    if Hp is None or Hp.number_of_nodes() == 0:
        st.warning("Subgrafo vazio.")
        return

    try:
        pos = nx.spring_layout(Hp, seed=seed, k=1.0)
    except Exception:
        pos = nx.kamada_kawai_layout(Hp)

    # largura por contagem
    counts = [d.get("count", 1) for _, _, d in Hp.edges(data=True)]
    mx = max(counts) if counts else 1
    widths = [0.8 + 3.2 * (c / mx) for c in counts]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(Hp, pos, node_size=700, alpha=0.9)
    nx.draw_networkx_edges(Hp, pos, width=widths, alpha=0.45, arrows=True)

    if show_labels:
        nx.draw_networkx_labels(Hp, pos, font_size=8)

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
Glm, Gl = build_league_graph(df)

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
        "Subgrafos",
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

elif menu == "Subgrafos":
    st.subheader("Subgrafos")

    modo = st.radio(
        "Escolha o tipo de subgrafo",
        ["Hubs das ligas", "Liga específica", "Transferências de um jogador"],
        horizontal=True
    )

    # =========================
    # 1) Hubs das ligas
    # =========================
    if modo == "Hubs das ligas":
        st.write("Subgrafo induzido pelas ligas mais conectadas (hubs) no grafo liga-liga.")
        k = st.slider("Qtd. de ligas (top K)", 5, 30, 10)

        hubs = top_league_hubs(Gl, k=k)
        st.dataframe(pd.DataFrame(hubs, columns=["liga", "grau"]), use_container_width=True)

        hub_nodes = [h[0] for h in hubs]
        H = Gl.to_undirected().subgraph(hub_nodes).copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Ligas no subgrafo", H.number_of_nodes())
        c2.metric("Arestas no subgrafo", H.number_of_edges())
        c3.metric("Densidade", f"{nx.density(H):.3f}")

        try:
            pos = nx.spring_layout(H, seed=42)
        except Exception:
            pos = nx.kamada_kawai_layout(H)

        fig = plt.figure(figsize=(9, 7))
        ax = plt.gca()
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        nx.draw_networkx_nodes(H, pos, node_size=650)
        nx.draw_networkx_edges(H, pos, alpha=0.55)
        nx.draw_networkx_labels(H, pos, font_size=8)

        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Interpretação: evidencia ligas com maior integração no mercado internacional "
            "(muitas rotas de negociação com outras ligas)."
        )

    # =========================
    # 2) Liga específica (ego)
    # =========================
    elif modo == "Liga específica":
        st.write("Vizinhança de uma liga no grafo liga-liga (quem negocia com ela).")
        Gu = Gl.to_undirected()
        ligas = sorted(list(Gu.nodes()))
        liga = st.selectbox("Escolha a liga", ligas)

        if liga:
            neighbors = list(Gu.neighbors(liga))
            nodes = [liga] + neighbors
            H = Gu.subgraph(nodes).copy()

            st.metric("Nº de ligas conectadas", len(neighbors))

            try:
                pos = nx.spring_layout(H, seed=42, k=0.9)
            except Exception:
                pos = nx.kamada_kawai_layout(H)

            fig = plt.figure(figsize=(9, 6))
            ax = plt.gca()
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            nx.draw_networkx_nodes(H, pos, nodelist=neighbors, node_size=420, alpha=0.8)
            nx.draw_networkx_nodes(H, pos, nodelist=[liga], node_size=950, node_color="orange", alpha=0.95)
            nx.draw_networkx_edges(H, pos, alpha=0.5)
            nx.draw_networkx_labels(H, pos, font_size=8)

            plt.axis("off")
            plt.tight_layout()
            st.pyplot(fig)

    # =========================
    # 3) Jogador (novo)
    # =========================
    else:
        st.write("Selecione um jogador e veja o subgrafo das transferências dele (clubes como nós).")

        # busca (para não jogar 200k nomes no selectbox)
        all_players = sorted(df["Name"].dropna().unique().tolist())

        query = st.text_input("Buscar jogador (digite parte do nome)", value="")
        if query.strip():
            filt = [p for p in all_players if query.lower() in p.lower()]
        else:
            filt = all_players[:500]  # fallback: mostra só os primeiros pra não travar

        player = st.selectbox("Jogador", filt)

        if st.button("Gerar subgrafo do jogador"):
            Hp, dpf = build_player_subgraph(df, player)

            if dpf.empty or Hp is None:
                st.warning("Não encontrei transferências para esse jogador na base.")
            else:
                st.success(f"Transferências encontradas: {len(dpf)}")

                # tabela
                show_cols = ["Season", "Team_from", "Team_to", "League_from", "League_to", "Transfer_fee"]
                st.dataframe(dpf[show_cols], use_container_width=True)

                # métricas do subgrafo
                c1, c2, c3 = st.columns(3)
                c1.metric("Clubes no subgrafo", Hp.number_of_nodes())
                c2.metric("Arestas (movimentos)", Hp.number_of_edges())
                c3.metric("Total pago (somado)", f"{dpf['Transfer_fee'].sum():,.0f}")

                st.divider()
                st.markdown("### Visualização do subgrafo do jogador")
                draw_player_graph(Hp, show_labels=True, seed=42)

                st.info(
                    "Interpretação: esse subgrafo mostra o 'trajeto' do jogador no mercado, "
                    "incluindo possíveis retornos e repetição de rotas entre clubes."
                )

elif menu == "Caminhos mínimos":
    st.subheader("Caminhos mínimos na rede de transferências")

    Hu = giant_undirected_graph(G)

    st.write(
        f"A análise é feita na componente gigante do grafo, "
        f"composta por {Hu.number_of_nodes()} clubes e "
        f"{Hu.number_of_edges()} arestas."
    )

    try:
        avg_len = nx.average_shortest_path_length(Hu)
        st.metric("Comprimento médio do menor caminho", f"{avg_len:.2f}")
    except Exception:
        st.warning("Não foi possível calcular o comprimento médio do caminho.")

    st.info(
        "Interpretação: em média, poucos intermediários separam dois clubes quaisquer, "
        "indicando uma rede bem integrada."
    )

    st.divider()
    st.markdown("### Menor caminho entre dois clubes")

    clubs = sorted(list(Hu.nodes()))
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs) - 1))
    with col3:
        calc = st.button("Calcular")

    if calc:
        try:
            path = nx.shortest_path(Hu, a, b)
            st.success(f"Comprimento do caminho: {len(path) - 1}")
            st.write("Caminho:")
            st.write(" → ".join(path))
        except nx.NetworkXNoPath:
            st.warning("Não existe caminho entre esses clubes.")

    st.divider()
    st.markdown("### Top 5 maiores e menores caminhos mínimos")

    min_dist = st.slider(
        "Distância mínima para o Top 5 menores caminhos",
        min_value=1,
        max_value=5,
        value=2,
        help="Use 1 para conexões diretas. Use 2 para evitar arestas diretas."
    )

    if st.button("Gerar Top 5"):
        with st.spinner("Calculando rankings..."):
            max_items, min_items, diam = top_k_extremes_shortest_paths(Hu, k=5, min_dist=min_dist)

        st.subheader("Top 5 maiores distâncias mínimas")
        if not max_items:
            st.warning("Não foi possível calcular.")
        else:
            df_max = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Distância": it["dist"],
                    "Par de clubes": f'{it["u"]} ↔ {it["v"]}',
                    "Caminho": " → ".join(it["path"])
                }
                for i, it in enumerate(max_items)
            ])
            st.dataframe(df_max, use_container_width=True)

        st.divider()
        st.subheader("Top 5 menores distâncias mínimas")
        if not min_items:
            st.warning("Não foram encontrados pares com esse filtro.")
        else:
            df_min = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Distância": it["dist"],
                    "Par de clubes": f'{it["u"]} ↔ {it["v"]}',
                    "Caminho": " → ".join(it["path"])
                }
                for i, it in enumerate(min_items)
            ])
            st.dataframe(df_min, use_container_width=True)

elif menu == "Ciclos":
    st.subheader("Ciclos")
    st.write("Ranking aproximado de clubes que mais aparecem em ciclos (baseado na componente gigante).")

    cycles_found, ranking = cycles_ranking(G)
    st.metric("Ciclos analisados (aprox.)", f"{cycles_found}")

    topn = st.slider("Top N", 5, 50, 15)
    st.dataframe(pd.DataFrame(ranking[:topn], columns=["clube", "participações_em_ciclos"]), use_container_width=True)

elif menu == "Cliques":
    st.subheader("Cliques e triângulos no grafo")
    st.write("A análise de cliques é feita na componente gigante (grafo não direcionado).")

    Hu, max_clique, tri_sorted = clique_analysis(G)

    st.markdown("### Maior clique")
    if max_clique:
        st.metric("Tamanho da maior clique", len(max_clique))
        st.write(", ".join(sorted(max_clique)))
    else:
        st.warning("Não foi possível identificar uma clique.")

    st.divider()
    st.markdown("### Participação em triângulos")
    topn = st.slider("Top N clubes por triângulos", 5, 50, 15)
    tri_df = pd.DataFrame(tri_sorted[:topn], columns=["clube", "número de triângulos"])
    st.dataframe(tri_df, use_container_width=True)

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
            "Interpretação: a árvore de abrangência mostra o mínimo necessário para manter a conectividade. "
            "O fato do grafo original ter muito mais arestas indica redundância estrutural."
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

    if Hu.number_of_nodes() == 0:
        st.warning("Grafo vazio.")
    else:
        with st.spinner("Calculando centro(s) e excentricidade..."):
            centers, ecc, min_ecc = center_nodes_and_eccentricity(Hu)

        if not centers:
            st.warning("Não foi possível calcular o centro.")
        else:
            st.metric("Excentricidade mínima (raio do grafo)", f"{min_ecc}")
            center_choice = st.selectbox("Escolha um centro", centers)
            Tcentral = central_tree_from_root(Hu, center_choice)

            st.divider()
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
            plt.axis("off")
            plt.tight_layout()
            st.pyplot(fig)

elif menu == "Cortes de arestas":
    st.subheader("Cortes de arestas (corte mínimo)")

    st.write(
        "O corte mínimo de arestas entre dois clubes indica quantas arestas precisam ser removidas "
        "para desconectá-los."
    )

    Hu = giant_undirected_graph(G)

    clubs = sorted(list(Hu.nodes()))
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Clube A", clubs, index=0)
    with col2:
        b = st.selectbox("Clube B", clubs, index=min(1, len(clubs) - 1))

    st.divider()
    col3, col4, col5 = st.columns([1.2, 1, 1.2])
    with col3:
        show_cut_edges = st.checkbox("Mostrar arestas do corte mínimo (tabela)", value=False)
    with col4:
        plot_cut = st.checkbox("Plotar zoom do corte (recomendado)", value=True)
    with col5:
        hops = st.slider("Zoom (saltos a partir de A e B)", 1, 3, 2)

    if st.button("Calcular corte mínimo"):
        if a == b:
            st.warning("Escolha dois clubes diferentes.")
        else:
            try:
                with st.spinner("Calculando conectividade por arestas..."):
                    kcut = nx.edge_connectivity(Hu, a, b)

                st.success(f"Tamanho do corte mínimo entre {a} e {b}: {kcut}")

                cut = None
                if plot_cut or show_cut_edges:
                    with st.spinner("Calculando conjunto de arestas do corte mínimo..."):
                        cut = nx.minimum_edge_cut(Hu, a, b)

                if plot_cut and cut is not None:
                    st.markdown("### Visualização do corte (zoom)")
                    st.write("Em vermelho, as arestas do corte mínimo.")
                    draw_edge_cut_zoom(Hu, a, b, cut_edges=cut, hops=hops, seed=42)

                if show_cut_edges and cut is not None:
                    cut_list = [(u, v) for (u, v) in cut]
                    st.write(f"Arestas no corte mínimo (total: {len(cut_list)})")
                    st.dataframe(pd.DataFrame(cut_list, columns=["u", "v"]), use_container_width=True)

            except nx.NetworkXError as e:
                st.error(f"Erro: {e}")
            except Exception as e:
                st.error(
                    "Não consegui calcular o corte mínimo (pode ser pesado em alguns grafos). "
                    f"Detalhe: {e}"
                )
