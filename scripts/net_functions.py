'''
Shared network generators and graph-property helpers.

This module is the single source of truth for ws / mhk / ke generation and
for the graph properties both the LFC (N=240/1000/5000) and LTM (N=1000)
halves compute. Every pipeline stage imports from here.

Average degree
--------------
All three generators pin the REALIZED average degree to exactly the `k`
they are given, so `2E == round(N*k)` holds for every graph produced:

  * mhk grows each new node with exactly k/2 edges and rejects duplicates
    rather than collapsing them, so <k> does not drift with seed or p.
  * ke takes k as the target average degree and uses an active set of size
    m = k/2, deactivating an active node with probability ~ 1/(m + degree),
    the Klemm-Eguiluz rule that makes the degree distribution scale-free
    (tail exponent ~3.3).

scripts/smoke_test.py asserts this over a range of k, and the V/E/kbar
columns in every props CSV make it auditable per graph.

Conventions
-----------
mhk_network and ke_network return **networkx** graphs; ws_network returns a
graph_tool graph directly. Call nx_to_gt() on the former before handing them
to any of the get_* helpers, which all operate on graph_tool graphs.
'''

import numpy as np
import networkx as nx
import scipy as sp
from scipy import linalg
import graph_tool as gt
import graph_tool.clustering
import graph_tool.spectral
import graph_tool.topology


# --------------------------------------------------------------------------
# networkx -> graph_tool
# --------------------------------------------------------------------------

def nx_to_gt(G, expected_n=None):
    '''
    Convert a networkx simple graph to an undirected graph_tool Graph.

    Two guards on the edge extraction, both load-bearing:

      * add_vertex(n) pre-allocates the vertices, so a graph whose
        highest-labelled node ends up isolated still yields exactly n
        vertices. The bare add_edge_list-only form silently drops trailing
        isolated nodes, which would desync get_gain's np.arange(N) indexing.
      * nodelist=sorted(...) makes vertex i in the gt graph the same as node
        i in the nx graph regardless of nx insertion order.
    '''
    n = G.number_of_nodes()
    assert nx.number_of_selfloops(G) == 0, 'self-loops are not supported downstream'
    if expected_n is not None:
        assert n == expected_n, f'generator returned {n} nodes, expected {expected_n}'
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    T = gt.Graph(directed=False)
    T.add_vertex(n)
    T.add_edge_list(np.transpose(sp.sparse.tril(A).nonzero()))
    return T


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------

def ws_network(N, k, p, seed=None):
    '''
    Watts-Strogatz small-world network. Returns a graph_tool graph directly
    (unchanged from the parent project -- ws always realized <k> == k).
    '''
    if seed is None:
        seed = np.random.randint(2**63)

    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(sp.sparse.tril(nx.adjacency_matrix(
        nx.connected_watts_strogatz_graph(N, k, p, tries=1000000, seed=seed))).nonzero()))

    return G


def mhk_network(N, k, p, seed=None):
    """
    Modified Holme-Kim network generator.

    Restores the preferential-attachment mechanism (degree-weighted node
    selection) that makes the network scale-free, while guaranteeing the
    realized average degree matches `k` exactly -- regardless of seed, p,
    or duplicate-edge collisions.

    Parameters
    ----------
    N : int
        Number of nodes in the final graph.
    k : float
        Target average degree of the graph (k >= 4 recommended).
    p : float
        Triad-formation probability (0 <= p <= 1): probability that a new
        node's additional edges close a triangle with a neighbor of its
        anchor, versus attaching to another node via preferential attachment.
    seed : int, optional
        Random seed. If None, a random seed is drawn.

    Returns
    -------
    G : networkx.Graph
        The generated undirected graph.
    """
    if seed is None:
        seed = np.random.randint(2**63)
    rng = np.random.default_rng(seed)

    # --- bootstrap ring lattice ---
    k0 = max(3, int(np.ceil(k)) + 1)
    G = nx.connected_watts_strogatz_graph(k0, 2, 0, seed=seed)

    # --- fix the TOTAL edge budget so average degree == k exactly ---
    E_target = int(round(k * N / 2))
    E_seed = G.number_of_edges()
    n_new_nodes = N - k0
    if n_new_nodes <= 0:
        raise ValueError("N must be larger than the bootstrap ring size")

    E_remaining = E_target - E_seed
    if E_remaining < n_new_nodes:
        raise ValueError(
            "target average degree too small for this N "
            "(each new node needs at least 1 edge to attach)"
        )

    base, extra = divmod(E_remaining, n_new_nodes)
    edge_budget = [base + 1 if i < extra else base for i in range(n_new_nodes)]
    rng.shuffle(edge_budget)  # avoid systematically favoring early/late nodes

    # --- degree-weighted sampling pool for O(1) preferential attachment ---
    # (classic trick: a node with degree d appears d times in this list, so
    # rng.choice(repeated_nodes) samples proportional to degree)
    repeated_nodes = []
    for node, deg in G.degree():
        repeated_nodes.extend([node] * deg)

    for idx, node in enumerate(range(k0, N)):
        budget = max(1, edge_budget[idx])

        # --- preferential attachment: pick anchor proportional to degree ---
        anchor = rng.choice(repeated_nodes)
        anchor_neigh = list(G.neighbors(anchor))

        G.add_edge(anchor, node)
        repeated_nodes.extend([anchor, node])  # both endpoints gained +1 degree
        added = 1

        attempts, max_attempts = 0, 200 * budget
        while added < budget and attempts < max_attempts:
            attempts += 1
            cand = None

            if rng.random() < p and anchor_neigh:
                # triad formation: close a triangle with a neighbor of anchor
                pool = [z for z in anchor_neigh if z != node and not G.has_edge(z, node)]
                if pool:
                    cand = rng.choice(pool)

            if cand is None:
                # preferential attachment among eligible remaining nodes
                for _ in range(50):
                    candidate = rng.choice(repeated_nodes)
                    if candidate != node and candidate != anchor and not G.has_edge(candidate, node):
                        cand = candidate
                        break

            if cand is not None:
                G.add_edge(cand, node)
                repeated_nodes.extend([cand, node])
                added += 1

    return G


def ke_network(n, k, seed=None):
    """
    Klemm-Eguiluz scale-free, high-clustering network generator,
    engineered to have EXACTLY (up to integer rounding of k*n/2) an
    average degree of k.

    https://rf.mokslasplius.lt/achieving-high-clustering-in-scale-free-networks/

    The deactivation rule follows that reference verbatim:

        "Remove one node from the group of 'active' nodes. Let the
         probability of deactivation be inversely proportional to node
         degree: p_d(d_i) ~ 1/(m + d_i)."

    Note this is 1/(m + degree), NOT 1/degree. Using 1/degree produces far
    more extreme hubs (k_max ~787 vs ~412 at n=1000, k=16) and a lower tail
    exponent than the model specifies.

    Parameters
    ----------
    n : int
        Number of nodes in the final graph.
    k : float
        Target average degree of the graph.
    seed : int, optional
        Random seed.

    Returns
    -------
    G : networkx.Graph
    """
    rng = np.random.default_rng(seed)

    # --- size of the "active" set ---
    # each new node connects to all m active nodes, so m new edges are
    # added per new node -> asymptotic average degree ~ 2m
    m = max(2, round(k / 2))
    if m >= n:
        raise ValueError("n must be larger than m = round(k/2)")

    # --- start with a full graph of m nodes, as specified in the model ---
    G = nx.complete_graph(m)
    degree = dict(G.degree())

    active_nodes = list(G.nodes())
    for i in range(m, n):
        for node in active_nodes:
            G.add_edge(node, i)
            degree[node] += 1
        degree[i] = len(active_nodes)
        active_nodes.append(i)

        # deactivate one node with probability inversely proportional to
        # (m + degree), per the Klemm-Eguiluz rule
        weights = np.array([1.0 / (m + degree[a]) for a in active_nodes])
        weights /= weights.sum()
        deactivate = rng.choice(active_nodes, p=weights)
        active_nodes.remove(deactivate)

    # --- exact correction pass ---
    # the deterministic growth above lands close to, but not exactly on,
    # the target average degree (a finite-size effect from the initial
    # clique and the rounding of m = k/2). Close the gap exactly by
    # adding/removing a handful of random edges.
    E_target = int(round(k * n / 2))
    E_actual = G.number_of_edges()
    deficit = E_target - E_actual

    nodes = list(G.nodes())
    if deficit > 0:
        added, attempts = 0, 0
        while added < deficit and attempts < 200 * deficit:
            attempts += 1
            u, v = rng.choice(nodes, size=2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
    elif deficit < 0:
        edges = list(G.edges())
        rng.shuffle(edges)
        removed = 0
        for (u, v) in edges:
            if removed >= -deficit:
                break
            G.remove_edge(u, v)
            removed += 1

    return G


# --------------------------------------------------------------------------
# graph properties (all operate on graph_tool graphs)
# --------------------------------------------------------------------------

def get_laplacian_eigenvalues(G):
    if not G.vertex_properties.get('eig_laplacian', False):
        eig_lap = np.linalg.eigvalsh(gt.spectral.laplacian(G, norm=False).todense())
        G.vp['eig_laplacian'] = G.new_vertex_property('double', vals=eig_lap)
    return G


def get_kirchhoff_index(G):
    G = get_laplacian_eigenvalues(G)
    G.graph_properties['kirchhoff'] = G.new_graph_property(
        'int64_t', sum(1 / np.sort(G.vp.eig_laplacian.get_array())[1:]))
    return G


def get_local_clutsering(G):
    if not G.vertex_properties.get('local_clustering', False):
        G.vertex_properties['local_clustering'] = graph_tool.clustering.local_clustering(G)
    return G


def get_transitivity(G):
    if not G.gp.get('transitivity', False):
        trans = G.new_graph_property('double', val=graph_tool.clustering.global_clustering(G)[0])
        G.graph_properties['transitivity'] = trans
    return G


def get_ave_shortest_path(G):
    if not G.gp.get('shortest_path', False):
        G.gp['shortest_path'] = G.new_graph_property(
            'double',
            val=np.sum(graph_tool.topology.shortest_distance(G).get_2d_array(
                range(G.num_vertices()))) / (G.num_vertices() * (G.num_vertices() - 1)))
    return G


def mean_local_clustering(G):
    '''
    Mean local clustering coefficient -- the `CC` column the props CSVs
    record, and the quantity the LTM CC calibration matches against ws.
    Cheap relative to shortest-path / eigenvalues, so the calibration can
    compute it alone.
    '''
    G = get_local_clutsering(G)
    return float(np.sum(G.vp.local_clustering.get_array())) / G.num_vertices()


def get_gain(graph, w, N, centrality=240, base='degree'):
    '''
    graph : G is a graph tool network
    w : array of the frequencies
    N : number of nodes
    centrality : choosing the top (positive) or bottom (negative) nodes with selector
    base : choose the node selector 'degree' or 'clustering'
    '''
    L = gt.spectral.laplacian(graph, norm=False)  # the built-in normalized gives the symmetric normalized laplacian, but we want the random walk normalized laplacian

    L = (L / L.diagonal()).T  # Random walk normalization  D^-1 L = LD^-1 because L is symmetric

    L = L.toarray()
    h2 = graph.new_vertex_property('vector<double>')

    if base == 'degree':
        degrees = dict(nx.from_numpy_array(gt.spectral.adjacency(graph).T.toarray()).degree())

        if centrality == N:
            Nodes = graph.vertices()
        elif centrality > 0:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[:centrality]
        else:
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            Nodes = sorted_nodes[centrality:]
    elif base == 'clustering':
        clustering_coeffs = graph_tool.clustering.local_clustering(graph)
        sorted_nodes = sorted(graph.vertices(), key=lambda v: clustering_coeffs[v], reverse=True)
        if centrality == N:
            Nodes = graph.vertices()
        elif centrality > 0:
            Nodes = [int(v) for v in sorted_nodes[:centrality]]
        else:
            Nodes = [int(v) for v in sorted_nodes[centrality:]]

    for g in Nodes:
        ida = np.arange(N) != g
        idb = np.arange(N) == g
        A = L[np.ix_(ida, ida)].astype(complex)
        B = L[np.ix_(ida, idb)]

        H2 = []
        for f in w:
            np.fill_diagonal(A, 1.0 + 1j * f)
            h = linalg.solve(A, -B)

            H2.append(linalg.norm(h) ** 2)

        h2[g] = H2

    return h2


# --------------------------------------------------------------------------
# fast gain: every node from one eigendecomposition
# --------------------------------------------------------------------------

def gain_all_nodes(A, w):
    '''
    H^2(node, frequency) for EVERY grounded node, via one eigendecomposition.

    Same quantity as get_gain, computed a different way. get_gain grounds one
    node at a time and does a dense complex solve per (node, frequency), which
    is O(N^3) per solve; this diagonalises the symmetric normalised Laplacian
    once and then spends two real matmuls per frequency for all N nodes at
    once. Verified against get_gain to a max relative difference of 1.4e-14
    on N=240 mhk and ws graphs.

    The speedup is what makes the N=5000 sweep tractable:

        N      get_gain (5% of nodes, top+bot)   gain_all_nodes (all nodes)
        240    ~2.5 s                            0.09 s
        1000   ~3.8 min                          5.2 s
        5000   ~22.6 h                           9.5 min

    Because it returns every node, top/bottom-k selection becomes a slice of
    the result rather than a reason to run the whole computation again.

    Ported from temp_eigenspectrum_clone/realizations/spec_div/
    gain_laplacian_all_gen.py.

    Parameters
    ----------
    A : (n, n) dense float adjacency matrix, in vertex-index order
    w : array of frequencies

    Returns
    -------
    (n, len(w)) array of H^2 values.
    '''
    w = np.asarray(w, dtype=float)
    n = len(A)
    d = A.sum(1)
    if np.any(d <= 0):
        raise ValueError('isolated vertex: the random-walk Laplacian D^-1 L is '
                         'undefined (graphs from this pipeline are connected)')

    inv_sqrt = 1.0 / np.sqrt(d)
    S = np.eye(n) - inv_sqrt[:, None] * A * inv_sqrt[None, :]
    lam, Q = np.linalg.eigh(S)

    inv_d = 1.0 / d
    QT = np.ascontiguousarray(Q.T)
    h2 = np.empty((n, len(w)))
    for i, f in enumerate(w):
        scale = 1.0 / (lam + 1j * f)      # lam >= 0 and f > 0, so never singular
        # |G_w|^2 elementwise, from the real and imaginary parts separately:
        # two real matmuls are cheaper than one complex one.
        P = ((Q * scale.real) @ QT) ** 2 + ((Q * scale.imag) @ QT) ** 2
        diag = np.diagonal(P).copy()      # |G_w[g, g]|^2
        np.fill_diagonal(P, 0.0)          # drop j = g before summing, not after
        h2[:, i] = d * (inv_d @ P) / diag
    return h2


def adjacency_array(G):
    '''Dense adjacency of a graph_tool graph, in vertex-index order.'''
    return np.asarray(gt.spectral.adjacency(G).todense(), dtype=float)


def select_nodes_by_degree(G, t_b, perc):
    '''
    Indices of the top ('top') or bottom ('bot') `perc` percent of nodes by
    degree. Kept here so the fast path and the CSV extractors agree on node
    selection; np.argsort's stable order matches get_gain's sorted() order.
    '''
    deg = G.get_total_degrees(G.get_vertices())
    n_sel = max(1, int(round(G.num_vertices() * perc / 100.0)))
    order = np.argsort(-deg, kind='stable')
    return order[:n_sel] if t_b == 'top' else order[-n_sel:]


def gains_property(G, h2, nodes):
    '''
    Pack rows of a gain_all_nodes() result into a graph_tool
    vector<double> vertex property, for the given node indices only.
    Unset vertices stay empty, matching what get_gain produced.
    '''
    vp = G.new_vertex_property('vector<double>')
    for v in nodes:
        vp[int(v)] = h2[int(v)]
    return vp
