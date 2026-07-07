"""Topological single-stroke solver.

Input: the artwork as a set of polylines ("ink"). Output: ONE continuous
polyline that draws every input line exactly once, plus the minimum extra
pen travel needed to make that possible:

  - RETRACE segments re-travel a line already drawn (invisible on screen),
  - CONNECTOR segments are new visible lines bridging disjoint pieces.

Stages: build a topological multigraph (nodes only at endpoints/junctions,
edges carry whole polylines) -> join components with short bridges that
prefer landing on degree-1 endpoints (connects AND fixes parity in one
move) -> Eulerize by min-cost matching of odd-degree nodes, choosing per
pair between retracing a graph path and adding a bridge -> Hierholzer
Euler path -> stitch edge polylines into one point list.

This replaces the old per-pixel-node approach (Art/graphMods.py) which
built 100k+ node multigraphs and Eulerized with distance-threshold
heuristics.
"""

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial import QhullError

from .geometry import as_polyline, dedupe, densify, polyline_length

# Segment classes in solver output
INK = 0
RETRACE = 1
CONNECTOR = 2

CLASS_NAMES = {INK: "ink", RETRACE: "retrace", CONNECTOR: "connector"}


@dataclass
class Edge:
    a: int                 # node id, pts[0] sits exactly on node a
    b: int                 # node id, pts[-1] sits exactly on node b
    pts: np.ndarray        # (N,2) float32
    length: float
    kind: int              # INK / RETRACE / CONNECTOR
    alive: bool = True


@dataclass
class SolveResult:
    points: np.ndarray       # (P,2) float32, the single stroke
    seg_class: np.ndarray    # (P-1,) uint8, class of segment i->i+1
    n_components: int = 1    # components before joining
    n_odd: int = 0           # odd nodes before Eulerization


class StrokeGraph:
    """Multigraph over polyline endpoints/junctions.

    Node identity comes from quantizing coordinates to a grid (default
    0.75px) so endpoints that were meant to coincide actually share a node.
    """

    def __init__(self, quant=0.75):
        self.quant = quant
        self.node_xy = []            # canonical coordinate per node
        self._key_to_node = {}
        self.adj = []                # node -> list of edge ids
        self.edges = []
        self._splits = {}            # dead edge id -> (cut_vidx, e1, e2)

    # -- construction ------------------------------------------------------

    def _node_for(self, p):
        key = (int(round(p[0] / self.quant)), int(round(p[1] / self.quant)))
        n = self._key_to_node.get(key)
        if n is None:
            n = len(self.node_xy)
            self._key_to_node[key] = n
            self.node_xy.append((float(p[0]), float(p[1])))
            self.adj.append([])
        return n

    def add_polyline(self, pts, kind=INK):
        pts = dedupe(as_polyline(pts))
        if len(pts) < 2:
            return None
        a = self._node_for(pts[0])
        b = self._node_for(pts[-1])
        pts = pts.copy()
        pts[0] = self.node_xy[a]
        pts[-1] = self.node_xy[b]
        length = polyline_length(pts)
        if a == b and length < self.quant * 2:
            return None  # degenerate speck
        eid = len(self.edges)
        self.edges.append(Edge(a, b, pts, length, kind))
        self.adj[a].append(eid)
        self.adj[b].append(eid)
        return eid

    # -- basic queries -----------------------------------------------------

    def degree(self, n):
        d = 0
        for eid in self.adj[n]:
            e = self.edges[eid]
            if not e.alive:
                continue
            d += 2 if e.a == e.b else 1
        return d

    def odd_nodes(self):
        return [n for n in range(len(self.node_xy)) if self.degree(n) % 2 == 1]

    def alive_edges(self):
        return [i for i, e in enumerate(self.edges) if e.alive]

    def other(self, eid, n):
        e = self.edges[eid]
        return e.b if e.a == n else e.a

    def n_alive_nodes(self):
        return sum(1 for n in range(len(self.node_xy)) if any(
            self.edges[eid].alive for eid in self.adj[n]))

    # -- mutation ----------------------------------------------------------

    def split_edge(self, eid, vertex_idx):
        """Split edge at one of its interior vertices; returns (new_node, e1, e2)."""
        e = self.edges[eid]
        assert e.alive and 0 < vertex_idx < len(e.pts) - 1
        p = e.pts[vertex_idx]
        n = self._node_for(p)
        if n == e.a or n == e.b:
            return None  # too close to an existing endpoint to be worth splitting
        pts1 = e.pts[:vertex_idx + 1].copy()
        pts2 = e.pts[vertex_idx:].copy()
        pts1[-1] = self.node_xy[n]
        pts2[0] = self.node_xy[n]
        e.alive = False
        self.adj[e.a].remove(eid)
        self.adj[e.b].remove(eid)
        e1 = len(self.edges)
        self.edges.append(Edge(e.a, n, pts1, polyline_length(pts1), e.kind))
        self.adj[e.a].append(e1)
        self.adj[n].append(e1)
        e2 = len(self.edges)
        self.edges.append(Edge(n, e.b, pts2, polyline_length(pts2), e.kind))
        self.adj[n].append(e2)
        self.adj[e.b].append(e2)
        self._splits[eid] = (vertex_idx, e1, e2)
        return n, e1, e2

    def resolve_sample(self, eid, vidx):
        """Follow splits so (edge, vertex) references stay valid."""
        while not self.edges[eid].alive:
            info = self._splits.get(eid)
            if info is None:
                # edge died by rewiring, not splitting; snap to nearest end
                e = self.edges[eid]
                vidx = 0 if vidx < len(e.pts) // 2 else len(e.pts) - 1
                return eid, vidx
            cut, e1, e2 = info
            if vidx <= cut:
                eid = e1
            else:
                eid, vidx = e2, vidx - cut
        return eid, vidx

    def add_connector(self, na, nb, pts=None, kind=CONNECTOR):
        if pts is None:
            pts = np.array([self.node_xy[na], self.node_xy[nb]], dtype=np.float32)
        else:
            pts = as_polyline(pts).copy()
            pts[0] = self.node_xy[na]
            pts[-1] = self.node_xy[nb]
        eid = len(self.edges)
        self.edges.append(Edge(na, nb, pts, polyline_length(pts), kind))
        self.adj[na].append(eid)
        self.adj[nb].append(eid)
        return eid

    def duplicate_edge(self, eid, kind=RETRACE):
        e = self.edges[eid]
        return self.add_connector(e.a, e.b, e.pts, kind=kind)

    # -- components --------------------------------------------------------

    def component_labels(self):
        """Union-find over nodes; returns (labels array, n_components).

        Nodes with no alive edges get label -1 and don't count.
        """
        parent = list(range(len(self.node_xy)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self.edges:
            if not e.alive:
                continue
            ra, rb = find(e.a), find(e.b)
            if ra != rb:
                parent[ra] = rb
        labels = np.full(len(self.node_xy), -1, dtype=np.int64)
        has_edge = [False] * len(self.node_xy)
        for e in self.edges:
            if e.alive:
                has_edge[e.a] = has_edge[e.b] = True
        roots = {}
        for n in range(len(self.node_xy)):
            if not has_edge[n]:
                continue
            r = find(n)
            if r not in roots:
                roots[r] = len(roots)
            labels[n] = roots[r]
        return labels, len(roots)


# ---------------------------------------------------------------------------
# Endpoint snapping (linemerge + weld-to-geometry)
# ---------------------------------------------------------------------------

def _project_point_to_segment(p, a, b):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    q = a + t * ab
    return q, t, float(np.hypot(*(p - q)))


def _split_at_projection(graph, eid, seg_idx, q):
    """Split edge at point q on segment seg_idx (inserting a vertex if
    needed). Returns (node_at_q, left_piece_edge_id)."""
    e = graph.edges[eid]
    pts = e.pts
    if np.hypot(*(pts[seg_idx] - q)) < graph.quant:
        vidx = seg_idx
    elif np.hypot(*(pts[seg_idx + 1] - q)) < graph.quant:
        vidx = seg_idx + 1
    else:
        e.pts = np.insert(pts, seg_idx + 1, q, axis=0).astype(np.float32)
        vidx = seg_idx + 1
    if vidx <= 0:
        return e.a, eid
    if vidx >= len(e.pts) - 1:
        return e.b, eid
    out = graph.split_edge(eid, vidx)
    if out is None:  # quantized onto an endpoint
        da = np.hypot(*(np.asarray(graph.node_xy[e.a]) - q))
        db = np.hypot(*(np.asarray(graph.node_xy[e.b]) - q))
        return (e.a if da <= db else e.b), eid
    n, e1, _ = out
    return n, e1


def _rewire_endpoint(graph, eid, n, m):
    """Move edge eid's endpoint from node n to node m (the snap)."""
    e = graph.edges[eid]
    if n == m or (e.a == n and e.b == n):
        return False
    other = e.b if e.a == n else e.a
    if m == other and e.length < graph.quant * 4:
        return False  # would collapse into a speck self-loop
    if e.a == n:
        e.pts = e.pts.copy()
        e.pts[0] = graph.node_xy[m]
        e.a = m
    else:
        e.pts = e.pts.copy()
        e.pts[-1] = graph.node_xy[m]
        e.b = m
    e.length = polyline_length(e.pts)
    graph.adj[n].remove(eid)
    graph.adj[m].append(eid)
    return True


def snap_endpoints(graph, snap_radius, sample_step=4.0):
    """Weld dangling polyline ends onto nearby geometry.

    For every degree-1 node within snap_radius of another polyline, the
    exact projection point is inserted into that polyline, the polyline is
    split there, and the dangling end is moved onto the new shared node.
    This merges components and creates proper T-junctions - the main
    defense against connector clutter. Returns number of snaps done.
    """
    if snap_radius <= 0:
        return 0
    step = min(sample_step, max(2.0, snap_radius))
    points, eids, vidx = _edge_samples(graph, step)
    if len(points) == 0:
        return 0
    tree = cKDTree(points)
    guard = int(np.ceil(3.0 * snap_radius / step)) + 1

    # collect candidate snaps first (graph unmodified while scanning)
    jobs = []
    for n in range(len(graph.node_xy)):
        if graph.degree(n) != 1:
            continue
        own = next(e for e in graph.adj[n] if graph.edges[e].alive)
        e = graph.edges[own]
        own_end = 0 if (e.a == n) else len(e.pts) - 1
        p = np.asarray(graph.node_xy[n])
        best = None
        for ci in tree.query_ball_point(p, snap_radius * 1.5):
            teid, tv = int(eids[ci]), int(vidx[ci])
            if teid == own and abs(tv - own_end) < guard:
                continue  # don't snap an end back onto its own tip
            tpts = graph.edges[teid].pts
            for seg in (tv - 1, tv):
                if not (0 <= seg < len(tpts) - 1):
                    continue
                q, t, d = _project_point_to_segment(p, tpts[seg], tpts[seg + 1])
                if best is None or d < best[0]:
                    best = (d, teid, seg, t, q)
        if best is not None and best[0] <= snap_radius:
            jobs.append((n, own, best))

    # group by target edge; process each edge's hits from far end backwards
    # so earlier indices stay valid through inserts/splits
    by_edge = {}
    for n, own, (d, teid, seg, t, q) in jobs:
        by_edge.setdefault(teid, []).append((seg, t, q, n, own))
    snapped = 0
    for teid, hits in by_edge.items():
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        cur = teid
        for seg, t, q, n, own in hits:
            if graph.degree(n) != 1:
                continue  # this endpoint got merged by an earlier snap
            # own edge may have been split; find which piece still ends at n
            own_live = next((e for e in graph.adj[n] if graph.edges[e].alive), None)
            if own_live is None:
                continue
            if not graph.edges[cur].alive:
                cur, _ = graph.resolve_sample(cur, seg)
            if cur == own_live:
                continue
            if seg >= len(graph.edges[cur].pts) - 1:
                seg = len(graph.edges[cur].pts) - 2
            m, left = _split_at_projection(graph, cur, seg, q)
            if _rewire_endpoint(graph, own_live, n, m):
                snapped += 1
            cur = left
    return snapped


# ---------------------------------------------------------------------------
# Component joining
# ---------------------------------------------------------------------------

def _edge_samples(graph, sample_step):
    """Sample points along every alive edge for nearest-pair queries.

    Edges are densified in place (shape-preserving) so any sample can later
    become a split point. Returns (points array, edge id per sample,
    vertex index per sample).
    """
    pts_list, eids, vidx = [], [], []
    for eid in graph.alive_edges():
        e = graph.edges[eid]
        e.pts = densify(e.pts, sample_step)
        e.length = polyline_length(e.pts)
        n = len(e.pts)
        pts_list.append(e.pts)
        eids.append(np.full(n, eid, dtype=np.int64))
        vidx.append(np.arange(n, dtype=np.int64))
    if not pts_list:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64))
    return np.concatenate(pts_list), np.concatenate(eids), np.concatenate(vidx)


def _candidate_pairs(points, labels):
    """Cross-component candidate connector pairs (i, j) into `points`.

    Uses Delaunay triangulation: the inter-point MST is a subset of the
    Delaunay graph, so cross-component Delaunay edges are enough to build
    the optimal component MST. Falls back to all-pairs for tiny inputs or
    degenerate geometry.
    """
    n = len(points)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    if n > 4:
        try:
            tri = Delaunay(np.asarray(points, dtype=np.float64), qhull_options="QJ")
            s = tri.simplices
            pairs = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
            pairs = np.sort(pairs, axis=1)
            pairs = np.unique(pairs, axis=0)
        except (QhullError, ValueError):
            pairs = None
    else:
        pairs = None
    if pairs is None:
        ii, jj = np.triu_indices(n, k=1)
        pairs = np.stack([ii, jj], axis=1)
    cross = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    return pairs[cross]


def _attach_node(graph, sample_pt, eid, vidx):
    """Turn a sample point into a graph node to hang a connector on."""
    e = graph.edges[eid]
    if vidx == 0:
        return e.a
    if vidx == len(e.pts) - 1:
        return e.b
    split = graph.split_edge(eid, vidx)
    if split is None:
        # quantized onto an endpoint; pick the closer one
        da = np.hypot(*(np.asarray(graph.node_xy[e.a]) - sample_pt))
        db = np.hypot(*(np.asarray(graph.node_xy[e.b]) - sample_pt))
        return e.a if da <= db else e.b
    return split[0]


def connect_components(graph, sample_step=4.0, endpoint_factor=0.7,
                       endpoint_radius_scale=1.5, router=None):
    """Join all components with near-minimal total connector length.

    Kruskal over cross-component Delaunay candidate pairs. For each chosen
    pair, nearby degree-1 endpoints are preferred as attachment points
    (slightly longer bridge accepted at `endpoint_factor` discount) because
    a bridge landing on two odd endpoints also fixes parity for free.
    """
    labels, n_comp = graph.component_labels()
    if n_comp <= 1:
        return n_comp
    points, eids, vidx = _edge_samples(graph, sample_step)
    slabels = labels[[graph.edges[e].a for e in eids]]
    pairs = _candidate_pairs(points, slabels)
    if len(pairs) == 0:
        return n_comp
    d = np.hypot(*(points[pairs[:, 0]] - points[pairs[:, 1]]).T)
    order = np.argsort(d)

    # Union-find over component labels
    parent = list(range(n_comp))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Endpoint preference: KD-tree over samples that are degree-1 nodes
    def deg1_samples():
        idx = []
        for i in range(len(points)):
            e = graph.edges[eids[i]]
            if not e.alive:
                continue
            if vidx[i] == 0 and graph.degree(e.a) == 1:
                idx.append(i)
            elif vidx[i] == len(e.pts) - 1 and graph.degree(e.b) == 1:
                idx.append(i)
        return np.array(idx, dtype=np.int64)

    d1 = deg1_samples()
    d1_tree = cKDTree(points[d1]) if len(d1) else None

    def prefer_endpoint(i, other_pt, base_d):
        """Maybe swap sample i for a nearby degree-1 endpoint sample."""
        if d1_tree is None:
            return i
        r = base_d * endpoint_radius_scale + 2.0
        cand = d1_tree.query_ball_point(points[i], r)
        best, best_cost = i, base_d
        for c in cand:
            gi = d1[c]
            e = graph.edges[eids[gi]]
            if slabels[gi] != slabels[i] or not e.alive:
                continue
            node = e.a if vidx[gi] == 0 else e.b
            if graph.degree(node) != 1:
                continue  # got a connector since the d1 list was built
            cost = np.hypot(*(points[gi] - other_pt)) * endpoint_factor
            if cost < best_cost:
                best, best_cost = gi, cost
        return best

    joined = 0
    for k in order:
        i, j = pairs[k]
        li, lj = find(slabels[i]), find(slabels[j])
        if li == lj:
            continue
        # samples may reference edges that were since split; follow the splits
        eids[i], vidx[i] = graph.resolve_sample(int(eids[i]), int(vidx[i]))
        eids[j], vidx[j] = graph.resolve_sample(int(eids[j]), int(vidx[j]))
        base_d = float(np.hypot(*(points[i] - points[j])))
        i2 = prefer_endpoint(i, points[j], base_d)
        j2 = prefer_endpoint(j, points[i2], np.hypot(*(points[i2] - points[j])))
        na = _attach_node(graph, points[i2], int(eids[i2]), int(vidx[i2]))
        nb = _attach_node(graph, points[j2], int(eids[j2]), int(vidx[j2]))
        pts = None
        if router is not None:
            pts = router(graph.node_xy[na], graph.node_xy[nb])
        graph.add_connector(na, nb, pts)
        parent[find(li)] = find(lj)
        joined += 1
        if joined == n_comp - 1:
            break
    return n_comp


# ---------------------------------------------------------------------------
# Eulerization
# ---------------------------------------------------------------------------

def _bounded_dijkstra(graph, src, targets, cutoff):
    """Graph distances from src to any of `targets`, ignoring paths > cutoff."""
    dist = {src: 0.0}
    heap = [(0.0, src)]
    tset = set(targets)
    found = {}
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        if u in tset and u not in found:
            found[u] = d
            if len(found) == len(tset):
                break
        if d > cutoff:
            break
        for eid in graph.adj[u]:
            e = graph.edges[eid]
            if not e.alive:
                continue
            v = e.b if e.a == u else e.a
            nd = d + e.length
            if nd < dist.get(v, np.inf) and nd <= cutoff:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return found


def _dijkstra_path(graph, src, dst, cutoff=np.inf):
    """Shortest path src->dst; returns list of edge ids (or None)."""
    dist = {src: 0.0}
    prev = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            path = []
            while u != src:
                eid, u = prev[u]
                path.append(eid)
            path.reverse()
            return path
        if d > dist.get(u, np.inf) or d > cutoff:
            continue
        for eid in graph.adj[u]:
            e = graph.edges[eid]
            if not e.alive:
                continue
            v = e.b if e.a == u else e.a
            nd = d + e.length
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = (eid, u)
                heapq.heappush(heap, (nd, v))
    return None


def _segment_brightness(tone, p, q, n_samples=8):
    """Mean brightness (0 dark .. 1 light) under segment p-q; 0.5 if no tone."""
    if tone is None:
        return 0.5
    h, w = tone.shape
    t = np.linspace(0.0, 1.0, n_samples)
    xs = np.clip(p[0] + t * (q[0] - p[0]), 0, w - 1).astype(np.int64)
    ys = np.clip(p[1] + t * (q[1] - p[1]), 0, h - 1).astype(np.int64)
    return float(tone[ys, xs].mean())


def eulerize(graph, tone=None, k_candidates=8, blossom_max=400,
             bridge_penalty=6.0, retrace_weight=0.35, keep_open=True,
             router=None):
    """Make all node degrees even (except up to 2) by pairing odd nodes.

    Per matched pair the solver picks the cheaper of:
      - RETRACE: duplicate the shortest graph path between them (invisible;
        cost = path length * retrace_weight), or
      - CONNECTOR: a new straight/routed bridge (visible; cost = euclidean
        * (1 + bridge_penalty * brightness under it)).
    With keep_open, the most expensive matched pair is skipped, leaving the
    two Euler-path endpoints there for free.
    Returns the number of odd nodes it started with.
    """
    odd = graph.odd_nodes()
    n_odd_initial = len(odd)
    if len(odd) <= 2:
        return n_odd_initial

    coords = np.array([graph.node_xy[n] for n in odd], dtype=np.float64)
    tree = cKDTree(coords)
    k = min(k_candidates + 1, len(odd))
    dists, nbrs = tree.query(coords, k=k)

    def bridge_cost(i, j):
        d = float(np.hypot(*(coords[i] - coords[j])))
        b = _segment_brightness(tone, coords[i], coords[j])
        return d * (1.0 + bridge_penalty * b), d

    # Pair costs over the sparse candidate set
    pair_info = {}
    cand_per_node = [[] for _ in range(len(odd))]
    for i in range(len(odd)):
        for jj in range(1, k):
            j = int(nbrs[i, jj])
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in pair_info:
                bc, _ = bridge_cost(a, b)
                pair_info[(a, b)] = {"bridge": bc}
            cand_per_node[i].append(j)

    # Retrace option: bounded Dijkstra per odd node to its candidates
    for i in range(len(odd)):
        targets = {odd[j]: j for j in cand_per_node[i] if i < j}
        if not targets:
            continue
        cutoff = max(pair_info[(i, j)]["bridge"] for j in cand_per_node[i] if i < j) / max(retrace_weight, 1e-6)
        found = _bounded_dijkstra(graph, odd[i], list(targets.keys()), cutoff)
        for node, d in found.items():
            j = targets[node]
            info = pair_info[(i, j)]
            rc = d * retrace_weight
            if rc < info["bridge"]:
                info["retrace"] = rc

    def pair_cost(a, b):
        info = pair_info.get((a, b))
        if info is None:
            bc, _ = bridge_cost(a, b)
            info = pair_info[(a, b)] = {"bridge": bc}
        rc = info.get("retrace")
        if rc is not None and rc <= info["bridge"]:
            return rc, "retrace"
        return info["bridge"], "bridge"

    # Matching
    matched = {}
    if len(odd) <= blossom_max:
        import networkx as nx
        M = nx.Graph()
        M.add_nodes_from(range(len(odd)))
        for (a, b) in pair_info:
            c, _ = pair_cost(a, b)
            M.add_edge(a, b, weight=-c)
        mate = nx.max_weight_matching(M, maxcardinality=True)
        for a, b in mate:
            a, b = (a, b) if a < b else (b, a)
            matched[a] = b
            matched[b] = a
    else:
        heap = []
        for (a, b) in pair_info:
            c, _ = pair_cost(a, b)
            heap.append((c, a, b))
        heapq.heapify(heap)
        while heap:
            c, a, b = heapq.heappop(heap)
            if a in matched or b in matched:
                continue
            matched[a] = b
            matched[b] = a

    # Leftovers (candidate graph had no perfect matching): greedy nearest
    left = [i for i in range(len(odd)) if i not in matched]
    if left:
        lcoords = coords[left]
        ltree = cKDTree(lcoords)
        used = set()
        for ii, i in enumerate(left):
            if i in used:
                continue
            ds, js = ltree.query(lcoords[ii], k=len(left))
            js = np.atleast_1d(js)
            partner = None
            for j in js:
                cj = left[int(j)]
                if cj != i and cj not in used:
                    partner = cj
                    break
            if partner is None:
                continue
            a, b = (i, partner) if i < partner else (partner, i)
            matched[a] = b
            matched[b] = a
            used.add(i)
            used.add(partner)

    # Collect unique pairs with final costs, optionally skip the worst
    pairs = sorted({(a, b) if a < b else (b, a) for a, b in matched.items()})
    costed = []
    for a, b in pairs:
        c, choice = pair_cost(a, b)
        costed.append((c, a, b, choice))
    if keep_open and costed:
        costed.sort()
        costed = costed[:-1]  # the most expensive pair becomes path endpoints

    for c, a, b, choice in costed:
        na, nb = odd[a], odd[b]
        if choice == "retrace":
            path = _dijkstra_path(graph, na, nb)
            if path is not None:
                for eid in path:
                    graph.duplicate_edge(eid, kind=RETRACE)
                continue
        pts = router(graph.node_xy[na], graph.node_xy[nb]) if router else None
        graph.add_connector(na, nb, pts)

    return n_odd_initial


# ---------------------------------------------------------------------------
# Euler path + stitching
# ---------------------------------------------------------------------------

def euler_path(graph):
    """Hierholzer. Returns ordered [(edge_id, forward)] covering every alive edge."""
    n_edges = len(graph.alive_edges())
    if n_edges == 0:
        return []
    odd = graph.odd_nodes()
    if odd:
        start = min(odd, key=lambda n: graph.node_xy[n])
    else:
        start = min((n for n in range(len(graph.node_xy))
                     if any(graph.edges[e].alive for e in graph.adj[n])),
                    key=lambda n: graph.node_xy[n])

    used = [not e.alive for e in graph.edges]
    ptr = [0] * len(graph.node_xy)
    stack = [(start, -1, True)]
    route = []
    while stack:
        v, in_eid, in_fwd = stack[-1]
        advanced = False
        while ptr[v] < len(graph.adj[v]):
            eid = graph.adj[v][ptr[v]]
            if used[eid]:
                ptr[v] += 1
                continue
            used[eid] = True
            e = graph.edges[eid]
            w = e.b if e.a == v else e.a
            stack.append((w, eid, e.a == v))
            advanced = True
            break
        if not advanced:
            stack.pop()
            if in_eid >= 0:
                route.append((in_eid, in_fwd))
    route.reverse()
    if len(route) != n_edges:
        raise RuntimeError(
            f"Euler path covered {len(route)}/{n_edges} edges - graph not connected?")
    return route


def stitch(graph, route):
    """Concatenate routed edge polylines into one stroke + per-segment classes."""
    if not route:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.uint8))
    parts = []
    classes = []
    first = True
    for eid, fwd in route:
        e = graph.edges[eid]
        pts = e.pts if fwd else e.pts[::-1]
        if not first:
            pts = pts[1:]
        if len(pts) == 0:
            continue
        parts.append(pts)
        n_segs = len(pts) if not first else len(pts) - 1
        classes.append(np.full(n_segs, e.kind, dtype=np.uint8))
        first = False
    points = np.concatenate(parts).astype(np.float32)
    seg_class = np.concatenate(classes)
    assert len(seg_class) == len(points) - 1
    return points, seg_class


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def solve(polylines, tone=None, quant=0.75, snap_radius=2.0, sample_step=4.0,
          endpoint_factor=0.7, k_candidates=8, blossom_max=400,
          bridge_penalty=6.0, retrace_weight=0.35, keep_open=True,
          router=None):
    """polylines -> SolveResult (one continuous stroke).

    tone: optional (H,W) brightness array (0 dark..1 light) in the same
    coordinate space as the polylines; used to route/penalize visible
    bridges through light areas.
    """
    graph = StrokeGraph(quant=quant)
    for pl in polylines:
        graph.add_polyline(pl, kind=INK)
    if not graph.alive_edges():
        return SolveResult(np.zeros((0, 2), np.float32), np.zeros(0, np.uint8), 0, 0)

    snap_endpoints(graph, snap_radius, sample_step=sample_step)
    n_comp = connect_components(graph, sample_step=sample_step,
                                endpoint_factor=endpoint_factor, router=router)
    n_odd = eulerize(graph, tone=tone, k_candidates=k_candidates,
                     blossom_max=blossom_max, bridge_penalty=bridge_penalty,
                     retrace_weight=retrace_weight, keep_open=keep_open,
                     router=router)
    route = euler_path(graph)
    points, seg_class = stitch(graph, route)
    return SolveResult(points, seg_class, n_comp, n_odd)
