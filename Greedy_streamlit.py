import streamlit as st
from typing import Dict, Tuple, List, Any, Optional
import heapq

# ---------- Greedy Best-First core algorithm ----------
def greedy_best_first_search(
    graph: Dict[str, Dict[str, float]],
    start: str,
    goal: Optional[str],
    h: Dict[str, float],
) -> Tuple[List[str], float, List[str], Dict[str, float], Dict[str, Optional[str]], List[Dict[str, Any]]]:
    """
    Run Greedy Best-First Search on a weighted directed graph.
    Priority = h(n). Reports actual path cost (sum of weights) if goal is found.

    Returns
    - path: list of nodes from start to goal (empty if no goal or not found)
    - total_cost: sum of edge weights along returned path (float('inf') if not found)
    - expanded_order: nodes expanded in order
    - g_cost: mapping node -> best seen path cost (not guaranteed optimal)
    - parent: mapping node -> predecessor (for path reconstruction)
    - trace: list of step-by-step snapshots (for UI/debug)
    """
    pq: List[Tuple[float, str]] = []  # (h, node)
    heapq.heappush(pq, (h.get(start, 0.0), start))

    g_cost: Dict[str, float] = {start: 0.0}
    parent: Dict[str, Optional[str]] = {start: None}
    expanded_order: List[str] = []
    visited: set[str] = set()
    trace: List[Dict[str, Any]] = []

    goal_found = goal is None  # if no goal, we’ll traverse reachable nodes

    while pq:
        cur_h, node = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        expanded_order.append(node)

        trace.append({
            "expanded": node,
            "h": cur_h,
            "g": g_cost.get(node, float('inf')),
            "frontier": [(hh, nn) for hh, nn in pq],
            "g_cost": dict(g_cost),
        })

        if goal is not None and node == goal:
            goal_found = True
            break

        for nbr, w in graph.get(node, {}).items():
            if w < 0:
                raise ValueError(f"Negative edge weight detected on {node}->{nbr}: {w}")

            new_g = g_cost[node] + float(w)
            # Standard GBFS doesn't revisit with better g; we keep first-come-best h-priority
            if nbr not in visited and new_g < g_cost.get(nbr, float("inf")):
                g_cost[nbr] = new_g
                parent[nbr] = node
                heapq.heappush(pq, (h.get(nbr, 0.0), nbr))

    path: List[str] = []
    total = float("inf")
    if goal is not None and goal_found:
        total = g_cost.get(goal, float("inf"))
        if total != float("inf"):
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()

    return path, total, expanded_order, g_cost, parent, trace


# ---------- Helpers ----------
def parse_edges(text: str, undirected: bool) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = {}

    def ensure_node(node: str):
        if node not in graph:
            graph[node] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        parts = [p for p in line.replace(',', ' ').split() if p]
        if len(parts) != 3:
            raise ValueError(f"Invalid edge line: '{line}'. Expected: src dst cost")
        u, v, w = parts[0], parts[1], parts[2]
        try:
            w_val = float(w)
        except ValueError:
            raise ValueError(f"Invalid weight in line: '{line}'. Got '{w}'")

        ensure_node(u)
        ensure_node(v)
        graph[u][v] = w_val
        if undirected:
            graph[v][u] = w_val
    return graph


def parse_heuristic(text: str) -> Dict[str, float]:
    h: Dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(',', ' ').split() if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid heuristic line: '{line}'. Expected: node value")
        n, v = parts
        try:
            h[n] = float(v)
        except ValueError:
            raise ValueError(f"Invalid heuristic value for node '{n}': {v}")
    return h


def all_nodes(graph: Dict[str, Dict[str, float]]) -> List[str]:
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        nodes.update(nbrs.keys())
    return sorted(nodes)


def to_graphviz(graph: Dict[str, Dict[str, float]], path: List[str], directed: bool = True) -> str:
    is_on_path = set()
    for i in range(len(path) - 1):
        is_on_path.add((path[i], path[i + 1]))
        if not directed:
            is_on_path.add((path[i + 1], path[i]))

    rankdir = "LR"
    gtype = "digraph" if directed else "graph"
    arrow = "->" if directed else "--"

    lines = [
        f"{gtype} G {{",
        f"  rankdir={rankdir};",
        "  node [shape=circle, fontsize=12, fontname=Helvetica];",
    ]

    nodes = all_nodes(graph)
    for n in nodes:
        lines.append(f'  "{n}";')

    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            on_path = (u, v) in is_on_path
            color = "#d32f2f" if on_path else "#4285f4"
            penwidth = "3" if on_path else "1"
            lines.append(f'  "{u}" {arrow} "{v}" [label="{w}", color="{color}", penwidth={penwidth}];')

    lines.append("}")
    return "\n".join(lines)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Greedy Best-First Search", page_icon="⚡", layout="wide")

st.title("Greedy Best-First Search (GBFS)")
st.caption("Prioritizes nodes with the smallest h(n). Fast but not optimal in general. Provide a heuristic h(n).")

with st.sidebar:
    st.header("Graph Input")
    input_mode = st.radio("Definition mode", ["Sample", "Custom"], index=0)
    undirected = st.checkbox("Treat edges as undirected", value=False)

    sample_edges = """
    # source,target,cost
    A,B,1
    A,C,4
    B,C,2
    B,D,5
    C,D,3
    """.strip()

    if input_mode == "Sample":
        edge_text = sample_edges
        st.image("UCS_img1.jpg", caption="Sample graph (if available)", use_container_width=True)
    else:
        edge_text = st.text_area(
            "Edges (one per line: src,dst,cost)",
            value=sample_edges,
            height=160,
            help="Use commas or spaces. Comments start with '#'.",
        )

    parse_ok = True
    graph: Dict[str, Dict[str, float]] = {}
    try:
        graph = parse_edges(edge_text, undirected=undirected)
    except Exception as e:
        parse_ok = False
        st.error(str(e))

    st.header("Heuristic h(n)")
    nodes_preview = sorted(set([*graph.keys(), *{k for d in graph.values() for k in d.keys()}]))
    sample_h = "\n".join([f"{n},0" for n in nodes_preview]) if nodes_preview else "A,0\nB,0\nC,0\nD,0"
    heur_text = st.text_area(
        "Heuristic lines (node,value). Missing nodes default to 0.",
        value=sample_h,
        height=160,
    )

if not parse_ok or not graph:
    st.stop()

nodes = all_nodes(graph)
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_node = st.selectbox("Start", nodes, index=0 if nodes else None)
with col2:
    goal_node = st.selectbox("Goal (optional)", ["<none>"] + nodes, index=(nodes.index("D") + 1) if "D" in nodes else 0)
with col3:
    run_btn = st.button("Run Greedy", type="primary")

st.divider()

if run_btn and start_node:
    goal_value = None if goal_node == "<none>" else goal_node
    try:
        h = parse_heuristic(heur_text)
        for n in nodes:
            h.setdefault(n, 0.0)

        path, total, expanded, g_cost, parent, trace = greedy_best_first_search(graph, start_node, goal_value, h)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Result")
        if goal_value is None:
            st.write("Explored reachable nodes (no specific goal).")
        elif path:
            st.success(f"Path (GBFS): {' → '.join(path)}  |  Reported path cost: {total}")
        else:
            st.warning("No path found to the specified goal.")

        st.write("Expanded order:", ", ".join(expanded) if expanded else "None")

        rows = [
            {"Node": n, "g(n) (accumulated)": (g_cost[n] if n in g_cost else float('inf')), "Parent": parent.get(n), "h(n)": h.get(n, 0.0)}
            for n in nodes
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Step-by-step frontier (trace)"):
            for i, snap in enumerate(trace, start=1):
                frontier_str = ", ".join(f"{nn}@h={hh:.2f}" for hh, nn in sorted(list(snap["frontier"]))[:10])
                st.write(f"{i}. expanded={snap['expanded']}  h={snap['h']:.2f}  g={snap['g']:.2f}  frontier=[{frontier_str}]")

    with right:
        st.subheader("Graph")
        gv = to_graphviz(graph, path if path else [], directed=not undirected)
        st.graphviz_chart(gv, use_container_width=True)

else:
    st.info("Set start/goal, edit heuristic if needed, and click Run Greedy to execute.")
