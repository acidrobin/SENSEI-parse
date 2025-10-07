import re
import time
from itertools import permutations

import networkx as nx
import numpy as np
from sklearn.metrics import f1_score 
from arglu.graph_processing import make_arg_dicts_from_graph, get_perspectives_dict
import rouge

# ---------------------------
# Graph Perspective Utilities
# ---------------------------

def get_gold_and_pred_perspectives(gold_graph: nx.DiGraph, pred_graph: nx.DiGraph):
    """
    Extract node perspectives from gold and predicted graphs.
    Maps 'red' -> 0, 'green' -> 1, 'black' -> 2.
    """
    try:
        gold_persp = get_perspectives_dict(*make_arg_dicts_from_graph(gold_graph))
        pred_persp = get_perspectives_dict(*make_arg_dicts_from_graph(pred_graph))
    except Exception as e:
        print("Error extracting perspectives:", e)
        exit()

    for key in ["main topic", "majorclaim 1"]:
        gold_persp.pop(key, None)
        pred_persp.pop(key, None)

    gold_keys = list(gold_persp.keys())
    gold_values = [gold_persp[k] for k in gold_keys]
    pred_values = [pred_persp.get(k, "black") for k in gold_keys]

    map_stance = {"red": 0, "green": 1, "black": 2}
    return [map_stance[v] for v in gold_values], [map_stance[v] for v in pred_values]

# ---------------------------
# Node Position Utilities
# ---------------------------

def get_all_root_paths(G: nx.DiGraph):
    """Return all paths from each node to the root."""
    root = next(n for n, d in G.out_degree() if d == 0)
    return [nx.shortest_path(G, node, root) for node in G]

def get_paths_dict(gold_graph: nx.DiGraph, pred_graph: nx.DiGraph):
    """Create a unique mapping of all paths in both graphs to integers."""
    all_paths = [str(p) for p in get_all_root_paths(gold_graph) + get_all_root_paths(pred_graph)]
    return {p: i for i, p in enumerate(set(all_paths))}

def get_pos_id(node, G: nx.DiGraph, paths_dict: dict):
    """Get the path ID of a node in the graph, or -1 if missing."""
    if node not in G:
        return -1
    root = next(n for n, d in G.out_degree() if d == 0)
    return paths_dict[str(nx.shortest_path(G, node, root))]

def get_node_pos_acc_f1(gold_graph: nx.DiGraph, pred_graph: nx.DiGraph):
    """Compute node positional accuracy and F1 score."""
    paths_dict = get_paths_dict(gold_graph, pred_graph)
    all_nodes = sorted(set(gold_graph.nodes()) | set(pred_graph.nodes()))

    node_ids_gold = [get_pos_id(n, gold_graph, paths_dict) for n in all_nodes]
    node_ids_pred = [get_pos_id(n, pred_graph, paths_dict) for n in all_nodes]

    pos_f1 = f1_score(node_ids_gold, node_ids_pred, average="macro")
    pos_acc = 1 - sum(np.array(node_ids_gold) != np.array(node_ids_pred)) / len(gold_graph.nodes())
    return pos_acc, pos_f1

# ---------------------------
# Node Stance Metrics
# ---------------------------

def node_stance_accuracy(gold: nx.DiGraph, pred: nx.DiGraph):
    gold_persp, pred_persp = get_gold_and_pred_perspectives(gold, pred)
    return np.mean(np.array(gold_persp) == np.array(pred_persp)) if gold_persp else 0

def node_stance_f1(gold: nx.DiGraph, pred: nx.DiGraph):
    gold_persp, pred_persp = get_gold_and_pred_perspectives(gold, pred)
    return f1_score(gold_persp, pred_persp, average="macro")

# ---------------------------
# Graph Parsing
# ---------------------------

def parse_text_to_networkx(text: str, main_topic: str = "") -> nx.DiGraph:
    """
    Parse a string representation of a text graph into a NetworkX DiGraph.
    Expected format for nodes: 'Comment X (relation parent) comment_text'
    """
    node_pattern = r"\s*([Cc]omment .+?)\s*\(\s*(\S+)\s*(.+?)\s*\)\s*(.+)"
    G = nx.DiGraph(rankdir="TB")
    G.add_node("main topic", node_name="main topic", text=main_topic)
    colon_trans = str.maketrans("", "", ":")

    for line in text.split("\n"):
        match = re.match(node_pattern, line)
        if not match:
            continue

        node_name, relation, parent, comment = [s.strip().lower() for s in match.groups()]
        if parent in G and node_name not in G and parent != node_name:
            G.add_node(node_name, node_name=node_name.translate(colon_trans),
                       text=comment.translate(colon_trans))
            G.add_edge(node_name, parent, label=relation.translate(colon_trans))
    return G

# ---------------------------
# Graph Edit Distance
# ---------------------------

def node_match(node1, node2):
    return node1["node_name"] == node2["node_name"]

def compute_node_stance_acc_f1_ged(references, predictions):
    """Compute stance, positional, and GED metrics for a list of graphs."""
    node_accs, node_f1s, pos_accs, pos_f1s, geds, geds_norm = [], [], [], [], [], []

    for i, (lab, pred) in enumerate(zip(references, predictions)):
        print(f"Computing metrics {i+1}/{len(references)}...", flush=True)
        gold_graph = parse_text_to_networkx(lab)
        pred_graph = parse_text_to_networkx(pred)

        node_accs.append(node_stance_accuracy(gold_graph, pred_graph))
        node_f1s.append(node_stance_f1(gold_graph, pred_graph))
        pos_acc, pos_f1 = get_node_pos_acc_f1(pred_graph, gold_graph)
        pos_accs.append(pos_acc)
        pos_f1s.append(pos_f1)

        ged = nx.graph_edit_distance(gold_graph, pred_graph, node_match=node_match, timeout=20)
        geds.append(ged)
        geds_norm.append(ged / len(gold_graph.nodes()))

    return (np.mean(node_accs), np.mean(node_f1s),
            np.mean(pos_accs), np.mean(pos_f1s),
            np.mean(geds), np.mean(geds_norm))

def compute_ged(references, predictions):
    """Compute only graph edit distance for a list of graph strings."""
    geds = []
    for i, (lab, pred) in enumerate(zip(references, predictions)):
        print(f"Computing GED {i+1}/{len(references)}...", flush=True)
        gold_graph = parse_text_to_networkx(lab)
        pred_graph = parse_text_to_networkx(pred)
        ged = nx.graph_edit_distance(gold_graph, pred_graph, node_match=node_match)
        geds.append(ged if ged is not None else 999)
    return np.mean(geds)

# ---------------------------
# Metrics Evaluation
# ---------------------------

scorer = rouge.Rouge(metrics=['rouge-n'], max_n=2, stemming=True)

def compute_metrics(predictions, references):
    """Compute ROUGE-2, stance, positional, and GED metrics for predicted graphs."""
    rouge_score = scorer.get_scores(references=references, hypothesis=predictions)["rouge-2"]["f"]
    node_acc, node_f1, pos_acc, pos_f1, ged, ged_norm = compute_node_stance_acc_f1_ged(predictions, references)

    return {
        'rouge2_f_measure': round(rouge_score, 4),
        'node stance f1': round(node_f1, 4),
        'node stance acc': round(node_acc, 4),
        'pos acc': round(pos_acc, 4),
        'pos f1': round(pos_f1, 4),
        'graph edit distance': round(ged, 4),
        'ged norm': round(ged_norm, 4)
    }