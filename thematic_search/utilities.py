import base64
from collections import defaultdict
import numpy as np

def topic_uid(tup: tuple) -> str:
    """Given a tuple `(layer, cluster_number)` returns its UID string."""
    a, b = tup
    a = int(a)
    b = int(b) + 1  # Because unclustered is -1 and we can't convert negative to unsigned.
    combined = (a << 10) | b  # pack into 20 bits
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b'=').decode()

def uid_to_ints(s: str) -> tuple:
    """Given a UID `s`, returns `(layer, cluster_number)`"""
    padded = s + '=' * (-len(s) % 4)
    combined = int.from_bytes(base64.urlsafe_b64decode(padded), "big")
    return combined >> 10, (combined & 0x3FF) - 1

def print_subtree(
        node: tuple,
        cluster_tree: dict[tuple, list[tuple]],
        cluster_labels: dict[tuple, str] = {},
        depth:int = 0
    ):
    """
    Print the subtree of a node in a cluster_tree.
    
    Parameters
    ----------
    node: tuple,
        A key in cluster_tree to print the subtree of.
    cluster_tree: dict[tuple, list[tuple]],
        The cluster tree to print,
    cluster_labels: dict[tuple, str], (optional, default={})
        A dictionary containing display names for the
    """
    print("--"*depth+cluster_labels.get(node, node))
    for child in cluster_tree[node]:
        print_subtree(
            child,
            cluster_tree,
            cluster_labels=cluster_labels,
            depth=depth+1
        )

def print_tree(cluster_tree: dict[tuple, list[tuple]], cluster_labels={}):
    """
    Print the cluster tree to the console.
    
    Parameters
    ----------
    cluster_tree: dict[tuple, list[tuple]],
        The cluster tree to print,
    cluster_labels: dict[tuple, str], (optional, default={})
        A dictionary containing display names for the
    """
    n_layers = max([
        l for l, _ in cluster_tree.keys()
    ])
    root = (n_layers, 0)
    print_subtree(root, cluster_tree, cluster_labels=cluster_labels, depth=0)


def compute_layers(tree):
    layers = {}
    def dfs(node):
        if node in layers:
            return layers[node]
        children = tree.get(node, [])
        if not children:
            layers[node] = 0
        else:
            layers[node] = max(dfs(c) for c in children) + 1
        return layers[node]
    for node in tree:
        dfs(node)
    return layers

def assign_cluster_tuples(tree, layers={}):
    if layers == {}:
        layers = compute_layers(tree)
    by_layer = defaultdict(list)
    for node, layer in layers.items():
        by_layer[layer].append(node)
    result = {}
    for layer in sorted(by_layer):
        for i, node in enumerate(sorted(by_layer[layer])):
            result[node] = (layer, i)
    return result

def convert_tree(tree:dict, layers:dict[any, int]={})->dict[tuple, list[tuple]]:
    """ Given an tree in the form of a dictionary containing `vertex:[list of children]`, 
    convert it to a cluster_tree. 
    
    Parameters
    ----------
    tree: dict
        The tree to convert. Must have pairs `vertex:[list of children]`
    layers: dict
        Custom layer assignment dictionary of the form `vertex:layer`. If not specified,
        leaves are assigned layer=0 and all other nodes are assigned layer=max_layer_of_children+1
    """
    uid_to_tup = assign_cluster_tuples(tree, layers)
    cluster_labels = {v:k for k,v in uid_to_tup.items()}
    cluster_tree = {uid_to_tup[n]:[uid_to_tup[x] for x in tree[n]] for n in tree}
    return cluster_tree, cluster_labels

def cluster_layers_from_leaf_matrix(
        cluster_tree:dict[tuple, list[tuple]],
          matrix:np.ndarray
        )->list[np.ndarray]:
    """ Given a cluster_tree and a matrix of inclusion strengths for the 0th layer,
        compute a set of cluster layers by summing over the children of each node. 
    """
    cluster_matrices = []
    n_layers = max([l for l, _ in cluster_tree.keys()])
    n_samples = matrix.shape[0]

    nodes_in_layer = [[
        (l,c) for l,c in cluster_tree.keys() if l ==layer 
    ] for layer in range(n_layers)]

    for layer in range(n_layers):
        if layer == 0:
            cluster_matrices.append(matrix)
            continue
        n_nodes_layer = len(nodes_in_layer[layer])
        layer_matrix = np.zeros((n_samples, n_nodes_layer))
        for node in nodes_in_layer[layer]:
            l,c = node
            for s,t in cluster_tree[node]:
                layer_matrix[:,c] += cluster_matrices[s][:, t]
        cluster_matrices.append(layer_matrix)

    return cluster_matrices