import base64
from collections import defaultdict

def topic_uid(tup) -> str:
    a, b = tup
    a = int(a)
    b = int(b) + 1  # Because unclustered is -1 and we can't convert negative to unsigned.
    combined = (a << 10) | b  # pack into 20 bits
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b'=').decode()

def uid_to_ints(s: str):
    """Returns (layer, cluster_number)"""
    padded = s + '=' * (-len(s) % 4)
    combined = int.from_bytes(base64.urlsafe_b64decode(padded), "big")
    return combined >> 10, (combined & 0x3FF) - 1

def print_children(node, cluster_tree, cluster_labels, depth=0):
    print("--"*depth+cluster_labels.get(node, node))
    for child in cluster_tree[node]:
        print_children(
            child,
            cluster_tree,
            cluster_labels=cluster_labels,
            depth=depth+1
        )

def print_tree(cluster_tree, cluster_labels={}):
    n_layers = max([
        l for l, _ in cluster_tree.keys()
    ])
    root = (n_layers, 0)
    print_children(root, cluster_tree, cluster_labels=cluster_labels, depth=0)


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

def convert_string_tree(tree, layers={}):
    uid_to_tup = assign_cluster_tuples(tree, layers)
    cluster_labels = {v:k for k,v in uid_to_tup.items()}
    cluster_tree = {uid_to_tup[n]:[uid_to_tup[x] for x in tree[n]] for n in tree}
    return cluster_tree, cluster_labels