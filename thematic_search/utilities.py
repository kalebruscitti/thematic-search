import base64

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