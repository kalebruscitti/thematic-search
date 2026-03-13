import numpy as np

def z_score(tag1, tag2, cooccurance, tag_matrix):
    q = tag_matrix.shape[0]
    q1 = np.mean(tag_matrix[:,tag1])
    q2 = np.mean(tag_matrix[:,tag2])
    # expected
    E = q1*q2/q
    # variance
    var = E*(q-q1)*(q-q2)/(q*(q-1))
    return (cooccurance[tag1,tag2]-E)/np.sqrt(var)

def compute_initial_forest(tag_matrix, omega=0.4):
    n_tags = tag_matrix.shape[1]
    cooccurance = np.zeros((
        n_tags, n_tags
    ))
    for row in tag_matrix:
        cooccurance += np.outer(row, row)

    zscores = np.zeros((n_tags, n_tags))
    strong_links = {}
    for tag1 in range(n_tags):
        maxcooc = np.amax(cooccurance[tag1,:])
        strong_links[tag1] = (
            cooccurance[tag1]>=omega*maxcooc
        ).nonzero()[0]

    parents = {}
    for tag1 in range(n_tags):
        strong_zscores = [
            (tag2, z_score(tag1,tag2,cooccurance,tag_matrix))
            for tag2 in strong_links[tag1]
        ]
        # get parent candidates in descending order
        # of their z-score
        zscores = np.argsort(
            np.array([x[1] for x in strong_zscores])
        )
        sort = np.argsort(zscores)[::-1]
        candidates = np.array([x[0] for x in strong_zscores])[sort]
        parent = None
        for tag2 in candidates:
            if tag1 not in strong_links[tag2]:
                parent = tag2
                break 
        parents[tag1] = parent

    return parents

def forest_to_children_tree(parents):
    # Assign all local roots to be under the a new global root
    n_tags = len(parents)
    for tag in parents:
        if parents[tag] is None:
            parents[tag] = n_tags

    children = {tag:[] for tag in parents}
    children[n_tags] = [] # add root node
    for tag in parents:
        children[parents[tag]].append(tag)
    
    return children

def compute_layer(tag, children):
    if len(children[tag])==0:
        return 0
    else:
        child_layers = [
            compute_layer(c, children)
            for c in children[tag]
        ]
        return max(child_layers)+1

def assemble_layers(children):
    layers = {tag:compute_layer(tag,children) for tag in children}

    ## Convert to cluster number format:
    n_layers = max(layers.values())
    clusters_per_layer = [[] for _ in range(n_layers)]
    for tag in layers:
        if layers[tag]==n_layers:
            # skip the root node.
            continue
        else:
            clusters_per_layer[layers[tag]].append(tag)

    layer_map = {}
    for l in range(n_layers):
        for c, tag in enumerate(clusters_per_layer[l]):
            layer_map[tag] = (l, c)
    n_tags = len(layers)-1
    layer_map[n_tags] = (n_layers, 0) # add root node
 
    return layer_map, clusters_per_layer

def build_cluster_tree_and_layers(
        tag_matrix,
        children,
        layer_map,
        clusters_per_layers,
    ):
    layers = [l for _, (l, _) in layer_map.items()]
    n_layers = max(layers)
    n_samples = tag_matrix.shape[0]
    cluster_layers = []
    for l in range(n_layers):
        n_tags = len(clusters_per_layers[l])
        cluster_layers.append(
            np.zeros((n_samples, n_tags))
        )
    
    for tag in layer_map:
        l, c = layer_map[tag]
        if l == n_layers:
            # skip root.
            continue
        else:
            cluster_layers[l][:,c] = tag_matrix[:,tag]
    
    cluster_tree = {
        layer_map[tag]:[
            layer_map[c] for c in children
        ] for tag, children in children.items()
    }
    reverse_map = {
        v:k for k,v in layer_map.items()
    }
    return cluster_layers, cluster_tree, reverse_map

def tag_hierarchy(tag_matrix, omega=0.6):
    parents = compute_initial_forest(tag_matrix, omega=omega)
    children = forest_to_children_tree(parents)
    layer_map, clusters_per_layers = assemble_layers(children)
    return build_cluster_tree_and_layers(
        tag_matrix, children, layer_map, clusters_per_layers
    )

