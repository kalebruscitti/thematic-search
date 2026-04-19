import numpy as np
from dataclasses import dataclass, field


# =============================================================================
# ThemeNode dataclass
# =============================================================================


@dataclass
class ThemeNode:
    """
    A node in a recursive theme formula tree.

    The formula represented by a ThemeNode is interpreted as:

        weight * (conjunction[0] AND conjunction[1] AND ... AND (disjuncts[0] OR disjuncts[1] OR ...))

    where each element of `conjunction` and `disjuncts` is itself a ThemeNode,
    allowing arbitrarily nested AND/OR structures.

    Parameters
    ----------
    topic_idx : int
        Index of the topic in the SoftClusterTree that this node represents.
    weight : float
        The pi_i weight at the point of splitting — the fraction of conditioned
        query mass assigned to this branch. 1.0 for the root node.
    conjunction : list of ThemeNode
        Ordered list of additional themes conjoined with this node's topic,
        from most to least conditionally surprising.
    disjuncts : list of ThemeNode
        Side branches spawned by a split at this node, each carrying their
        own weight and sub-formula. Empty if no split occurred here.
    """

    topic_idx: int
    weight: float = 1.0
    conjunction: list = field(default_factory=list)
    disjuncts: list = field(default_factory=list)

    def _topic_name(self, db) -> str:
        """Look up a display name for this node's topic from topic_df."""
        try:
            name = db.topic_df.loc[self.topic_idx, "name"]
            if isinstance(name, str) and name:
                return name
        except (KeyError, AttributeError):
            pass
        # Fallback: use (layer, cluster_number) tuple
        loc = db.soft_cluster_tree.idx_to_loc.get(self.topic_idx)
        if loc is not None:
            return f"topic({loc[0]},{loc[1]})"
        return f"topic({self.topic_idx})"

    def to_string(self, db, indent: int = 0) -> str:
        """
        Render the formula tree as an indented string.

        Parameters
        ----------
        db : TopicDatabase
            Used to look up topic names.
        indent : int
            Current indentation level (used in recursion).

        Returns
        -------
        str
        """
        pad = "  " * indent
        name = self._topic_name(db)

        # Weight prefix: omit for root (weight == 1.0) and for conjunction members
        weight_str = f"[{self.weight:.2f}] " if self.weight < 1.0 else ""
        lines = [f"{pad}{weight_str}{name}"]

        # Conjunction terms (AND chain)
        for node in self.conjunction:
            lines.append(f"{pad}  AND")
            lines.append(node.to_string(db, indent=indent + 2))

        # Disjunct branches (OR branches)
        for node in self.disjuncts:
            lines.append(f"{pad}  OR")
            lines.append(node.to_string(db, indent=indent + 2))

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_conj = len(self.conjunction)
        n_disj = len(self.disjuncts)
        return (
            f"ThemeNode(topic_idx={self.topic_idx}, weight={self.weight:.2f}, "
            f"conjunction={n_conj}, disjuncts={n_disj})"
        )

    def pprint(self, db):
        """Pretty-print the formula tree to stdout."""
        print(self.to_string(db))


# =============================================================================
# Core computational functions
# =============================================================================


def _surprise_score(
    query_strengths: np.ndarray,
    global_strengths: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute the KL-divergence-style surprise score for a topic.

        score = query_coverage * log(query_coverage / global_freq)

    Positive when the topic is over-represented in the query relative to the
    global corpus, zero when coverage matches global frequency.

    Parameters
    ----------
    query_strengths : np.ndarray
        Conditioned float strength vector over query documents, shape (n_query,).
    global_strengths : np.ndarray
        Conditioned float strength vector over all documents, shape (n_docs,).
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
    """
    query_coverage = query_strengths.mean()
    if query_coverage < eps:
        return 0.0
    global_freq = global_strengths.mean()
    if global_freq < eps:
        return 0.0
    return float(query_coverage * np.log(query_coverage / global_freq))


def _condition_strengths(
    raw: np.ndarray,
    cond_weights: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Apply Bayesian conditioning to a raw strength vector.

    Computes s(d, T | prev) = min(cond_weight[d], s(d,T)) / cond_weight[d]
    for each document d, treating documents with near-zero weight as having
    zero conditioned strength.

    Uses explicit masking rather than np.where to avoid divide-by-zero
    warnings from NumPy evaluating both branches eagerly before selecting.

    Parameters
    ----------
    raw : np.ndarray
        Raw float strength vector, shape (n,).
    cond_weights : np.ndarray
        Accumulated conditioning weights, shape (n,).

    Returns
    -------
    np.ndarray, shape (n,)
    """
    result = np.zeros_like(raw)
    mask = cond_weights > eps
    result[mask] = np.minimum(cond_weights[mask], raw[mask]) / cond_weights[mask]
    return result


def _child_pi(
    children: list,
    query_indices: np.ndarray,
    all_strengths: np.ndarray,
    cond_weights_query: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute the normalised query weight distribution pi over a set of children.

    For each child C_i, the conditioned query weight is the mean conditioned
    strength over query documents. pi_i is then the fraction of total weight
    on child i.

    Parameters
    ----------
    children : list of int
        Child topic indices.
    query_indices : np.ndarray
        Document indices in the query set.
    all_strengths : np.ndarray
        Full float strength matrix, shape (n_docs, n_topics).
    cond_weights_query : np.ndarray
        Current conditioning weights for query documents, shape (n_query,).

    Returns
    -------
    np.ndarray
        Normalised weight vector pi, shape (n_children,). Sums to 1.0,
        or is all-zeros if all children have zero conditioned coverage.
    """
    weights = np.zeros(len(children))
    for i, child_idx in enumerate(children):
        raw = all_strengths[query_indices, child_idx]
        cond = _condition_strengths(raw, cond_weights_query)
        weights[i] = cond.mean()

    total = weights.sum()
    if total < eps:
        return weights
    return weights / total


def _normalised_entropy(pi: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute the normalised entropy H_tilde = H / log(n) of a distribution pi.

    Returns 0.0 for distributions over fewer than 2 elements.

    Parameters
    ----------
    pi : np.ndarray
        Probability vector, shape (n,). Need not sum to 1 (zero-weight children
        are ignored).

    Returns
    -------
    float in [0, 1]
    """
    n = len(pi)
    if n < 2:
        return 0.0
    # Only consider children with non-negligible weight
    p = pi[pi > eps]
    if len(p) < 2:
        return 0.0
    H = -np.sum(p * np.log(p))
    return float(H / np.log(n))


def _best_topic(
    searchable: list,
    query_indices: np.ndarray,
    all_strengths: np.ndarray,
    cond_weights_all: np.ndarray,
    cond_weights_query: np.ndarray,
) -> tuple[int | None, float]:
    """
    Find the topic in `searchable` with the highest conditional surprise score.

    Parameters
    ----------
    searchable : list of int
        Candidate topic indices.
    query_indices : np.ndarray
        Document indices in the query.
    all_strengths : np.ndarray
        Full float strength matrix, shape (n_docs, n_topics).
    cond_weights_all : np.ndarray
        Conditioning weights over all documents, shape (n_docs,).
    cond_weights_query : np.ndarray
        Conditioning weights over query documents, shape (n_query,).

    Returns
    -------
    (best_idx, best_score) : (int or None, float)
    """
    best_idx = None
    best_score = -np.inf

    for topic_idx in searchable:
        raw_all = all_strengths[:, topic_idx]
        raw_query = all_strengths[query_indices, topic_idx]

        cond_all = _condition_strengths(raw_all, cond_weights_all)
        cond_query = _condition_strengths(raw_query, cond_weights_query)

        score = _surprise_score(cond_query, cond_all)
        if score > best_score:
            best_score = score
            best_idx = topic_idx

    return best_idx, best_score


def _all_ancestors(tree, topic_idx: int) -> set:
    """
    Return all ancestor indices of a topic using the adjacency closure.

    adjacency_closure[i, j] == 1 means i is an ancestor of j (i can reach j),
    so ancestors of topic_idx are the rows where adjacency_closure[:, topic_idx] == 1.

    Parameters
    ----------
    tree : SoftClusterTree
    topic_idx : int

    Returns
    -------
    set of int
    """
    return set(int(i) for i in np.where(tree.adjacency_closure[:, topic_idx])[0])


def _all_descendants(tree, topic_idx: int) -> set:
    """
    Return all descendant indices of a topic using the adjacency closure.

    adjacency_closure[i, j] == 1 means i is an ancestor of j, so descendants
    of topic_idx are the columns where adjacency_closure[topic_idx, :] == 1.

    Parameters
    ----------
    tree : SoftClusterTree
    topic_idx : int

    Returns
    -------
    set of int
    """
    return set(int(j) for j in np.where(tree.adjacency_closure[topic_idx])[0])


def build_theme_tree(
    db,
    query_indices: np.ndarray,
    cond_weights_all: np.ndarray,
    cond_weights_query: np.ndarray,
    excluded: set,
    z_threshold: float,
    entropy_threshold: float,
    min_disjunct_weight: float,
    max_disjuncts: float,
    max_conjunction: int,
    depth: int = 0,
) -> ThemeNode:
    """
    Recursively build a ThemeNode formula tree for a set of query documents.

    At each step:
    1. Find the most conditionally surprising non-excluded topic T.
    2. If the surprise falls below the noise threshold, stop.
    3. Compute normalised entropy of T's children under conditioned query weights.
    4. If entropy is HIGH (heterogeneous query): accept T into the conjunction,
       update conditioning, exclude T, all its descendants, and all its ancestors
       (since ancestor AND T = T by inclusion consistency), then continue.
    5. If entropy is LOW (concentrated query): drill into the dominant child,
       treating it as the new candidate (re-evaluating from step 1). Side branches
       with sufficient pi weight are spawned as weighted disjuncts, each recursing
       independently with inherited conditioning weights.

    Parameters
    ----------
    db : TopicDatabase
    query_indices : np.ndarray
        Document indices in this branch's query set.
    cond_weights_all : np.ndarray
        Conditioning weights over all documents, shape (n_docs,).
    cond_weights_query : np.ndarray
        Conditioning weights over query documents, shape (n_query,).
    excluded : set of int
        Topic indices excluded from search at this level.
    z_threshold : float
        Stopping sensitivity — stop when best surprise < z / sqrt(k).
    entropy_threshold : float
        Normalised entropy threshold in [0, 1]. Above this, T is accepted as-is.
        Below this, the query is considered concentrated and a split is attempted.
    min_disjunct_weight : float
        Minimum pi_i for a side branch to be spawned as a disjunct.
    max_disjuncts : float
        Maximum number of disjunct side branches at any single split.
        np.inf means unlimited.
    max_conjunction : int
        Maximum number of terms in the conjunction chain.
    depth : int
        Current recursion depth.

    Returns
    -------
    ThemeNode
        Falls back to a node pointing at the root topic if nothing passes
        the noise threshold.
    """
    tree = db.soft_cluster_tree
    all_strengths = db._all_strengths  # precomputed float matrix, shape (n_docs, n_topics)
    n_query = len(query_indices)
    noise_threshold = z_threshold / np.sqrt(max(n_query, 1))

    all_topic_indices = list(tree.idx_to_loc.keys())

    # ---- Conjunction loop ----
    # We build the conjunction chain iteratively. The first term becomes the
    # root ThemeNode; subsequent terms are appended to its conjunction list.
    root_node: ThemeNode | None = None
    current_node: ThemeNode | None = None  # tail of the conjunction chain
    local_excluded = set(excluded)
    local_cond_all = cond_weights_all.copy()
    local_cond_query = cond_weights_query.copy()

    for _ in range(max_conjunction):
        searchable = [t for t in all_topic_indices if t not in local_excluded]
        if not searchable:
            break

        best_idx, best_score = _best_topic(
            searchable, query_indices, all_strengths,
            local_cond_all, local_cond_query,
        )

        if best_idx is None or best_score < noise_threshold:
            break

        # ---- Split check ----
        # Drill down through children as long as entropy is low,
        # re-evaluating the candidate at each level.
        candidate_idx = best_idx
        disjuncts = []

        while True:
            children = tree.children(candidate_idx)
            if not children:
                # Leaf node — accept as-is, no split possible
                break

            pi = _child_pi(
                children, query_indices, all_strengths, local_cond_query
            )
            H_tilde = _normalised_entropy(pi)

            if H_tilde >= entropy_threshold:
                # Query is heterogeneous across children — accept candidate as-is
                break

            # Query is concentrated — dominant child becomes new candidate
            dominant_i = int(np.argmax(pi))
            dominant_child = children[dominant_i]

            # Spawn side branches for non-dominant children with sufficient weight,
            # sorted by descending weight, up to max_disjuncts - 1
            side_children = [
                (children[i], pi[i])
                for i in range(len(children))
                if i != dominant_i and pi[i] >= min_disjunct_weight
            ]
            side_children.sort(key=lambda x: -x[1])
            if max_disjuncts < np.inf:
                side_children = side_children[:int(max_disjuncts) - 1]

            for side_idx, side_weight in side_children:
                side_node = build_theme_tree(
                    db=db,
                    query_indices=query_indices,
                    cond_weights_all=local_cond_all.copy(),
                    cond_weights_query=local_cond_query.copy(),
                    excluded=local_excluded | {candidate_idx},
                    z_threshold=z_threshold,
                    entropy_threshold=entropy_threshold,
                    min_disjunct_weight=min_disjunct_weight,
                    max_disjuncts=max_disjuncts,
                    max_conjunction=max_conjunction,
                    depth=depth + 1,
                )
                side_node.weight = float(side_weight)
                disjuncts.append(side_node)

            # Drill into the dominant child
            candidate_idx = dominant_child

        # ---- Accept candidate into the conjunction ----
        new_node = ThemeNode(
            topic_idx=candidate_idx,
            weight=1.0,  # weight set by caller for disjuncts; 1.0 within conjunction
            conjunction=[],
            disjuncts=disjuncts,
        )

        if root_node is None:
            root_node = new_node
            current_node = new_node
        else:
            current_node.conjunction.append(new_node)
            current_node = new_node

        # Update conditioning weights
        raw_best_all = all_strengths[:, candidate_idx]
        raw_best_query = all_strengths[query_indices, candidate_idx]
        local_cond_all = np.minimum(local_cond_all, raw_best_all)
        local_cond_query = np.minimum(local_cond_query, raw_best_query)

        # Exclude candidate, all its descendants, and all its ancestors.
        # Descendants excluded: T AND descendant = descendant (adding T is redundant).
        # Ancestors excluded:   ancestor AND T = T (adding ancestor is redundant).
        local_excluded.add(candidate_idx)
        local_excluded.update(_all_descendants(tree, candidate_idx))
        local_excluded.update(_all_ancestors(tree, candidate_idx))

    # Fallback: if nothing passed the noise threshold, return the root topic.
    if root_node is None:
        root_node = ThemeNode(topic_idx=tree.root_idx, weight=1.0)

    return root_node


def recursive_theme(
    db,
    indices: np.ndarray,
    z_threshold: float = 2.0,
    entropy_threshold: float = 0.4,
    min_disjunct_weight: float = 0.1,
    max_disjuncts: float = np.inf,
    max_conjunction: int = 10,
) -> ThemeNode:
    """
    Find a recursive theme formula for a set of query documents.

    The result is a ThemeNode representing a formula of the form:
        (A AND B AND ...) OR (C AND D AND ...) OR ...

    where each branch is weighted by the fraction of conditioned query mass
    it captures at the point of splitting.

    Parameters
    ----------
    db : TopicDatabase
        The database to query.
    indices : np.ndarray
        Document indices forming the query set. Must be non-empty.
    z_threshold : float, optional (default=2.0)
        Stopping sensitivity. Recursion stops when the best remaining
        conditional surprise score falls below z_threshold / sqrt(k).
        Higher values stop earlier (fewer themes).
    entropy_threshold : float, optional (default=0.4)
        Normalised entropy threshold in [0, 1]. When a candidate topic T's
        children have normalised entropy above this value, the query is
        considered too heterogeneous to split and T is accepted into the
        conjunction as-is. Below this threshold, the algorithm drills into
        the dominant child and spawns side branches as disjuncts.
    min_disjunct_weight : float, optional (default=0.1)
        Minimum pi_i weight for a side branch to be spawned. Branches below
        this threshold are silently dropped, preventing spurious tiny disjuncts.
    max_disjuncts : float, optional (default=np.inf)
        Maximum number of disjunct branches at any single split point.
        np.inf means unlimited. Set to 1 to suppress all disjuncts (purely
        conjunctive output, fastest).
    max_conjunction : int, optional (default=10)
        Maximum number of terms in any single conjunction chain.

    Returns
    -------
    ThemeNode
        The root of the formula tree. Falls back to a root-topic node if
        no topic clears the noise threshold.

    Raises
    ------
    ValueError
        If indices is empty.
    """
    if len(indices) == 0:
        raise ValueError(
            "recursive_theme requires a non-empty query set. "
            "indices must contain at least one document index."
        )

    tree = db.soft_cluster_tree

    # _all_strengths is precomputed in TopicDatabase.__init__ and cached on db.
    # The check here is a safety net for databases constructed before this
    # attribute was added.
    if not hasattr(db, "_all_strengths"):
        db._all_strengths = (
            tree.cluster_matrix.toarray().astype(np.float64) / 255.0
        )

    cond_weights_all = np.ones(tree.n_docs, dtype=np.float64)
    cond_weights_query = np.ones(len(indices), dtype=np.float64)
    excluded = {tree.root_idx}

    return build_theme_tree(
        db=db,
        query_indices=indices,
        cond_weights_all=cond_weights_all,
        cond_weights_query=cond_weights_query,
        excluded=excluded,
        z_threshold=z_threshold,
        entropy_threshold=entropy_threshold,
        min_disjunct_weight=min_disjunct_weight,
        max_disjuncts=max_disjuncts,
        max_conjunction=max_conjunction,
        depth=0,
    )