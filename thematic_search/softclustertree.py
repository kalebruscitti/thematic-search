import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import floyd_warshall
import warnings
from typing import Union
from .utilities import *

# =================== Index Expressions ===================

class IndexExpr:
    """
    Base class for symbolic cluster expressions.
    Supports &, | and ~ operators corresponding to meet,
    join and Heyting negation respectively.

    In the Heyting algebra of inclusion strengths:
    - a & b  evaluates to elementwise min of inclusion strengths
    - a | b  evaluates to elementwise max of inclusion strengths
    - ~a     evaluates to 1 where strength == 0, else 0  (Heyting negation)
    - ~~a    evaluates to 1 where strength > 0, else 0   (double negation)
    """
    def __and__(self, other):
        return IndexAnd(self, other)

    def __or__(self, other):
        return IndexOr(self, other)

    def __invert__(self):
        return IndexNot(self)

    def __repr__(self):
        raise NotImplementedError


class Cluster(IndexExpr):
    """A single cluster node identified by its index."""
    def __init__(self, idx: int, name: str=None):
        self.idx = idx
        self.name = name

    def __repr__(self):
        string_rep = f"ClusterLeaf({self.idx})"
        if self.name:
            string_rep += f" `{self.name}`"
        return string_rep


class IndexAnd(IndexExpr):
    """Conjunction (meet) of two cluster expressions: elementwise min."""
    def __init__(self, left: IndexExpr, right: IndexExpr):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} & {self.right})"


class IndexOr(IndexExpr):
    """Disjunction (join) of two cluster expressions: elementwise max."""
    def __init__(self, left: IndexExpr, right: IndexExpr):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} | {self.right})"


class IndexNot(IndexExpr):
    """
    Heyting negation of a cluster expression.
    ~a  = 255 where strength == 0, else 0
    ~~a = 255 where strength > 0,  else 0
    """
    def __init__(self, operand: IndexExpr):
        self.operand = operand

    def __repr__(self):
        return f"~{self.operand}"

# =================== Soft Cluster Tree ===================

class SoftClusterTree:
    """
    A hierarchical soft clustering structure storing inclusion strengths
    as uint8 sparse matrices (0-255, divide by 255 to recover floats).

    The cluster hierarchy may be a DAG (directed acyclic graph), meaning
    a node may have multiple parents. Edges must respect the layer ordering:
    if (s, k) is a parent of (l, i) then s > l. The inclusion strength
    consistency assumption must hold for every edge: c^s_k(r) >= c^l_i(r)
    for all records r.

    Parameters
    ----------
    cluster_matrices : list of np.ndarray
        List of L dense float arrays, one per layer.
        cluster_matrices[l] has shape (n_docs, n_clusters_at_layer_l),
        with values in [0, 1].
    cluster_tree : dict
        A dict mapping (layer, cluster_number) tuples to lists of children
        tuples, e.g. {(2, 0): [(1, 0), (1, 1)], (2, 1): [(1, 2)], ...}
        The unique root node is the key with no parents, i.e. the node
        that does not appear in any value list.
    sparsity_threshold : float, optional (default=0.0)
        Inclusion strengths below this value are set to zero before
        sparsification. Useful for cleaning up near-zero soft memberships.
    """

    def __init__(
        self,
        cluster_matrices: list,
        cluster_tree: dict,
        sparsity_threshold: float = 0.0,
    ):
        self.n_docs = cluster_matrices[0].shape[0]
        self.n_layers = len(cluster_matrices)

        self._validate(cluster_matrices, cluster_tree)

        self.idx_to_loc = {}   # idx -> (layer, col_index)
        self.loc_to_idx = {}   # (layer, col_index) -> idx
        idx = 0
        for l, matrix in enumerate(cluster_matrices):
            for j in range(matrix.shape[1]):
                self.idx_to_loc[idx] = (l, j)
                self.loc_to_idx[(l, j)] = idx
                idx += 1
        # don't forget the root node.
        self.loc_to_idx[(l+1, 0)] = idx
        self.idx_to_loc[idx] = [(l+1, 0)]
        self.n_topics = idx+1

        self.layers = []
        for m in cluster_matrices:
            if scipy.sparse.issparse(m):
                warnings.warn("You passed sparse matrices," \
                " SoftClusterTree is assuming they are ranged in [0,255]")
                self.layers.append(m)
            else:
                self.layers.append(
                    self._sparsify(m, sparsity_threshold)
                )

        self.children_map = {}  # idx -> list of child indices
        self.parent_map = {}    # idx -> list of parent indices
        for node, children in cluster_tree.items():
            node_idx = self.loc_to_idx[node]
            child_indices = [self.loc_to_idx[c] for c in children]
            self.children_map[node_idx] = child_indices
            for child_idx in child_indices:
                self.parent_map.setdefault(child_idx, [])
                self.parent_map[child_idx].append(node_idx)

        # Identify the root: the unique node in cluster_tree with no parents
        all_children = {self.loc_to_idx[c] for children in cluster_tree.values() for c in children}
        roots = [self.loc_to_idx[n] for n in cluster_tree if self.loc_to_idx[n] not in all_children]
        if len(roots) != 1:
            raise ValueError(
                f"cluster_tree must have exactly one root (a node with no parents), "
                f"but found {len(roots)}: {roots}"
            )
        self.root_idx = roots[0]
        self.idx_to_loc[self.root_idx] = (self.n_layers, 0)

        cluster_matrix = np.zeros(
            (self.n_docs, self.n_topics), dtype=np.uint8
        )
        for col_idx in range(self.n_topics):
            layer, layer_idx = self.idx_to_loc[idx]
            if layer < self.n_layers:
                layer_matrix =  (
                    cluster_matrices[layer].toarray()
                    if scipy.sparse.issparse(cluster_matrices[layer])
                    else np.asarray(cluster_matrices[layer])
                )
                cluster_matrix[:,col_idx] = layer_matrix[:,layer_idx]
            else:
                cluster_matrix[:,col_idx] = 1
        self.cluster_matrix = self._sparsify(cluster_matrix, sparsity_threshold)
        # Compute transitive closure matrix of the tree
        # This needs to be stored for quick indexed colimits
        A = np.zeros((self.n_topics, self.n_topics), dtype=bool)
        for i, children in self.children_map.items():
            for j in children:
                A[i, j] = True
        self.adjacency_closure = (floyd_warshall(
            scipy.sparse.csr_matrix(A)
        )<np.inf).astype(int)  


    # =================== Utilities ===================

    @staticmethod
    def to_float(uint8_value):
        """Convert uint8 inclusion strength to float in [0, 1]."""
        return uint8_value / 255

    @staticmethod
    def to_int(float_value):
        """Convert float inclusion strength in [0, 1] to uint8."""
        return np.round(float_value * 255).astype(np.uint8)

    def _sparsify(self, dense_matrix: np.ndarray, threshold: float) -> scipy.sparse.csr_matrix:
        """Quantize a float matrix to uint8 and sparsify."""
        quantized = np.round(dense_matrix * 255).astype(np.uint8)
        quantized[quantized <= self.to_int(threshold)] = 0
        return scipy.sparse.csr_matrix(quantized)

    def _validate(self, cluster_matrices, cluster_tree):
        """Warn if tree nodes are inconsistent with matrix dimensions."""
        for (layer, cluster_number), children in cluster_tree.items():
            if (layer >= self.n_layers) and (cluster_number != 0):
                warnings.warn(
                    f"Tree node ({layer}, {cluster_number}) references layer {layer} "
                    f"but only {self.n_layers} layers were provided."
                )
                continue
            if layer == self.n_layers:
                # The top layer contains only the root, so the rest of 
                # the validation steps don't apply.
                continue
            n_clusters = cluster_matrices[layer].shape[1]
            if cluster_number >= n_clusters:
                warnings.warn(
                    f"Tree node ({layer}, {cluster_number}) references cluster "
                    f"{cluster_number} but layer {layer} only has {n_clusters} clusters."
                )
            for (cl, cn) in children:
                if cl >= self.n_layers:
                    warnings.warn(
                        f"Child node ({cl}, {cn}) references layer {cl} "
                        f"but only {self.n_layers} layers were provided."
                    )
                    continue
                n_child_clusters = cluster_matrices[cl].shape[1]
                if cn >= n_child_clusters:
                    warnings.warn(
                        f"Child node ({cl}, {cn}) references cluster {cn} "
                        f"but layer {cl} only has {n_child_clusters} clusters."
                    )

    def _get_strength_vector(self, idx: int) -> np.ndarray:
        """
        Return the dense uint8 inclusion strength vector (shape: n_docs,)
        for a given cluster idx.
        """
        layer, col = self.idx_to_loc[idx]
        if layer == self.n_layers:
            # root node.
            return np.full((self.n_docs,1), 255, dtype=np.uint8)
        else:
            return np.array(self.layers[layer].getcol(col).toarray()).flatten()

    def _evaluate(self, expr: IndexExpr) -> np.ndarray:
        """
        Recursively evaluate a Cluster expression tree,
        returning a dense uint8 vector of shape (n_docs,).
        """
        if isinstance(expr, Cluster):
            return self._get_strength_vector(expr.idx)

        elif isinstance(expr, IndexAnd):
            left = self._evaluate(expr.left)
            right = self._evaluate(expr.right)
            return np.minimum(left, right)

        elif isinstance(expr, IndexOr):
            left = self._evaluate(expr.left)
            right = self._evaluate(expr.right)
            return np.maximum(left, right)

        elif isinstance(expr, IndexNot):
            operand = self._evaluate(expr.operand)
            # Heyting negation: 255 where strength == 0, else 0
            result = np.zeros(self.n_docs, dtype=np.uint8)
            result[operand == 0] = 255
            return result

        else:
            raise ValueError(f"Unknown Cluster expression type: {type(expr)}")

    # =================== Query Methods ===================

    def inside(
        self,
        expr: Union[IndexExpr, str],
        min_strength: float = 1.0,
    ) -> np.ndarray:
        """
        Return indices of documents satisfying the cluster expression
        with inclusion strength >= min_strength.

        Parameters
        ----------
        expr : Cluster or int
            A Cluster expression or an index for a single cluster.
        min_strength : float, optional (default=1.0)
            Minimum inclusion strength in [0, 1].
            - min_strength=1.0 means full membership (strength == 255)
            - min_strength just above 0 combined with ~~a gives partial membership

        Returns
        -------
        np.ndarray
            Array of document indices.
        """
        if not isinstance(expr, IndexExpr):
            expr = Cluster(expr)
        threshold = self.to_int(min_strength)
        strengths = self._evaluate(expr)
        return np.where(strengths >= threshold)[0]

    def parents(self, idx: int) -> list:
        """
        Return the parent indices of a cluster, or an empty list if it is the root.

        Parameters
        ----------
        idx : str
            The idx of the cluster.

        Returns
        -------
        list of str
            A list of parent indices, or empty if idx is the root.
        """
        return self.parent_map.get(idx, [])

    def children(self, idx: int) -> list:
        """
        Return the child indices of a cluster.

        Parameters
        ----------
        idx : str
            The idx of the cluster.

        Returns
        -------
        list of str
            List of child indices, empty if idx is a leaf.
        """
        return self.children_map.get(idx, [])

    def join(self, indices: list) -> list:
        """
        Return the indices of the least upper bounds (LUBs) of a set of clusters,
        i.e. their lowest common ancestors in the DAG. There may be multiple
        incomparable LUBs at the same minimum layer.

        Parameters
        ----------
        indices : list of str
            List of cluster indices.

        Returns
        -------
        list of str
            The indices of the LUB clusters.
        """
        if len(indices) == 1:
            return indices

        def ancestors(idx):
            visited = set()
            queue = [idx]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                queue.extend(self.parent_map.get(current, []))
            return visited

        ancestor_sets = [ancestors(idx) for idx in indices]
        common = ancestor_sets[0].intersection(*ancestor_sets[1:])

        if not common:
            return [self.root_idx]

        min_layer = min(self.idx_to_loc[u][0] for u in common)
        return [u for u in common if self.idx_to_loc[u][0] == min_layer]

    #=== API/Utilities ===#

    def strengths(
        self,
        expr: Union[IndexExpr, int],
        indices: np.ndarray = None,
        as_float: bool = True,
    ) -> np.ndarray:
        """
        Return the inclusion strengths of a set of documents for a given
        cluster expression.

        Parameters
        ----------
        indices : np.ndarray
            Array of document indices (e.g. as returned by inside()).
        expr : Cluster or int
            A Cluster expression or the index for a single cluster.
        as_float : bool, optional (default=True)
            If True, return strengths as floats in [0, 1].
            If False, return raw uint8 values in [0, 255].

        Returns
        -------
        np.ndarray
            Array of inclusion strengths for the given documents.
        """
        if indices is None:
            indices = np.arange(self.n_docs)
        if isinstance(expr, int):
            expr = Cluster(expr)
        strength_vector = self._evaluate(expr)
        result = strength_vector[indices]
        if as_float:
            return self.to_float(result)
        return result

    def cluster(self, layer: int, cluster_number: int) -> Cluster:
        """
        Convenience method to construct a Cluster from
        a (layer, cluster_number) pair.

        Parameters
        ----------
        layer : int
        cluster_number : int

        Returns
        -------
        Cluster
        """
        return Cluster(self.loc_to_idx[(layer, cluster_number)])

    @property
    def topics(self):
        return [self.cluster(idx) for idx in self.idx_to_loc.keys()]

    @property
    def cluster_matrices(self):
        """Reconstruct the dense cluster matrices for saving purposes. """
        return [matrix.todense() for matrix in self.layers]
    

    