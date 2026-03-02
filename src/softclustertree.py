import numpy as np
import scipy.sparse
import warnings
import base64
from typing import Union


# =================== UID Utilities ===================

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


# =================== Cluster Expression Tree ===================

class Cluster:
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
        return ClusterAnd(self, other)

    def __or__(self, other):
        return ClusterOr(self, other)

    def __invert__(self):
        return ClusterNot(self)

    def __repr__(self):
        raise NotImplementedError


class ClusterLeaf(Cluster):
    """A single cluster node identified by its uid string."""
    def __init__(self, uid: str, name: str=None):
        self.uid = uid
        self.name = name

    def __repr__(self):
        string_rep = f"ClusterLeaf({self.uid})"
        if self.name:
            string_rep += f" `{self.name}`"
        return string_rep


class ClusterAnd(Cluster):
    """Conjunction (meet) of two cluster expressions: elementwise min."""
    def __init__(self, left: Cluster, right: Cluster):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} & {self.right})"


class ClusterOr(Cluster):
    """Disjunction (join) of two cluster expressions: elementwise max."""
    def __init__(self, left: Cluster, right: Cluster):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} | {self.right})"


class ClusterNot(Cluster):
    """
    Heyting negation of a cluster expression.
    ~a  = 255 where strength == 0, else 0
    ~~a = 255 where strength > 0,  else 0
    """
    def __init__(self, operand: Cluster):
        self.operand = operand

    def __repr__(self):
        return f"~{self.operand}"


# =================== Soft Cluster Tree ===================

class SoftClusterTree:
    """
    A hierarchical soft clustering structure storing inclusion strengths
    as uint8 sparse matrices (0-255, divide by 255 to recover floats).

    Parameters
    ----------
    cluster_matrices : list of np.ndarray
        List of L dense float arrays, one per layer.
        cluster_matrices[l] has shape (n_docs, n_clusters_at_layer_l),
        with values in [0, 1].
    cluster_tree : dict
        A dict mapping (layer, cluster_number) tuples to lists of children
        tuples, e.g. {(2, 0): [(1, 0), (1, 1)], (2, 1): [(1, 2)], ...}
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
        self.layers = [
            self._sparsify(m, sparsity_threshold) for m in cluster_matrices
        ]
        
        self.uid_to_loc = {}   # uid -> (layer, col_index)
        self.loc_to_uid = {}   # (layer, col_index) -> uid
        for l, matrix in enumerate(cluster_matrices):
            for j in range(matrix.shape[1]):
                uid = topic_uid((l, j))
                self.uid_to_loc[uid] = (l, j)
                self.loc_to_uid[(l, j)] = uid

        self.children_map = {}  # uid -> list of child uids
        self.parent_map = {}    # uid -> parent uid
        for node, children in cluster_tree.items():
            node_uid = topic_uid(node)
            child_uids = [topic_uid(c) for c in children]
            self.children_map[node_uid] = child_uids
            for child_uid in child_uids:
                self.parent_map[child_uid] = node_uid

        # Add root node (L, 0) with all-255 inclusion strengths
        root_tup = (self.n_layers, 0)
        root_uid = topic_uid(root_tup)
        self.root_uid = root_uid
        root_vector = scipy.sparse.csr_matrix(
            np.full((self.n_docs, 1), 255, dtype=np.uint8)
        )
        self.root_vector = root_vector
        self.uid_to_loc[root_uid] = (self.n_layers, 0)

        # Root's children are the top-layer clusters
        top_layer_uids = [self.loc_to_uid[(self.n_layers - 1, j)]
                          for j in range(cluster_matrices[-1].shape[1])]
        self.children_map[root_uid] = top_layer_uids
        for uid in top_layer_uids:
            self.parent_map[uid] = root_uid

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

    def _get_strength_vector(self, uid: str) -> np.ndarray:
        """
        Return the dense uint8 inclusion strength vector (shape: n_docs,)
        for a given cluster uid.
        """
        if uid == self.root_uid:
            return self.root_vector.toarray().flatten()
        layer, col = self.uid_to_loc[uid]
        return np.array(self.layers[layer].getcol(col).toarray()).flatten()

    def _evaluate(self, expr: Cluster) -> np.ndarray:
        """
        Recursively evaluate a Cluster expression tree,
        returning a dense uint8 vector of shape (n_docs,).
        """
        if isinstance(expr, ClusterLeaf):
            return self._get_strength_vector(expr.uid)

        elif isinstance(expr, ClusterAnd):
            left = self._evaluate(expr.left)
            right = self._evaluate(expr.right)
            return np.minimum(left, right)

        elif isinstance(expr, ClusterOr):
            left = self._evaluate(expr.left)
            right = self._evaluate(expr.right)
            return np.maximum(left, right)

        elif isinstance(expr, ClusterNot):
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
        expr: Union[Cluster, str],
        min_strength: float = 1.0,
    ) -> np.ndarray:
        """
        Return indices of documents satisfying the cluster expression
        with inclusion strength >= min_strength.

        Parameters
        ----------
        expr : Cluster or str
            A Cluster expression or a uid string for a single cluster.
        min_strength : float, optional (default=1.0)
            Minimum inclusion strength in [0, 1].
            - min_strength=1.0 means full membership (strength == 255)
            - min_strength just above 0 combined with ~~a gives partial membership

        Returns
        -------
        np.ndarray
            Array of document indices.
        """
        if isinstance(expr, str):
            expr = ClusterLeaf(expr)
        threshold = self.to_int(min_strength)
        strengths = self._evaluate(expr)
        return np.where(strengths >= threshold)[0]

    def parents(self, uid: str) -> list:
        """
        Return the parent uid of a cluster, or an empty list if it is the root.

        Parameters
        ----------
        uid : str
            The uid of the cluster.

        Returns
        -------
        list of str
            A list containing the parent uid, or empty if uid is the root.
        """
        if uid == self.root_uid:
            return []
        return [self.parent_map[uid]]

    def children(self, uid: str) -> list:
        """
        Return the child uids of a cluster.

        Parameters
        ----------
        uid : str
            The uid of the cluster.

        Returns
        -------
        list of str
            List of child uids, empty if uid is a leaf.
        """
        return self.children_map.get(uid, [])

    def join(self, uids: list) -> str:
        """
        Return the uid of the least upper bound (LUB) of a set of clusters
        in the tree, i.e. their lowest common ancestor.

        Parameters
        ----------
        uids : list of str
            List of cluster uids.

        Returns
        -------
        str
            The uid of the LUB cluster.
        """
        if len(uids) == 1:
            return uids[0]

        def ancestors(uid):
            path = [uid]
            while uid in self.parent_map:
                uid = self.parent_map[uid]
                path.append(uid)
            return path

        # Find ancestors of each uid, then find the first common ancestor
        ancestor_lists = [ancestors(uid) for uid in uids]
        # Walk up the first list and check if each ancestor is in all others
        anchor_sets = [set(a) for a in ancestor_lists[1:]]
        for candidate in ancestor_lists[0]:
            if all(candidate in s for s in anchor_sets):
                return candidate

        return self.root_uid

    #=== API/Utilities ===#

    def strengths(
        self,
        expr: Union[Cluster, str],
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
        expr : Cluster or str
            A Cluster expression or a uid string for a single cluster.
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
        if isinstance(expr, str):
            expr = ClusterLeaf(expr)
        strength_vector = self._evaluate(expr)
        result = strength_vector[indices]
        if as_float:
            return self.to_float(result)
        return result

    def leaf(self, layer: int, cluster_number: int) -> ClusterLeaf:
        """
        Convenience method to construct a ClusterLeaf from
        a (layer, cluster_number) pair.

        Parameters
        ----------
        layer : int
        cluster_number : int

        Returns
        -------
        ClusterLeaf
        """
        return ClusterLeaf(topic_uid((layer, cluster_number)))

    @property
    def topics(self):
        return [self.leaf(uid) for uid in self.uid_to_loc.keys()]
        
