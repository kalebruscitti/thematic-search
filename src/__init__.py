from .topicdatabase import TopicDatabase, IndexQuery, TopicQuery, RootQuery
from .softclustertree import SoftClusterTree, Cluster, ClusterLeaf, ClusterAnd, ClusterOr, ClusterNot

__all__ = [
    # Primary entry point
    "TopicDatabase",
    # Query classes (users receive these from chained calls, so they need to be importable for type hints)
    "IndexQuery",
    "TopicQuery",
    "RootQuery",
    # Cluster expression tree (users build these directly with & | ~ operators, but may need the types for isinstance checks or type hints)
    "SoftClusterTree",
    "Cluster",
    "ClusterLeaf",
    "ClusterAnd",
    "ClusterOr",
    "ClusterNot",
]