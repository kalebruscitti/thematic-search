from .topicdatabase import TopicDatabase
from .softclustertree import (
    SoftClusterTree,
    IndexExpr,
    Cluster,
    IndexAnd,
    IndexOr,
    IndexNot,
)
from .topic_formula import TopicFormula
from .robust_topic import robust_topic
from .queries import SampleQuery, TopicQuery, RootQuery, FuzzyQuery
import thematic_search.utilities as utils

__all__ = [
    # Primary entry point
    "TopicDatabase",
    # Query classes (users receive these from chained calls, so they need to be importable for type hints)
    "SampleQuery",
    "TopicQuery",
    "RootQuery",
    "FuzzyQuery",
    # Formula representation
    "TopicFormula",
    # Core operations
    "robust_topic",
    # Tree structure
    "SoftClusterTree",
    "IndexExpr",
    "Cluster",
    "IndexAnd",
    "IndexOr",
    "IndexNot",
    # Utilities
    "utils",
]
