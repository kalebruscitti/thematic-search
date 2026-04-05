import warnings
import numpy as np
import pandas as pd
import pynndescent
from typing import Union, List, Optional
from pathlib import Path
from .softclustertree import SoftClusterTree
from .serialization import (
    save_topic_database,
    load_topic_database,
    save_topic_database_lance,
    load_topic_database_lance,
)
from .queries import *

class TopicDatabase:
    """
    A hierarchical soft-clustering database for thematic search.

    Packages together:
    - A SoftClusterTree for the hierarchical intuitionistic database
    - A pynndescent NNDescent index for the vector database
    - A document metadata DataFrame
    - A topic metadata DataFrame

    Parameters
    ----------
    soft_cluster_tree : SoftClusterTree
        The hierarchical soft clustering structure.
    embedding_vectors : np.ndarray
        The embedding vectors of the documents, shape (n_docs, n_features).
    reduced_vectors : np.ndarray
        The reduced vectors of the documents, shape (n_docs, n_reduced_features).
    document_df : pd.DataFrame, optional
        Document metadata. If None, a minimal DataFrame with just indices is created.
    topic_df : pd.DataFrame, optional
        Topic metadata. Must have an 'index' column as primary key if provided.
        If None, a minimal DataFrame with idx, layer and cluster_number is created.
    cluster_labels : dict, optional
        A dict mapping (layer, cluster_number) tuples to original node labels,
        as returned by convert_tree(). If provided, topic_df.index is expected
        to use those original labels and will be re-indexed to numeric indices internally.
        The mapping is stored as self.cluster_labels for display purposes.
    embedding_model : optional
        A SentenceTransformer model for use with topicdb.q.search().
        If None, search() will raise a helpful error.
    default_k : int, optional (default=15)
        Default number of nearest neighbours for nearby() queries.
    """

    def __init__(
        self,
        soft_cluster_tree: SoftClusterTree,
        embedding_vectors: np.ndarray,
        reduced_vectors: np.ndarray = None,
        document_df: pd.DataFrame = None,
        topic_df: pd.DataFrame = None,
        cluster_labels: dict = None,
        embedding_model=None,
        default_k: int = 15,
    ):
        self.soft_cluster_tree = soft_cluster_tree
        self.embedding_vectors = embedding_vectors
        self.reduced_vectors = reduced_vectors
        self.embedding_model = embedding_model
        self.default_k = default_k
        self.cluster_labels = cluster_labels

        # Build nearest neighbour index
        self.nn_index = pynndescent.NNDescent(embedding_vectors)

        # Document metadata
        if document_df is None:
            self.document_df = pd.DataFrame(
                {"idx": range(len(embedding_vectors))}
            )
        else:
            self.document_df = document_df.reset_index(drop=True)

        # Topic metadata
        if topic_df is None:
            self.topic_df = self._minimal_topic_df()
        else:
            # Re-index from original node labels to indices
            # cluster_labels should be a map of the form location tuple -> labels
            if cluster_labels is None:
                # In this case we are assuming that the topic_df is indexed by tuples
                label_to_idx = self.soft_cluster_tree.loc_to_idx
            if cluster_labels is not None:
                # In this case we assume the index is some set of labels
                label_to_idx = {v: i for i, v in enumerate(cluster_labels.values())}

            unknown = set(topic_df.index) - set(label_to_idx.keys())
            if unknown:
                warnings.warn(
                    f"topic_df contains {len(unknown)} row(s) with labels not "
                    f"found in cluster_labels; they will be dropped: {unknown}"
                )
                topic_df = topic_df.loc[topic_df.index.isin(label_to_idx)]
            topic_df = topic_df.copy()
            topic_df.index = topic_df.index.map(label_to_idx)
            self.topic_df = topic_df

    def _minimal_topic_df(self) -> pd.DataFrame:
        """Build a minimal topic metadata DataFrame from the SoftClusterTree."""
        rows = []
        for idx, (layer, cluster_number) in self.soft_cluster_tree.idx_to_loc.items():
            rows.append({
                "index": idx,
                "layer": layer,
                "cluster_number": cluster_number,
            })
        return pd.DataFrame(rows).set_index("index")

    @property
    def q(self) -> RootQuery:
        """Entry point for all queries."""
        return RootQuery(self)
        
    @property
    def topics(self):
        try:
            return [self.leaf(idx, self.topic_df.loc[idx].name) for idx in self.idx_to_loc.keys()]
        except:
            return self.soft_cluster_tree.topics

    # =================== Internal Methods ===================

    def _nearby(self, vector: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest neighbours of a vector."""
        vector = np.atleast_2d(vector)
        indices, _ = self.nn_index.query(vector, k=k)
        return indices.flatten()

    def _embeddings(self, indices: np.ndarray) -> np.ndarray:
        """Return embedding vectors for a set of document indices."""
        return self.embedding_vectors[indices]

    def _documents(self, indices: np.ndarray) -> pd.DataFrame:
        """Return document metadata rows for a set of indices."""
        return self.document_df.iloc[indices]

    def _info(self, indices: list) -> pd.DataFrame:
        """Return topic metadata rows for a set of indices."""
        return self.topic_df.loc[self.topic_df.index.isin(indices)]

    def _docs_where(self, indices: np.ndarray, query: str) -> np.ndarray:
        """Filter document indices by metadata column values."""
        df = self.document_df.iloc[indices]
        result = df.query(query)
        return result.index.to_numpy()
    
    def _topics_where(self, indices, query: str) -> list:
        """Filter topics by metadata column values."""
        df = self.topic_df.loc[indices]
        result = df.query(query)
        return result.index.to_list()

    def _topics(self, indices: np.ndarray, min_strength: float = 1, logic: str = "OR") -> list:
        """
        Return the topics containing a set of document indices.
        Uses the finest (lowest layer) topic for each document.
        """
        topic_sets = []
        for i in indices:
            doc_topics = set()
            for layer in range(self.soft_cluster_tree.n_layers):
                col_vec = self.soft_cluster_tree.layers[layer].getrow(i)
                nonzero_cols = (col_vec>=255*min_strength).nonzero()[1]
                if len(nonzero_cols) > 0:
                    for col in nonzero_cols:
                        doc_topics.add(self.soft_cluster_tree.loc_to_idx[(layer, col)])
                    break  # use finest layer only
            topic_sets.append(doc_topics)

        if logic == "OR":
            return list(set().union(*topic_sets))
        elif logic == "AND":
            return list(set().intersection(*topic_sets))
        else:
            raise ValueError("`logic` must be 'OR' or 'AND'")

    def _theme(self, indices: np.ndarray, vector: np.ndarray = None) -> str:
        """
        Find the most specific topic that best covers a set of document indices,
        weighted by their soft membership strengths and topic depth.

        Balances two objectives:
        - Coverage: what fraction of the query documents are inside this topic
        - Specificity: how deep the topic is in the tree

        Uses soft membership strengths to weight the coverage score,
        so documents with higher inclusion strength contribute more.

        Parameters
        ----------
        indices : np.ndarray
            Document indices (typically nearest neighbours of a query vector).
        vector : np.ndarray, optional
            Unused, reserved for future strength-weighted aggregation.

        Returns
        -------
        str
            The idx of the best matching topic.
        """
        if len(indices) == 0:
            return self.soft_cluster_tree.root_idx

        tree = self.soft_cluster_tree
        all_indices = list(tree.idx_to_loc.keys())

        best_idx = None
        best_score = -1

        for topic_idx in all_indices:
            # Get soft membership strengths for all query documents
            strengths = tree.strengths(topic_idx, indices, as_float=True)

            # Weighted coverage: mean strength over query documents
            coverage = strengths.mean()
            if coverage == 0:
                continue

            # Layer score: prefer lower layer (more specific) topics
            layer_score = ((tree.n_layers+1) - tree.idx_to_loc[topic_idx][0])/(tree.n_layers+1)

            score = coverage * layer_score

            if score > best_score:
                best_score = score
                best_idx = topic_idx

        return best_idx if best_idx is not None else tree.root_idx
    
    @property 
    def tree(self):
        """ Return the database tree as a dictionary. (self.soft_cluster_tree.children_map) """
        return self.soft_cluster_tree.children_map
    
    @property
    def cluster_matrix(self):
        """ Return the Fuzzy inclusion cluster matrix. """
        return self.soft_cluster_tree.cluster_matrix
    
    def __repr__(self):
        n_topics = self.soft_cluster_tree.n_topics
        n_docs = self.soft_cluster_tree.n_docs
        n_layers = self.soft_cluster_tree.n_layers
        string = "TopicDatabase("
        string += f"{n_docs} samples, {n_topics} topics, "
        string += f"{n_layers} layers)"
        return string


    def to_file(self, path: str):
        """ Save a TopicDatbase to a `tm.zip` file. """
        save_topic_database(self, path)

    def to_lance(self, path: str):
        """ Save a TopicDatabase to a LanceDB folder. """
        save_topic_database_lance(self, path)

    @classmethod
    def from_file(cls, path: str):
        """ Load a TopicDatabase from a `tm.zip` file. """
        return load_topic_database(path, SoftClusterTree, cls)
    
    @classmethod
    def from_lance(cls, path: str):
        """ Load a TopicDatabase from a LanceDB folder. """
        return load_topic_database_lance(path, SoftClusterTree, cls)
    
    @classmethod
    def from_topic_model(cls, topic_model):
        """ Integration with Toponymy's TopicModel class. """
        return cls(
            SoftClusterTree(
                topic_model.cluster_layers,
                topic_model.cluster_tree,
                sparsity_threshold = 0.01,
            ),
            embedding_vectors = topic_model.embedding_vectors,
            reduced_vectors = topic_model.reduced_vectors,
            document_df = topic_model.document_df,
            topic_df = topic_model.topic_df.set_index('idx'),
        )