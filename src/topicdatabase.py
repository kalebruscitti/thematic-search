import numpy as np
import pandas as pd
import pynndescent
import warnings
from typing import Union, List, Optional
from softclustertree import SoftClusterTree, Cluster, ClusterLeaf
from pathlib import Path
from serialization import (
    save_topic_database,
    load_topic_database,
    save_topic_database_lance,
    load_topic_database_lance,
)
from utilities import topic_uid
import os
import json
import tempfile
import zipfile


# =================== Query Classes ===================

class IndexQuery:
    """
    A query object carrying a set of document indices.
    Supports navigation to topics or retrieval of document data.

    Typical usage:
        topicdb.q.search("jazz music").nearby().topics().theme()
    """
    def __init__(self, db, indices: np.ndarray, vector: np.ndarray = None):
        self.db = db
        self.indices = indices
        self._vector = vector  # carry the query vector for theme()

    def __repr__(self):
        return f"IndexQuery({len(self.indices)} documents)"

    def unwrap(self) -> np.ndarray:
        return self._resolve_indices()

    def _resolve_indices(self) -> np.ndarray:
        """
        If indices is empty but a query vector is available,
        automatically fetch nearby indices using default_k.
        This allows chains like topicdb.q.search("...").theme()
        without requiring an explicit .nearby() call.
        """
        if len(self.indices) == 0 and self._vector is not None:
            return self.db._nearby(self._vector, self.db.default_k)
        return self.indices

    def nearby(self, k: int = None) -> "IndexQuery":
        """
        Find the k nearest neighbours of the query vector in the embedding space.
        Requires that this IndexQuery was created via search() or vector() or from_docs().

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbours. Defaults to topicdb.default_k.
        """
        if self._vector is None:
            raise ValueError(
                "nearby() requires a query vector. "
                "Use topicdb.q.search(), topicdb.q.vector(), or topicdb.q.from_docs() "
                "as your entry point."
            )
        k = k or self.db.default_k
        indices = self.db._nearby(self._vector, k)
        return IndexQuery(self.db, indices, vector=self._vector)

    def topics(self, logic: str = "OR") -> "TopicQuery":
        """
        Return the topics containing these document indices.
        Uses weighted nearest neighbour aggregation over soft membership vectors.

        Parameters
        ----------
        logic : str, optional (default='OR')
            'OR'  - return topics that contain any of the indices
            'AND' - return topics that contain all of the indices
        """
        topics = self.db._topics(self._resolve_indices(), logic=logic)
        return TopicQuery(self.db, topics)

    def theme(self) -> "TopicQuery":
        """
        Find the most specific topic that best covers these document indices,
        weighted by their soft membership strengths.
        Equivalent to optimal_join in the boolean implementation.
        """
        topic_uid = self.db._theme(self._resolve_indices(), self._vector)
        return TopicQuery(self.db, [topic_uid])

    def documents(self) -> pd.DataFrame:
        """Return the document metadata rows for these indices."""
        return self.db._documents(self._resolve_indices())

    def embeddings(self) -> np.ndarray:
        """Return the embedding vectors for these indices."""
        return self.db._embeddings(self._resolve_indices())

    def strengths(self, expr: Union[Cluster, str]) -> np.ndarray:
        """
        Return the inclusion strengths of these documents for a cluster expression.

        Parameters
        ----------
        expr : Cluster or str
            A Cluster expression or uid string.

        Returns
        -------
        np.ndarray of floats in [0, 1]
        """
        return self.db.soft_cluster_tree.strengths(expr, self._resolve_indices())

    def where(self, **kwargs) -> "IndexQuery":
        """
        Filter documents by metadata column values.

        Example
        -------
        query.where(author="Alice")
        """
        indices = self.db._where(self._resolve_indices(), kwargs)
        return IndexQuery(self.db, indices, vector=self._vector)


class TopicQuery:
    """
    A query object carrying a set of topic uids.
    Supports navigation within the topic tree or retrieval of topic data.

    Typical usage:
        topicdb.q.search("jazz music").theme().parents().info()
    """
    def __init__(self, db, uids: list):
        self.db = db
        self.uids = uids

    def __repr__(self):
        return f"TopicQuery({self.uids})"

    def unwrap(self) -> list:
        return self.uids

    def parents(self) -> "TopicQuery":
        """Return the parent topics of these topics."""
        result = set()
        for uid in self.uids:
            result.update(self.db.soft_cluster_tree.parents(uid))
        return TopicQuery(self.db, list(result))

    def children(self) -> "TopicQuery":
        """Return the child topics of these topics."""
        result = set()
        for uid in self.uids:
            result.update(self.db.soft_cluster_tree.children(uid))
        return TopicQuery(self.db, list(result))

    def join(self) -> "TopicQuery":
        """Return the least upper bound (lowest common ancestor) of these topics."""
        uid = self.db.soft_cluster_tree.join(self.uids)
        return TopicQuery(self.db, [uid])

    def inside(
        self,
        min_strength: float = 1.0,
    ) -> "IndexQuery":
        """
        Return document indices inside these topics, combined with OR logic.

        Parameters
        ----------
        min_strength : float, optional (default=1.0)
            Minimum inclusion strength in [0, 1].
        """
        indices = set()
        for uid in self.uids:
            indices.update(
                self.db.soft_cluster_tree.inside(uid, min_strength=min_strength).tolist()
            )
        return IndexQuery(self.db, np.array(sorted(indices)))

    def info(self) -> pd.DataFrame:
        """Return topic metadata rows for these topics."""
        return self.db._info(self.uids)


# =================== Root Query ===================

class RootQuery:
    """
    Entry point for all queries on a TopicDatabase.
    Access via topicdb.q
    """
    def __init__(self, db):
        self.db = db

    def search(self, text: str) -> IndexQuery:
        """
        Embed a text string and return an IndexQuery carrying the query vector.
        Requires a sentence transformer model to be provided at construction time.

        Parameters
        ----------
        text : str
            The query string to embed.
        """
        if self.db.embedding_model is None:
            raise ValueError(
                "search() requires an embedding model. "
                "Pass a SentenceTransformer model as the `embedding_model` "
                "parameter when constructing TopicDatabase."
            )
        vec = self.db.embedding_model.encode(text)
        return IndexQuery(self.db, np.array([]), vector=vec)

    def vector(self, vec: np.ndarray) -> IndexQuery:
        """
        Use a raw embedding vector as the query entry point.

        Parameters
        ----------
        vec : np.ndarray
            The query embedding vector.
        """
        return IndexQuery(self.db, np.array([]), vector=vec)

    def from_docs(self, indices: Union[List[int], np.ndarray]) -> IndexQuery:
        """
        Use a set of document indices as the query entry point.
        Their embedding vectors are averaged to form a single query vector.

        Parameters
        ----------
        indices : list or np.ndarray
            Document indices to use as the query.
        """
        indices = np.array(indices)
        vec = self.db._embeddings(indices).mean(axis=0)
        return IndexQuery(self.db, indices, vector=vec)

    def topic(self, layer: int, cluster_number: int) -> TopicQuery:
        """
        Enter the query via a known topic node.

        Parameters
        ----------
        layer : int
        cluster_number : int
        """
        uid = topic_uid((layer, cluster_number))
        return TopicQuery(self.db, [uid])

    def topic_uid(self, uid: str) -> TopicQuery:
        """
        Enter the query via a known topic uid string.

        Parameters
        ----------
        uid : str
        """
        return TopicQuery(self.db, [uid])


# =================== Topic Database ===================

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
    document_df : pd.DataFrame, optional
        Document metadata. If None, a minimal DataFrame with just indices is created.
    topic_df : pd.DataFrame, optional
        Topic metadata. Must have a 'uid' column as primary key if provided.
        If None, a minimal DataFrame with uid, layer and cluster_number is created.
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
        embedding_model=None,
        default_k: int = 15,
    ):
        self.soft_cluster_tree = soft_cluster_tree
        self.embedding_vectors = embedding_vectors
        self.reduced_vectors = reduced_vectors
        self.embedding_model = embedding_model
        self.default_k = default_k

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
            if "uid" not in topic_df.columns:
                raise ValueError("topic_df must have a 'uid' column as primary key.")
            self.topic_df = topic_df.set_index("uid")

    def _minimal_topic_df(self) -> pd.DataFrame:
        """Build a minimal topic metadata DataFrame from the SoftClusterTree."""
        rows = []
        for uid, (layer, cluster_number) in self.soft_cluster_tree.uid_to_loc.items():
            rows.append({
                "uid": uid,
                "layer": layer,
                "cluster_number": cluster_number,
            })
        return pd.DataFrame(rows).set_index("uid")

    @property
    def q(self) -> RootQuery:
        """Entry point for all queries."""
        return RootQuery(self)
        
    @property
    def topics(self):
        try:
            return [self.leaf(uid, self.topic_df.loc[uid].name) for uid in self.uid_to_loc.keys()]
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

    def _info(self, uids: list) -> pd.DataFrame:
        """Return topic metadata rows for a set of uids."""
        return self.topic_df.loc[self.topic_df.index.isin(uids)]

    def _where(self, indices: np.ndarray, selectors: dict) -> np.ndarray:
        """Filter document indices by metadata column values."""
        df = self.document_df.iloc[indices]
        for col, value in selectors.items():
            df = df[df[col] == value]
        return df.index.to_numpy()

    def _topics(self, indices: np.ndarray, logic: str = "OR") -> list:
        """
        Return the topics containing a set of document indices.
        Uses the finest (lowest layer) topic for each document.
        """
        topic_sets = []
        for i in indices:
            doc_topics = set()
            for layer in range(self.soft_cluster_tree.n_layers):
                col_vec = self.soft_cluster_tree.layers[layer].getrow(i)
                nonzero_cols = col_vec.nonzero()[1]
                if len(nonzero_cols) > 0:
                    for col in nonzero_cols:
                        doc_topics.add(self.soft_cluster_tree.loc_to_uid[(layer, col)])
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
            The uid of the best matching topic.
        """
        if len(indices) == 0:
            return self.soft_cluster_tree.root_uid

        tree = self.soft_cluster_tree
        all_uids = list(tree.uid_to_loc.keys())

        best_uid = None
        best_score = -1

        for uid in all_uids:
            # Get soft membership strengths for all query documents
            strengths = tree.strengths(uid, indices, as_float=True)

            # Weighted coverage: mean strength over query documents
            coverage = strengths.mean()
            if coverage == 0:
                continue

            # Layer score: prefer lower layer (more specific) topics
            layer_score = ((tree.n_layers+1) - tree.uid_to_loc[uid][0])/(tree.n_layers+1)

            score = coverage * layer_score

            if score > best_score:
                best_score = score
                best_uid = uid

        return best_uid if best_uid is not None else tree.root_uid

    def to_file(self, path: str):
        save_topic_database(self, path)

    def to_lance(self, path: str):
        save_topic_database_lance(self, path)

    @classmethod
    def from_file(cls, path: str):
        return load_topic_database(path, SoftClusterTree, cls)
    
    @classmethod
    def from_lance(cls, path: str):
        return load_topic_database_lance(path, SoftClusterTree, cls)