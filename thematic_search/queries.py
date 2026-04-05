import numpy as np
from typing import Union, List, Optional
import pandas as pd
from .softclustertree import IndexExpr

class FuzzyQuery:
    """
    A query object operating on the entire soft cluster matrix at once.
    """
    def __init__(self, db, matrix: np.ndarray):
        self.db = db
        self.matrix = matrix
        self.idx_to_idx = self.db.soft_cluster_tree.idx_to_idx

    def samples_where(self, query: str):
        indices = self.db._docs_where(
            np.arange(self.matrix.shape[0]),
            query
        )
        self.matrix[~indices,:] = 0
        return FuzzyQuery(self, self.db, self.matrix)

    def topics_where(self, query: str):
        indices = self.db._topics_where(
            self.db.soft_cluster_tree.indices, 
            query
        )
        indices = [self.idx_to_idx[idx] for idx in indices]
        self.matrix[:, ~indices] = 0
        A = self.db.soft_cluster_tree.adjacency_closure
        indexed_colimit = np.max(
            self.matrix[:, :, None] * A[None, :, :],
            axis=1
        )
        return FuzzyQuery(self, self.db, indexed_colimit)

    def samples(self, threshold: float=1.0):
        sample_vec = np.max(self.matrix, axis=1)
        indices = (sample_vec>=threshold).nonzero()[0]
        return SampleQuery(self.db, indices)
    
    def topics(self, threshold: float=1.0):
        topic_vec = np.max(self.matrix, axis=0)
        indices = (topic_vec>=threshold).nonzero()[0]  
        idx_to_idx = {v:k for k,v in self.idx_to_idx}
        indices = np.array([idx_to_idx[i] for i in indices])
        return TopicQuery(self.db, indices)

class SampleQuery:
    """
    A query object carrying a set of document indices.
    Supports navigation to topics or retrieval of document data.

    Typical usage:
        topicdb.q.search("jazz music").nearby().topics().theme()
    """
    def __init__(self, db, indices: np.ndarray):
        self.db = db
        self.indices = indices

    def __repr__(self):
        return f"SampleQuery({len(self.indices)} samples)"

    def unwrap(self) -> np.ndarray:
        return self.indices

    def neighbours(self, k: int = None) -> "SampleQuery":
        """
        Average the embeddings of this SampleQuery's indices, then find the k nearest neighbours.

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbours. Defaults to topicdb.default_k.
        """
        k = k or self.db.default_k 
        vector = self.embeddings().mean(axis=0)
        indices = self.db._nearby(vector, k)
        return SampleQuery(self.db, indices)

    def topics(self, min_strength: float = 1, logic: str = "OR") -> "TopicQuery":
        """
        Return the topics containing these document indices.
        Uses weighted nearest neighbour aggregation over soft membership vectors.

        Parameters
        ----------
        logic : str, optional (default='OR')
            'OR'  - return topics that contain any of the indices
            'AND' - return topics that contain all of the indices
        """
        topics = self.db._topics(
            self.indices,
            min_strength=min_strength,
            logic=logic
        )
        return TopicQuery(self.db, topics)

    def theme(self) -> "TopicQuery":
        """
        Find the most specific topic that best covers these document indices,
        weighted by their soft membership strengths.
        """
        topic_idx = self.db._theme(self.indices)
        return TopicQuery(self.db, [topic_idx])

    def metadata(self) -> pd.DataFrame:
        """Return the document metadata rows for these indices."""
        return self.db._documents(self.indices)

    def embeddings(self) -> np.ndarray:
        """Return the embedding vectors for these indices."""
        return self.db._embeddings(self.indices)

    def strengths(self, expr: Union[IndexExpr, str]) -> np.ndarray:
        """
        Return the inclusion strengths of these documents for a cluster expression.

        Parameters
        ----------
        expr : Cluster or str
            A Cluster expression or idx string.

        Returns
        -------
        np.ndarray of floats in [0, 1]
        """
        return self.db.soft_cluster_tree.strengths(expr, self.indices)

    def where(self, query:str ) -> "SampleQuery":
        """
        Filter documents by metadata column values.

        Example
        -------
        query.where("author=='Alice'")
        """
        indices = self.db._docs_where(self.indices, query)
        return SampleQuery(self.db, indices)


class TopicQuery:
    """
    A query object carrying a set of topic indices.
    Supports navigation within the topic tree or retrieval of topic data.

    Typical usage:
        topicdb.q.search("jazz music").theme().parents().info()
    """
    def __init__(self, db, indices: list):
        self.db = db
        self.indices = indices

    def __repr__(self):
        return f"TopicQuery({len(self.indices)} topics)"

    def unwrap(self) -> list:
        return self.indices

    def parents(self) -> "TopicQuery":
        """Return the parent topics of these topics."""
        result = set()
        for idx in self.indices:
            result.update(self.db.soft_cluster_tree.parents(idx))
        return TopicQuery(self.db, list(result))

    def children(self) -> "TopicQuery":
        """Return the child topics of these topics."""
        result = set()
        for idx in self.indices:
            result.update(self.db.soft_cluster_tree.children(idx))
        return TopicQuery(self.db, list(result))

    def least_upper_bound(self) -> "TopicQuery":
        """Return the least upper bound (lowest common ancestor) of these topics."""
        return TopicQuery(self.db, self.db.soft_cluster_tree.join(self.indices))

    def inside(
        self,
        min_strength: float = 1.0,
    ) -> "SampleQuery":
        """
        Return document indices inside these topics, combined with OR logic.

        Parameters
        ----------
        min_strength : float, optional (default=1.0)
            Minimum inclusion strength in [0, 1].
        """
        indices = set()
        for idx in self.indices:
            indices.update(
                self.db.soft_cluster_tree.inside(idx, min_strength=min_strength).tolist()
            )
        return SampleQuery(self.db, np.array(sorted(indices)))

    def info(self) -> pd.DataFrame:
        """Return topic metadata rows for these topics."""
        return self.db._info(self.indices)
    
    def where(self, query:str ) -> "SampleQuery":
        """
        Filter topics by metadata column values.

        Example
        -------
        query.where("layer>=1")
        """
        topics = self.db._topics_where(self.indices, query)
        return TopicQuery(self.db, topics)


# =================== Root Query ===================

class RootQuery:
    """
    Entry point for all queries on a TopicDatabase.
    Access via topicdb.q
    """
    def __init__(self, db):
        self.db = db

    def neighbours(self, text: str, k: int=15) -> SampleQuery:
        """
        Embed a text string and return an SampleQuery carrying the query vector.
        Requires a sentence transformer model to be provided at construction time.

        Parameters
        ----------
        text : str
            The query string to embed.
        """
        if self.db.embedding_model is None:
            raise ValueError(
                "neighbours() requires an embedding model. "
                "Pass a SentenceTransformer model as the `embedding_model` "
                "parameter when constructing TopicDatabase."
            )
        vec = self.db.embedding_model.encode(text)
        indices = self.db._nearby(vec, k=k)
        return SampleQuery(self.db, indices)
    
    def from_docs(self, indices: Union[List[int], np.ndarray]) -> SampleQuery:
        """
        Use a set of document indices as the query entry point.
        Their embedding vectors are averaged to form a single query vector.

        Parameters
        ----------
        indices : list or np.ndarray
            Document indices to use as the query.
        """
        indices = np.array(indices)
        return SampleQuery(self.db, indices)

    def topic(self, layer: int, cluster_number: int) -> TopicQuery:
        """
        Enter the query via a known topic node.

        Parameters
        ----------
        layer : int
        cluster_number : int
        """
        idx = self.db.soft_cluster_tree.loc_to_idx[(layer, cluster_number)]
        return TopicQuery(self.db, [idx])

    def topic_idx(self, idx: str) -> TopicQuery:
        """
        Enter the query via a known topic index.

        Parameters
        ----------
        idx : int
        """
        return TopicQuery(self.db, [idx])

    def topic_name(self, name: str) -> TopicQuery:
        """
        Entry the query via a known topic name string.

        Parameters
        ----------
        name : str
        """
        match_df = self.db.topic_df[self.db.topic_df['name']==name]
        if len(match_df)==1:
            idx = match_df.index[0]
            return TopicQuery(self.db, [idx])
        elif len(match_df)==0:
            print(f"No topics with name '{name}' found.")
            return TopicQuery(self.db, [])
        elif len(match_df)>1:
            print(f"Multiple topics with name '{name}' found:")
            print(match_df)
            print(f"Please select a topic by index to disambiguate.")
            return TopicQuery(self.db, [])

    def docs_where(self, query: str) -> SampleQuery:
        """
        Request the set of documents whose metadata matches a Pandas query.

        Parameters
        ----------
        query : str
            A query string following `pandas.DataFrame.query` syntax.
        """
        all_indices = self.db.document_df.index.to_numpy()
        return SampleQuery(self.db, self.db._docs_where(all_indices, query))
    
    def topics_where(self, query: str) -> TopicQuery:
        """
        Request the set of topics whose metadata matches a Pandas query.

        Parameters
        ----------
        query : str
            A query string following `pandas.DataFrame.query` syntax.
        """      
        all_indices = self.db.topic_df.index.to_numpy()
        return TopicQuery(self.db,self.db._topics_where(all_indices, query))

    def index_expr(self, expr: IndexExpr, min_strength: float=1.0) -> SampleQuery:
        """
        Initialize an SampleQuery with the indices inside an IndexExpr 

        Parameters
        ----------
        expr: IndexExpr
            The expression to evaluate 
        min_strength: float (optional default=1.0)
            Minimum inclusion strength in [0, 1].
        """
        indices = self.db.soft_cluster_tree.inside(expr, min_strength=min_strength)
        return SampleQuery(self.db, indices)
