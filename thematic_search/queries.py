import numpy as np
from typing import Union, List, Optional
import pandas as pd
from .softclustertree import Cluster
from .utilities import topic_uid

class IndexQuery:
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
        return f"IndexQuery({len(self.indices)} documents)"

    def unwrap(self) -> np.ndarray:
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
        k = k or self.db.default_k 
        vector = self.embeddings().mean(axis=0)
        indices = self.db._nearby(vector, k)
        return IndexQuery(self.db, indices)

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
        topic_uid = self.db._theme(self.indices)
        return TopicQuery(self.db, [topic_uid])

    def documents(self) -> pd.DataFrame:
        """Return the document metadata rows for these indices."""
        return self.db._documents(self.indices)

    def embeddings(self) -> np.ndarray:
        """Return the embedding vectors for these indices."""
        return self.db._embeddings(self.indices)

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
        return self.db.soft_cluster_tree.strengths(expr, self.indices)

    def where(self, query:str ) -> "IndexQuery":
        """
        Filter documents by metadata column values.

        Example
        -------
        query.where("author=='Alice'")
        """
        indices = self.db._docs_where(self.indices, query)
        return IndexQuery(self.db, indices)


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
        return f"TopicQuery({len(self.uids)} topics)"

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

    def least_upper_bound(self) -> "TopicQuery":
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

    def search(self, text: str, k: int=15) -> IndexQuery:
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
        indices = self.db._nearby(vec, k=k)
        return IndexQuery(self.db, indices)
    
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
        return IndexQuery(self.db, indices)

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

    def topic_name(self, name: str) -> TopicQuery:
        """
        Entry the query via a known topic name string.

        Parameters
        ----------
        name : str
        """
        match_df = self.db.topic_df[self.db.topic_df['name']==name]
        if len(match_df)==1:
            uid = match_df.index[0]
            return TopicQuery(self.db, [uid])
        elif len(match_df)==0:
            print(f"No topics with name '{name}' found.")
            return TopicQuery(self.db, [])
        elif len(match_df)>1:
            print(f"Multiple topics with name '{name}' found:")
            print(match_df)
            print(f"Please select a topic by UID to disambiguate.")
            return TopicQuery(self.db, [])

    def docs_where(self, query: str) -> IndexQuery:
        """
        Request the set of documents whose metadata matches a Pandas query.

        Parameters
        ----------
        query : str
            A query string following `pandas.DataFrame.query` syntax.
        """
        all_indices = self.db.document_df.index.to_numpy()
        return IndexQuery(self.db, self.db._docs_where(all_indices, query))
    
    def topics_where(self, query: str) -> TopicQuery:
        """
        Request the set of topics whose metadata matches a Pandas query.

        Parameters
        ----------
        query : str
            A query string following `pandas.DataFrame.query` syntax.
        """      
        return TopicQuery(self.db, self.db._topics_where(query))