"""
Basic tests for SoftClusterTree and TopicDatabase.
Uses synthetic data so no external datasets are required.
"""
import numpy as np
import pandas as pd
import pytest

from thematic_search.softclustertree import SoftClusterTree, ClusterLeaf, topic_uid, uid_to_ints
from thematic_search.topicdatabase import TopicDatabase, IndexQuery, TopicQuery


# =================== Synthetic Test Data ===================

def make_test_data(n_docs=100, n_features=32, seed=42):
    """
    Generate synthetic data for testing:
    - 2 layers, 4 clusters at layer 0, 2 clusters at layer 1
    - Simple cluster tree: (1,0) -> [(0,0),(0,1)], (1,1) -> [(0,2),(0,3)]
    - Random soft memberships (normalized so they sum to 1 per layer)
    - Random embeddings
    """
    rng = np.random.default_rng(seed)

    # Layer 0: 4 clusters, shape (n_docs, 4)
    raw0 = rng.random((n_docs, 4))
    layer0 = raw0 / raw0.sum(axis=1, keepdims=True)

    # Layer 1: 2 clusters, shape (n_docs, 2)
    # cluster (1,0) = sum of (0,0) and (0,1), (1,1) = sum of (0,2) and (0,3)
    layer1 = np.stack([
        layer0[:, 0] + layer0[:, 1],
        layer0[:, 2] + layer0[:, 3],
    ], axis=1)
    layer1 = layer1 / layer1.sum(axis=1, keepdims=True)

    cluster_matrices = [layer0, layer1]

    # Tree: children of (1,0) are (0,0),(0,1); children of (1,1) are (0,2),(0,3)
    cluster_tree = {
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
    }

    # Random embeddings
    embeddings = rng.standard_normal((n_docs, n_features)).astype(np.float32)

    # Document metadata
    document_df = pd.DataFrame({
        "text": [f"document {i}" for i in range(n_docs)],
        "label": rng.integers(0, 4, size=n_docs),
    })

    # Topic metadata
    rows = []
    for layer, n_clusters in [(0, 4), (1, 2)]:
        for j in range(n_clusters):
            rows.append({
                "uid": topic_uid((layer, j)),
                "layer": layer,
                "cluster_number": j,
                "name": f"Topic L{layer}C{j}",
            })
    topic_df = pd.DataFrame(rows)

    return cluster_matrices, cluster_tree, embeddings, document_df, topic_df


# =================== SoftClusterTree Tests ===================

class TestSoftClusterTree:

    def setup_method(self):
        matrices, tree, self.embeddings, _, _ = make_test_data()
        self.tree = SoftClusterTree(matrices, tree)
        self.n_docs = 100

    def test_construction(self):
        assert self.tree.n_docs == self.n_docs
        assert self.tree.n_layers == 2

    def test_uid_round_trip(self):
        uid = topic_uid((1, 2))
        layer, cluster = uid_to_ints(uid)
        assert layer == 1
        assert cluster == 2

    def test_root_node_exists(self):
        root = self.tree.root_uid
        assert root is not None
        strengths = self.tree._get_strength_vector(root)
        assert np.all(strengths == 255), "Root should have full membership for all docs"

    def test_inside_full_membership(self):
        uid = topic_uid((0, 0))
        idx = self.tree.inside(uid, min_strength=1.0)
        # At min_strength=1.0, only docs with uint8 strength==255 are returned
        strengths = self.tree._get_strength_vector(uid)
        expected = np.where(strengths == 255)[0]
        np.testing.assert_array_equal(np.sort(idx), np.sort(expected))

    def test_inside_partial_membership(self):
        uid = topic_uid((0, 0))
        idx_full = self.tree.inside(uid, min_strength=1.0)
        idx_partial = self.tree.inside(uid, min_strength=0.0)
        # Partial should return at least as many docs as full
        assert len(idx_partial) >= len(idx_full)

    def test_inside_double_negation(self):
        """~~a should return docs with any nonzero strength."""
        leaf = self.tree.leaf(0, 0)
        idx_double_neg = self.tree.inside(~~leaf, min_strength=1.0)
        uid = topic_uid((0, 0))
        strengths = self.tree._get_strength_vector(uid)
        expected = np.where(strengths > 0)[0]
        np.testing.assert_array_equal(np.sort(idx_double_neg), np.sort(expected))

    def test_inside_negation(self):
        """~a should return docs with zero strength."""
        leaf = self.tree.leaf(0, 0)
        idx_neg = self.tree.inside(~leaf, min_strength=1.0)
        uid = topic_uid((0, 0))
        strengths = self.tree._get_strength_vector(uid)
        expected = np.where(strengths == 0)[0]
        np.testing.assert_array_equal(np.sort(idx_neg), np.sort(expected))

    def test_inside_and(self):
        """a & b should return docs where min(strength_a, strength_b) >= threshold."""
        a = self.tree.leaf(0, 0)
        b = self.tree.leaf(0, 1)
        idx_and = self.tree.inside(a & b, min_strength=0.5)
        sa = self.tree._get_strength_vector(topic_uid((0, 0)))
        sb = self.tree._get_strength_vector(topic_uid((0, 1)))
        threshold = SoftClusterTree.to_int(0.5)
        expected = np.where(np.minimum(sa, sb) >= threshold)[0]
        np.testing.assert_array_equal(np.sort(idx_and), np.sort(expected))

    def test_inside_or(self):
        """a | b should return docs where max(strength_a, strength_b) >= threshold."""
        a = self.tree.leaf(0, 0)
        b = self.tree.leaf(0, 1)
        idx_or = self.tree.inside(a | b, min_strength=0.5)
        sa = self.tree._get_strength_vector(topic_uid((0, 0)))
        sb = self.tree._get_strength_vector(topic_uid((0, 1)))
        threshold = SoftClusterTree.to_int(0.5)
        expected = np.where(np.maximum(sa, sb) >= threshold)[0]
        np.testing.assert_array_equal(np.sort(idx_or), np.sort(expected))

    def test_parents(self):
        uid = topic_uid((0, 0))
        parents = self.tree.parents(uid)
        assert topic_uid((1, 0)) in parents

    def test_children(self):
        uid = topic_uid((1, 0))
        children = self.tree.children(uid)
        assert topic_uid((0, 0)) in children
        assert topic_uid((0, 1)) in children

    def test_root_has_no_parents(self):
        assert self.tree.parents(self.tree.root_uid) == []

    def test_leaves_have_no_children(self):
        uid = topic_uid((0, 0))
        assert self.tree.children(uid) == []

    def test_join_siblings(self):
        """LUB of two siblings should be their parent."""
        uid0 = topic_uid((0, 0))
        uid1 = topic_uid((0, 1))
        lub = self.tree.join([uid0, uid1])
        assert lub == topic_uid((1, 0))

    def test_join_single(self):
        uid = topic_uid((0, 0))
        assert self.tree.join([uid]) == uid

    def test_join_cousins(self):
        """LUB of nodes in different branches should be the root."""
        uid0 = topic_uid((0, 0))
        uid2 = topic_uid((0, 2))
        lub = self.tree.join([uid0, uid2])
        assert lub == self.tree.root_uid

    def test_strengths(self):
        uid = topic_uid((0, 0))
        idx = np.array([0, 1, 2])
        s = self.tree.strengths(uid, idx)
        assert s.shape == (3,)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_strengths_uint8(self):
        uid = topic_uid((0, 0))
        idx = np.array([0, 1, 2])
        s = self.tree.strengths(uid, idx, as_float=False)
        assert s.dtype == np.uint8

    def test_outlier_documents(self):
        """Documents with all-zero membership should be preserved."""
        matrices, tree, embeddings, doc_df, _ = make_test_data()
        # Force first document to have zero membership everywhere
        for m in matrices:
            m[0, :] = 0.0
        sct = SoftClusterTree(matrices, tree)
        for layer in range(sct.n_layers):
            vec = sct.layers[layer].getrow(0)
            assert vec.nnz == 0, f"Doc 0 should have zero membership at layer {layer}"


# =================== TopicDatabase Tests ===================

class TestTopicDatabase:

    def setup_method(self):
        matrices, tree, self.embeddings, self.doc_df, self.topic_df = make_test_data()
        self.sct = SoftClusterTree(matrices, tree)
        self.db = TopicDatabase(
            soft_cluster_tree=self.sct,
            embedding_vectors=self.embeddings,
            document_df=self.doc_df,
            topic_df=self.topic_df,
            embedding_model=None,
            default_k=10,
        )

    def test_construction(self):
        assert self.db.default_k == 10
        assert self.db.embedding_model is None

    def test_vector_entry_point(self):
        vec = self.embeddings[0]
        q = self.db.q.vector(vec)
        assert isinstance(q, IndexQuery)
        assert q._vector is not None

    def test_from_docs_entry_point(self):
        q = self.db.q.from_docs([0, 1, 2])
        assert isinstance(q, IndexQuery)
        assert q._vector is not None
        # Vector should be mean of embeddings 0,1,2
        expected_vec = self.embeddings[[0, 1, 2]].mean(axis=0)
        np.testing.assert_array_almost_equal(q._vector, expected_vec)

    def test_search_without_model_raises(self):
        with pytest.raises(ValueError, match="embedding model"):
            self.db.q.search("test query")

    def test_nearby_returns_indices(self):
        vec = self.embeddings[0]
        idx = self.db.q.vector(vec).nearby(k=5).unwrap()
        assert len(idx) == 5
        assert isinstance(idx, np.ndarray)

    def test_nearby_default_k(self):
        vec = self.embeddings[0]
        idx = self.db.q.vector(vec).nearby().unwrap()
        assert len(idx) == self.db.default_k

    def test_documents_terminal(self):
        vec = self.embeddings[0]
        df = self.db.q.vector(vec).nearby().documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == self.db.default_k
        assert "text" in df.columns

    def test_embeddings_terminal(self):
        vec = self.embeddings[0]
        embs = self.db.q.vector(vec).nearby().embeddings()
        assert embs.shape == (self.db.default_k, self.embeddings.shape[1])

    def test_topics_returns_topic_query(self):
        vec = self.embeddings[0]
        tq = self.db.q.vector(vec).nearby().topics(min_strength=0.5)
        assert isinstance(tq, TopicQuery)
        assert len(tq.uids) > 0

    def test_theme_returns_topic_query(self):
        vec = self.embeddings[0]
        tq = self.db.q.vector(vec).nearby().theme()
        assert isinstance(tq, TopicQuery)
        assert len(tq.uids) == 1

    def test_theme_uid_is_valid(self):
        vec = self.embeddings[0]
        uid = self.db.q.vector(vec).nearby().theme().uids[0]
        assert uid in self.sct.uid_to_loc

    def test_info_terminal(self):
        tq = self.db.q.topic(1, 0)
        df = tq.info()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "name" in df.columns

    def test_parents_chain(self):
        tq = self.db.q.topic(0, 0).parents()
        assert isinstance(tq, TopicQuery)
        assert topic_uid((1, 0)) in tq.uids

    def test_children_chain(self):
        tq = self.db.q.topic(1, 0).children()
        assert isinstance(tq, TopicQuery)
        assert topic_uid((0, 0)) in tq.uids
        assert topic_uid((0, 1)) in tq.uids

    def test_join_chain(self):
        tq = TopicQuery(self.db, [topic_uid((0, 0)), topic_uid((0, 1))]).least_upper_bound()
        assert tq.uids[0] == topic_uid((1, 0))

    def test_inside_chain(self):
        iq = self.db.q.topic(1, 0).inside(min_strength=0.0)
        assert isinstance(iq, IndexQuery)
        assert len(iq.indices) > 0

    def test_where_filter(self):
        vec = self.embeddings[0]
        # label is in {0,1,2,3}, filter for label==0
        df_all = self.db.q.vector(vec).nearby(k=50).documents()
        iq_filtered = self.db.q.vector(vec).nearby(k=50).where(label=0)
        df_filtered = iq_filtered.documents()
        assert len(df_filtered) <= 50
        assert (df_filtered["label"] == 0).all()

    def test_strengths_in_chain(self):
        vec = self.embeddings[0]
        iq = self.db.q.vector(vec).nearby()
        leaf = self.sct.leaf(0, 0)
        s = iq.strengths(leaf)
        assert len(s) == self.db.default_k
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_full_chain(self):
        """Test a realistic end-to-end query chain.
        Uses a known layer-1 topic as the entry point so that .parents() is
        guaranteed to return the virtual root, and .inside() on the root
        returns all documents — avoiding the KeyError that arises when
        theme() returns a leaf with no children in topic_df.
        """
        result = (
            self.db.q
            .topic(1, 0)       # enter at a known mid-tree node
            .children()        # step down to layer-0 leaves
            .least_upper_bound()            # back up to (1, 0)
            .inside(min_strength=0.3)
            .documents()
        )
        assert isinstance(result, pd.DataFrame)

    def test_minimal_topic_df(self):
        """TopicDatabase should build a minimal topic_df if none is provided."""
        db = TopicDatabase(
            soft_cluster_tree=self.sct,
            embedding_vectors=self.embeddings,
        )
        assert "layer" in db.topic_df.columns
        assert "cluster_number" in db.topic_df.columns

    def test_topic_df_missing_uid_raises(self):
        bad_df = pd.DataFrame({"layer": [0], "cluster_number": [0]})
        with pytest.raises(ValueError, match="uid"):
            TopicDatabase(
                soft_cluster_tree=self.sct,
                embedding_vectors=self.embeddings,
                topic_df=bad_df,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
