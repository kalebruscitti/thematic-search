"""
Tests for TopicDatabase serialization (zip and Lance backends).

Run with:
    pytest test_serialization.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from thematic_search import SoftClusterTree, TopicDatabase

# =============================================================================
# Shared fixtures
# =============================================================================

N_DOCS = 40
EMB_DIM = 8
RED_DIM = 2

# Two-layer cluster tree:
#
#        (root)
#       /      \
#   (1,0)    (1,1)          <- layer 1 (2 clusters)
#   /   \       \
# (0,0) (0,1)  (0,2)        <- layer 0 (3 clusters)

CLUSTER_TREE = {
    (2, 0): [(1, 0), (1, 1)],
    (1, 0): [(0, 0), (0, 1)],
    (1, 1): [(0, 2)],
}


def _make_cluster_matrices(rng: np.random.Generator):
    """
    Two float matrices with values in [0, 1].
    Layer 0: 40 docs x 3 clusters  (sparse-ish)
    Layer 1: 40 docs x 2 clusters  (denser)
    """
    # Layer 0: each doc belongs strongly to exactly one cluster
    layer0 = np.zeros((N_DOCS, 3), dtype=np.float32)
    assignments = rng.integers(0, 3, size=N_DOCS)
    for doc, cluster in enumerate(assignments):
        layer0[doc, cluster] = rng.uniform(0.7, 1.0)

    # Layer 1: soft memberships, some overlap
    layer1 = np.zeros((N_DOCS, 2), dtype=np.float32)
    layer1[:, 0] = np.where(assignments < 2, rng.uniform(0.5, 1.0, N_DOCS), 0.0)
    layer1[:, 1] = np.where(assignments >= 1, rng.uniform(0.4, 1.0, N_DOCS), 0.0)

    return [layer0, layer1]


def _make_topic_database(with_reduced: bool = True) -> TopicDatabase:
    rng = np.random.default_rng(42)

    cluster_matrices = _make_cluster_matrices(rng)
    soft_cluster_tree = SoftClusterTree(
        cluster_matrices=cluster_matrices,
        cluster_tree=CLUSTER_TREE,
    )

    embedding_vectors = rng.standard_normal((N_DOCS, EMB_DIM)).astype(np.float32)
    reduced_vectors = rng.standard_normal((N_DOCS, RED_DIM)).astype(np.float32) \
        if with_reduced else None

    document_df = pd.DataFrame({
        "title": [f"doc_{i}" for i in range(N_DOCS)],
        "year":  rng.integers(2000, 2024, size=N_DOCS),
    })

    return TopicDatabase(
        soft_cluster_tree=soft_cluster_tree,
        embedding_vectors=embedding_vectors,
        reduced_vectors=reduced_vectors,
        sample_df=document_df,
        embedding_model=None,
    )


# =============================================================================
# Assertion helpers  (backend-agnostic, called for both zip and Lance)
# =============================================================================

def _assert_round_trip(original: TopicDatabase, loaded: TopicDatabase):
    """Assert that all serializable fields survive a round-trip."""

    # --- embedding_model is intentionally dropped ---
    assert loaded.embedding_model is None

    # --- embedding vectors ---
    np.testing.assert_array_almost_equal(
        original.embedding_vectors,
        loaded.embedding_vectors,
        decimal=5,
        err_msg="embedding_vectors mismatch",
    )

    # --- reduced vectors ---
    if original.reduced_vectors is None:
        assert loaded.reduced_vectors is None, "reduced_vectors should be None"
    else:
        assert loaded.reduced_vectors is not None, "reduced_vectors should not be None"
        np.testing.assert_array_almost_equal(
            original.reduced_vectors,
            loaded.reduced_vectors,
            decimal=5,
            err_msg="reduced_vectors mismatch",
        )

    # --- cluster matrices (sparse, compare as dense uint8) ---
    orig_layers = original.soft_cluster_tree.layers
    load_layers = loaded.soft_cluster_tree.layers
    assert len(orig_layers) == len(load_layers), "number of cluster layers mismatch"
    for i, (orig_mat, load_mat) in enumerate(zip(orig_layers, load_layers)):
        np.testing.assert_array_equal(
            orig_mat.toarray(),
            load_mat.toarray(),
            err_msg=f"cluster matrix mismatch at layer {i}",
        )

    # --- tree topology ---
    assert original.soft_cluster_tree.children_map == loaded.soft_cluster_tree.children_map, \
        "children_map mismatch"
    assert original.soft_cluster_tree.parent_map == loaded.soft_cluster_tree.parent_map, \
        "parent_map mismatch"

    # --- document metadata ---
    pd.testing.assert_frame_equal(
        original.sample_df.reset_index(drop=True),
        loaded.sample_df.reset_index(drop=True),
        check_like=True,
    )

    # --- topic metadata ---
    orig_topic = original.topic_df.sort_index()
    load_topic = loaded.topic_df.sort_index()
    pd.testing.assert_frame_equal(orig_topic, load_topic, check_like=True)


def _assert_query_works(db: TopicDatabase):
    """Smoke-test a basic query chain on a loaded database."""
    random_indices = np.random.default_rng().choice(len(db.sample_df),size=5)
    result = db.q.samples(random_indices).topics().metadata()
    assert isinstance(result, pd.DataFrame)


# =============================================================================
# Zip backend tests
# =============================================================================

class TestZipBackend:

    def test_round_trip_with_reduced(self):
        db = _make_topic_database(with_reduced=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            db.to_file(path)
            loaded = TopicDatabase.from_file(path)
        _assert_round_trip(db, loaded)

    def test_round_trip_without_reduced(self):
        db = _make_topic_database(with_reduced=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            db.to_file(path)
            loaded = TopicDatabase.from_file(path)
        _assert_round_trip(db, loaded)

    def test_query_after_load(self):
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            db.to_file(path)
            loaded = TopicDatabase.from_file(path)
        _assert_query_works(loaded)

    def test_output_is_single_file(self):
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            db.to_file(path)
            assert path.is_file(), "zip backend should produce a single file"

    def test_wrong_serial_version_raises(self):
        import zipfile, json
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.zip"
            db.to_file(path)

            # Corrupt the serial version inside the zip
            corrupted = Path(tmp) / "corrupted.zip"
            with zipfile.ZipFile(path) as zin, \
                 zipfile.ZipFile(corrupted, "w") as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "metadata.json":
                        meta = json.loads(data)
                        meta["serial_version"] = "9.9"
                        data = json.dumps(meta).encode()
                    zout.writestr(item, data)

            with pytest.raises(ValueError, match="serial version"):
                TopicDatabase.from_file(corrupted)

    def from_serialized_toponymy(self):
        """ We should be able to init a TopicDB from Toponymy """
        topicdb = TopicDatabase.from_file("20ng-topicdb.tm.zip")
        q1=topicdb.q.neighbours("Recent advancements in space exploration").neighbours().theme().metadata()
        assert q1.name.values[0] == "sci.space"
        q2=topicdb.q.samples([0,7,8,33,18132]).theme().metadata()
        assert q2.name.values[0] == "rec.sport"

# =============================================================================
# Lance backend tests
# =============================================================================

lance = pytest.importorskip("lance", reason="lance not installed; skipping Lance tests")


class TestLanceBackend:

    def test_round_trip_with_reduced(self):
        db = _make_topic_database(with_reduced=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)
            loaded = TopicDatabase.from_lance(path)
        _assert_round_trip(db, loaded)

    def test_round_trip_without_reduced(self):
        db = _make_topic_database(with_reduced=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)
            loaded = TopicDatabase.from_lance(path)
        _assert_round_trip(db, loaded)

    def test_query_after_load(self):
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)
            loaded = TopicDatabase.from_lance(path)
        _assert_query_works(loaded)

    def test_output_is_directory(self):
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)
            assert path.is_dir(), "Lance backend should produce a directory"
            subtables = {p.name for p in path.iterdir()}
            assert {"documents.lance", "topics.lance", "clusters.lance", "config.lance"} \
                <= subtables, f"Missing expected Lance tables, found: {subtables}"

    def test_existing_path_raises(self):
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)
            with pytest.raises(FileExistsError):
                db.to_lance(path)

    def test_wrong_serial_version_raises(self):
        import lance
        import pyarrow as pa
        db = _make_topic_database()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.lancedb"
            db.to_lance(path)

            # Overwrite config.lance with a bad serial version
            config_path = str(path / "config.lance")
            config = lance.dataset(config_path).to_table().to_pydict()
            config["serial_version"] = ["9.9"]
            lance.write_dataset(
                pa.table(config),
                config_path,
                mode="overwrite",
            )

            with pytest.raises(ValueError, match="serial version"):
                TopicDatabase.from_lance(path)