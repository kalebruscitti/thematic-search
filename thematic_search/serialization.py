import os
import json
import tempfile
import zipfile
from pathlib import Path
from copy import deepcopy

import scipy.sparse as sp
import pandas as pd
import numpy as np

from .utilities import uid_to_ints, topic_uid


_SERIAL_VERSION = "0.1"


# =============================================================================
# Zip backend
# =============================================================================

def save_topic_database(topicdb, path):
    """
    Save a TopicDatabase to a zip archive.

    Parameters
    ----------
    topicdb : TopicDatabase
    path : str or Path
        Destination path, e.g. "mydb.zip".
    """
    path = Path(path)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "topicdb"
        matrices_dir = root / "cluster_matrices"
        root.mkdir()
        matrices_dir.mkdir()

        # --- DataFrames ---
        topicdb.document_df.to_parquet(root / "document_df.parquet")
        topic_df = deepcopy(topicdb.topic_df)
        topic_df["uid"] = [topic_uid(topicdb.soft_cluster_tree.idx_to_loc[i]) for i in topic_df.index]
        topic_df.to_parquet(root / "topic_df.parquet")

        # --- Vectors ---
        np.save(root / "embedding_vectors.npy", topicdb.embedding_vectors)
        has_reduced = False
        if topicdb.reduced_vectors is not None:
            np.save(root / "reduced_vectors.npy", topicdb.reduced_vectors)  # bugfix: was saving embedding_vectors to cwd
            has_reduced = True

        # --- Sparse cluster matrices ---
        for i, matrix in enumerate(topicdb.soft_cluster_tree.layers):
            sp.save_npz(matrices_dir / f"layer_{i}.npz", matrix)

        # --- Cluster tree topology ---
        idx_to_uid = {
            i:topic_uid(topicdb.soft_cluster_tree.idx_to_loc[i])
            for i in range(topicdb.soft_cluster_tree.n_topics)
        }
        uid_tree = {
            idx_to_uid[k]:[idx_to_uid[c] for c in children] 
            for k, children in topicdb.soft_cluster_tree.children_map.items()
        }
        with open(root / "cluster_tree.json", "w") as f:
            json.dump(uid_tree, f)

        # --- Metadata ---
        metadata = {
            "serial_version": _SERIAL_VERSION,
            "n_layers": len(topicdb.soft_cluster_tree.layers),
            "has_reduced": has_reduced,
        }
        with open(root / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # --- Bundle into zip ---
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in root.rglob("*"):
                z.write(file, file.relative_to(root))


def load_topic_database(path, SoftClusterTree, TopicDatabase):
    """
    Load a TopicDatabase from a zip archive produced by save_topic_database().

    Parameters
    ----------
    path : str or Path
    SoftClusterTree : class
    TopicDatabase : class

    Returns
    -------
    TopicDatabase
        The embedding_model attribute will be None; re-attach it after loading.
    """
    path = Path(path)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        with zipfile.ZipFile(path) as z:
            z.extractall(root)

        with open(root / "metadata.json") as f:
            metadata = json.load(f)

        serial_version = metadata["serial_version"]
        if serial_version != _SERIAL_VERSION:
            raise ValueError(
                f"The file's serial version ({serial_version}) does not match "
                f"the current version ({_SERIAL_VERSION})."
            )

        has_reduced = metadata["has_reduced"]

        # --- DataFrames ---
        document_df = pd.read_parquet(root / "document_df.parquet")
        topic_df = pd.read_parquet(root / "topic_df.parquet")
        topic_df['index'] = topic_df['uid'].map(uid_to_ints)
        topic_df.drop(columns='uid', inplace=True)
        topic_df.set_index('index', drop=True, inplace=True)

        # --- Vectors ---
        embedding_vectors = np.load(root / "embedding_vectors.npy")
        reduced_vectors = None
        if has_reduced:
            reduced_vectors = np.load(root / "reduced_vectors.npy")  # bugfix: was loading from cwd

        # --- Sparse cluster matrices ---
        matrices_dir = root / "cluster_matrices"
        layer_files = sorted(
            matrices_dir.glob("layer_*.npz"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        matrices = [sp.load_npz(f) for f in layer_files]

        # --- Cluster tree topology ---
        with open(root / "cluster_tree.json") as f:
            raw_tree = json.load(f)

        cluster_tree = {
            uid_to_ints(k): [uid_to_ints(child) for child in v]
            for k, v in raw_tree.items()
        }

        # --- Reconstruct ---
        soft_cluster_tree = SoftClusterTree(
            cluster_matrices=matrices,
            cluster_tree=cluster_tree,
        )
        return TopicDatabase(
            soft_cluster_tree=soft_cluster_tree,
            embedding_vectors=embedding_vectors,
            reduced_vectors=reduced_vectors,
            document_df=document_df,
            topic_df=topic_df,
            embedding_model=None,
        )


# =============================================================================
# Lance backend
# =============================================================================

def save_topic_database_lance(topicdb, path):
    """
    Save a TopicDatabase to a directory of Lance tables.

    Layout
    ------
    <path>/
      documents.lance   one row per document; columns = document_df columns
                        + 'embedding' (fixed-size vector)
                        + 'reduced_embedding' (fixed-size vector, if present)
      topics.lance      one row per topic; columns = topic_df columns + 'uid'
      clusters.lance    COO rows: layer (int16), row_idx (int32),
                        col_idx (int16), value (int32 — Lance has no uint8)
      config.lance      single row: serial_version, n_layers,
                        has_reduced, cluster_tree (JSON string)

    Parameters
    ----------
    topicdb : TopicDatabase
    path : str or Path
        Destination directory path, e.g. "mydb.lancedb".
        The directory must not already exist.
    """
    import lance
    import pyarrow as pa

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Remove it first or choose a different path."
        )
    path.mkdir(parents=True)

    # --- documents.lance ---
    doc_dict = {col: topicdb.document_df[col].tolist()
                for col in topicdb.document_df.columns}

    emb_dim = topicdb.embedding_vectors.shape[1]
    doc_dict["embedding"] = topicdb.embedding_vectors.tolist()
    schema_fields = [
        *[pa.field(col, _pandas_col_to_arrow(topicdb.document_df[col]))
          for col in topicdb.document_df.columns],
        pa.field("embedding", pa.list_(pa.float32(), emb_dim)),
    ]

    has_reduced = topicdb.reduced_vectors is not None
    if has_reduced:
        red_dim = topicdb.reduced_vectors.shape[1]
        doc_dict["reduced_embedding"] = topicdb.reduced_vectors.tolist()
        schema_fields.append(
            pa.field("reduced_embedding", pa.list_(pa.float32(), red_dim))
        )

    doc_schema = pa.schema(schema_fields)
    doc_table = pa.table(doc_dict, schema=doc_schema)
    lance.write_dataset(doc_table, str(path / "documents.lance"))

    # --- topics.lance ---
    topic_df = deepcopy(topicdb.topic_df)
    topic_df["uid"] = [topic_uid(topicdb.soft_cluster_tree.idx_to_loc[i]) for i in topic_df.index]
    topic_dict = {col: topic_df[col].tolist() for col in topic_df.columns}
    topic_table = pa.table(topic_dict)
    lance.write_dataset(topic_table, str(path / "topics.lance"))

    # --- clusters.lance ---
    # Flatten all sparse layers to COO and tag each row with its layer index.
    # Lance has no uint8 column type, so values are stored as int32.
    coo_layers, coo_rows, coo_cols, coo_vals = [], [], [], []
    for layer_idx, matrix in enumerate(topicdb.soft_cluster_tree.layers):
        coo = matrix.tocoo()
        n = len(coo.data)
        coo_layers.append(np.full(n, layer_idx, dtype=np.int16))
        coo_rows.append(coo.row.astype(np.int32))
        coo_cols.append(coo.col.astype(np.int16))
        coo_vals.append(coo.data.astype(np.int32))

    clusters_table = pa.table({
        "layer":   pa.array(np.concatenate(coo_layers), type=pa.int16()),
        "row_idx": pa.array(np.concatenate(coo_rows),   type=pa.int32()),
        "col_idx": pa.array(np.concatenate(coo_cols),   type=pa.int16()),
        "value":   pa.array(np.concatenate(coo_vals),   type=pa.int32()),
    })
    lance.write_dataset(clusters_table, str(path / "clusters.lance"))

    idx_to_uid = {
        i:topic_uid(topicdb.soft_cluster_tree.idx_to_loc[i])
        for i in range(topicdb.soft_cluster_tree.n_topics)
    }
    uid_tree = {
        idx_to_uid[k]:[idx_to_uid[c] for c in children] 
        for k, children in topicdb.soft_cluster_tree.children_map.items()
    }
    # --- config.lance ---
    config_table = pa.table({
        "serial_version": pa.array([_SERIAL_VERSION],  type=pa.string()),
        "n_layers":       pa.array([len(topicdb.soft_cluster_tree.layers)], type=pa.int32()),
        "has_reduced":    pa.array([has_reduced],       type=pa.bool_()),
        "cluster_tree":   pa.array(
            [json.dumps(uid_tree)],
            type=pa.string(),
        ),
    })
    lance.write_dataset(config_table, str(path / "config.lance"))


def load_topic_database_lance(path, SoftClusterTree, TopicDatabase):
    """
    Load a TopicDatabase from a Lance directory produced by
    save_topic_database_lance().

    Parameters
    ----------
    path : str or Path
    SoftClusterTree : class
    TopicDatabase : class

    Returns
    -------
    TopicDatabase
        The embedding_model attribute will be None; re-attach it after loading.
    """
    import lance

    path = Path(path)

    # --- config ---
    config = lance.dataset(str(path / "config.lance")).to_table().to_pydict()
    serial_version = config["serial_version"][0]
    if serial_version != _SERIAL_VERSION:
        raise ValueError(
            f"The file's serial version ({serial_version}) does not match "
            f"the current version ({_SERIAL_VERSION})."
        )
    n_layers = config["n_layers"][0]
    has_reduced = config["has_reduced"][0]
    raw_tree = json.loads(config["cluster_tree"][0])

    cluster_tree = {
        uid_to_ints(k): [uid_to_ints(child) for child in v]
        for k, v in raw_tree.items()
    }

    # --- documents ---
    doc_table = lance.dataset(str(path / "documents.lance")).to_table().to_pydict()
    embedding_vectors = np.array(doc_table.pop("embedding"), dtype=np.float32)
    reduced_vectors = None
    if has_reduced:
        reduced_vectors = np.array(doc_table.pop("reduced_embedding"), dtype=np.float32)
    document_df = pd.DataFrame(doc_table)

    # --- topics ---
    topic_dict = lance.dataset(str(path / "topics.lance")).to_table().to_pydict()
    topic_df = pd.DataFrame(topic_dict)
    topic_df['index'] = topic_df['uid'].map(uid_to_ints)
    topic_df.drop(columns='uid', inplace=True)
    topic_df.set_index('index', drop=True, inplace=True)
    
    # --- clusters: reconstruct one csr_matrix per layer ---
    coo_dict = lance.dataset(str(path / "clusters.lance")).to_table().to_pydict()
    layers_arr   = np.array(coo_dict["layer"],   dtype=np.int16)
    rows_arr     = np.array(coo_dict["row_idx"],  dtype=np.int32)
    cols_arr     = np.array(coo_dict["col_idx"],  dtype=np.int16)
    vals_arr     = np.array(coo_dict["value"],    dtype=np.uint8)  # safe: values are 0-255
    n_docs       = len(document_df)

    matrices = []
    for layer_idx in range(n_layers):
        mask = layers_arr == layer_idx
        n_cols = int(cols_arr[mask].max()) + 1 if mask.any() else 0
        csr = sp.coo_matrix(
            (vals_arr[mask], (rows_arr[mask], cols_arr[mask])),
            shape=(n_docs, n_cols),
            dtype=np.uint8,
        ).tocsr()
        matrices.append(csr)

    # --- Reconstruct ---
    soft_cluster_tree = SoftClusterTree(
        cluster_matrices=matrices,
        cluster_tree=cluster_tree,
    )
    return TopicDatabase(
        soft_cluster_tree=soft_cluster_tree,
        embedding_vectors=embedding_vectors,
        reduced_vectors=reduced_vectors,
        document_df=document_df,
        topic_df=topic_df,
        embedding_model=None,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _pandas_col_to_arrow(series: pd.Series):
    """Infer a PyArrow type from a pandas Series for schema construction."""
    import pyarrow as pa
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return pa.int64()
    if pd.api.types.is_float_dtype(dtype):
        return pa.float64()
    if pd.api.types.is_bool_dtype(dtype):
        return pa.bool_()
    return pa.string()