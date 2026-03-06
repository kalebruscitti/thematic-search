import os
import json
import tempfile
import zipfile
from pathlib import Path
from copy import deepcopy

import scipy.sparse as sp
import pandas as pd
import numpy as np
from utilities import (
    uid_to_ints
)

def save_topic_database(topicdb, path):
    path = Path(path)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "topicdb"
        matrices_dir = root / "cluster_matrices"

        root.mkdir()
        matrices_dir.mkdir()

        topicdb.document_df.to_parquet(root / "document_df.parquet")
        topic_df = deepcopy(topicdb.topic_df)
        topic_df['uid'] = topic_df.index
        topic_df.to_parquet(root / "topic_df.parquet")

        np.save(root / "embedding_vectors.npy", topicdb.embedding_vectors)
        has_reduced = 'N'
        if topicdb.reduced_vectors is not None:
            np.save("reduced_vectors.npy", topicdb.embedding_vectors)
            has_reduced = 'Y'

        for i, matrix in enumerate(topicdb.soft_cluster_tree.layers):
            #np.save(matrices_dir / f"layer_{i}.npy", matrix)
            sp.save_npz(matrices_dir / f"layer_{i}.npz", matrix)

        with open(root / "cluster_tree.json", "w") as f:
            json.dump(topicdb.soft_cluster_tree.children_map, f)

        metadata = {
            "serial_version": "0.1",
            "n_layers": len(topicdb.soft_cluster_tree.layers),
            "has_reduced": has_reduced,
        }

        with open(root / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in root.rglob("*"):
                z.write(file, file.relative_to(root))
                
def load_topic_database(path, SoftClusterTree, TopicDatabase):
    path = Path(path)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        with zipfile.ZipFile(path) as z:
            z.extractall(root)
        with open(root / "metadata.json") as f:
            metadata = json.load(f)
        serial_version = metadata['serial_version']
        has_reduced = (metadata['has_reduced']=='Y')
        if serial_version != "0.1":
            raise ValueError(
                "The file's serial version ({serial_version}) does not equal the current version (0.1)."
            ) 

        document_df = pd.read_parquet(root / "document_df.parquet")
        topic_df = pd.read_parquet(root / "topic_df.parquet")

        embedding_vectors = np.load(root / "embedding_vectors.npy")
        if has_reduced:
            reduced_vectors = np.load("reduced_vectors.npy")


        matrices = []
        matrices_dir = root / "cluster_matrices"

        layer_files = sorted(
            matrices_dir.glob("layer_*.npz"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        for f in layer_files:
            #matrices.append(np.load(f))
            matrices.append(sp.load_npz(f))

        with open(root / "cluster_tree.json") as f:
            raw_tree = json.load(f)
        
        cluster_tree = {}
        for k, v in raw_tree.items():
            children = [uid_to_ints(child) for child in v]
            cluster_tree[uid_to_ints(k)] = children

        soft_cluster_tree = SoftClusterTree(
            cluster_matrices=matrices,
            cluster_tree=cluster_tree,
        )

        topicdb = TopicDatabase(
            soft_cluster_tree=soft_cluster_tree,
            embedding_vectors=embedding_vectors,
            document_df=document_df,
            topic_df=topic_df,
            embedding_model=None,
        )

        return topicdb