import json
import math
import pickle

import faiss
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from retrieval.utils import load_jsonl


def create_kb(
    key_margin_steps: int = 10,
    value_margin_steps: int = 10,
    trajectory_path: str = "trajectories.h5",
    kb_path: str = "knowledge_base.h5",
    faiss_index_path: str = "faiss_index.bin",
    prompts_from_path: str = "prompts.jsonl",
    prompts_to_path: str = "new_prompts.jsonl",
    use_pca: bool = False,
    embedding_dim: int = 512,
    pca_path: str = "compression.pkl",
    use_pq: bool = False,
    n_pq_centroids: int = 256,
    n_pq_subvectors: int = 16,
) -> None:
    """
    Create a knowledge base from a trajectory file and a prompt file.

    Args:
        key_margin_steps (int): The number of steps from the begginging for the key.
        value_margin_steps (int): The number of steps to use for the value from the end. I.e. 20 steps from the end.
        trajectory_path (str): The path to the trajectory file.
        kb_path (str): The path to the knowledge base file.
        faiss_index_path (str): The path to the faiss index file.
        prompts_from_path (str): The path to the prompts file.
        prompts_to_path (str): The path to the new prompts file.
        use_pca (bool): Whether to use PCA for compression.
        embedding_dim (int): The dimension of the embedding space.
        pca_path (str): The path to save the PCA compression model.

    Returns:
        None

    """
    if not use_pca:
        embedding_dim = 1 * 4 * 64 * 64

    prompts_from = load_jsonl(prompts_from_path)
    prompts_to = {}

    with h5py.File(trajectory_path, "r") as trajectory_file:
        traj_names = list(trajectory_file.keys())

        # PCA keys compression
        if use_pca:
            all_keys = []
            pca = PCA(n_components=embedding_dim, svd_solver="auto")
            for name in traj_names:
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)
                all_keys.append(flat_key)

            all_keys = np.vstack(all_keys)
            print("Fitting PCA...")
            pca.fit(all_keys)
            with open(pca_path, "wb") as pca_file:
                pickle.dump(pca, pca_file)

        # PQ keys compression
        if use_pq:
            n_bits = math.log2(n_pq_centroids)
            if n_bits % 1 != 0:
                raise ValueError("n_pq_centroids must be a power of 2.")
            index = faiss.IndexPQ(embedding_dim, n_pq_subvectors, int(n_bits))

            all_keys = []
            for name in traj_names:
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)
                if use_pca:
                    flat_key = pca.transform(flat_key)
                all_keys.append(flat_key)

            all_keys = np.vstack(all_keys)
            print("Training PQ index...")
            index.train(all_keys)

        else:
            index = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIDMap(index)

        with h5py.File(kb_path, "a") as kb_file:
            for id, name in tqdm(enumerate(traj_names)):
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)
                if use_pca:
                    flat_key = pca.transform(flat_key)

                index.add_with_ids(flat_key, np.array([id], dtype=np.int64))
                kb_file.create_dataset(
                    str(id),
                    data=trajectory_file[name][-value_margin_steps],
                    compression="gzip",
                    compression_opts=4,
                )

                prompts_to[str(id)] = prompts_from[name]
            faiss.write_index(index, faiss_index_path)

    with open(prompts_to_path, "w") as f:
        json.dump(prompts_to, f, indent=4)


def clean_kb(
    kb_path: str = "knowledge_base.h5",
    faiss_index_path: str = "faiss_index.bin",
    prompts_path: str = "prompts.jsonl",
    compression_path: str | None = "compression.pkl",
) -> None:
    """
    Clean the knowledge base by removing the trajectory file and the faiss index file.

    Args:
        kb_path (str): The path to the knowledge base file.
        faiss_index_path (str): The path to the faiss index file.
        prompts_path (str): The path to the prompts file.
        compression_path (str): The path to the PCA compression model.

    Returns:
        None
    """
    import os

    os.remove(kb_path)
    os.remove(faiss_index_path)
    os.remove(prompts_path)
    if compression_path is not None:
        os.remove(compression_path)
    print(f"Removed {kb_path}, {faiss_index_path}, {prompts_path}, {compression_path}")
