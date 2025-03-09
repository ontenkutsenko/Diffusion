import h5py
import faiss
from sklearn.decomposition import PCA
import pickle
import numpy as np
import json
from redi.utils import load_jsonl

def create_kb(
        key_margin_steps: int = 10,
        value_margin_steps: int = 10,
        trajectory_path: str = "trajectories.h5",
        kb_path: str = "knowledge_base.h5",
        faiss_index_path: str = "faiss_index.bin",

        prompts_from_path: str = "prompts.jsonl",
        prompts_to_path: str = "new_prompts.jsonl",

        key_compression: str | None = None,
        embedding_dim: int = 512,
        compression_path: str = "compression.pkl"
):  
    if key_compression is None:
        embedding_dim = 1 * 4 * 64 * 64

    prompts_from = load_jsonl(prompts_from_path)
    prompts_to = {}

    with h5py.File(trajectory_path, "r") as trajectory_file:
        traj_names = list(trajectory_file.keys())

        all_keys = []

        # PCA keys compression
        if key_compression == "pca":
            pca = PCA(n_components=embedding_dim, svd_solver="auto")
            for name in traj_names:
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)
                all_keys.append(flat_key)

            all_keys = np.vstack(all_keys)
            pca.fit(all_keys)
            with open(compression_path, "wb") as pca_file:
                pickle.dump(pca, pca_file)

        index = faiss.IndexFlatL2(embedding_dim)

        with h5py.File(kb_path, "a") as kb_file:
            for name in traj_names:
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)

                if key_compression == "pca":
                    flat_key = pca.transform(flat_key)

                index.add(flat_key)
                kb_file.create_dataset(str(index.ntotal-1), 
                                data=trajectory_file[name][-value_margin_steps], 
                                compression="gzip", 
                                compression_opts=4)
                faiss.write_index(index, faiss_index_path)

                prompts_to[str(index.ntotal-1)] = prompts_from[name]

    with open(prompts_to_path, "w") as f:
        json.dump(prompts_to, f, indent=4)
