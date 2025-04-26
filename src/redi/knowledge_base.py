import h5py
import faiss
from sklearn.decomposition import PCA
import pickle
import numpy as np
import json
from redi.utils import load_jsonl
from tqdm import tqdm

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
        key_compression (str | None): The type of compression to use for the keys. 
                                      If None, no compression is used. 
                                      If "pca", PCA compression is used.
        embedding_dim (int): The dimension of the embedding space.
        compression_path (str): The path to save the PCA compression model.
        
    Returns:
        None
    
    """
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
        index = faiss.IndexIDMap(index)

        with h5py.File(kb_path, "a") as kb_file:
            for id, name in tqdm(enumerate(traj_names)):
                flat_key = trajectory_file[name][key_margin_steps].reshape(1, -1)

                if key_compression == "pca":
                    flat_key = pca.transform(flat_key)

                index.add_with_ids(flat_key, np.array([id], dtype=np.int64))
                kb_file.create_dataset(str(id), 
                                data=trajectory_file[name][-value_margin_steps], 
                                compression="gzip", 
                                compression_opts=4)
                faiss.write_index(index, faiss_index_path)

                prompts_to[str(id)] = prompts_from[name]

    with open(prompts_to_path, "w") as f:
        json.dump(prompts_to, f, indent=4)
        
def clean_kb(
        kb_path: str = "knowledge_base.h5",
        faiss_index_path: str = "faiss_index.bin",
        prompts_path: str = "prompts.jsonl",
        compression_path: str = "compression.pkl",
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
    os.remove(compression_path)
    print(f"Removed {kb_path}, {faiss_index_path}, {prompts_path}, {compression_path}")
