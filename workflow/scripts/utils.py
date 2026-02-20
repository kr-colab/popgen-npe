"""
Utility functions for the workflow.
"""
import os
import torch

def get_least_busy_gpu() -> int:
    """
    Get the least busy GPU.
    """
    if not torch.cuda.is_available():
        return None
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return free_memory.index(max(free_memory))


def parse_paths(config: dict) -> dict:
    """
    Given config dict, return the paths to embedding network, normalizing flow,
    and zarr containing data.
    """
    project_dir = config["project_dir"]
    random_seed = int(config["random_seed"])
    n_train = int(config["n_train"])
    train_separately = bool(config["train_embedding_net_separately"])
    simulator_name = config["simulator"]["class_name"]
    processor_name = config["processor"]["class_name"]
    embedding_name = config["embedding_network"]["class_name"]
    train_path = (
        f"{project_dir}/"
        f"{simulator_name}-"
        f"{processor_name}-"
        f"{embedding_name}-"
        f"{random_seed}-"
        f"{n_train}"
    )
    if train_separately:
        train_path += "-sep/"
        embedding_net = os.path.join(train_path, "pretrain_embedding_network")
        normalizing_flow = os.path.join(train_path, "pretrain_normalizing_flow")
    else:
        train_path += "-e2e/"
        embedding_net = os.path.join(train_path, "embedding_network")
        normalizing_flow = os.path.join(train_path, "normalizing_flow")
    zarr_data = os.path.join(train_path, "tensors", "zarr")
    return {
        "embedding_net": embedding_net, 
        "normalizing_flow": normalizing_flow,
        "zarr": zarr_data,
    }



