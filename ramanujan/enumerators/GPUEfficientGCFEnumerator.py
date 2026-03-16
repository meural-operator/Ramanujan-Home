import torch
from time import time
from typing import List

from .EfficientGCFEnumerator import EfficientGCFEnumerator, Match
from ramanujan.constants import g_N_initial_search_terms

class GPUEfficientGCFEnumerator(EfficientGCFEnumerator):
    """
    GPU-accelerated version of EfficientGCFEnumerator using PyTorch.
    Evaluates Generalized Continued Fractions in massive parallel batches.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _first_enumeration(self, verbose: bool) -> List[Match]:
        start = time()
        a_coef_iter = list(self.get_an_iterator())
        b_coef_iter = list(self.get_bn_iterator())
        
        # 1. Build batched tensors for all a_n and b_n
        a_series_list = []
        a_coef_list = []
        for coef in a_coef_iter:
            an = self.create_an_series(coef, g_N_initial_search_terms)
            if 0 not in an[1:]:
                a_series_list.append(an)
                a_coef_list.append(coef)

        b_series_list = []
        b_coef_list = []
        for coef in b_coef_iter:
            bn = self.create_bn_series(coef, g_N_initial_search_terms)
            if 0 not in bn[1:]:
                b_series_list.append(bn)
                b_coef_list.append(coef)

        if not a_series_list or not b_series_list:
            if verbose:
                print("No valid polynomial sequences found.")
            return []

        # Move sequences to Tensor
        a_tensor = torch.tensor(a_series_list, dtype=torch.float64, device=self.device) # (N_a, N_terms)
        b_tensor = torch.tensor(b_series_list, dtype=torch.float64, device=self.device) # (N_b, N_terms)
        
        N_a = a_tensor.shape[0]
        N_b = b_tensor.shape[0]
        N_terms = a_tensor.shape[1]
        
        num_iterations = N_a * N_b
        if verbose:
            print(f"Created final enumerations filters after {time() - start:.2f}s")
            print(f"Batch evaluating {num_iterations} combinations on {self.device}...")
        
        results = []
        key_factor = round(1 / self.threshold)
        
        # Batch chunks to prevent VRAM OOM on massive search spaces
        CHUNK_A = 1000
        CHUNK_B = 10000
        
        for i in range(0, N_a, CHUNK_A):
            a_chunk = a_tensor[i : i + CHUNK_A] 
            chunk_a_size = a_chunk.shape[0]
            
            for j in range(0, N_b, CHUNK_B):
                b_chunk = b_tensor[j : j + CHUNK_B] 
                chunk_b_size = b_chunk.shape[0]
                
                # Cross product using broadcasting and reshape
                a_expanded = a_chunk.unsqueeze(1).expand(chunk_a_size, chunk_b_size, N_terms).reshape(-1, N_terms)
                b_expanded = b_chunk.unsqueeze(0).expand(chunk_a_size, chunk_b_size, N_terms).reshape(-1, N_terms)
                
                bsz = a_expanded.shape[0]
                
                prev_q = torch.zeros(bsz, dtype=torch.float64, device=self.device)
                q = torch.ones(bsz, dtype=torch.float64, device=self.device)
                prev_p = torch.ones(bsz, dtype=torch.float64, device=self.device)
                p = a_expanded[:, 0].clone()
                
                # Batched convergent evaluation
                for k in range(1, N_terms):
                    tmp_q = q.clone()
                    tmp_p = p.clone()
                    q = a_expanded[:, k] * q + b_expanded[:, k] * prev_q
                    p = a_expanded[:, k] * p + b_expanded[:, k] * prev_p
                    prev_q = tmp_q
                    prev_p = tmp_p
                    
                    # Periodic scaling to prevent float64 overflow, taking max magnitude
                    scale = q.abs().clamp_min(1.0)
                    q /= scale
                    p /= scale
                    prev_q /= scale
                    prev_p /= scale

                dist = key_factor * p / q
                dist = torch.nan_to_num(dist, nan=0.0)
                hash_keys = dist.trunc().long()
                
                # Check for hits in CPU memory hash table
                hash_keys_cpu = hash_keys.cpu().numpy()
                
                for idx in range(bsz):
                    key = hash_keys_cpu[idx]
                    if key in self.hash_table:
                        a_idx = i + (idx // chunk_b_size)
                        b_idx = j + (idx % chunk_b_size)
                        results.append(Match(key, a_coef_list[a_idx], b_coef_list[b_idx]))
                        
        if verbose:
            print(f"Created results after {time() - start:.2f}s. Found {len(results)} preliminary matches.")
            
        return results
