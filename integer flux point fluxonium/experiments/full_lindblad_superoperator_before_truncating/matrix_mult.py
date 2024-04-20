
import multiprocessing
chunk_size = 50000
spre_a_path = "spre_a.npz"
spost_ad_path = "spost_ad.npz"
chunks_dir = "chunks"

import os
import scipy as sp

def worker(i, j):
    spre_a_chunk = sp.sparse.load_npz(os.path.join(chunks_dir, f"chunk_{i}.npz"))
    spost_ad_chunk = sp.sparse.load_npz(os.path.join(chunks_dir, f"column_strips_chunk_{j}.npz"))
    result_chunk = spre_a_chunk @ spost_ad_chunk
    sp.sparse.save_npz(os.path.join(chunks_dir, f"result_chunk_{i}_{j}.npz"), result_chunk)
    print(f"done {i}-{j}")

def multiply_and_store():
    num_rows = 102
    num_cols = 102
    pool = multiprocessing.Pool(10)
    
    pool.starmap(worker, [(i, j) for i in range(num_rows)  for j in range(num_cols)])
    
    # Close the pool to free up resources
    pool.close()
    pool.join()

multiply_and_store()
