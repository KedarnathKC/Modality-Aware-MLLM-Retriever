import os
import faiss
import json
import numpy as np
from tqdm import tqdm
from utils.pathUtils import get_model_path


def run_retrieval(Args):

    evaluation_path = get_model_path('Evaluation', Args)

    run_qrel_path = evaluation_path + '/run_qrel.json'

    if not os.path.exists(run_qrel_path):

        index_path = evaluation_path + f'/{Args.Common.DataSet.Name}.index'

        if not os.path.exists(index_path):
            raise Exception(f"Missing index file {index_path}")

        query_embedding_path = evaluation_path + '/embeddings/'

        index_cpu = faiss.read_index(index_path)

        try:
            query_embeddings = np.load(query_embedding_path + 'query_embed.npy').astype("float32")
            query_ids = np.load(query_embedding_path + 'query_ids.npy')
        except Exception:
            raise Exception(f"Missing embedding files {query_embedding_path}")

        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            indexer = faiss.index_cpu_to_all_gpus(index_cpu, ngpu=ngpus)
        else:
            faiss.omp_set_num_threads(1)
            indexer = index_cpu

        all_distances = []
        all_indices = []

        batch_size = Args.Retrieval.BatchSize
        top_n = Args.Retrieval.Top
        # Process in batches
        for i in tqdm(range(0, len(query_embeddings), batch_size), desc="Batched Index Search"):
            batch = query_embeddings[i: i + batch_size]
            distances, indices = indexer.search(batch, top_n)
            all_distances.append(distances)
            all_indices.append(indices)

        # Stack results for distances and indices
        final_distances = np.vstack(all_distances)
        final_indices = np.vstack(all_indices)

        print(f"Completed search. Saving qrel run file.")

        save_run_qrel(run_qrel_path, query_ids, final_distances, final_indices)


def save_run_qrel(run_qrel_path, query_ids, retrieved_cand_dist, retrieved_indices):

    run_qrel = {}
    for idx, (distances, indices) in tqdm(enumerate(zip(retrieved_cand_dist, retrieved_indices)), desc="Scoring"):
        qid_map = {}
        for rank, (doc_id, score) in enumerate(zip(indices, distances), start=1):
            qid_map[doc_id.item()] = 1 / rank
        run_qrel[query_ids[idx].item()] = qid_map

    with open(run_qrel_path, "w") as f:
       json.dump(run_qrel, f, indent=2)

    print(f"Run file saved to {run_qrel_path}")



