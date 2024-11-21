import os
import numpy as np
import faiss
import gc
from utils.pathUtils import get_model_path


def create_index(Args):

    evaluation_path = get_model_path('Evaluation', Args)

    index_path = evaluation_path + f'/{Args.Common.DataSet.Name}.index'

    if not os.path.exists(index_path):
        print(f'Creating index.')

        candidate_embedding_path = evaluation_path + '/embeddings/'

        candidate_embeddings = np.load(candidate_embedding_path+'candidate_embed.npy')
        candidate_ids = np.load(candidate_embedding_path+'candidate_ids.npy')

        N, D = candidate_embeddings.shape

        cpu_index = faiss.index_factory(D, f"IDMap,Flat", faiss.METRIC_INNER_PRODUCT)

        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, ngpu=ngpus)
            index_cpu = faiss.index_gpu_to_cpu(index_gpu)
            del index_gpu
        else:
            faiss.omp_set_num_threads(1)
            index_cpu = cpu_index

        index_cpu.add_with_ids(candidate_embeddings, candidate_ids)
        assert N == index_cpu.ntotal, "Issue with indexing"
        faiss.write_index(index_cpu, index_path)
        print(f"Successfully indexed {index_cpu.ntotal} documents")
        print(f"Index saved to: {index_path}")

        del candidate_embeddings
        del candidate_ids
        del cpu_index
        del index_cpu

        gc.collect()
    else:
        print(f"Index already exists: {index_path}")





