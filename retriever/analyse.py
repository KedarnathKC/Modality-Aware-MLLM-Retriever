import pytrec_eval
import os
import json
from tqdm import tqdm
from utils.pathUtils import get_model_path
from utils.commonUtils import unhash_qid

dataset_mapping = {
    "0": "VisualNews",
    "9": "MSCOCO",
    "1": "Fashion200K",
}

task_mapping = {
    "0": "Text_2_Image",
    "3": "Image_2_Text",
}


def get_scores(qrels, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.5,10'})
    result = evaluator.evaluate(run)
    metrics = ['ndcg_cut', 'recall']
    cutoffs = [5, 10]
    scores = {f'{metric}_{cutoff}': 0 for metric in metrics for cutoff in cutoffs if
              not (metric == 'ndcg_cut' and cutoff == 5)}
    for key in result:
        for metric in metrics:
            for cutoff in cutoffs:
                if not (metric == 'ndcg_cut' and cutoff == 5):
                    scores[f'{metric}_{cutoff}'] += result[key][f'{metric}_{cutoff}']
    run_length = len(run)
    for score in scores:
        scores[score] *= 100/run_length
        scores[score] = round(scores[score], 5)
    return scores


def pretty_print_save(results_path, results):
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved pytrac evaluations to {results_path}")


def score_results(Args):
    evaluation_path = get_model_path('Evaluation', Args)

    run_qrel_path = evaluation_path + '/run_qrel.json'
    qrels_path = 'datasetUtils/qrels.json'
    tasks_path = 'datasetUtils/task_map.json'
    results_path = evaluation_path + '/results.json'

    for file_paths in [run_qrel_path, qrels_path, tasks_path]:
        if not os.path.exists(file_paths):
            raise Exception(f'File not found: {file_paths}')

    with open(run_qrel_path, 'r') as f:
        run_qrel = json.load(f)

    with open(qrels_path, 'r') as f:
        qrels = json.load(f)

    with open(tasks_path, 'r') as f:
        tasks_map = json.load(f)

    print(f"Running pytrac evaluations.")
    results = {}
    for task in ['0', '3']:
        for dataset_id in ['0', '1', '9']:
            subset_qrels = {}
            subset_run_qrels = {}
            filtered_qid = list(
                map(str, filter(lambda qid: unhash_qid(qid).startswith(f'{dataset_id}:'), tasks_map[task])))
            for qid in tqdm(filtered_qid, desc=f'Task {task} Dataset {dataset_id}'):
                if qid in run_qrel:
                    subset_run_qrels[qid] = run_qrel[qid]
                    subset_qrels[qid] = qrels[qid]
            if len(subset_run_qrels) > 0:
                scores = get_scores(subset_qrels, subset_run_qrels)
                scores["NDCG@10"] = scores.pop("ndcg_cut_10")
                scores["Recall@5"] = scores.pop("recall_5")
                scores["Recall@10"] = scores.pop("recall_10")
                results[task_mapping[task]] = {dataset_mapping[dataset_id]: scores}

    print(f"Completed pytrac evaluations.")
    print(json.dumps(results, indent=2))

    pretty_print_save(results_path, results)
