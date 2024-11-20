from transformers import TrainingArguments, CLIPModel
from clipTrainer import ClipTrainer
from utils.pathUtils import prepare_output_path, get_checkpoint_path, get_model_path
from utils.commonUtils import start_prediction, save_config
from datasetUtils.prepare_dataset import Builder
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def get_evaluation_args(output_path, hyperparameters):
    return TrainingArguments(
        output_dir=output_path + 'evaluation/',
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        remove_unused_columns=False,
        do_predict=True,
        push_to_hub=False,
        seed=42,
        label_names=['modalities'],
    )


def save_embeddings(output_path, embeddings, ids):
    if "query" in output_path:
        embed = "query"
    else:
        embed = "candidate"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"\n{embed} embedding size: {embeddings.shape}, ids: {ids.shape}")

    np.save(output_path+f"{embed}_embed.npy", embeddings)
    np.save(output_path+f"{embed}_ids.npy", ids)

    print(f"\n{embed} embeddings saved to {output_path}")


def evaluate(Args):

    builder = Builder(Args, onlyPrediction=True)

    output_path = prepare_output_path('Evaluation', Args)

    eval_args = get_evaluation_args(output_path, Args.Evaluate.Hyperparameters)

    if Args.Evaluate.Model.LoadCheckPoint:
        model_path = get_checkpoint_path('FineTuned', Args)
    elif Args.Evaluate.Model.UseLocal:
        model_path = get_model_path('FineTuned', Args)
    else:
        model_path = Args.Evaluate.Model.Name

    model = CLIPModel.from_pretrained(model_path, cache_dir=Args.Evaluate.Model.CachePath, attn_implementation='flash_attention_2')

    testing_data = builder.get_eval_dataset()
    compute_metrics = builder.get_compute()

    evaluation_trainer = ClipTrainer(
        model=model,
        configArgs=Args,
        onlyPrediction=True,
        args=eval_args,
        compute_metrics=compute_metrics,
        data_collator=builder.get_collate_fn()
    )

    save_config(output_path + 'evaluation/config.yaml', Args)

    query_embeddings, candidate_embeddings, query_ids, candidate_ids = start_prediction(evaluation_trainer, testing_data)

    save_embeddings(output_path + 'query_embeddings/', query_embeddings, query_ids)

    save_embeddings(output_path + 'candidate_embeddings/', candidate_embeddings, candidate_ids)





