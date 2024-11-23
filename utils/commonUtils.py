import yaml
import re

IGNORE_KEYS = []


def represent_bool(self, data):
    if data:
        return self.represent_scalar('tag:yaml.org,2002:bool', 'True')
    return self.represent_scalar('tag:yaml.org,2002:bool', 'False')


yaml.add_representer(bool, represent_bool)

DATASET_CAN_NUM_UPPER_BOUND = 10000000  # Maximum number of candidates per dataset
DATASET_QUERY_NUM_UPPER_BOUND = 500000  # Maximum number of queries per dataset


def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"


def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id


def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id


def save_config(output_path, Args):
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml_data = yaml.dump(Args, sort_keys=False,  default_flow_style=False)
        cleaned_yaml = re.sub(r'(\s*!!.*$)|(^!!.*$)', '', yaml_data, flags=re.MULTILINE)
        f.write(cleaned_yaml)


def start_training(Args, trainer, loadCheckpoint, model, output_path, model_output, testing_data):

    save_config(output_path + '/training/config.yaml', Args)

    if loadCheckpoint:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS, resume_from_checkpoint=model)
    else:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS)

    trainer.save_model(output_dir=output_path + model_output)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    start_evaluation(trainer, testing_data)


def start_evaluation(trainer, testing_data):
    metrics = trainer.evaluate(testing_data, ignore_keys=IGNORE_KEYS)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def start_prediction(trainer, testing_data):

    outputs, _, metrics = trainer.predict(testing_data, ignore_keys=IGNORE_KEYS)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return outputs[1:-1]

