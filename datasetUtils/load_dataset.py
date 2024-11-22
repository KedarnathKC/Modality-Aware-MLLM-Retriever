import yaml
import json
from datasets import load_dataset, concatenate_datasets, Features, Value
from utils.argUtils import CustomObject, get_yaml_loader


domain_mapping = {
    "visualnews": 0,
    "fashion200k": 1,
    "mscoco": 9,
}


def get_training_data(split_perc='', domains=None, show_details=False):

    print(f"Loading training data for {domains}")

    data_files_train = {
        'train': ['query/train/' + f'*{file}*' + '.jsonl' for file in domains],
    }

    ds_train = load_dataset("TIGER-Lab/M-BEIR",
                            cache_dir='dataset/train',
                            data_files=data_files_train,
                            name='query', split=f'train')

    if show_details:
        print("Training data:")
        print(ds_train)

    # ds_train_part0 = ds_train.filter(lambda x: x['qid'].startswith("0:")).select(range(10))
    # ds_train_part1 = ds_train.filter(lambda x: x['qid'].startswith("1:")).select(range(10))
    # ds_train_part2 = ds_train.filter(lambda x: x['qid'].startswith("9:")).select(range(10))
    # ds_train = concatenate_datasets([ds_train_part0, ds_train_part1, ds_train_part2])

    ds_train = ds_train.shuffle(seed=42)

    if split_perc != '':
        split_perc = int(split_perc)
        ds_train = ds_train.select(range(split_perc))

    return ds_train


def get_validation_data(split_perc='', domains=None, show_details=False):
    ds_validate_tasks = []
    if split_perc != '': split_perc = int(split_perc) // 2

    print(f"Loading validation data for {domains}")

    for task_n in ['task0*', 'task3*']:
        data_files_validate_task_n = {
            'val': ['query/val/' + f'*{file}*' + task_n + '.jsonl' for file in domains]
        }

        ds_validate_tasks.append(load_dataset("TIGER-Lab/M-BEIR",
                                              cache_dir='dataset/val',
                                              data_files=data_files_validate_task_n,
                                              name='query', split=f'val'))
    ds_validate = concatenate_datasets(ds_validate_tasks)

    if show_details:
        print("Validation data:")
        print(ds_validate)

    # ds_validate_part0 = ds_validate.filter(lambda x: len(x['pos_cand_list']) > 3).select(range(6))
    # ds_validate_part1 = ds_validate.filter(lambda x: len(x['pos_cand_list']) == 3).select(range(9))
    # ds_validate_part2 = ds_validate.filter(lambda x: len(x['pos_cand_list']) == 2).select(range(4))
    # ds_validate_part3 = ds_validate.filter(lambda x: len(x['pos_cand_list']) == 1).select(range(5))

    # ds_validate_part0 = ds_validate.filter(lambda x: x['qid'].startswith("0:") and x['query_modality'] == 'image').select(range(5))
    # ds_validate_part1 = ds_validate.filter(lambda x: x['qid'].startswith("0:") and x['query_modality'] == 'text').select(range(5))
    # ds_validate_part2 = ds_validate.filter(lambda x: x['qid'].startswith("1:") and x['query_modality'] == 'image').select(range(5))
    # ds_validate_part3 = ds_validate.filter(lambda x: x['qid'].startswith("9:") and x['query_modality'] == 'text').select(range(5))
    # ds_validate = concatenate_datasets([ds_validate_part0, ds_validate_part1, ds_validate_part2, ds_validate_part3])

    ds_validate = ds_validate.shuffle(seed=42)

    if split_perc != '':
        split_perc = int(split_perc)
        ds_validate = ds_validate.select(range(split_perc))

    return ds_validate


def get_candidate_dataset(split_perc='', domains=None, show_details=False):
    ds_candidate_tasks = []

    features = Features({
        "did": Value('string'),
        'txt': Value('string'),
        'img_path': Value('string'),
        'modality': Value('string'),
        'src_content': Value('string'),
    })
    if split_perc != '': split_perc = int(split_perc) // 2

    for split in ['*val*', '*train*']:

        ds_candidate_tasks.append(load_dataset("TIGER-Lab/M-BEIR",
                                               cache_dir='dataset/cand',
                                               features=features,
                                               data_files={'cand_pool': [f'cand_pool/global/{split}.jsonl']},
                                               name='cand_pool', split=f'cand_pool[:{split_perc}]'))

    ds_candidate = concatenate_datasets(ds_candidate_tasks)

    if domains:
        dataset_ids = [str(domain_mapping[domain]) for domain in domains]
        print(f"Loading candidate data for {domains}")
    else:
        dataset_ids = ['0', '1', '9']

    ds_candidate = ds_candidate.filter(lambda candidate: candidate['did'].split(':')[0] in dataset_ids)

    if show_details:
        print("Candidate data:")
        print(ds_candidate)

    return ds_candidate


def get_dataset(train_perc='', valid_perc='', cand_perc='', domains=None, show_details=False):

    ds_train = get_training_data(train_perc, domains, show_details)

    ds_validate = get_validation_data(valid_perc, domains, show_details)

    ds_candidate = get_candidate_dataset(cand_perc, domains, show_details)

    return ds_train, ds_validate, ds_candidate


def validate(ds_train, ds_validate, ds_candidate):

    train_cand_dids = []
    for pos_cand in ds_train['pos_cand_list']:
        train_cand_dids += pos_cand

    print(f"Total Training candidates: {len(train_cand_dids)}")

    val_cand_dids = []
    for pos_cand in ds_validate['pos_cand_list']:
        val_cand_dids += pos_cand

    print(f"Total Validation candidates: {len(val_cand_dids)}")

    candidate_dids = ds_candidate['did']

    diff = set(train_cand_dids).difference(set(candidate_dids))
    print(f"Missing training candidates: {len(diff)}")

    diff = set(val_cand_dids).difference(set(candidate_dids))
    print(f"Missing validation candidates: {len(diff)}")

    print(f"Confirm candidate has all dids :{set(candidate_dids) == set(train_cand_dids).union(set(val_cand_dids))}")


if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    ds_train, ds_validate, ds_candidate = get_dataset(domains=Args.Common.DataSet.FilterDomains, show_details=True)

    validate(ds_train, ds_validate, ds_candidate)