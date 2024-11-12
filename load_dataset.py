from datasets import load_dataset, load_dataset_builder, concatenate_datasets, Features, Value

files = ['*visualnews*','*fashion200k*','*mscoco*']


def get_training_data():

    data_files_train = {
        'train': ['query/train/' + file + '.jsonl' for file in files],
    }

    ds_train = load_dataset("TIGER-Lab/M-BEIR",
                            cache_dir='dataset/train',
                            data_files=data_files_train,
                            name='query', split='train')

    print("Training data:")
    print(ds_train)

    return ds_train


def get_validation_data():
    ds_validate_tasks = []
    for task_n in ['task0*', 'task3*']:
        data_files_validate_task_n = {
            'val': ['query/val/' + file + task_n + '.jsonl' for file in files]
        }

        ds_validate_tasks.append(load_dataset("TIGER-Lab/M-BEIR",
                                              cache_dir='dataset/val',
                                              data_files=data_files_validate_task_n,
                                              name='query', split='val'))
    ds_validate = concatenate_datasets(ds_validate_tasks)
    print("Validation data:")
    print(ds_validate)

    return ds_validate


def get_candidate_dataset():
    ds_candidate_tasks = []

    features = Features({
        "did": Value('string'),
        'txt': Value('string'),
        'img_path': Value('string'),
        'modality': Value('string'),
        'src_content': Value('string'),
    })

    for task_n in ['task0*', 'task3*']:
        data_files_validate_task_n = {
            'cand_pool': ['cand_pool/local/' + file + task_n + '.jsonl' for file in files]
        }

        ds_candidate_tasks.append(load_dataset("TIGER-Lab/M-BEIR",
                                               cache_dir='dataset/cand',
                                               data_files=data_files_validate_task_n,
                                               features=features,
                                               name='cand_pool', split='cand_pool'))
    ds_candidate = concatenate_datasets(ds_candidate_tasks)
    print("Candidate data:")
    print(ds_candidate)

    return ds_candidate


def get_dataset():

    ds_train = get_training_data()

    ds_validate = get_validation_data()

    ds_candidate = get_candidate_dataset()

    return ds_train, ds_validate, ds_candidate


if __name__ == '__main__':
    ds_train, ds_validate, ds_candidate = get_dataset()
