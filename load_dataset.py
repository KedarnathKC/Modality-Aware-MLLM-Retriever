from datasets import load_dataset, load_dataset_builder, concatenate_datasets, Features, Value

files = ['*visualnews*','*fashion200k*','*mscoco*']


def get_dataset():

    data_files_train = {
        'train': ['query/train/' + file + '.jsonl' for file in files],
    }

    ds_train = load_dataset("TIGER-Lab/M-BEIR",
                            cache_dir='dataset/train',
                            data_files=data_files_train,
                            name='query', split='train')

    print("Training data:")
    print(ds_train)

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
            'cand_pool': ['cand_pool/local/'+ file + task_n + '.jsonl' for file in files]
        }

        ds_candidate_tasks.append(load_dataset("TIGER-Lab/M-BEIR",
                                   cache_dir='dataset/cand',
                                   data_files=data_files_validate_task_n,
                                   features=features,
                                   name='cand_pool', split='cand_pool'))

    ds_candidate = concatenate_datasets(ds_candidate_tasks)

    print("Candidate data:")
    print(ds_candidate)

    return ds_train, ds_validate, ds_candidate


if __name__ == '__main__':
    ds_train, ds_validate, ds_candidate = get_dataset()
