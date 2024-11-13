import json
from load_dataset import get_dataset


## For filtering out only the images needed for training and evaluation
def get_json_file():
    ds_train, ds_validate, ds_candidate = get_dataset()

    combined = set([img_path for img_path in ds_train['query_img_path'] if img_path is not None])

    combined |= set([img_path for img_path in ds_validate['query_img_path'] if img_path is not None])

    combined |= set([img_path for img_path in ds_candidate['img_path'] if img_path is None])

    total_images = len(combined)
    print(f"{total_images} images to be crawled")

    with open('dataset/image_paths.jsonl', 'w') as f:
        for entry in combined:
            f.write(json.dumps(entry) + '\n')


def filter_images():
    data = []
    with open('dataset/image_paths.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    ##TODO:: Remove images not needed from the dataset


if __name__ == '__main__':
    get_json_file()
