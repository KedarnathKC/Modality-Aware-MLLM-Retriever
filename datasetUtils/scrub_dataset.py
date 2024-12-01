import os
import shutil
import logging
import json
import argparse

from tqdm import tqdm
from load_dataset import get_dataset, get_test_data

logging.basicConfig(
    filename='../dataset/logs/output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class LoggingTqdm(tqdm):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger or logging.getLogger()

    def display(self, msg=None, pos=None):
        if msg is None:
            msg = self.__str__()
        logger.info(msg)
        super().display(msg, pos)


def prepare_json_file(image_paths):

    if not os.path.exists(image_paths):

        logger.info(f"Preparing image paths file")

        ds_train, ds_validate, ds_candidate = get_dataset(domains=['mscoco','visualnews','fashion200k'])

        combined = set(ds_train.filter(lambda x: x['query_img_path'] is not None)['query_img_path'])

        combined |= set(ds_validate.filter(lambda x: x['query_img_path'] is not None)['query_img_path'])

        combined |= set(ds_candidate.filter(lambda x: x['img_path'] is not None)['img_path'])

        combined = list(combined)
        t = len(combined)
        logger.info(f"{t} images to be crawled")

        with open(image_paths, 'w') as f:
            json.dump(combined[0], f)
            for entry in tqdm(combined[1:], total=t-1):
                f.write('\n' + json.dumps(entry))

        return combined

    else:

        logger.info(f"Image paths file exists")
        images_paths_model = []
        with open(image_paths, 'r') as f:
            for line in f:
                images_paths_model.append(json.loads(line))

        logger.info(f"{len(images_paths_model)} images to be filtered")

        return images_paths_model


def filter_images(images_paths_model, source, destination):

    imagePaths = set()
    logger.info("Crawling image paths ...")
    for root, dirs, files in os.walk(f"{source}"):
        if len(files) > 0 and (".jpg" in files[0]):
            prefix = root[root.index("mbeir_images"):] + '/'
            img_paths = set(prefix + f for f in files)
            imagePaths |= img_paths

    logger.info(f"{len(imagePaths)} Data Crawled")

    image_paths_dir = imagePaths.intersection(images_paths_model)

    logger.info(f"{len(images_paths_model)} (training) = {len(image_paths_dir)} (crawled)")

    logger.info(f"Confirmed, copying {len(image_paths_dir)} images.")

    fashion200k_count, news_count, coco_count = 0, 0, 0
    for path in image_paths_dir:
        if "/mscoco_images/" in path:
            coco_count += 1
        elif "/fashion200k_images/" in path:
            fashion200k_count += 1
        elif "/visualnews_images/" in path:
            news_count += 1

    logger.info(f"{fashion200k_count}\t Fashion 200k images to be copied.")
    logger.info(f"{news_count}\t News images to be copied.")
    logger.info(f"{coco_count}\t Coco images to be copied.")

    logger.info("Copying files ...")

    count = 0
    with LoggingTqdm(image_paths_dir) as pbar:
        for path in image_paths_dir:
            file_path = source + path.replace("mbeir_images/", "")
            dest_path = destination + path
            if os.path.exists(file_path):
                os.makedirs("/".join(dest_path.split("/")[:-1])+'/', exist_ok=True)
                shutil.copy(file_path, dest_path)
                logger.debug(f"{file_path} to {dest_path}")
                count += 1
            if count % 10000 == 0:
                pbar.set_postfix({"copied": f"{count//10000}K"})

            pbar.update(1)

    logger.info(f"Copied {count} files.")
    logger.info("Completed !")


def prepare_json_file_test(test_image_paths):

    if not os.path.exists(test_image_paths):

        logger.info(f"Preparing test image paths file")

        ds_test, ds_test_candidate = get_test_data(domains=['mscoco', 'visualnews', 'fashion200k'])

        combined = set(ds_test.filter(lambda x: x['query_img_path'] is not None)['query_img_path'])

        combined |= set(ds_test_candidate.filter(lambda x: x['img_path'] is not None)['img_path'])

        combined = list(combined)
        t = len(combined)
        logger.info(f"{t} test images to be crawled")

        with open(test_image_paths, 'w') as f:
            json.dump(combined[0], f)
            for entry in tqdm(combined[1:], total=t - 1):
                f.write('\n' + json.dumps(entry))

        return combined

    else:

        logger.info(f"Test Image paths file exists")
        images_paths_model = []
        with open(test_image_paths, 'r') as f:
            for line in f:
                images_paths_model.append(json.loads(line))

        logger.info(f"{len(images_paths_model)} test images to be filtered")

        return images_paths_model


def filter_images_test(images_test_paths_model, source, destination):
    testimagePaths = set()
    logger.info("Crawling image paths ...")
    for root, dirs, files in os.walk(f"{source}"):
        if len(files) > 0 and (".jpg" in files[0]):
            prefix = root[root.index("mbeir_images"):] + '/'
            img_paths = set(prefix + f for f in files)
            testimagePaths |= img_paths

    logger.info(f"{len(testimagePaths)} Data Crawled")

    image_paths_dir = testimagePaths.intersection(images_test_paths_model)

    logger.info(f"{len(images_test_paths_model)} (testing) = {len(image_paths_dir)} (crawled)")

    logger.info(f"Confirmed, copying {len(image_paths_dir)} images.")

    fashion200k_count, news_count, coco_count = 0, 0, 0
    for path in image_paths_dir:
        if "/mscoco_images/" in path:
            coco_count += 1
        elif "/fashion200k_images/" in path:
            fashion200k_count += 1
        elif "/visualnews_images/" in path:
            news_count += 1

    logger.info(f"{fashion200k_count}\t Fashion 200k images to be copied.")
    logger.info(f"{news_count}\t News images to be copied.")
    logger.info(f"{coco_count}\t Coco images to be copied.")

    logger.info("Copying files ...")

    count = 0
    with LoggingTqdm(image_paths_dir) as pbar:
        for path in image_paths_dir:
            file_path = source + path.replace("mbeir_images/", "")
            dest_path = destination + path
            if os.path.exists(file_path):
                os.makedirs("/".join(dest_path.split("/")[:-1]) + '/', exist_ok=True)
                # shutil.copy(file_path, dest_path)
                logger.debug(f"{file_path} to {dest_path}")
                count += 1
            if count % 10000 == 0:
                pbar.set_postfix({"copied": f"{count // 10000}K"})

            pbar.update(1)

    logger.info(f"Copied {count} files.")
    logger.info("Completed !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_paths', help='Image path needed by model',
                        default="../dataset/logs/image_paths.jsonl")

    parser.add_argument('--source', help='Source directory',
                        default="../dataset/mbeir_images/")

    parser.add_argument('--destination', help='Destination directory',
                        default="../dataset/mbeir_imagesV1/")

    parser.add_argument('--test', help='Test mode', default=True)

    parser.add_argument('--test_image_paths', help='Test image path needed by model',
                        default="dataset/logs/test_image_paths.jsonl")

    args = parser.parse_args()

    if args.test:
        images_test_paths_model = prepare_json_file_test(args.test_image_paths)

        filter_images_test(images_test_paths_model, source=args.source, destination=args.destination)

    else:
        images_paths_model = prepare_json_file(args.image_paths)

        filter_images(images_paths_model, source=args.source, destination=args.destination)



