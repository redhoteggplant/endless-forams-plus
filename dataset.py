## Prepare the dataset
from segments import SegmentsClient
from segments import SegmentsDataset
from segments.utils import export_dataset

import json
from sklearn.model_selection import train_test_split

import os

def get_segments_dataset(dataset_name='nadiairwanto/PRISM', release_name='v0.5.4', annotation_file='annotations.json'):
    """
    Obtain reviewed images from the specified Segments.ai dataset release.
    """
    if os.path.exists(annotation_file):
        return annotation_file, os.path.join(dataset_name, release_name)

    # Set up the client
    API_KEY = '2eea99b9f8e79adaf162c8736a6969277e1a2c47'
    client = SegmentsClient(API_KEY)

    # Initialize a dataset from the release file
    release = client.get_release(dataset_name, release_name)
    dataset = SegmentsDataset(release, task='segmentation', filter_by=['reviewed']) # 'labeled'

    annotation_json, image_dir = export_dataset(dataset)
    os.rename(annotation_json, annotation_file)
    return annotation_file, image_dir


def save_coco(file, info, images, annotations, categories):
    ids = [im['id'] for im in images]
    annotations = [a for a in annotations if a['image_id'] in ids]
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({
            'info': info,
            'images': images,
            'annotations': annotations,
            'categories': categories
            }, coco, indent=2, sort_keys=True)


def split_dataset(images, val_size, test_size, random_state=123):
    # split dataset https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py
    images_train, images_test = train_test_split(images, test_size=test_size,
                                                    random_state=random_state)
    images_train, images_val = train_test_split(images_train, test_size=val_size/(1-test_size),
                                                    random_state=random_state)
    return images_train, images_val, images_test


def prepare_dataset(val_size=0.15, test_size=0.15):
    # split_ratio = f"{int((1-val_size-test_size)*100)}-{int(val_size*100)}-{int(test_size*100)}"
    json_train = f"annotation_train.json"
    json_val = f"annotation_val.json"
    json_test = f"annotation_test.json"

    if os.path.exists(json_train):
        return json_train, json_val, json_test

    # download dataset
    annotation_json, image_dir = get_segments_dataset()

    # save train, val, and test datasets
    with open(annotation_json, 'rt') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_train, images_val, images_test = split_dataset(images, val_size, test_size)

        save_coco(json_train, info, images_train, annotations, categories)
        save_coco(json_val, info, images_val, annotations, categories)
        save_coco(json_test, info, images_test, annotations, categories)

    print("Saved {} entries in {}, {} in {}, and {} in {}".format(
        len(images_train), json_train, len(images_val), json_val, len(images_test), json_test))

    return json_train, json_val, json_test


if __name__ == "__main__":
    prepare_dataset()
    exit(0)
