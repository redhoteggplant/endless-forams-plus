from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch

def custom_mapper(dataset_dict):
    """
    Custom mapper function to perform custom transformations to data.
    The mapper transforms the lightweight representation of a dataset item into
    a format that is ready for the model to consume.

    Code adapted from Detectron2 dataloader tutorial, modified to customize
    transformations and handle RLE masks.
    https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Custom transformations
    transform_list = [
        T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
        T.RandomContrast(intensity_min=0.6, intensity_max=1.3),
        T.ResizeScale(min_scale=0.2, max_scale=2.0, target_height=image.shape[0], target_width=image.shape[1]),
        T.RandomRotation(angle=[90, 90], expand=False),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(
        annos, image.shape[:2], mask_format='bitmask' # self.instance_mask_format
    )

    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict
