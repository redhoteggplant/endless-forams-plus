import cv2
import os
from detectron2.utils.visualizer import Visualizer, ColorMode

def output(vis, fname, dirname, show_images):
    """
    Take the VisImage
    """
    if show_images:
        print(fname)
        cv2.imshow("window", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
    else:
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)


def visualize_predictions(model, cfg, show_images):
    data_dir = "./dataset"
    image_dir = DEFAULT_IMAGE_DIR
    annotation_filename = "annotation"
    dataset_name = DEFAULT_DATASET_NAME

    for split in ["train", "val", "test"]:
        dataset_dicts = load_coco_json(f"{data_dir}/{annotation_filename}_{split}.json", DEFAULT_IMAGE_DIR)
        segments_metadata = MetadataCatalog.get(f"{dataset_name}_{split}").set(thing_classes=['planktonic foraminifera'])

        pred_dir = os.path.join(cfg.OUTPUT_DIR, "predictions", split)
        os.makedirs(pred_dir)
        for d in dataset_dicts:
            im = cv2.imread(d["file_name"])
            outputs = model.predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=segments_metadata,
                        scale=0.2,
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output(vis, d["file_name"].split("/")[-1], pred_dir, show_images)

        print(f"{len(os.listdir(pred_dir))} visualizations saved to {pred_dir}")


from detectron2.data import build_detection_train_loader
from custom_mapper import custom_mapper
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils

def visualize_data_augmentations(cfg, show_images):
    """
	Visualize the data augmentations applied.

    Code adapted from Detectron2 visualize_data.py.
    https://github.com/facebookresearch/detectron2/blob/master/tools/visualize_data.py
    """
    dirname = os.path.join(cfg.OUTPUT_DIR, "augmentations")
    os.makedirs(dirname, exist_ok=True)\

    train_data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    scale = 1.0

    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = detection_utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            v = Visualizer(img, metadata=metadata, scale=scale)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = v.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, str(per_image["image_id"]) + ".jpg", dirname, show_images)
