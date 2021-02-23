# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from dataset import prepare_dataset

DEFAULT_MODEL = "R50-FPN"
DEFAULT_CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DEFAULT_BASE_LR = 0.0025
DEFAULT_IMS_PER_BATCH = 2


# register dataset
def register_dataset(dataset_name="prism", annotation_file="annotation", data_dir="./dataset", image_dir="./segments/nadiairwanto_PRISM/v0.5.4/"):
    for d in ["train", "val", "test"]:
        try:
            register_coco_instances(f"{dataset_name}_{d}", {}, f"{data_dir}/{annotation_file}_{d}.json", image_dir)
        except:
            print(f"Dataset {dataset_name}_{d} is already registered")
        MetadataCatalog.get(f"{dataset_name}_{d}").set(thing_classes=['planktonic foraminifera']) # [c['name'] for c in dataset.categories])


## Train!

# %cp -r AdelaiDet/adet .
# !wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O SOLOv2_R50_3x.pth

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "coco"
        return COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_dir=output_folder)



def train_model(cfg, args):
    skip_train = args.eval_only or os.path.exists(cfg.OUTPUT_DIR)
    if os.path.exists(cfg.OUTPUT_DIR):
        print(f"{cfg.OUTPUT_DIR} already exists. Skipping training")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if not skip_train:
        trainer.train()
    return trainer


"""Then, we randomly select several samples to visualize the prediction results."""

def visualize_predictions(dataset_name, annotation_json, image_dir):
    dataset_dicts = load_coco_json(annotation_json, image_dir)
    metadata = MetadataCatalog.get(dataset_name)
    for d in dataset_dicts: # random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        print(d["file_name"].split("/")[-1])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.2,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(out.get_image()[:, :, ::-1])

"""
We can also evaluate its performance using AP metric implemented in COCO API.
"""

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def evaluate(trainer, cfg, dataset_name):
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold; default = 0.05

    cfg.DATASETS.TEST = (dataset_name,)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, dataset_name)
    cfg.freeze()
    metrics = Trainer.test(cfg, trainer.model)

    # evaluator = COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_dir=os.path.join(cfg.OUTPUT_DIR, dataset_name))
    # test_loader = build_detection_test_loader(cfg, dataset_name)
    # metrics = inference_on_dataset(trainer.model, test_loader, evaluator)
    return metrics


def get_mrcnn_cfg(model=DEFAULT_MODEL, config_path=DEFAULT_CONFIG_PATH,
    lr=DEFAULT_BASE_LR, ims_per_batch=DEFAULT_IMS_PER_BATCH,
    dataset_name="prism"
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)  # Let training initialize from model zoo
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch

    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = (dataset_name + "_val",)
    cfg.TEST.EVAL_PERIOD = 40
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.SOLVER.MAX_ITER = 500    # 500 iterations seems good enough for the baseline
    cfg.SOLVER.STEPS = (400,)
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512); try 256 next
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (foram). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    exp_string = f"{model.lower()}.lr_{lr}" # .ims_per_batch_{ims_per_batch}"
    cfg.OUTPUT_DIR = os.path.join("./tmp/output", exp_string)
    return cfg


def setup_cfgs(args):
    """
    Create configs and perform basic setups.
    """
    models = {
        "R50-FPN":  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "R50-DC5": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        "X101-FPN": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "R101-FPN": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    }
    cfgs = []
    if args.expt == "arch":
        for model, config_path in models.items():
            cfg = get_mrcnn_cfg(model, config_path=config_path)
            cfgs.append(cfg)
    elif args.expt == "hparam":
        lrs = [0.025, 0.0025, 0.00025, 0.000025]
        for lr in lrs:
            cfg = get_mrcnn_cfg(lr=lr)
            cfgs.append(cfg)
    else:
        cfg = get_mrcnn_cfg()
        cfgs.append(cfg)
        # cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}.warmup_200_decay_400"
    for cfg in cfgs:
        cfg.merge_from_list(args.opts)
    return cfgs


import pandas as pd

def run_data_expts(args, num_reps=5):
    # split_ratios = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2)]
    # split_ratios = [(0.1, 0.1)]
    # split_ratios = [(0.15, 0.15)]
    split_ratios = [(0.2, 0.2)]
    for val_size, test_size in split_ratios:
        metrics_list = []
        max_iter, output_dir = 0, ""
        split_ratio_str = f"{int((1-val_size-test_size)*100)}-{int(val_size*100)}-{int(test_size*100)}" # e.g. 70-15-15
        for i in range(num_reps):
            dataset_name = f"prism_{split_ratio_str}_{i}"
            annotation_file = f"annotation_{split_ratio_str}_{i}"
            prepare_dataset(annotation_file, val_size, test_size, random_state=123+i)
            register_dataset(dataset_name, annotation_file)

            cfg = get_mrcnn_cfg(dataset_name=dataset_name)
            cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}.split_{split_ratio_str}/{i}"
            cfg.merge_from_list(args.opts)
            cfg.freeze()
            max_iter, output_dir = cfg.SOLVER.MAX_ITER, os.path.dirname(cfg.OUTPUT_DIR)

            # Train model
            trainer = train_model(cfg, args)

            # Evaluate on test data
            metrics_i = evaluate(trainer, cfg, f"{dataset_name}_test")
            metrics_list.append(flatten_results_dict(metrics_i))

        # Calculate average metrics across repeated runs
        df = pd.DataFrame.from_dict(metrics_list)
        s = df.mean(axis=0)
        metrics = s.to_dict()
        save_metrics(metrics, output_dir, max_iter):


from detectron2.utils.events import EventStorage, JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.testing import flatten_results_dict

def save_metrics(metrics, output_dir, iter):
    PathManager.mkdirs(output_dir)
    with EventStorage(iter) as storage:
        writer = JSONWriter(os.path.join(output_dir, "metrics.json"))
        storage.put_scalars(**metrics)
        writer.write()
        writer.close()


def main(args):
    setup_logger()

    if args.expt == "data":
        run_data_expts(args)
    else:
        prepare_dataset()
        register_dataset()
        cfgs = setup_cfgs(args)
        for cfg in cfgs:
            trainer = train_model(cfg, args)
            metrics = evaluate(trainer, cfg, "prism_test")
            save_metrics(metrics, cfg.OUTPUT_DIR, cfg.SOLVER.MAX_ITER)


from detectron2.engine import default_argument_parser, launch

def parse_args():
    parser = default_argument_parser()
    parser.add_argument(
        "--expt",
        default="",
        type=str,
        choices=["arch", "hparam", "data"],
        help="whether to tune the model architecture, hyperparameters, or none",
    )
    parser.add_argument(
        "--max-iter", type=int,
        default=500, help="the maximum number of training iterations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    """
    Command Line Args: (config_file='', dist_url='tcp://127.0.0.1:50158', eval_only=False, machine_rank=0, model='mask_rcnn', num_gpus=1, num_machines=1, opts=[], resume=False)
    """