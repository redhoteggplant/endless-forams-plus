# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

# data functions
from dataset import prepare_dataset, sample_from_train, get_segments_dataset

# defaults
DEFAULT_MODEL = "R50-FPN"
DEFAULT_CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DEFAULT_BASE_LR = 0.0025
DEFAULT_IMS_PER_BATCH = 2
DEFAULT_IMAGE_DIR = "./segments/nadiairwanto_PRISM/v0.5.4/"
DEFAULT_DATASET_NAME="prism"
DEFAULT_OUTPUT_DIR="./output"

def register_dataset(dataset_name=DEFAULT_DATASET_NAME,
    annotation_filename="annotation", data_dir="./dataset",
    image_dir=DEFAULT_IMAGE_DIR):
    # Register a dataset with detectron2
    for d in ["train", "val", "test"]:
        try:
            register_coco_instances(f"{dataset_name}_{d}", {},
                f"{data_dir}/{annotation_filename}_{d}.json", image_dir)
        except:
            print(f"Dataset {dataset_name}_{d} is already registered")
        MetadataCatalog.get(f"{dataset_name}_{d}").set(thing_classes=['planktonic foraminifera']) # [c['name'] for c in dataset.categories])


def get_mrcnn_cfg(model=DEFAULT_MODEL, config_path=DEFAULT_CONFIG_PATH,
    lr=DEFAULT_BASE_LR, ims_per_batch=DEFAULT_IMS_PER_BATCH,
    dataset_name=DEFAULT_DATASET_NAME, output_dir=DEFAULT_OUTPUT_DIR
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
    cfg.SOLVER.STEPS = (400,)   # lr decay
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512); try 256 next
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (foram). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    exp_string = f"{model.lower()}.lr_{lr}" # .ims_per_batch_{ims_per_batch}"
    cfg.OUTPUT_DIR = os.path.join(output_dir, exp_string)
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
            cfg = get_mrcnn_cfg(model, config_path=config_path, output_dir=args.output_dir)
            cfgs.append(cfg)
    elif args.expt == "hparam":
        lrs = [0.025, 0.0025, 0.00025, 0.000025]
        for lr in lrs:
            cfg = get_mrcnn_cfg(lr=lr, output_dir=args.output_dir)
            cfgs.append(cfg)
    else:
        cfg = get_mrcnn_cfg(output_dir=args.output_dir)
        cfgs.append(cfg)
    for cfg in cfgs:
        cfg.merge_from_list(args.opts)
    return cfgs


## Train!

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from custom_mapper import custom_mapper

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

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def build_and_train_model(cfg, args):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    skip_train = os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    if skip_train:
        print(f"{cfg.OUTPUT_DIR} already contains final model weights. Skipping training")
    if not skip_train and not args.eval_only:
        trainer.train()
    return trainer


from utils import CustomModel

def load_model(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.TEST.DETECTIONS_PER_IMAGE = 100 # is this needed?
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    model = CustomModel(predictor)
    return model


def evaluate(trainer, cfg, dataset_name=f"{DEFAULT_DATASET_NAME}_test"):
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold; default = 0.05

    cfg.DATASETS.TEST = (dataset_name,)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, dataset_name)
    cfg.freeze()
    metrics = Trainer.test(cfg, trainer.model)
    return metrics


from detectron2.utils.events import EventStorage, JSONWriter
from detectron2.utils.file_io import PathManager

def save_metrics(metrics, output_dir, iter):
    PathManager.mkdirs(output_dir)
    with EventStorage(iter) as storage:
        writer = JSONWriter(os.path.join(output_dir, "metrics.json"))
        storage.put_scalars(**metrics)
        writer.write()
        writer.close()


from detectron2.evaluation.testing import flatten_results_dict
import pandas as pd

def run_data_expts(args, num_reps=5):
    prepare_dataset()
    register_dataset()

    sample_ratios = map(float, args.sample_ratios.split(','))
    for sample_ratio in sample_ratios:
        metrics_list = []
        max_iter, output_dir = 0, ""
        for i in range(num_reps):
            annotation_train = sample_from_train(sample_ratio, i)
            dataset_train = f"prism_{sample_ratio}_{i}_train"
            register_coco_instances(dataset_train, {}, annotation_train, DEFAULT_IMAGE_DIR)
            MetadataCatalog.get(dataset_train).set(thing_classes=['planktonic foraminifera']) # [c['name'] for c in dataset.categories])

            cfg = get_mrcnn_cfg()
            cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}.sample_{sample_ratio}/{i}"
            cfg.DATASETS.TRAIN = (dataset_train,)
            cfg.merge_from_list(args.opts)
            cfg.freeze()
            max_iter, output_dir = cfg.SOLVER.MAX_ITER, os.path.dirname(cfg.OUTPUT_DIR)

            # Train model
            trainer = build_and_train_model(cfg, args)

            # Evaluate on test data
            metrics_i = evaluate(trainer, cfg)
            metrics_list.append(flatten_results_dict(metrics_i))

        # Calculate average metrics across repeated runs
        df = pd.DataFrame.from_dict(metrics_list)
        s = df.mean(axis=0)
        metrics = s.to_dict()
        save_metrics(metrics, output_dir, max_iter)

from visualization import visualize_predictions, visualize_data_augmentations

def main(args):
    setup_logger()

    if args.expt == "data":
        run_data_expts(args)
    else:
        prepare_dataset()
        register_dataset()
        cfgs = setup_cfgs(args)
        for cfg in cfgs:
            if args.vis_preds:
                model = load_model(cfg)
                visualize_predictions(model, cfg, args.show_images)
            elif args.vis_augs:
                visualize_data_augmentations(cfg, args.show_images)
            else:
                trainer = build_and_train_model(cfg, args)
                metrics = flatten_results_dict(evaluate(trainer, cfg))
                save_metrics(metrics, cfg.OUTPUT_DIR, cfg.SOLVER.MAX_ITER)


from detectron2.engine import default_argument_parser, launch

def parse_args():
    parser = default_argument_parser(epilog="Train an instance segmentation model on PRISM data")
    parser.add_argument(
        "--expt",
        default="",
        type=str,
        choices=["arch", "hparam", "data"],
        help="whether to tune the model architecture, hyperparameters, amount of training data, or none",
    )
    parser.add_argument(
        "--max-iter", type=int,
        default=500, help="the maximum number of training iterations"
    )
    parser.add_argument(
        "--sample-ratios",
        default="0.4,0.5,0.6,0.7,0.8,0.9",
        type=str,
        help="proportions of training images to use",
    )
    parser.add_argument(
        "--vis-preds", default=False, action='store_true',
        help="load trained model and visualize predictions"
    )
    parser.add_argument(
        "--vis-augs", default=False, action='store_true',
        help="visualize data augmentations"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="path to output directory")
    parser.add_argument("--show-images", action="store_true", help="show visualization output in a window")
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
