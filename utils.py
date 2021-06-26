import numpy as np

class CustomModel:
    """
    Taken from Segments.ai's fast labeling workflow tutorial.
    https://github.com/segments-ai/fast-labeling-workflow/blob/master/utils.py
    """
    def __init__(self, predictor):
        self.predictor = predictor

    def _convert_to_segments_format(self, image, outputs):
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
        annotations = []
        counter = 1
        instances = outputs['instances']
        for i in range(len(instances.pred_classes)):
            category_id = int(instances.pred_classes[i])
            instance_id = counter
            mask = instances.pred_masks[i].cpu()
            segmentation_bitmap[mask] = instance_id
            annotations.append({'id': instance_id, 'category_id': category_id})
            counter += 1
        return segmentation_bitmap, annotations

    def __call__(self, image):
        image = np.array(image)
        outputs = self.predictor(image)
        label, label_data = self._convert_to_segments_format(image, outputs)

        return label, label_data
