import torch
import detectron2
from torch import nn

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
import matplotlib.colors as mcolors
import tqdm
import requests

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch import GeneralizedRCNN
setup_logger()

from nebula3_videoprocessing.videoprocessing.bboxes_interface import IBBoxerInitter

#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

class BBoxesBaseImplementation(IBBoxerInitter):

    def compute_bbox_proposals_url(self, url: str, text: list[str]):
        image = self.load_image_url(url)
        return self.compute_bbox_proposals(image, text)

def inference_rpn(
        self,
        batched_inputs, #: List[Dict[str, torch.Tensor]],
        detected_instances = None,
        do_postprocess = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        return proposals


class DetectronBBInitter(BBoxesBaseImplementation): # Inherits from TrackerModel ?

    def __init__(self, topk_rpn = 50):
        self.topk_rpn = topk_rpn
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(self.cfg)
        self.model.inference = inference_rpn
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

    def load_image_url(self, url: str):
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            print("Image load was unsuccessful.")
        return image

    def get_predictions(self, img):
        aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.      
            height, width = img.shape[:2]
            image = aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model.inference(self.model, [inputs])[0]                

            return predictions

    def compute_bbox_proposals(self, img, uniquify=False):

        preds = self.get_predictions(img)

        meta_data_det = list()
        for top_proposal_rpn in range(self.topk_rpn):
            pred_instance = preds[top_proposal_rpn].to("cpu")
            meta_data_det.append(pred_instance.__getattr__('proposal_boxes').tensor.numpy())
        sz = pred_instance[0].image_size
        out = img[:, :, ::-1] # Detectron2 BGR to RGB
        bbox_build_in = False
        meta_data_det = [l[0].tolist() for l in meta_data_det]
        return {'meta_data_det' :meta_data_det, 'draw_inst_pred': out, 'image_size': sz, 'bbox_build_in': bbox_build_in}
        # return [l[0].tolist() for l in meta_data_det]


def main():

    full_fname = '/datasets/visualgenome/VG/2316634.jpg'
    im = cv2.imread(full_fname)

    det = DetectronBBInitter()
    outputs = det.compute_bbox_proposals(im)
    print(outputs)




if __name__ == '__main__':
    main()
