import requests
import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from torchvision.transforms.functional import crop as crop_f, resize, to_pil_image
import cv2

from OwlViT import OWL_ViT
from segment_anything import build_sam, SamPredictor

import time

class OWL_SAM(nn.Module):

    def __init__(self, device: str = 'cpu') -> None:
        super().__init__()

        self.device = device
        self.detector = OWL_ViT(device)
        self.segmentator = SamPredictor(build_sam(checkpoint=os.path.join(Path(__file__).parent, "sam_vit_h_4b8939.pth")))
        self.segmentator.model.to(device=device)
    
    def detect_object(self, x: dict):
        return self.detector(x)
    
    def subpart_suppression(self, masks, threshold=0.2):
        # For any pair of objects, if (subpart_threshold) of one is inside the other, keep the other.
        remove_idxs = []
        for i in range(len(masks)):
            curr_mask = masks[i]
            curr_area = curr_mask.sum()
            for j in range(i + 1, len(masks)):
                other_mask = masks[j]
                other_area = other_mask.sum()
                intersection = (curr_mask & other_mask).sum()
                if intersection / curr_area > threshold or intersection / other_area > threshold:
                    # Remove the smaller one.
                    smaller_area_idx = i if curr_area < other_area else j
                    remove_idxs.append(smaller_area_idx)

        keep_idxs = [i for i in range(len(masks)) if i not in remove_idxs]
        masks = [masks[i] for i in keep_idxs]
        return masks

    def large_obj_suppression(self, masks, img, threshold=0.3):
        img_area = img.shape[0] * img.shape[1]
        masks = [mask for mask in masks if mask.sum() / img_area <= threshold]
        return masks

    def small_obj_suppression(self, masks, img, threshold=0.005):
        img_area = img.shape[0] * img.shape[1]
        masks = [mask for mask in masks if mask.sum() / img_area >= threshold]
        return masks

    # Keeps only masks which are connected components (no multiple islands).
    def disconnected_components_suppression(self, masks):
        masks = [mask for mask in masks if cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))[0] == 2]
        return masks
    
    def forward(self, x: dict):
        bbox = self.detect_object(x)
        
        images = x["images"]
        masks = np.zeros((images.shape[:-1]))
        for image_id in range(images.shape[0]):
            self.segmentator.set_image(images[image_id])
            transformed_boxes = self.segmentator.transform.apply_boxes_torch(bbox, images.shape[1:3])
            masks_output, _, _ = self.segmentator.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = True,
            )
            masks_output = self.disconnected_components_suppression(masks_output[image_id])
            masks_output = self.large_obj_suppression(masks_output, images[image_id]) # To remove bground obj.
            masks_output = self.subpart_suppression(masks_output)
            # masks_output = self.small_obj_suppression(masks_output, images[image_id]) # To remove small objs which cannot be grasped anyway.
            masks[image_id,:,:] = masks_output[0].cpu().numpy()
        
        if masks.shape[0] == 1:
            return masks[0]
        return masks

    # def forward(self, x: dict):
    #     bbox = self.detect_object(x)
    #     bbox = bbox.unsqueeze(0)
        
    #     images = x["images"]
    #     masks = np.zeros((images.shape[:-1]))
    #     for image_id in range(images.shape[0]):
    #         self.segmentator.set_image(images[image_id])
    #         transformed_boxes = self.segmentator.transform.apply_boxes_torch(bbox, images.shape[1:3])
    #         mask, _, _ = self.segmentator.predict_torch(
    #             point_coords = None,
    #             point_labels = None,
    #             boxes = transformed_boxes,
    #             multimask_output = False,
    #         )
    #         masks[image_id,:,:] = mask.cpu().numpy()
        
    #     if masks.shape[0] == 1:
    #         return masks[0]
    #     return masks


if __name__ == "__main__":
    from utils import draw_mask, open_image_numpy
    import matplotlib.pyplot as plt
    import sys

    device = 'cuda:0'

    if len(sys.argv) > 1:

        image_number = sys.argv[1]
        # image_dir = "/home/pita/Documents/PhD/OwlViT/fixed_present/head_camera_rgb_"
        image_dir = "/home/pita/Documents/Projects/LangSeg/fixed_present/head_camera_rgb_"
        image_dir += f"{image_number}.png"
        image = open_image_numpy(image_dir)

        text = [sys.argv[2]]
        time_stamp = time.time()

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # text = ["cat", "a photo of a remote"]

        segmentator = OWL_SAM(device)
        print(f"Model initialisation time: {time.time() - time_stamp} seconds")
        time_stamp = time.time()

        # x = {
        #     "texts": text,
        #     "images": image.copy(),
        # }
        x = {
            "texts": text,
            "images": np.stack((image.copy(), image.copy())),
            # "images": image.copy()
        }

        mask = segmentator(x)
        print(f"Inference time: {time.time() - time_stamp} seconds")
        time_stamp = time.time()

        draw_mask(image, mask)
    else:
        owl = OWL_SAM()