import sys
from utils import draw_mask
import numpy as np
import cv2

import torch
import torch.nn as nn
from PIL import Image
from lang_sam import LangSAM as LangSAM_model

class LangSAM(nn.Module):

    def __init__(self, device: str = 'cuda') -> None:
        super().__init__()

        self.device = device
        self.lang_segmentatator = LangSAM_model(device=device)
    
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
    
    def backgorund_suppression(self, masks, bboxes, img, threshold=0.95):
        print("\n\n\n\n\n")
        print(img.shape)
        height, width, _ = img.shape

        print(masks[0].shape)

        print(bboxes)
        print(bboxes[0,3] / height)
        print(bboxes[1,3] / height)
        print(bboxes[0,2] / width)
        print(bboxes[0,2] / width)

        print((len(masks)))
        # print(masks)

        masks = [masks[id] for id in range(len(masks)) if bboxes[id,3] / height <= threshold]
        print(len(masks))
        masks = [masks[id] for id in range(len(masks)) if bboxes[id,2] / width <= threshold]
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
        if len(x["images"].shape) == 3:
            x["images"] = np.expand_dims(x["images"], 0)

        images = x["images"]
        bs = images.shape[0]
        texts = [x["texts"]] * bs

        # text_prompt = x["texts"]
        masks = np.zeros((images.shape[:-1]))
        
        for image_id in range(images.shape[0]):
            with torch.no_grad():
                image_pil = Image.fromarray(images[image_id])
                masks_output, bbox_output, phrases, logits = self.lang_segmentatator.predict(image_pil, text_prompt)
                bbox_output[:, 2:] = bbox_output[:, 2:] - bbox_output[:, :2]

            print("\n\n")
            print(masks.shape)
            print(phrases)
            print(logits)
            print(f"Masks before: {masks_output.shape}")
            if masks_output.shape[0] > 1:
                # masks_output = self.disconnected_components_suppression(masks_output)
                masks_output = self.backgorund_suppression(masks_output, bbox_output, images[image_id]) # To remove bground obj.
                masks_output = self.subpart_suppression(masks_output)
                masks_output = self.small_obj_suppression(masks_output, images[image_id]) # To remove small objs which cannot be grasped anyway.
            print(f"Masks after: {len(masks_output)} {masks_output[0].shape}")
            masks[image_id,:,:] = masks_output[0].cpu().numpy()
        
        if masks.shape[0] == 1:
            return masks[0]
        return masks
    
    # def forward(self, x: dict):
    #     images = x["images"]
    #     text_prompt = x["texts"]
    #     masks = np.zeros((images.shape[:-1]))
        
    #     for image_id in range(images.shape[0]):
    #         self.segmentator.set_image(images[image_id])
    #         transformed_boxes = self.segmentator.transform.apply_boxes_torch(bbox, images.shape[1:3])
    #         masks_output, _, _ = self.segmentator.predict_torch(
    #             point_coords = None,
    #             point_labels = None,
    #             boxes = transformed_boxes,
    #             multimask_output = True,
    #         )
    #         print(masks_output.shape)
    #         masks_output = self.disconnected_components_suppression(masks_output[image_id])
    #         masks_output = self.large_obj_suppression(masks_output, images[image_id]) # To remove bground obj.
    #         masks_output = self.subpart_suppression(masks_output)
    #         # masks_output = self.small_obj_suppression(masks_output, images[image_id]) # To remove small objs which cannot be grasped anyway.
    #         masks[image_id,:,:] = masks_output[0].cpu().numpy()
        
    #     if masks.shape[0] == 1:
    #         return masks[0]
    #     return masks


model = LangSAM()
image_number = sys.argv[1]
image_dir_root = "/home/pita/Documents/Projects/LangSeg/fixed_present/head_camera_rgb_"
try:
    image_dir = image_dir_root + f"{image_number}.png"
    image_pil = Image.open(image_dir).convert("RGB")
except:
    image_dir = image_dir_root + f"{image_number}.jpg"
    image_pil = Image.open(image_dir).convert("RGB")
image_np = np.array(Image.open(image_dir).convert("RGB"))
text_prompt = "everything"

# model = LangSAM_model()
# masks, boxes_out, phrases, logits = model.predict(image_pil, text_prompt)
# mask = masks[0]

x = {"images": image_np,
     "texts": text_prompt}

mask = model(x)
draw_mask(image_pil, mask)