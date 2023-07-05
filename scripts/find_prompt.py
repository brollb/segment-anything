# Given a model and correct set of annotations/masks, try to find a prompt to produce the masks
# For now, we will use the `no_mask_embed` layer in the prompt encoder. In other words, we will
# feed in no prompt and train the (default/empty) prompt embedding

import cv2
import torch
import json
from torchvision.ops import sigmoid_focal_loss
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor

torch.set_grad_enabled(False)  # This will not apply to the embedding to train since it is hard-coded

# TODO: load the model
model_type = 'default'
checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'
device = 'cuda'
input = '../../scm/dataset/Images/full/bq24072t-bq24075t-bq24079t.jpg'
target_path = '../../scm/dataset/ti/annotations/bq24072t-bq24075t-bq24079t.json'

sam = sam_model_registry[model_type](checkpoint=checkpoint)
_ = sam.to(device=device)
predictor = SamPredictor(sam)

# predict w/o any inputs (train the no_mask embedding)

# First, prepare the targets
def anno_to_mask(anno):
    pos_str = anno['target']['selector']['value']
    x, y, w, h = [ float(n) for n in pos_str.split(':')[1].split(',') ]
    # TODO: should I normalize this?
    return [x, y, x + w, y + h]

with open(target_path, 'r') as f:
    resistors = [ anno for anno in json.load(f) if 'Resistor' in [ b['value'] for b in anno['body'] ]]
    print(len(resistors))
    print(resistors[0])
    target_boxes = torch.tensor([ anno_to_mask(r) for r in resistors ])

# Next, load the image
image = cv2.imread(input)
print('image', image.shape)
predictor.set_image(image)

# predict using the expected annotations
masks, confs, _ = predictor.predict_torch(None, None, return_logits=True)
print('masks', masks.shape)
print('confs', confs.unsqueeze(0))
conf_masks = masks * confs

# convert masks to bboxes
loss = sigmoid_focal_loss(masks, targets)

