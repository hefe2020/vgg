import torch
from torchvision import transforms
import json
from PIL import Image
from .model import TinyVGGSample

def load_model(model_path, class_to_idx_path):
  class_idx = json.load(open(class_to_idx_path))
  num_class = len(class_idx)
  model = TinyVGGSample(in_channels=3, hidden=20, numb_classes=num_class)
  model.torch.load_state_dict(torch.load(model_path, map_location='cpu'))
  model.eval()
  return model, class_idx

transform = transforms.compose([
  transforms.Resize((32, 32)),
  transforms.ToTensor()
])

def predict(model, img_path, class_idx):
  img = Image.open(img_path).convert("RGB")
  x = transform(img).unsqueeze(0)
  with torch.inference_mode():
    preds = model(x)
    prob = preds.softmax(dim=1)[0]
    top = prob.argmax().item()
  inv_idx = {v: k for k, v in class_idx.items()}
  return inv_idx[top], prob[top].item()

