import torch
from utils import compute_accuracy

ckpt_data = torch.load("path/to/your/checkpoint.pt", map_location="cpu")

target_logits = ckpt_data['target_test']['logits'].numpy()
target_gts = ckpt_data['target_test']['gts'].numpy()

target_acc = compute_accuracy(target_logits, target_gts)
