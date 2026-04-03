import torch
from utils import compute_accuracy

ckpt_data = torch.load("path/to/your/checkpoint.pt", map_location="cpu")

source_logits = ckpt_data['source_test']['logits'].numpy()
source_gts = ckpt_data['source_test']['gts'].numpy()

src_acc_score = compute_acc(source_logits, source_gts)
