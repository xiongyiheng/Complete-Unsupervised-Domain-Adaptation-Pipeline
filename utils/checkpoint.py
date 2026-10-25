"""
Checkpoint saving for model selection.

A checkpoint stores the model weights together with the outputs every validator needs, so that validators which
must re-run the model (MixVal: forward passes on mixed target inputs) or need the classifier weights
(TransferScore: uniformity of the final classifier layer) are computed exactly as in their original papers:

    checkpoint = {
        'state_dict':   model.state_dict(),                 # full model (backbone(s), heads, discriminators, memory buffers)
        'head_weights': [W_1, ...],                         # weight of the final nn.Linear of each classifier head (C x D)
        'meta':         {...},                              # anything passed as **meta (iteration, lambda, fold, ...)
        'source_train': {'features': ..., 'head_features': ..., 'logits': ..., 'gts': ...},
        'target_train': {...}, 'source_test': {...}, 'target_test': {...},
    }

features      = backbone output (concatenated over backbones for two-backbone models such as CoUDA);
head_features = input of the final nn.Linear of each head, shape (N, D) for one head or (H, N, D) for H heads
                (MCD, CoUDA) -- the representation TransferScore's Hopkins statistic is computed on;
logits        = model.predict(x) (the averaged prediction for two-head models).

Loaders must iterate the split in a fixed order (shuffle=False) and yield (x, y) or (x, y, index).
"""
import torch
import torch.nn as nn

HEAD_NAMES = ('classifier', 'classifier1', 'classifier2', 'cls_head', 'cls_head1', 'cls_head2')
BACKBONE_NAMES = ('backbone', 'backbone1', 'backbone2')


def final_linears(model: nn.Module):
    """The last nn.Linear of every classifier head of the model, in HEAD_NAMES order."""
    out = []
    for name in HEAD_NAMES:
        head = getattr(model, name, None)
        if head is None:
            continue
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        if linears:
            out.append(linears[-1])
    if not out:
        raise ValueError('model has no classifier head among ' + ', '.join(HEAD_NAMES))
    return out


def head_weights(model: nn.Module):
    return [lin.weight.detach().float().cpu() for lin in final_linears(model)]


@torch.no_grad()
def extract_split(model: nn.Module, loader, device):
    """features, head_features, logits and labels of one split (model in eval mode)."""
    backbones = [getattr(model, n) for n in BACKBONE_NAMES if getattr(model, n, None) is not None]
    linears = final_linears(model)
    cache = {'bb': [], 'head': []}
    hooks = [b.register_forward_hook(lambda m, i, o: cache['bb'].append(o)) for b in backbones]
    hooks += [lin.register_forward_hook(lambda m, i, o: cache['head'].append(i[0])) for lin in linears]

    was_training = model.training
    model.eval()
    feats, head_feats, logits, gts = [], [], [], []
    try:
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            cache['bb'].clear(); cache['head'].clear()
            logits.append(model.predict(x).float().cpu())
            feats.append(torch.cat([f.flatten(1) for f in cache['bb'][:len(backbones)]], dim=1).float().cpu())
            h = [f.float().cpu() for f in cache['head'][:len(linears)]]
            head_feats.append(h[0] if len(h) == 1 else torch.stack(h))
            gts.append(torch.as_tensor(y).view(-1).cpu())
    finally:
        for hk in hooks:
            hk.remove()
        model.train(was_training)
    cat_dim = 0 if head_feats[0].dim() == 2 else 1
    return {'features': torch.cat(feats), 'head_features': torch.cat(head_feats, dim=cat_dim),
            'logits': torch.cat(logits), 'gts': torch.cat(gts)}


def save_checkpoint(path, model: nn.Module, loaders: dict, device, **meta):
    """loaders: {'source_train': ..., 'target_train': ..., 'source_test': ..., 'target_test': ...}."""
    ckpt = {
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'head_weights': head_weights(model),
        'meta': meta,
    }
    for split, loader in loaders.items():
        ckpt[split] = extract_split(model, loader, device)
    torch.save(ckpt, path)
    return ckpt


def load_weights(model: nn.Module, ckpt_or_path, device='cpu'):
    """Restore the weights saved by save_checkpoint into a freshly built model of the same configuration."""
    ckpt = torch.load(ckpt_or_path, map_location=device) if isinstance(ckpt_or_path, (str, bytes)) or hasattr(
        ckpt_or_path, '__fspath__') else ckpt_or_path
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device)
