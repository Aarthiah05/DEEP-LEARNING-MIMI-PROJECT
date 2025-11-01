import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def save_checkpoint(state, fname="checkpoint.pth"):
    torch.save(state, fname)

def load_checkpoint(fname, model, optimizer=None):
    ckpt = torch.load(fname)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    return model, optimizer

def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb,yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds += out.argmax(dim=1).cpu().numpy().tolist()
            targets += yb.cpu().numpy().tolist()
    print(classification_report(targets,preds))
    print("Confusion matrix:")
    print(confusion_matrix(targets,preds))
    return preds, targets
