###########################################################################
# Computer vision - Embedded person tracking demo software by HyperbeeAI. #
# Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai #
###########################################################################
import torch

def compute_batch_accuracy(pred, label):
    correct = (pred == label).sum()
    return correct,label.size(0)

def compute_set_accuracy(model, test_loader):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total   = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            
            inputs    = inputs.to(device)
            labels    = labels.to(device)
            outputs   = model(inputs)
            
            correct_batch, total_batch = compute_batch_accuracy(torch.argmax(outputs, dim=1), labels)
            correct += correct_batch
            total   += total_batch
            
    return correct/total