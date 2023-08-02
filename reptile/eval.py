import torch
from tqdm import tqdm

from reptile import Reptile


def evaluate(model,
             optimizer,
             criterion,
             device,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             fn=Reptile):
    reptile = fn(model, optimizer, criterion, device, transductive)
    total_acc = torch.Tensor(0)
    for _ in tqdm(range(num_samples)):
        total_acc += reptile.evaluate(dataset,
                                      num_classes=num_classes, num_shots=num_shots,
                                      inner_batch_size=eval_inner_batch_size,
                                      inner_iters=eval_inner_iters, replacement=replacement)
    return (total_acc / num_samples).item()
