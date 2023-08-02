import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .reptile import Reptile
from .checkpoints import save_ckpt_best_corr, resume_from_ckpt


def train(model,
          optimizer,
          criterion,
          device,
          train_set,
          test_set,
          checkpoint,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          train_shots=None,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          output='outputs',
          transductive=False,
          fn=Reptile):
    start_epoch = 0

    if checkpoint and os.path.isfile(checkpoint):
        start_epoch = resume_from_ckpt(checkpoint, model, optimizer)
    elif checkpoint and not os.path.exists(checkpoint):
        os.makedirs(checkpoint, exist_ok=True)

    reptile = fn(model, optimizer, criterion, device, transductive)

    train_writer = SummaryWriter(os.path.join('/root/tf-logs', 'train'))
    test_writer = SummaryWriter(os.path.join('/root/tf-logs', 'test'))

    for epoch in range(start_epoch, meta_iters):
        frac_done = epoch / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        reptile.train(train_set, num_classes=num_classes,
                      num_shots=(train_shots or num_shots),
                      meta_batch_size=meta_batch_size, inner_batch_size=inner_batch_size,
                      inner_iters=inner_iters, replacement=replacement,
                      meta_step_size=cur_meta_step_size)
        if epoch % eval_interval == 0:
            accuracies = []
            for data_set, writer in [(train_set, train_writer), (test_set, test_writer)]:
                acc = reptile.evaluate(data_set, num_classes=num_classes,
                                       num_shots=num_shots, inner_batch_size=eval_inner_batch_size,
                                       inner_iters=eval_inner_iters, replacement=replacement)
                writer.add_scalar('accuracy', acc, epoch)
                accuracies.append(acc)
            print(f'EPOCH: {epoch}/{meta_iters} train acc: {accuracies[0]} test acc: {accuracies[1]}')
            save_ckpt_best_corr(checkpoint, accuracies[1], epoch, model, optimizer)
        if epoch == meta_iters - 1:
            if not os.path.exists(output):
                os.makedirs(output)
            torch.save(model.state_dict(), os.path.join(output, 'model.pth'))
