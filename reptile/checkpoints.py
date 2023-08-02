import logging
import os.path
from typing import Optional

import torch


logger = logging.getLogger(__name__)

__all__ = ['resume_from_ckpt', 'save_ckpt_best_corr']
best_corr: int = -1


def resume_from_ckpt(ckpt_path: Optional[str], model, optimizer=None, scheduler=None):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(state_dict=ckpt['model'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['lr_scheduler'])

    logger.info(f'load ckpt from {ckpt_path}')
    logger.info(f"model was trained for {ckpt['epoch']} epoch")
    return ckpt['epoch']


def save_ckpt_best_corr(ckpt_path: Optional[str], corr: float, epoch: int, model, optimizer=None, scheduler=None):
    if ckpt_path is None:
        return
    if os.path.isfile(ckpt_path):
        ckpt_path = os.path.dirname(ckpt_path)

    global best_corr

    if best_corr < 0:
        best_corr = corr
        save_ckpt(ckpt_path, epoch, model, optimizer, scheduler)
    elif corr >= best_corr:
        best_corr = corr
        save_ckpt(ckpt_path, epoch, model, optimizer, scheduler)


def save_ckpt(ckpt_path: Optional[str], epoch: int, model, optimizer=None, scheduler=None):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt = {
        'model': model.state_dict(),
        'epoch': epoch
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        ckpt['lr_scheduler'] = scheduler.state_dict()
    torch.save(ckpt, os.path.join(ckpt_path, f'ckpt_best.ckpt'))
