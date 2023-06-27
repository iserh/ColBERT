import os
import random
import torch
import numpy as np
from logging import getLogger

logger = getLogger("colbert")

ALREADY_INITALIZED = False

# TODO: Consider torch.distributed.is_initialized() instead


def init(rank, gpus):
    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)
    is_distributed = (nranks > 1) or ('WORLD_SIZE' in os.environ)

    global ALREADY_INITALIZED
    if ALREADY_INITALIZED:
        return nranks, is_distributed

    ALREADY_INITALIZED = True

    if is_distributed:
        if len(gpus) > 0:
            logger.info(f'nranks = {nranks} \t gpus = {gpus} \t device={gpus[rank]}')

            assert torch.cuda.device_count() >= len(gpus), f"{gpus} not visible to pytorch"

            torch.cuda.set_device(gpus[rank])
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        else:
            torch.distributed.init_process_group(backend='gloo', init_method='env://')

    return nranks, is_distributed


def barrier(rank):
    nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
    nranks = max(1, nranks)

    if rank >= 0 and nranks > 1:
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            torch.distributed.barrier(device_ids=[gpus[rank]])
        else:
            torch.distributed.barrier()
