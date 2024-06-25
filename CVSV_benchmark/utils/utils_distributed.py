import torch
import torch.distributed as dist
from collections import Iterable


def flatten(a):
    """
    :param a: list
    :return: iterator of items in a
        print(list(flatten(a)))

    """
    for each in a:
        if not isinstance(each, Iterable) or isinstance(each, str):
            yield each
        else:
            yield from flatten(each)

def flatten2(l):
    return [item for sublist in l for item in sublist]


def all_gather_concat(tensor, num_total_examples=-1):
    torch.cuda.synchronize()
    output_tensors = [torch.zeros_like(tensor).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    if num_total_examples != -1:
        return concat[:num_total_examples]
    else:
        return concat

def all_reduce_mean(x):
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= float(world_size)
        return x_reduce.item()
    else:
        return x

def all_reduce_sum(x):
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        return x_reduce.item()
    else:
        return x

@torch.no_grad()
def all_gather_object(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    gather_data = flatten2(gather_data)
    return gather_data

