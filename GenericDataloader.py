import torch

class SkipIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, data_iterator, args):
        self.bs = batch_size
        self.iterator = data_iterator
        self.args = args
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.iterator(start_value=None, batch_size=None, skip_value=None, *self.args)
        else:
            worker_id = worker_info.id # these are 0-indexed
            num_workers = worker_info.num_workers
            start_value = worker_id * self.bs
            skip_value = ((num_workers) * self.bs)
            return self.iterator(start_value, skip_value, self.bs, *self.args)


# return True if you are not skipping, False otherwise
def skipFunction(current_index, start_value, batch_size, skip_value):
    if start_value is None:
        return True

    elif current_index < start_value:
        return False

    elif start_value <= current_index < start_value + batch_size:
        return True

    remainder = current_index % batch_size
    current_interval_begin = current_index-remainder
    LHS = current_interval_begin-start_value
    RHS = skip_value

    if LHS % RHS == 0:
        return True
    else:
        return False


def sampleIterator(start_value, batch_size, skip_value, *args):
    for i in range(200):
        if skipFunction(i, start_value, batch_size, skip_value):
            yield i


def getSkipIterableDataLoader(bs, iterator, args, num_workers, collate_func, pinned_memory, persistent_workers, prefetch_factor):
    ds = SkipIterableDataset(batch_size=bs, data_iterator=iterator, args=args)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=num_workers, collate_fn=collate_func, pin_memory=pinned_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return dl

"""
if __name__ == "__main__":
    func = sampleIterator
    bs = 5
    ds = SkipIterableDataset(batch_size=bs, data_iterator=func, args=[])
    dl = getSkipIterableDataLoader(ds=ds, bs=bs, num_workers=5, collate_func=None, pinned_memory=False, persistent_workers=True)
"""


class RepeatedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_iterator, args):
        self.iterator = data_iterator
        self.args = args

    def __iter__(self):
        return self.iterator(*self.args)

def getRepeatedIterableDataLoader(bs, iterator, args, num_workers, collate_func, pinned_memory, persistent_workers):
    ds = RepeatedIterableDataset(data_iterator=iterator, args=args)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=num_workers, collate_fn=collate_func, pin_memory=pinned_memory, persistent_workers=persistent_workers)
    return dl