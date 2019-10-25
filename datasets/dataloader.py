import random
import torch
import time

class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indexes)

        self.i = 0

    def __iter__(self):
        return self

    def np2torch(self, samples):
        samples_ret = []
        for sample in samples:
            if type(sample) != type([]) and sample is not None and type(sample) != type({}):
                sample = torch.from_numpy(sample).unsqueeze(0)
            samples_ret.append(sample)
        return samples_ret

    def get_sample(self, index):
        return self.np2torch(self.dataset[index])

    def _build_batch(self, batch):
        ret_batch = list(map(list, zip(*batch)))
        for i in range(len(ret_batch) - 1):
            ret_batch[i] = torch.cat(ret_batch[i], 0)
        ret_batch[-1] = self.cat_batch(ret_batch[-1])
        return ret_batch

    def cat_sparse_matrices(self, matrices):
        t_i = []
        t_v = []
        for i, matrix in enumerate(matrices):
            t_i.append(matrix._indices() + i * matrix._values().shape[0])
            t_v.append(matrix._values())
        t_i = torch.cat(t_i, 1).transpose(0, 1)
        t_v = torch.cat(t_v)
        return torch.sparse.FloatTensor(t_i.t(), t_v)

    def cat_batch(self, batch_adjs):
        adjs_batch = list(map(list, zip(*batch_adjs)))
        adjs = []
        for adj_batch in adjs_batch:
            adj = self.cat_sparse_matrices(adj_batch)
            adjs.append(adj)
        return adjs

    def next_sample(self):
        sample = self.get_sample(self.indexes[self.i])
        self.i += 1
        return sample

    def __next__(self): # Python 3: def __next__(self)

        if self.i >= len(self.indexes):
            self.i = 0
            if self.shuffle:
                random.shuffle(self.indexes)
            raise StopIteration
        else:
            return self.next_sample()

    def __len__(self):
        return len(self.dataset)
