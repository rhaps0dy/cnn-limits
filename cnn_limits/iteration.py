from torch.utils.data import DataLoader, Subset
import numpy as np
import itertools
import time

__all__ = ('create_h5py_dataset', 'ProductIterator', 'DiagIterator', 'PrintTimings')


def create_h5py_dataset(f, batch_size, name, diag, N, N2, dtype=np.float32):
    """
    Creates a dataset named `name` on `f`, with chunks of size `batch_size`.
    The chunks have leading dimension 1, so as to accommodate future resizing
    of the leading dimension of the dataset (which starts at 1).
    """
    if diag:
        chunk_shape = (1, min(batch_size, N))
        shape = (1, N)
        maxshape = (None, N)
    else:
        chunk_shape = (1, min(batch_size, N), min(batch_size, N2))
        shape = (1, N, N2)
        maxshape = (None, N, N2)
    return f.create_dataset(name, shape=shape, dtype=dtype,
                            fillvalue=np.nan, chunks=chunk_shape,
                            maxshape=maxshape)


def _product_generator(N_batches_X, N_batches_X2, same):
    for i in range(N_batches_X):
        for j in range(0, i if same else N_batches_X2):
            yield (False, i, j)
        if same:
            yield (True, i, i)


def _round_up_div(a, b):
    return (a+b-1)//b


class ProductIterator:
    """Returns an iterator for loading data from both X and X2. It divides the
    load equally among `n_workers`, returning only the one that belongs to
    `worker_rank`.

    if `X2` is None, it only iterates over the lower-triangular part of the
    kernel matrix of the data set.
    """
    def __init__(self, batch_size, X, X2=None):
        N_batches_X = _round_up_div(len(X), batch_size)
        if X2 is None:
            same = True
            X2 = X
            N_batches_X2 = N_batches_X
            N_batches = max(1, N_batches_X * (N_batches_X+1) // 2)
        else:
            same = False
            N_batches_X2 = _round_up_div(len(X2), batch_size)
            N_batches = N_batches_X * N_batches_X2

        self.idx_iter = _product_generator(N_batches_X, N_batches_X2, same)
        self.prev_j = -2  # this + 1 = -1, which is not a valid j
        self.X_loader = None
        self.X2_loader = None
        self.x_batch = None
        self.X = X
        self.X2 = X2
        self.same = same
        self.batch_size = batch_size
        self.N_batches = N_batches

    def __len__(self):
        return self.N_batches

    def __iter__(self):
        return self

    def dataloader_for(self, dataset):
        return iter(DataLoader(dataset, batch_size=self.batch_size))

    def __next__(self):
        same, i, j = next(self.idx_iter)

        if self.X_loader is None:
            assert i==0
            self.X_loader = self.dataloader_for(self.X)

        if j != self.prev_j+1:
            assert j==0
            self.X2_loader = self.dataloader_for(self.X2)
            self.x_batch = next(self.X_loader)
        self.prev_j = j

        return (same,
                (i*self.batch_size, self.x_batch),
                (j*self.batch_size, next(self.X2_loader)))


class DiagIterator:
    def __init__(self, batch_size, X, X2=None):
        self.batch_size = batch_size
        dl = DataLoader(X, batch_size=batch_size)
        if X2 is None:
            self.same = True
            self.it = iter(enumerate(dl))
            self.length = len(dl)
        else:
            dl2 = DataLoader(X2, batch_size=batch_size)
            self.same = False
            self.it = iter(enumerate(zip(dl, dl2)))
            self.length = min(len(dl), len(dl2))

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.same:
            i, xy = next(self.it)
            xy2 = xy
        else:
            i, xy, xy2 = next(self.it)
        ib = i*self.batch_size
        return (self.same, (ib, xy), (ib, xy2))


class PrintTimings:
    """Prints the elapsed and total time for an iterator. Like tqdm, but does not
    overwrite parts of `cout`.

    The total time calculation only uses the time taken for the last iteration,
    so it is only good for relatively slow iterations with very regular
    workloads.

    The attributes can be modified during runtime
    `desc`: a description string for the periodic prints
    `print_interval`: the minimum time between prints
    `data`: a list of (key, value) to be added to the printed information
    """
    @staticmethod
    def hhmmss(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h == 0.0:
            return f"{m:02d}:{s:02d}"
        else:
            return f"{h:02d}:{m:02d}:{s:02d}"

    def __init__(self, desc="time", print_interval=2.):
        self.desc = desc
        self.print_interval = print_interval
        self.data = []

    def __call__(self, iterator, total=None):
        """
        Prints the current total number of iterations, speed of iteration, and
        elapsed time.

        Meant as a rudimentary replacement for `tqdm` that prints a new line at
        each iteration, and thus can be used in multiple parallel processes in the
        same terminal.
        """
        start_time = time.perf_counter()
        if total is None:
            total = len(iterator)
        last_printed = -self.print_interval
        prev_i = -1
        for i, value in enumerate(iterator):
            yield value
            cur_time = time.perf_counter()
            elapsed = cur_time - start_time
            if elapsed > last_printed + self.print_interval:
                it_s = (i - prev_i)/(elapsed - last_printed)
                it_left = total - i
                total_s = elapsed + it_left/it_s

                print((f"{self.desc}: {i+1}/{total} it, {it_s:.02f} it/s, "
                       f"[{self.hhmmss(elapsed)}<{self.hhmmss(total_s)}] ")
                      + ", ".join(f"{k}={v}" for k, v in self.data))
                last_printed = elapsed
                prev_i = i
