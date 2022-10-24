import sys
import threading
import queue
import random
import collections
import itertools
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch._utils import ExceptionWrapper
from utils import worker


if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

class _MHDataSingleLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        self.dataset = loader.dataset
        self.scale = loader.scale
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        if len(self.scale) > 1 and self.dataset.train:
            idx_scale = random.randrange(0, len(self.scale))
            self.dataset.set_scale(idx_scale)

        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

class _MHDataMultiLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        self.dataset = loader.dataset
        self.scale = loader.scale
        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self.scale))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    # def _next_data(self):
    #     while True:
    #         # If the worker responsible for `self._rcvd_idx` has already ended
    #         # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
    #         # we try to advance `self._rcvd_idx` to find the next valid index.
    #         #
    #         # This part needs to run in the loop because both the `self._get_data()`
    #         # call and `_IterableDatasetStopIteration` check below can mark
    #         # extra worker(s) as dead.

    #         while self._rcvd_idx < self._send_idx:
    #             info = self._task_info[self._rcvd_idx]
    #             worker_id = info[0]
    #             if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
    #                 break
    #             del self._task_info[self._rcvd_idx]
    #             self._rcvd_idx += 1
    #         else:
    #             # no valid `self._rcvd_idx` is found (i.e., didn't break)
    #             if not self._persistent_workers:
    #                 self._shutdown_workers()
    #             raise StopIteration

    #         # Now `self._rcvd_idx` is the batch index we want to fetch

    #         # Check if the next sample has already been generated
    #         if len(self._task_info[self._rcvd_idx]) == 2:
    #             data = self._task_info.pop(self._rcvd_idx)[1]
    #             return self._process_data(data)

    #         assert not self._shutdown and self._tasks_outstanding > 0
    #         idx, data = self._get_data()
    #         self._tasks_outstanding -= 1
    #         if self._dataset_kind == _DatasetKind.Iterable:
    #             # Check for _IterableDatasetStopIteration
    #             if isinstance(data, _utils.worker._IterableDatasetStopIteration):
    #                 if self._persistent_workers:
    #                     self._workers_status[data.worker_id] = False
    #                 else:
    #                     self._mark_worker_as_unavailable(data.worker_id)
    #                 self._try_put_index()
    #                 continue

    #         if idx != self._rcvd_idx:
    #             # store out-of-order samples
    #             self._task_info[idx] += (data,)
    #         else:
    #             del self._task_info[idx]
    #             return self._process_data(data)

    # def _process_data(self, data):
    #     self._rcvd_idx += 1
    #     self._try_put_index()

    #     if isinstance(data, ExceptionWrapper):
    #         data.reraise()
    #     return data                

class MHDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=_utils.collate.default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MHDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    def __iter__(self):
        if self.num_workers == 0:
            return _MHDataSingleLoaderIter(self)
        else :
            return _MHDataMultiLoaderIter(self)
        # if len(self.scale) > 1 and self.dataset.train:
            # idx_scale = random.randrange(0, len(self.scale))
            # self.dataset.set_scale(idx_scale)
            # return super().__iter__()

