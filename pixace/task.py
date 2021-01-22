import json
import traceback
import threading
from functools import partial
import multiprocessing as mp
from queue import Full, Empty
import numpy as np

class WorkQueues(object):
    __quemap__ = {}

    class StopWork(BaseException):
        pass

    def __init__(self, maxsize=None, group=None, timeout=1):
        (self.todo, self.done, self._running) = self._bind_queues(maxsize=maxsize, group=group)
        self.qmap = {"todo": self.todo, "done": self.done}
        self.timeout = timeout

    def get_running(self):
        return bool(self._running.value)
    def set_running(self, val):
        self._running.value = int(val)
    running = property(get_running, set_running)

    def stop(self):
        self.running = False
        self.flush()

    @classmethod
    def _bind_queues(cls, maxsize=None, group=None):
        group = group or "__default__"
        if group not in cls.__quemap__:
            running = mp.Value('B', 1)
            todo = mp.Queue(maxsize)
            done = mp.Queue(maxsize)
            cls.__quemap__[group] = (todo, done, running)
        return cls.__quemap__[group]

    def flush(self):
        for key in self.qmap:
            que = self.qmap[key]
            while not que.empty():
                try:
                    que.get_nowait()
                except Empty:
                    break

    def put(self, qname, item):
        que = self.qmap[qname]
        while self.running:
            try:
                que.put(item, True, self.timeout)
                break
            except Full:
                continue
        if not self.running:
            raise self.StopWork

    def get(self, qname):
        que = self.qmap[qname]
        while self.running:
            try:
                item = que.get(True, self.timeout)
                break
            except Empty:
                continue
        if not self.running:
            raise self.StopWork
        return item

class QueueTask(mp.Process):
    def __init__(self, group=None, worker_id=None, timeout=1):
        super().__init__()
        self.daemon = True
        self.group = group
        self.worker_id = worker_id
        self.timeout = timeout
        self.queues = WorkQueues(group=self.group)

    def get_work(self):
        return self.queues.get("todo")

    def return_result(self, job):
        self.queues.put("done", job)

    def process_job(self, job):
        return job

    def loop(self):
        job = self.get_work()
        result = self.process_job(job)
        self.return_result(result)

    def run(self):
        msg = f"{self.__class__.__name__}({self.worker_id}) Stating {self.group} worker loop ({id(self.queues.todo)})"
        print(msg)
        while self.queues.running:
            msg = f"{self.__class__.__name__}({self.worker_id}) loop"
            #print(msg)
            try:
                self.loop()
            except self.queues.StopWork:
                break
            except:
                traceback.print_exc()
                self.queues.stop()
                raise
        msg = f"{self.__class__.__name__}({self.worker_id}) Stopping {self.group} worker loop ({id(self.queues.todo)})"
        print(msg)
        self.queues.flush()

class BatchTask(QueueTask):
    def __init__(self, batch_size=None, **kw):
        super().__init__(**kw)
        assert batch_size is not None
        self.batch_size = batch_size
        self._current_batch = []

    def return_result(self, job):
        self._current_batch.append(job)
        assert len(self._current_batch) <= self.batch_size, \
            f"{len(self._current_batch)} > {self.batch_size}"
        if len(self._current_batch) == self.batch_size:
            batch = [np.vstack(it) for it in zip(*self._current_batch)]
            assert batch[0].shape[0] == self.batch_size
            super().return_result(batch)
            self._current_batch = []

class DatasetGenerator(threading.Thread):
    def __init__(self, data, group=None, n_workers=4, qsize=64, worker_ctor=None):
        super().__init__()
        self.daemon = True
        self.data = data
        self.group = group
        self.ques = WorkQueues(qsize, group=self.group)
        self.n_workers = n_workers
        self.worker_ctor = worker_ctor
        self._workers = self.build_workers()

    def build_workers(self):
        workers = []
        for worker_id in range(self.n_workers):
            worker = self.worker_ctor(
                worker_id=worker_id,
                group=self.group
            )
            workers.append(worker)
            worker.start()
        return workers

    def run(self):
        msg = f"{self.__class__.__name__} Stating dispatch loop ({id(self.ques.todo)})"
        print(msg)
        for item in self.data:
            try:
                self.ques.put("todo", item)
            except self.ques.StopWork:
                break
        self.ques.flush()
        msg = f"{self.__class__.__name__} Stopping dispatch loop ({id(self.ques.todo)})"
        print(msg)

    def stop(self):
        if not self.ques.running:
            return
        self.ques.stop()
        #for worker in self._workers:
            #worker.join()
        #if self.is_alive():
            #self.join()

    def __iter__(self):
        self.start()
        try:
            while self.ques.running:
                try:
                    yield self.ques.get("done")
                except self.ques.StopWork:
                    break
        finally:
            self.stop()
        raise StopIteration

class TokenizeTask(BatchTask):
    def __init__(self, tokenizer=None, **kw):
        super().__init__(**kw)
        self.tokenizer = tokenizer

    def process_job(self, job):
        work = self.tokenizer.encode(job)
        work = np.array(work).astype(np.int32)
        weights = (work != 0).astype(np.float)
        assert work.shape == weights.shape
        return (work, work, weights)

class TaskManager(object):
    def __init__(self, data=None, worker_ctor=None, group=None, seed=0):
        self.data = tuple(data)
        self.group = group
        self.seed = seed
        self.worker_ctor = worker_ctor
        self.dsgen = DatasetGenerator(
            data=self.shuffle_forever(self.data),
            group=self.group,
            worker_ctor=self.worker_ctor,
        )

    def __iter__(self):
        return iter(self.dsgen)
    
    def __del__(self):
        self.stop()

    def stop(self):
        self.dsgen.stop()

    def shuffle_forever(self, items):
        assert len(items) > 0
        item_order = np.arange(len(items), dtype=np.int32)
        
        next_seed = self.seed
        while True:
            np.random.seed(next_seed)
            next_seed = np.random.randint(2 ** 31)
            np.random.shuffle(item_order)

            for idx in item_order:
                yield items[idx]

def batch_generator(json_file=None, group=None, task=TokenizeTask, **kw):
    with open(json_file) as fh:
        data = json.load(fh)

    worker_ctor = partial(task, **kw)

    task_manager = TaskManager(
        data=data, 
        group=group,
        worker_ctor=worker_ctor
    )
    
    return task_manager
