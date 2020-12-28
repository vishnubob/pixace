import multiprocessing as mp
import threading

class WorkQueues(object):
    __quemap__ = {}

    def __init__(self, maxsize=None, group=None):
        (self.todo, self.done) = self._bind_queues(maxsize=maxsize, group=group)

    @classmethod
    def _bind_queues(cls, maxsize=None, group=None):
        group = group or "__default__"
        if group not in cls.__quemap__:
            todo = mp.Queue(maxsize)
            done = mp.Queue(maxsize)
            cls.__quemap__[group] = (todo, done)
        return cls.__quemap__[group]

class QueueWorker(mp.Process):
    def __init__(self, group=None, worker_id=None):
        super().__init__()
        self.daemon = True
        self.group = group
        self.worker_id = worker_id
        self._running = False
        self.queues = WorkQueues(group=self.group)

    def get_work(self):
        return self.queues.todo.get()

    def return_result(self, job):
        self.queues.done.put(job)

    def process_job(self, job):
        return job

    def loop(self):
        job = self.get_job_new()
        result = self.process(job)
        self.return_result(result)

    def run(self):
        msg = f"{self.__class__.__name__}({self.worker_id}) Stating {self.group} worker loop ({id(self.queues.todo)})"
        print(msg)
        self._running = True
        while self._running:
            self.loop()

class BatchWorker(QueueWorker):
    def __init__(self, batch_size=None, **kw):
        super().__init__(**kw)
        self.batch_size = batch_size
        self._current_batch = []

    def return_result(self, job):
        self._current_batch.append(job)
        if len(self._current_batch) == self.batch_size:
            batch = [np.vstack(it) for it in zip(*self._current_batch)]
            super().return_batch(batch)
            self._current_batch = []

class DatasetGenerator(threading.Thread):
    DefaultWorkerClass = QueueWorker

    def __init__(self, data, group=None, n_workers=4, qsize=64, worker_class=None, worker_config=None):
        super().__init__()
        self.daemon = True
        self.data = data
        self.group = group
        self.ques = DataQueues(qsize, group=self.group)
        self.n_workers = n_workers
        self.worker_class = worker_class or self.DefaultWorkerClass
        self.worker_config = worker_config or dict()
        self._workers = self.build_workers()
        self._running = False

    def build_workers(self):
        workers = []
        for worker_id in range(self.n_workers):
            conf = self.worker_config.copy()
            conf["worker_id"] = worker_id
            conf["group"] = self.group
            worker = DataWorker(**conf)
            worker.start()
            workers.append(worker)
        return workers

    def run(self):
        msg = f"{self.__class__.__name__} Stating dispatch loop ({id(self.ques.todo)})"
        self._running = True
        for item in self.data:
            self.ques.todo.put(item)
            if not self._running:
                break

    def __iter__(self):
        # XXX: this can leave data in the queue
        self.start()
        while self._running:
            yield self.ques.done.get()
        raise StopIteration
