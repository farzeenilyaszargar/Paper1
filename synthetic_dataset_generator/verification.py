"""Bounded process-based classical checks; Z3 contexts never cross threads."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from .logic import classical_label


def _verify_batch(batch):
    for facts, rules, query, target in batch:
        if classical_label(facts, rules, query) != target:
            raise ValueError(f"Classical solver disagrees with label for query {query}")


class SolverVerifier:
    def __init__(self, workers: int):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.executor = None
        self.batch = []
        self.pending = deque()

    def __enter__(self):
        if self.workers > 1:
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"))
        return self

    def check(self, facts, rules, query, target):
        item = (facts, rules, query, target)
        if self.executor is None:
            _verify_batch([item])
            return
        self.batch.append(item)
        if len(self.batch) >= 256:
            self._submit()

    def _submit(self):
        if self.batch:
            self.pending.append(self.executor.submit(_verify_batch, self.batch))
            self.batch = []
        if len(self.pending) >= 2 * self.workers:
            self.pending.popleft().result()

    def finish(self):
        if self.executor is not None:
            self._submit()
            while self.pending:
                self.pending.popleft().result()

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.finish()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
