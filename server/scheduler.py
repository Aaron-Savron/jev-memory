"""Bounded foreground-priority scheduler with deadlines and per-token quotas."""
import asyncio
import itertools
import time
from dataclasses import dataclass


class Busy(Exception):
    pass


@dataclass
class Job:
    records: list
    tokens: int
    deadline: float
    owner: str
    future: asyncio.Future
    started: float


class Scheduler:
    def __init__(self, backend, capacity=64, per_owner=8, batch_tokens=16384):
        self.backend = backend
        self.queue = asyncio.PriorityQueue(maxsize=capacity)
        self.counts = {}
        self.per_owner = per_owner
        self.batch_tokens = batch_tokens
        self.sequence = itertools.count()
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def close(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            _, _, job = self.queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(Busy("Service stopping"))

    async def submit(self, request, owner):
        if self.queue.full() or self.counts.get(owner, 0) >= self.per_owner:
            raise Busy("Queue full")
        started = time.monotonic()
        records, tokens, _ = self.backend.prepare(request)
        if tokens > self.batch_tokens:
            raise ValueError("Request exceeds the batch token budget")
        future = asyncio.get_running_loop().create_future()
        job = Job(records, tokens, started + request.deadlineMs / 1000, owner, future, started)
        self.counts[owner] = self.counts.get(owner, 0) + 1
        try:
            self.queue.put_nowait((0 if request.operation == "relevance" else 1, next(self.sequence), job))
            return await asyncio.wait_for(future, max(0.001, job.deadline - time.monotonic()))
        except asyncio.TimeoutError as error:
            raise Busy("Deadline exceeded") from error
        finally:
            future.cancel()
            self.counts[owner] -= 1

    async def run(self):
        while True:
            first = await self.queue.get()
            jobs = [first[2]]
            # Delay only up to 10 ms, and never intentionally exceed the first deadline.
            await asyncio.sleep(max(0, min(0.01, jobs[0].deadline - time.monotonic())))
            tokens = jobs[0].tokens
            while not self.queue.empty():
                entry = self.queue.get_nowait()
                if tokens + entry[2].tokens > self.batch_tokens:
                    self.queue.put_nowait(entry)
                    break
                jobs.append(entry[2]); tokens += entry[2].tokens
            active = []
            for job in jobs:
                if job.future.done():
                    continue
                if time.monotonic() >= job.deadline:
                    job.future.set_exception(Busy("Deadline exceeded"))
                else:
                    active.append(job)
            if not active:
                continue
            started = time.monotonic()
            try:
                results = await asyncio.to_thread(self.backend.infer, [job.records for job in active])
                if len(results) != len(active):
                    raise RuntimeError("Incomplete backend response")
                for job, decisions in zip(active, results):
                    if job.future.done():
                        continue
                    if time.monotonic() >= job.deadline:
                        job.future.set_exception(Busy("Deadline exceeded"))
                    else:
                        job.future.set_result({"model": self.backend.identity, "decisions": decisions,
                                               "timing": {"queueMs": (started-job.started)*1000,
                                                          "inferenceMs": (time.monotonic()-started)*1000}})
            except Exception:
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(Busy("Inference unavailable"))
