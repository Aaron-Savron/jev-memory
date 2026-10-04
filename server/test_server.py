import asyncio
import os
import time
import unittest
from fastapi.testclient import TestClient
from app import create_app
from protocol import DecisionRequest, compile_items
from scheduler import Scheduler, Busy

TOKEN = "synthetic-test-token-only-0123456789"


class FakeBackend:
    identity = "synthetic-test-model"

    def prepare(self, request):
        if any(len(c.text) > 2000 for c in request.candidates):
            raise ValueError("Oversized sequence")
        return [c.model_dump() for c in request.candidates], len(request.candidates) * 10, 10

    def infer(self, groups):
        return [[{"id": c["id"], "score": 0.95} for c in group] for group in groups]


def request(deadline=500, operation="relevance"):
    return DecisionRequest(operation=operation, context="deployment", candidates=[{"id": "a", "text": "Uses systemd"}], deadlineMs=deadline)


class ApiTests(unittest.TestCase):
    def setUp(self):
        os.environ["JEV_MEMORY_TOKEN"] = TOKEN
        self.client = TestClient(create_app(FakeBackend))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, body, auth=True):
        return self.client.post("/v1/memory/decide", json=body, headers={"authorization": "Bearer " + TOKEN} if auth else {})

    def test_authentication_and_readiness(self):
        self.assertEqual(self.client.get("/ready").status_code, 401)
        self.assertEqual(self.post(request().model_dump(), False).status_code, 401)
        self.assertEqual(self.client.get("/ready", headers={"authorization": "Bearer " + TOKEN}).json()["model"], FakeBackend.identity)

    def test_response_uses_supplied_ids(self):
        result = self.post(request().model_dump())
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["decisions"], [{"id": "a", "score": 0.95}])

    def test_duplicate_ids_and_invalid_operations(self):
        value = request().model_dump(); value["candidates"] *= 2
        self.assertEqual(self.post(value).status_code, 422)
        value = request().model_dump(); value["operation"] = "write_memory"
        self.assertEqual(self.post(value).status_code, 422)

    def test_payload_limit(self):
        result = self.client.post("/v1/memory/decide", content=b"x" * 70000,
                                  headers={"authorization": "Bearer " + TOKEN})
        self.assertEqual(result.status_code, 413)

    def test_no_silent_token_truncation(self):
        value = request().model_dump(); value["candidates"][0]["text"] = "x" * 3000
        self.assertEqual(self.post(value).status_code, 422)

    def test_relationship_requires_previous_evidence(self):
        value = request().model_dump(); value["operation"] = "relationship"
        self.assertEqual(self.post(value).status_code, 422)

    def test_candidate_context_does_not_repeat_the_whole_collection(self):
        value = request().model_dump(); value["candidates"].append({"id": "b", "text": "Private unrelated fact"})
        captured = []
        def compile_request(state, questions):
            captured.append(state)
            return [questions]
        compile_items(DecisionRequest(**value), compile_request)
        self.assertNotIn("Private unrelated fact", str(captured[0]))
        self.assertNotIn("Uses systemd", str(captured[1]))


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_ignores_a_late_result(self):
        class Slow(FakeBackend):
            def infer(self, groups):
                time.sleep(0.06)
                return super().infer(groups)
        scheduler = Scheduler(Slow()); scheduler.start()
        try:
            with self.assertRaises(Busy):
                await scheduler.submit(request(5), "owner")
            await asyncio.sleep(0.08)
            self.assertEqual(scheduler.counts["owner"], 0)
        finally:
            await scheduler.close()

    async def test_per_owner_quota(self):
        scheduler = Scheduler(FakeBackend(), per_owner=1)
        first = asyncio.create_task(scheduler.submit(request(), "owner"))
        await asyncio.sleep(0)
        with self.assertRaises(Busy):
            await scheduler.submit(request(), "owner")
        scheduler.start()
        try:
            self.assertEqual((await first)["model"], FakeBackend.identity)
        finally:
            await scheduler.close()

    async def test_microbatch_results_remain_aligned(self):
        scheduler = Scheduler(FakeBackend()); scheduler.start()
        try:
            results = await asyncio.gather(*(scheduler.submit(request(), str(i)) for i in range(8)))
            self.assertEqual(len(results), 8)
            self.assertTrue(all(r["decisions"][0]["id"] == "a" for r in results))
        finally:
            await scheduler.close()


if __name__ == "__main__":
    unittest.main()
