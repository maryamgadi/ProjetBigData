"""Tests unitaires du Feature Store (sans Redis ni Spark) : python -m unittest discover tests"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streaming"))
import redis_sink as rs  # noqa: E402


class FakePipeline:
    def __init__(self, store):
        self.store, self.ops, self.executions = store, [], 0

    def hset(self, key, mapping):
        self.ops.append(("hset", key, mapping))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    def hgetall(self, key):
        self.ops.append(("hgetall", key))

    def execute(self):
        self.executions += 1
        results = []
        for op in self.ops:
            if op[0] == "hset":
                self.store.setdefault(op[1], {}).update(op[2])
                results.append(len(op[2]))
            elif op[0] == "expire":
                self.store.setdefault("__ttl__", {})[op[1]] = op[2]
                results.append(True)
            else:
                results.append(dict(self.store.get(op[1], {})))
        self.ops = []
        return results


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.last_pipeline = None

    def pipeline(self, transaction=False):
        self.last_pipeline = FakePipeline(self.store)
        return self.last_pipeline


class TestConversion(unittest.TestCase):
    def test_to_redis_value(self):
        self.assertIsNone(rs.to_redis_value(None))
        self.assertEqual(rs.to_redis_value(3), "3")
        self.assertEqual(rs.to_redis_value(1.23456), "1.23")
        self.assertEqual(rs.to_redis_value(True), "1")
        self.assertEqual(rs.to_redis_value(datetime(2026, 1, 1, 12, 0)), "2026-01-01T12:00:00")

    def test_to_mapping_excludes_id_and_none(self):
        mapping = rs.to_mapping({"nameOrig": "C1", "tx_count": 2, "amount_max": None})
        self.assertEqual(mapping, {"tx_count": "2"})


class TestWriteAndRead(unittest.TestCase):
    def test_write_records_sets_hash_and_ttl(self):
        client = FakeRedis()
        records = [
            {"nameOrig": "C1", "tx_count": 3, "amount_sum": 150.0},
            {"nameOrig": None, "tx_count": 1},       # ignoré : pas d'identifiant
            {"nameOrig": "C2", "tx_count": None},    # ignoré : aucun champ
        ]
        written = rs.write_records(client, records, rs.window_prefix("5m"), 300)
        self.assertEqual(written, 1)
        self.assertEqual(client.store["features:5m:C1"], {"tx_count": "3", "amount_sum": "150.0"})
        self.assertEqual(client.store["__ttl__"]["features:5m:C1"], 300)

    def test_write_records_chunks_pipeline(self):
        client = FakeRedis()
        records = [{"nameOrig": f"C{i}", "tx_count": i} for i in range(5)]
        rs.write_records(client, records, "account:", 60, chunk_size=2)
        self.assertEqual(client.last_pipeline.executions, 3)
        self.assertEqual(len([k for k in client.store if k.startswith("account:")]), 5)

    def test_get_features_merges_windows_with_defaults(self):
        client = FakeRedis()
        client.store["account:C1"] = {"last_type": "TRANSFER", "last_amount": "500.0"}
        client.store["features:5m:C1"] = {"tx_count": "2", "amount_sum": "800.5"}
        features = rs.get_features(client, "C1")

        self.assertTrue(features["known_account"])
        self.assertEqual(features["last_type"], "TRANSFER")
        self.assertEqual(features["last_amount"], 500.0)
        self.assertEqual(features["tx_count_5m"], 2.0)
        self.assertEqual(features["amount_sum_5m"], 800.5)
        self.assertEqual(features["tx_count_1h"], 0.0)   # fenêtre expirée -> 0
        self.assertEqual(features["tx_count_24h"], 0.0)
        self.assertEqual(set(rs.feature_names()) | {"known_account"}, set(features))

    def test_get_features_unknown_account(self):
        features = rs.get_features(FakeRedis(), "C404")
        self.assertFalse(features["known_account"])
        self.assertIsNone(features["last_amount"])
        self.assertEqual(features["tx_count_24h"], 0.0)


if __name__ == "__main__":
    unittest.main()

