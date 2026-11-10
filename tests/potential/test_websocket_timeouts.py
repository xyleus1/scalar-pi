"""Finite transport waits are infrastructure controls, not robot success criteria."""

from openpi_client import websocket_client_policy as client
import pytest


def test_metadata_and_cold_warm_queries_have_distinct_finite_timeouts(monkeypatch):
    observed = []

    class Connection:
        def recv(self, *, timeout):
            observed.append(timeout)
            return {}

        def send(self, value):
            pass

    monkeypatch.setattr(client.websockets.sync.client, "connect", lambda *a, **kw: Connection())
    monkeypatch.setattr(client.msgpack_numpy, "unpackb", lambda value: value)
    policy = client.WebsocketClientPolicy(metadata_timeout=20, inference_timeout=30, cold_inference_timeout=90)
    policy.infer({})
    policy.infer({})
    assert 0 < observed[0] <= 20
    assert observed[1:] == [90, 30]


def test_metadata_timeout_closes_the_connection(monkeypatch):
    closed = []

    class Connection:
        def recv(self, *, timeout):
            raise TimeoutError("metadata")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(client.websockets.sync.client, "connect", lambda *a, **kw: Connection())
    with pytest.raises(TimeoutError, match="metadata"):
        client.WebsocketClientPolicy()
    assert closed == [True]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_transport_rejects_nonfinite_or_nonpositive_deadlines(value):
    with pytest.raises(ValueError, match="finite and positive"):
        client.WebsocketClientPolicy(metadata_timeout=value)
