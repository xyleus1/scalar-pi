import logging
import math
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        metadata_timeout: float = 60.0,
        inference_timeout: float = 120.0,
        cold_inference_timeout: float = 900.0,
    ) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (metadata_timeout, inference_timeout, cold_inference_timeout)
        ):
            raise ValueError("WebSocket timeouts must be finite and positive")
        self._metadata_timeout = metadata_timeout
        self._inference_timeout = inference_timeout
        self._cold_inference_timeout = cold_inference_timeout
        self._first_query = True
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._metadata_timeout
        while time.monotonic() < deadline:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=max(0.001, deadline - time.monotonic()),
                )
                try:
                    metadata = msgpack_numpy.unpackb(conn.recv(timeout=max(0.001, deadline - time.monotonic())))
                except BaseException:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(min(5, max(0, deadline - time.monotonic())))
        raise TimeoutError("Policy server metadata deadline exceeded")

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        timeout = self._cold_inference_timeout if self._first_query else self._inference_timeout
        response = self._ws.recv(timeout=timeout)
        self._first_query = False
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
