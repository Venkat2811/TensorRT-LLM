import asyncio
import hashlib
import hmac
import os
import pickle  # nosec B403
import threading
import time
import traceback
from queue import Queue
from typing import Any, Optional

import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import nvtx_mark, nvtx_range_debug
from ..llmapi.utils import (ManagedThread, enable_llm_debug, logger_debug,
                            print_colored)


_MYELON_INSTRUMENT = os.environ.get('TRTLLM_MYELON_INSTRUMENT', '0') == '1'

# Use time.perf_counter_ns for nanosecond resolution when available
_perf_ns = time.perf_counter_ns


class _IpcStats:
    """Nanosecond-granular IPC timing accumulator.

    Tracks per-phase breakdown: pickle, hmac, zmq_send/recv, unpickle.
    Also tracks payload sizes in bytes. All active only when TRTLLM_MYELON_INSTRUMENT=1.
    """

    __slots__ = (
        'name',
        # put (send) stats — all in nanoseconds
        'put_count', 'put_total_ns', 'put_max_ns', 'put_min_ns',
        'pickle_total_ns', 'pickle_max_ns',
        'hmac_sign_total_ns', 'hmac_sign_max_ns',
        'zmq_send_total_ns', 'zmq_send_max_ns',
        'put_bytes_total', 'put_bytes_max',
        # get (recv) stats — all in nanoseconds
        'get_count', 'get_total_ns', 'get_max_ns', 'get_min_ns',
        'zmq_recv_total_ns', 'zmq_recv_max_ns',
        'hmac_verify_total_ns', 'hmac_verify_max_ns',
        'unpickle_total_ns', 'unpickle_max_ns',
        'get_bytes_total', 'get_bytes_max',
        # async variants
        'put_async_count', 'put_async_total_ns',
        'get_async_count', 'get_async_total_ns',
    )

    def __init__(self, name: str):
        self.name = name
        self.put_count = 0
        self.put_total_ns = 0
        self.put_max_ns = 0
        self.put_min_ns = 2**63
        self.pickle_total_ns = 0
        self.pickle_max_ns = 0
        self.hmac_sign_total_ns = 0
        self.hmac_sign_max_ns = 0
        self.zmq_send_total_ns = 0
        self.zmq_send_max_ns = 0
        self.put_bytes_total = 0
        self.put_bytes_max = 0
        self.get_count = 0
        self.get_total_ns = 0
        self.get_max_ns = 0
        self.get_min_ns = 2**63
        self.zmq_recv_total_ns = 0
        self.zmq_recv_max_ns = 0
        self.hmac_verify_total_ns = 0
        self.hmac_verify_max_ns = 0
        self.unpickle_total_ns = 0
        self.unpickle_max_ns = 0
        self.get_bytes_total = 0
        self.get_bytes_max = 0
        self.put_async_count = 0
        self.put_async_total_ns = 0
        self.get_async_count = 0
        self.get_async_total_ns = 0

    def record_put(self, total_ns: int, pickle_ns: int, hmac_ns: int,
                   zmq_ns: int, payload_bytes: int):
        self.put_count += 1
        self.put_total_ns += total_ns
        if total_ns > self.put_max_ns:
            self.put_max_ns = total_ns
        if total_ns < self.put_min_ns:
            self.put_min_ns = total_ns
        self.pickle_total_ns += pickle_ns
        if pickle_ns > self.pickle_max_ns:
            self.pickle_max_ns = pickle_ns
        self.hmac_sign_total_ns += hmac_ns
        if hmac_ns > self.hmac_sign_max_ns:
            self.hmac_sign_max_ns = hmac_ns
        self.zmq_send_total_ns += zmq_ns
        if zmq_ns > self.zmq_send_max_ns:
            self.zmq_send_max_ns = zmq_ns
        self.put_bytes_total += payload_bytes
        if payload_bytes > self.put_bytes_max:
            self.put_bytes_max = payload_bytes

    def record_get(self, total_ns: int, zmq_ns: int, hmac_ns: int,
                   unpickle_ns: int, payload_bytes: int):
        self.get_count += 1
        self.get_total_ns += total_ns
        if total_ns > self.get_max_ns:
            self.get_max_ns = total_ns
        if total_ns < self.get_min_ns:
            self.get_min_ns = total_ns
        self.zmq_recv_total_ns += zmq_ns
        if zmq_ns > self.zmq_recv_max_ns:
            self.zmq_recv_max_ns = zmq_ns
        self.hmac_verify_total_ns += hmac_ns
        if hmac_ns > self.hmac_verify_max_ns:
            self.hmac_verify_max_ns = hmac_ns
        self.unpickle_total_ns += unpickle_ns
        if unpickle_ns > self.unpickle_max_ns:
            self.unpickle_max_ns = unpickle_ns
        self.get_bytes_total += payload_bytes
        if payload_bytes > self.get_bytes_max:
            self.get_bytes_max = payload_bytes

    def record_put_async(self, total_ns: int):
        self.put_async_count += 1
        self.put_async_total_ns += total_ns

    def record_get_async(self, total_ns: int):
        self.get_async_count += 1
        self.get_async_total_ns += total_ns

    def _avg(self, total: int, count: int) -> int:
        return total // count if count > 0 else 0

    def _ns_to_str(self, ns: int) -> str:
        """Format nanoseconds as the most readable unit."""
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.1f}ms"
        if ns >= 1_000:
            return f"{ns / 1_000:.1f}us"
        return f"{ns}ns"

    def dump(self):
        if self.put_count == 0 and self.get_count == 0 and self.put_async_count == 0 and self.get_async_count == 0:
            return
        lines = [f"[MyelonInstr] === {self.name} IPC Stats ==="]
        if self.put_count > 0:
            avg = self._avg(self.put_total_ns, self.put_count)
            avg_bytes = self._avg(self.put_bytes_total, self.put_count)
            lines.append(
                f"[MyelonInstr]   put(sync): n={self.put_count} "
                f"avg={self._ns_to_str(avg)} min={self._ns_to_str(self.put_min_ns)} max={self._ns_to_str(self.put_max_ns)} "
                f"avg_bytes={avg_bytes} max_bytes={self.put_bytes_max}"
            )
            lines.append(
                f"[MyelonInstr]     pickle:   avg={self._ns_to_str(self._avg(self.pickle_total_ns, self.put_count))} "
                f"max={self._ns_to_str(self.pickle_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     hmac:     avg={self._ns_to_str(self._avg(self.hmac_sign_total_ns, self.put_count))} "
                f"max={self._ns_to_str(self.hmac_sign_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     zmq_send: avg={self._ns_to_str(self._avg(self.zmq_send_total_ns, self.put_count))} "
                f"max={self._ns_to_str(self.zmq_send_max_ns)}"
            )
        if self.get_count > 0:
            avg = self._avg(self.get_total_ns, self.get_count)
            avg_bytes = self._avg(self.get_bytes_total, self.get_count)
            lines.append(
                f"[MyelonInstr]   get(sync): n={self.get_count} "
                f"avg={self._ns_to_str(avg)} min={self._ns_to_str(self.get_min_ns)} max={self._ns_to_str(self.get_max_ns)} "
                f"avg_bytes={avg_bytes} max_bytes={self.get_bytes_max}"
            )
            lines.append(
                f"[MyelonInstr]     zmq_recv: avg={self._ns_to_str(self._avg(self.zmq_recv_total_ns, self.get_count))} "
                f"max={self._ns_to_str(self.zmq_recv_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     hmac_v:   avg={self._ns_to_str(self._avg(self.hmac_verify_total_ns, self.get_count))} "
                f"max={self._ns_to_str(self.hmac_verify_max_ns)}"
            )
            lines.append(
                f"[MyelonInstr]     unpickle: avg={self._ns_to_str(self._avg(self.unpickle_total_ns, self.get_count))} "
                f"max={self._ns_to_str(self.unpickle_max_ns)}"
            )
        if self.put_async_count > 0:
            lines.append(
                f"[MyelonInstr]   put(async): n={self.put_async_count} "
                f"avg={self._ns_to_str(self._avg(self.put_async_total_ns, self.put_async_count))}"
            )
        if self.get_async_count > 0:
            lines.append(
                f"[MyelonInstr]   get(async): n={self.get_async_count} "
                f"avg={self._ns_to_str(self._avg(self.get_async_total_ns, self.get_async_count))}"
            )
        for line in lines:
            logger.info(line)


class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
        zmq.ROUTER: "ROUTER",
        zmq.DEALER: "DEALER",
    }

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 use_hmac_encryption: bool = True):
        '''
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address (tcp-ip_port, hmac_auth_key) for the IPC. Defaults to None. If hmac_auth_key is None and use_hmac_encryption is False, the queue will not use HMAC encryption.
            socket_type (int): The type of socket to use. Defaults to zmq.PAIR.
            is_server (bool): Whether the current process is the server or the client.
            is_async (bool): Whether to use asyncio for the socket. Defaults to False.
            name (str, optional): The name of the queue. Defaults to None.
            use_hmac_encryption (bool): Whether to use HMAC encryption for pickled data. Defaults to True.
        '''

        self._stats = _IpcStats(name or "unnamed") if _MYELON_INSTRUMENT else None
        self.socket_type = socket_type
        self.address_endpoint = address[
            0] if address is not None else "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket = self.context.socket(socket_type)
        self.socket.set_hwm(0)

        # For ROUTER sockets, track the last identity to enable replies. For now we assume there is only one client in our case.
        self._last_identity = None

        self.hmac_key = address[1] if address is not None else None
        self.use_hmac_encryption = use_hmac_encryption

        self._setup_lock = threading.Lock()

        # Thread safety debugging
        self._zmq_thread_id = None
        self._zmq_debug_enabled = os.environ.get('TLLM_LLMAPI_ZMQ_DEBUG',
                                                 '0') != '0'

        # Check HMAC key condition
        if self.use_hmac_encryption and not self.is_server and self.hmac_key is None:
            raise ValueError(
                "Client must receive HMAC key when encryption is enabled")
        elif not self.use_hmac_encryption and self.hmac_key is not None:
            raise ValueError(
                "Server and client should not receive HMAC key when encryption is disabled"
            )

        if self.should_bind_socket():
            self.socket.bind(
                self.address_endpoint
            )  # Binds to the address and occupy a port immediately
            self.address_endpoint = self.socket.getsockopt(
                zmq.LAST_ENDPOINT).decode()
            logger_debug(
                f"Server [{name}] bound to {self.address_endpoint} in {self.socket_type_str[socket_type]}\n",
                "green")

            if self.use_hmac_encryption and not self.hmac_key:
                # Initialize HMAC key for pickle encryption
                logger.info(f"Generating a new HMAC key for server {self.name}")
                self.hmac_key = os.urandom(32)

            self.address = (self.address_endpoint, self.hmac_key)

    def should_bind_socket(self) -> bool:
        """
        Determine if socket should bind vs connect based on type and role.

        ZMQ binding conventions:
        - PAIR: server binds, client connects (1-to-1 bidirectional)
        - PULL: server binds to receive from multiple PUSH sockets
        - PUSH: server binds when acting as message source
        - ROUTER: always binds to handle multiple clients

        Returns:
            True if socket should bind, False if it should connect
        """
        # Server binds for PAIR, PULL, PUSH patterns
        if self.is_server and self.socket_type in (zmq.PAIR, zmq.PULL,
                                                   zmq.PUSH):
            return True

        # ROUTER always binds (multi-client pattern)
        if self.socket_type == zmq.ROUTER:
            return True

        # Client connects for all other cases
        return False

    def setup_lazily(self):
        # Early return if setup is already done
        if self._setup_done:
            return

        with self._setup_lock:
            if self._setup_done:
                return
            self._setup_done = True

            if not self.is_server:
                logger_debug(
                    f"Client [{self.name}] connecting to {self.address_endpoint} in {self.socket_type_str[self.socket_type]}\n",
                    "green")
                self.socket.connect(self.address_endpoint)

            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)

    def _check_thread_safety(self):
        """Check if the current thread is the same as the thread that first used the socket."""
        if not self._zmq_debug_enabled:
            return

        current_thread_id = threading.get_ident()

        if self._zmq_thread_id is None:
            # First call - capture the thread ID
            self._zmq_thread_id = current_thread_id
            logger_debug(
                f"ZMQ socket [{self.name}] initialized on thread {current_thread_id}",
                "cyan")
        elif self._zmq_thread_id != current_thread_id:
            # Thread mismatch - raise error
            raise RuntimeError(
                f"ZMQ thread safety violation detected in [{self.name}]: "
                f"Socket created on thread {self._zmq_thread_id}, "
                f"but accessed from thread {current_thread_id}. "
                f"ZMQ sockets are not thread-safe!")

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()
        self._check_thread_safety()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def put(self, obj: Any, routing_id: Optional[bytes] = None):
        self.setup_lazily()
        self._check_thread_safety()
        with nvtx_range_debug("send", color="blue", category="IPC"):
            if self._stats is not None:
                t0 = _perf_ns()
                data = pickle.dumps(obj)  # nosec B301
                t_pickle = _perf_ns()
                if self.use_hmac_encryption:
                    data = self._sign_data(data)
                t_hmac = _perf_ns()
                self._send_data(data, routing_id=routing_id)
                t_end = _perf_ns()
                self._stats.record_put(
                    t_end - t0,
                    t_pickle - t0,
                    t_hmac - t_pickle,
                    t_end - t_hmac,
                    len(data),
                )
            elif self.use_hmac_encryption or self.socket_type == zmq.ROUTER:
                data = self._prepare_data(obj)
                self._send_data(data, routing_id=routing_id)
            else:
                self.socket.send_pyobj(obj)

    def put_noblock(self,
                    obj: Any,
                    *,
                    retry: int = 1,
                    wait_time: float = 0.001):
        '''
        Put an object into the queue without blocking, and retry if the send fails.
        NOTE: It won't raise any error if the send fails.

        Parameters:
            obj (Any): The object to send.
            retry (int): The number of times to retry sending the object.
            wait_time (float): The time to wait before retrying.
        '''

        assert retry >= 0 and retry <= 10, "Retry must be between 0 and 10, adjust the wait_time if needed"

        self.setup_lazily()
        self._check_thread_safety()
        with nvtx_range_debug("send", color="blue", category="IPC"):

            data = self._prepare_data(obj)
            try:
                self._send_data(data, flags=zmq.NOBLOCK)
            except zmq.Again:
                if retry > 0:
                    time.sleep(wait_time)
                    self.put_noblock(obj, retry=retry - 1, wait_time=wait_time)
                else:
                    logger.error(f"Failed to send object: {obj}")

    async def put_async(self, obj: Any, routing_id: Optional[bytes] = None):
        self.setup_lazily()
        self._check_thread_safety()
        t0 = _perf_ns() if self._stats is not None else 0
        try:
            if self.use_hmac_encryption or self.socket_type == zmq.ROUTER:
                data = self._prepare_data(obj)
                await self._send_data_async(data, routing_id=routing_id)
            else:
                await self.socket.send_pyobj(obj)
        except TypeError as e:
            logger.error(f"Cannot pickle {obj}")
            raise e
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e
        if self._stats is not None:
            self._stats.record_put_async(_perf_ns() - t0)
        nvtx_mark("ipc.send", color="blue", category="IPC")

    async def put_async_noblock(self, obj: Any):
        self.setup_lazily()
        self._check_thread_safety()
        try:
            if self.use_hmac_encryption:
                data = pickle.dumps(obj)  # nosec B301
                signed_data = self._sign_data(data)
                await self.socket.send(signed_data, flags=zmq.NOBLOCK)
            else:
                await self.socket.send_pyobj(obj, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

    def get(self) -> Any:
        self.setup_lazily()
        self._check_thread_safety()
        if self._stats is not None:
            t0 = _perf_ns()
            # Inline the recv path for per-phase timing
            if self.socket_type == zmq.ROUTER:
                identity, raw_data = self.socket.recv_multipart()
                self._last_identity = identity
            else:
                raw_data = self.socket.recv() if self.use_hmac_encryption else None
            t_recv = _perf_ns()
            if raw_data is not None:
                payload_bytes = len(raw_data)
                if self.use_hmac_encryption:
                    message_data = raw_data[:-32]
                    actual_hmac = raw_data[-32:]
                    if not self._verify_hmac(message_data, actual_hmac):
                        raise RuntimeError("HMAC verification failed")
                    t_hmac = _perf_ns()
                    result = pickle.loads(message_data)  # nosec B301
                    t_unpickle = _perf_ns()
                else:
                    t_hmac = t_recv
                    result = pickle.loads(raw_data)  # nosec B301
                    t_unpickle = _perf_ns()
            else:
                # recv_pyobj path (no HMAC)
                result = self.socket.recv_pyobj()
                t_recv = _perf_ns()  # update after actual recv
                t_hmac = t_recv
                t_unpickle = t_recv
                payload_bytes = 0
            t_end = _perf_ns()
            self._stats.record_get(
                t_end - t0,
                t_recv - t0,
                t_hmac - t_recv,
                t_unpickle - t_hmac,
                payload_bytes,
            )
            return result
        return self._recv_data()

    async def get_async(self) -> Any:
        self.setup_lazily()
        self._check_thread_safety()
        t0 = _perf_ns() if self._stats is not None else 0
        result = await self._recv_data_async()
        if self._stats is not None:
            self._stats.record_get_async(_perf_ns() - t0)
        return result

    async def get_async_noblock(self,
                                timeout: float = 0.5,
                                return_identity: bool = False) -> Any:
        """Get data with timeout using polling to avoid message drops.

        This method uses ZMQ's NOBLOCK flag with polling instead of asyncio.wait_for
        to prevent cancelling recv operations which can cause message drops.

        Args:
            timeout: Timeout in seconds
            return_identity: Whether to return the identity of the sender (for ROUTER sockets)

        Returns:
            The received object, or (object, identity) if return_identity is True

        Raises:
            asyncio.TimeoutError: If timeout is reached without receiving data
        """
        self.setup_lazily()
        self._check_thread_safety()

        # Use polling loop instead of asyncio.wait_for to avoid cancelling recv
        # which can cause message drops
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                # Try non-blocking receive
                if self.socket_type == zmq.ROUTER:
                    identity, data = await self.socket.recv_multipart(
                        flags=zmq.NOBLOCK)
                    self._last_identity = identity
                    obj = self._parse_data(data)
                    if return_identity:
                        return obj, identity
                    else:
                        return obj
                else:
                    if self.use_hmac_encryption:
                        data = await self.socket.recv(flags=zmq.NOBLOCK)
                        obj = self._parse_data(data)
                    else:
                        obj = await self.socket.recv_pyobj(flags=zmq.NOBLOCK)

                    if return_identity:
                        return obj, None
                    else:
                        return obj
            except zmq.Again:
                # No message available yet
                if asyncio.get_event_loop().time() >= deadline:
                    raise asyncio.TimeoutError()
                # Short sleep to avoid busy-waiting
                await asyncio.sleep(0.01)

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _verify_hmac(self, data: bytes, actual_hmac: bytes) -> bool:
        """Verify the HMAC of received pickle data."""
        expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected_hmac, actual_hmac)

    def _sign_data(self, data_before_encoding: bytes) -> bytes:
        """Generate HMAC for data."""
        hmac_signature = hmac.new(self.hmac_key, data_before_encoding,
                                  hashlib.sha256).digest()
        return data_before_encoding + hmac_signature

    def __del__(self):
        if self._stats is not None:
            self._stats.dump()
            self._stats = None
        self.close()

    def _prepare_data(self, obj: Any) -> bytes:
        """Serialize object and optionally add HMAC signature."""
        data = pickle.dumps(obj)  # nosec B301
        if self.use_hmac_encryption:
            return self._sign_data(data)
        return data

    def _parse_data(self, data: bytes) -> Any:
        """Parse data and optionally verify HMAC signature."""
        if self.use_hmac_encryption:
            # Split data and HMAC
            message_data = data[:-32]
            actual_hmac = data[-32:]

            # Verify HMAC
            if not self._verify_hmac(message_data, actual_hmac):
                raise RuntimeError("HMAC verification failed")

            return pickle.loads(message_data)  # nosec B301
        else:
            return pickle.loads(data)  # nosec B301

    def _send_data(self,
                   data: bytes,
                   flags: int = 0,
                   routing_id: Optional[bytes] = None):
        """Send data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            identity = routing_id if routing_id is not None else self._last_identity
            if identity is None:
                raise ValueError("ROUTER socket requires identity")
            self.socket.send_multipart([identity, data], flags=flags)
        else:
            self.socket.send(data, flags=flags)

    async def _send_data_async(self,
                               data: bytes,
                               routing_id: Optional[bytes] = None):
        """Async version of _send_data."""
        if self.socket_type == zmq.ROUTER:
            identity = routing_id if routing_id is not None else self._last_identity
            if identity is None:
                raise ValueError("ROUTER socket requires identity")
            await self.socket.send_multipart([identity, data])
        else:
            await self.socket.send(data)

    def _recv_data(self, return_identity: bool = False) -> Any:
        """Receive data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            identity, data = self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            obj = self._parse_data(data)
            if return_identity:
                return obj, identity
            return obj
        else:
            if self.use_hmac_encryption:
                data = self.socket.recv()
                obj = self._parse_data(data)
            else:
                obj = self.socket.recv_pyobj()

            if return_identity:
                return obj, None
            return obj

    async def _recv_data_async(self, return_identity: bool = False) -> Any:
        """Async version of _recv_data."""
        if self.socket_type == zmq.ROUTER:
            identity, data = await self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            obj = self._parse_data(data)
            if return_identity:
                return obj, identity
            return obj
        else:
            if self.use_hmac_encryption:
                data = await self.socket.recv()
                obj = self._parse_data(data)
            else:
                obj = await self.socket.recv_pyobj()

            if return_identity:
                return obj, None
            return obj

    def notify_with_retry(self, message, max_retries=5, timeout=1):
        """
        Notify with automatic retry on failure (for DEALER socket pattern).

        Args:
            message: Message to send
            max_retries: Maximum retry attempts (default: 5)
            timeout: Timeout in seconds for each attempt (default: 1)

        Returns:
            bool: True if acknowledgment received, False if failed after all retries
        """
        if self.socket_type != zmq.DEALER:
            raise ValueError(
                "notify_with_retry is only supported for DEALER socket for now")

        self._check_thread_safety()
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.put(message)
                # Wait for ACK with timeout
                if self.poll(timeout):
                    self.get()
                    return True
                else:
                    retry_count += 1

            except Exception as e:
                logger.error(f"Failed to notify with retry: {e}")
                retry_count += 1

        return False


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(obj)
        else:
            batch = obj if isinstance(obj, list) else [obj]
            self.queue.put(batch)

    def get(self) -> Any:
        return self.queue.get()

    @property
    def address(self) -> tuple[str, Optional[bytes]]:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
