# Copyright (C) 2019-2025 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Connection pool for gRPC channels.

This module provides connection pooling for gRPC channels to enable better load balancing
when connecting to Milvus clusters with multiple proxy instances.

By default, gRPC uses HTTP/2 multiplexing which means all requests go through a single
TCP connection. When connecting through a load balancer to multiple Milvus proxies,
this can result in uneven load distribution.

The ChannelPool class creates multiple independent gRPC channels and distributes
requests across them using round-robin selection, allowing better utilization of
multiple proxy instances.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import grpc
from grpc._cython import cygrpc

logger = logging.getLogger(__name__)


def _get_default_channel_options() -> list[tuple[str, int]]:
    """Get default gRPC channel options."""
    return [
        (cygrpc.ChannelArgKey.max_send_message_length, -1),
        (cygrpc.ChannelArgKey.max_receive_message_length, -1),
        ("grpc.enable_retries", 1),
        ("grpc.keepalive_time_ms", 55000),
    ]


class ChannelPool:
    """Thread-safe gRPC channel pool with round-robin load balancing.

    This class manages a pool of gRPC channels and provides round-robin
    channel selection for distributing requests across multiple connections.

    Args:
        address: The server address in "host:port" format.
        pool_size: Number of channels to create in the pool. Default is 1.
        secure: Whether to use TLS. Default is False.
        server_name: Server name override for TLS. Default is empty string.
        server_pem_path: Path to server certificate for one-way TLS.
        client_pem_path: Path to client certificate for mutual TLS.
        client_key_path: Path to client key for mutual TLS.
        ca_pem_path: Path to CA certificate for mutual TLS.

    Example:
        >>> pool = ChannelPool("localhost:19530", pool_size=4)
        >>> channel = pool.get_channel()  # Round-robin selection
        >>> pool.close()
    """

    def __init__(
        self,
        address: str,
        pool_size: int = 1,
        *,
        secure: bool = False,
        server_name: str = "",
        server_pem_path: str = "",
        client_pem_path: str = "",
        client_key_path: str = "",
        ca_pem_path: str = "",
    ) -> None:
        if pool_size < 1:
            msg = "pool_size must be at least 1"
            raise ValueError(msg)

        self._address = address
        self._pool_size = pool_size
        self._secure = secure
        self._server_name = server_name
        self._server_pem_path = server_pem_path
        self._client_pem_path = client_pem_path
        self._client_key_path = client_key_path
        self._ca_pem_path = ca_pem_path

        self._channels: list[grpc.Channel] = []
        self._index = 0
        self._lock = threading.Lock()
        self._closed = False

        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize all channels in the pool."""
        opts = _get_default_channel_options()

        if self._secure and self._server_name:
            opts.append(("grpc.ssl_target_name_override", self._server_name))

        creds = self._get_credentials() if self._secure else None

        for i in range(self._pool_size):
            channel = self._create_channel(opts, creds)
            self._channels.append(channel)
            logger.debug("Created channel %d/%d for %s", i + 1, self._pool_size, self._address)

    def _get_credentials(self) -> grpc.ChannelCredentials | None:
        """Get SSL credentials for secure channels."""
        root_cert, private_k, cert_chain = None, None, None

        if self._server_pem_path:
            with Path(self._server_pem_path).open("rb") as f:
                root_cert = f.read()
        elif self._client_pem_path and self._client_key_path and self._ca_pem_path:
            with Path(self._ca_pem_path).open("rb") as f:
                root_cert = f.read()
            with Path(self._client_key_path).open("rb") as f:
                private_k = f.read()
            with Path(self._client_pem_path).open("rb") as f:
                cert_chain = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=root_cert,
            private_key=private_k,
            certificate_chain=cert_chain,
        )

    def _create_channel(
        self,
        opts: list[tuple[str, int]],
        creds: grpc.ChannelCredentials | None,
    ) -> grpc.Channel:
        """Create a single gRPC channel."""
        if self._secure and creds is not None:
            return grpc.secure_channel(self._address, creds, options=opts)
        return grpc.insecure_channel(self._address, options=opts)

    def get_channel(self) -> grpc.Channel:
        """Get the next channel using round-robin selection.

        Returns:
            A gRPC channel from the pool.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        with self._lock:
            if self._closed:
                msg = "Channel pool has been closed"
                raise RuntimeError(msg)

            channel = self._channels[self._index]
            self._index = (self._index + 1) % self._pool_size
            return channel

    @property
    def pool_size(self) -> int:
        """Return the number of channels in the pool."""
        return self._pool_size

    def close(self) -> None:
        """Close all channels in the pool."""
        with self._lock:
            if self._closed:
                return

            self._closed = True
            for channel in self._channels:
                channel.close()
            self._channels.clear()
            logger.debug("Closed channel pool for %s", self._address)


class AsyncChannelPool:
    """Async gRPC channel pool with round-robin load balancing.

    This class manages a pool of async gRPC channels and provides round-robin
    channel selection for distributing requests across multiple connections.

    Args:
        address: The server address in "host:port" format.
        pool_size: Number of channels to create in the pool. Default is 1.
        secure: Whether to use TLS. Default is False.
        server_name: Server name override for TLS. Default is empty string.
        server_pem_path: Path to server certificate for one-way TLS.
        client_pem_path: Path to client certificate for mutual TLS.
        client_key_path: Path to client key for mutual TLS.
        ca_pem_path: Path to CA certificate for mutual TLS.

    Example:
        >>> pool = AsyncChannelPool("localhost:19530", pool_size=4)
        >>> channel = pool.get_channel()  # Round-robin selection
        >>> await pool.close()
    """

    def __init__(
        self,
        address: str,
        pool_size: int = 1,
        *,
        secure: bool = False,
        server_name: str = "",
        server_pem_path: str = "",
        client_pem_path: str = "",
        client_key_path: str = "",
        ca_pem_path: str = "",
    ) -> None:
        if pool_size < 1:
            msg = "pool_size must be at least 1"
            raise ValueError(msg)

        self._address = address
        self._pool_size = pool_size
        self._secure = secure
        self._server_name = server_name
        self._server_pem_path = server_pem_path
        self._client_pem_path = client_pem_path
        self._client_key_path = client_key_path
        self._ca_pem_path = ca_pem_path

        self._channels: list[grpc.aio.Channel] = []
        self._index = 0
        self._lock = threading.Lock()
        self._closed = False

        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize all channels in the pool."""
        opts = _get_default_channel_options()

        if self._secure and self._server_name:
            opts.append(("grpc.ssl_target_name_override", self._server_name))

        creds = self._get_credentials() if self._secure else None

        for i in range(self._pool_size):
            channel = self._create_channel(opts, creds)
            self._channels.append(channel)
            logger.debug(
                "Created async channel %d/%d for %s", i + 1, self._pool_size, self._address
            )

    def _get_credentials(self) -> grpc.ChannelCredentials | None:
        """Get SSL credentials for secure channels."""
        root_cert, private_k, cert_chain = None, None, None

        if self._server_pem_path:
            with Path(self._server_pem_path).open("rb") as f:
                root_cert = f.read()
        elif self._client_pem_path and self._client_key_path and self._ca_pem_path:
            with Path(self._ca_pem_path).open("rb") as f:
                root_cert = f.read()
            with Path(self._client_key_path).open("rb") as f:
                private_k = f.read()
            with Path(self._client_pem_path).open("rb") as f:
                cert_chain = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=root_cert,
            private_key=private_k,
            certificate_chain=cert_chain,
        )

    def _create_channel(
        self,
        opts: list[tuple[str, int]],
        creds: grpc.ChannelCredentials | None,
    ) -> grpc.aio.Channel:
        """Create a single async gRPC channel."""
        if self._secure and creds is not None:
            return grpc.aio.secure_channel(self._address, creds, options=opts)
        return grpc.aio.insecure_channel(self._address, options=opts)

    def get_channel(self) -> grpc.aio.Channel:
        """Get the next channel using round-robin selection.

        Returns:
            An async gRPC channel from the pool.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        with self._lock:
            if self._closed:
                msg = "Async channel pool has been closed"
                raise RuntimeError(msg)

            channel = self._channels[self._index]
            self._index = (self._index + 1) % self._pool_size
            return channel

    @property
    def pool_size(self) -> int:
        """Return the number of channels in the pool."""
        return self._pool_size

    async def close(self) -> None:
        """Close all channels in the pool."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels_to_close = list(self._channels)
            self._channels.clear()

        for channel in channels_to_close:
            await channel.close()
        logger.debug("Closed async channel pool for %s", self._address)
