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

"""Tests for the ChannelPool and AsyncChannelPool classes."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.client.channel_pool import AsyncChannelPool, ChannelPool


class TestChannelPool:
    """Tests for the synchronous ChannelPool class."""

    @patch("pymilvus.client.channel_pool.grpc.insecure_channel")
    def test_init_creates_channels(self, mock_insecure_channel):
        """Test that ChannelPool creates the specified number of channels."""
        mock_channel = MagicMock()
        mock_insecure_channel.return_value = mock_channel

        pool = ChannelPool("localhost:19530", pool_size=4)

        assert pool.pool_size == 4
        assert mock_insecure_channel.call_count == 4
        pool.close()

    @patch("pymilvus.client.channel_pool.grpc.insecure_channel")
    def test_round_robin_selection(self, mock_insecure_channel):
        """Test that get_channel returns channels in round-robin order."""
        channels = [MagicMock() for _ in range(3)]
        mock_insecure_channel.side_effect = channels

        pool = ChannelPool("localhost:19530", pool_size=3)

        # First round
        assert pool.get_channel() == channels[0]
        assert pool.get_channel() == channels[1]
        assert pool.get_channel() == channels[2]
        # Second round (wrap around)
        assert pool.get_channel() == channels[0]
        assert pool.get_channel() == channels[1]

        pool.close()

    @patch("pymilvus.client.channel_pool.grpc.insecure_channel")
    def test_thread_safety(self, mock_insecure_channel):
        """Test that get_channel is thread-safe."""
        mock_insecure_channel.return_value = MagicMock()

        pool = ChannelPool("localhost:19530", pool_size=4)
        results = []
        errors = []

        def get_channels():
            try:
                for _ in range(100):
                    channel = pool.get_channel()
                    results.append(channel)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_channels) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000  # 10 threads * 100 calls
        pool.close()

    @patch("pymilvus.client.channel_pool.grpc.insecure_channel")
    def test_close_pool(self, mock_insecure_channel):
        """Test that close properly closes all channels."""
        mock_channel = MagicMock()
        mock_insecure_channel.return_value = mock_channel

        pool = ChannelPool("localhost:19530", pool_size=2)
        pool.close()

        assert mock_channel.close.call_count == 2

    @patch("pymilvus.client.channel_pool.grpc.insecure_channel")
    def test_get_channel_after_close_raises(self, mock_insecure_channel):
        """Test that get_channel raises after pool is closed."""
        mock_insecure_channel.return_value = MagicMock()

        pool = ChannelPool("localhost:19530", pool_size=2)
        pool.close()

        with pytest.raises(RuntimeError, match="closed"):
            pool.get_channel()

    def test_invalid_pool_size(self):
        """Test that invalid pool_size raises ValueError."""
        with pytest.raises(ValueError, match="pool_size must be at least 1"):
            ChannelPool("localhost:19530", pool_size=0)

        with pytest.raises(ValueError, match="pool_size must be at least 1"):
            ChannelPool("localhost:19530", pool_size=-1)

    @patch("pymilvus.client.channel_pool.grpc.secure_channel")
    @patch("pymilvus.client.channel_pool.grpc.ssl_channel_credentials")
    def test_secure_channel_creation(self, mock_ssl_creds, mock_secure_channel):
        """Test that secure channels are created when secure=True."""
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds
        mock_channel = MagicMock()
        mock_secure_channel.return_value = mock_channel

        pool = ChannelPool("localhost:19530", pool_size=2, secure=True)

        assert mock_secure_channel.call_count == 2
        pool.close()


class TestAsyncChannelPool:
    """Tests for the asynchronous AsyncChannelPool class."""

    @patch("pymilvus.client.channel_pool.grpc.aio.insecure_channel")
    def test_init_creates_channels(self, mock_insecure_channel):
        """Test that AsyncChannelPool creates the specified number of channels."""
        mock_channel = MagicMock()
        mock_insecure_channel.return_value = mock_channel

        pool = AsyncChannelPool("localhost:19530", pool_size=4)

        assert pool.pool_size == 4
        assert mock_insecure_channel.call_count == 4

    @patch("pymilvus.client.channel_pool.grpc.aio.insecure_channel")
    def test_round_robin_selection(self, mock_insecure_channel):
        """Test that get_channel returns channels in round-robin order."""
        channels = [MagicMock() for _ in range(3)]
        mock_insecure_channel.side_effect = channels

        pool = AsyncChannelPool("localhost:19530", pool_size=3)

        # First round
        assert pool.get_channel() == channels[0]
        assert pool.get_channel() == channels[1]
        assert pool.get_channel() == channels[2]
        # Second round (wrap around)
        assert pool.get_channel() == channels[0]

    def test_invalid_pool_size(self):
        """Test that invalid pool_size raises ValueError."""
        with pytest.raises(ValueError, match="pool_size must be at least 1"):
            AsyncChannelPool("localhost:19530", pool_size=0)

    @patch("pymilvus.client.channel_pool.grpc.aio.insecure_channel")
    @pytest.mark.asyncio
    async def test_close_pool(self, mock_insecure_channel):
        """Test that close properly closes all channels."""
        mock_channel = MagicMock()
        mock_channel.close = MagicMock(return_value=None)
        # Make close() return a coroutine
        async def async_close():
            pass
        mock_channel.close = MagicMock(side_effect=async_close)
        mock_insecure_channel.return_value = mock_channel

        pool = AsyncChannelPool("localhost:19530", pool_size=2)
        await pool.close()

        assert mock_channel.close.call_count == 2
