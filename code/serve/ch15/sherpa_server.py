"""sherpa-onnx's streaming WebSocket server (python-api-examples/streaming_server.py at v2.0.1), unchanged in
its batching and decoding, with two fields added to each result message so a load generator can time it:

  n      number of audio messages this connection had sent when the result was computed
  final  true on the last message of the connection (sent after the client's "Done")

The vendor files are fetched by ch15/fetch_vendor.sh into $BOOK_DATA/vendor/ and are
Apache-2.0 (Copyright Xiaomi Corp.). The protocol is the vendor's: binary float32 little-endian 16 kHz samples,
then the text message "Done".

    python ch15/sherpa_server.py --port 6006 --max-batch-size 8 --num-threads 4 --use-endpoint 0 \
        --encoder ... --decoder ... --joiner ... --tokens ...
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import paths  # noqa: E402

VENDOR = f"{paths.BOOK_DATA}/vendor/sherpa-onnx-v2.0.1"  # filled by ch15/fetch_vendor.sh
sys.path.insert(0, VENDOR)

import streaming_server as vs  # noqa: E402  (vendor module)
import websockets  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402


class _NoHttp:  # the vendor HttpServer reads the browser demo pages at start-up; this server serves none
    def __init__(self, doc_root):
        pass


vs.HttpServer = _NoHttp


class TimedStreamingServer(vs.StreamingServer):
    async def handle_connection(self, socket):
        self.current_active_connections += 1
        try:
            await self.handle_connection_impl(socket)
        except websockets.exceptions.ConnectionClosedError:
            logging.info("client disconnected")
        finally:
            self.current_active_connections -= 1

    async def handle_connection_impl(self, socket):
        # Same as the vendor loop, plus the 'n' and 'final' fields.
        stream = self.recognizer.create_stream()
        segment = 0
        n = 0
        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break
            n += 1
            stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
            while self.recognizer.is_ready(stream):
                await self.compute_and_decode(stream)
                result = self.recognizer.get_result(stream)
                message = {"text": result, "segment": segment, "n": n, "final": False}
                if self.recognizer.is_endpoint(stream):
                    self.recognizer.reset(stream)
                    segment += 1
                await socket.send(json.dumps(message))
        tail_padding = np.zeros(int(self.sample_rate * 0.3)).astype(np.float32)
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=tail_padding)
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            await self.compute_and_decode(stream)
        result = self.recognizer.get_result(stream)
        await socket.send(json.dumps({"text": result, "segment": segment, "n": n, "final": True}))

    async def run(self, port: int):
        # The vendor run() uses the legacy websockets API (process_request(path, headers)); this one uses the
        # current API and serves no HTML pages. Batching is the vendor's stream_consumer_task.
        tasks = [asyncio.create_task(self.stream_consumer_task()) for _ in range(self.nn_pool_size)]
        async with serve(self.handle_connection, "127.0.0.1", port, max_size=self.max_message_size,
                         max_queue=self.max_queue_size, ping_interval=None):
            logging.info("listening on ws://127.0.0.1:%d", port)
            await asyncio.Future()
        await asyncio.gather(*tasks)


def main():
    args = vs.get_args()
    vs.check_args(args)
    recognizer = vs.create_recognizer(args)
    server = TimedStreamingServer(
        recognizer=recognizer,
        nn_pool_size=args.nn_pool_size,
        max_wait_ms=args.max_wait_ms,
        max_batch_size=args.max_batch_size,
        max_message_size=args.max_message_size,
        max_queue_size=args.max_queue_size,
        max_active_connections=args.max_active_connections,
        doc_root=args.doc_root,
        certificate=None,
    )
    print("READY", flush=True)
    asyncio.run(server.run(args.port))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
