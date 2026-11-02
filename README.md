# Streaming speech: the code

The code and the measurement results behind the book **A Hands-on Guide to Streaming Speech Recognition and
Synthesis**, which is read online at https://book.rockyshi.dev/streaming-speech/.

The book explains how streaming speech recognition and streaming speech synthesis work, how to measure their
accuracy, quality and latency, how to serve them, and the latency budget of a real-time voice agent. This repository
holds what the chapters ask you to run, and the harness that produced the book's own measurements. A path written in
a chapter as `code/...` is the same path in this repository.

Every example is in English and Mandarin, and every hands-on part has a CPU track and a GPU track. Every measured
number in the book names the result file it comes from (`code/**/results/`), so you can check a number, or repeat the
measurement on your own machine and compare. The book's measurements were made on one machine: an Intel Core
i5-12600K and one NVIDIA RTX 4090 (23,028 MiB usable).

## Where to start

| You want to | Read |
|---|---|
| run the recognition exercises (chapters 1 to 9) | `code/README.md` |
| run the synthesis exercises (chapters 10 to 14) | `code/tts/README.md` |
| run the serving, transport and cascade exercises (chapters 15 to 18) | `code/serve/README.md` |

Each README describes the environments, the data sets and models to fetch, and the exact command of every
measurement.

## Licence

The code is licensed under the Apache License 2.0 (`LICENSE`). Third-party material and its terms are listed in
`NOTICE`; the licences of the toolkits, models and data sets the book uses are in Appendix C of the book.
