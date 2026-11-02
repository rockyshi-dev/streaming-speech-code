# Streaming speech: the code

The code, the result files and the figure sources behind the book **A Hands-on Guide to Streaming Speech
Recognition and Synthesis**, which is read online at https://book.rockyshi.dev/streaming-speech/.

The book explains how streaming speech recognition and streaming speech synthesis work, how to measure their
accuracy, quality and latency, how to serve them, and the latency budget of a real-time voice agent. Every example
is in English and Mandarin, and every hands-on part has a CPU track and a GPU track. This repository holds what the
chapters ask the reader to run, and the harness that produced the book's own measurements. A path written in a
chapter as `code/...` or `figures/...` is the same path in this repository.

## Repository layout

| Path | What it holds |
|---|---|
| `code/` | the code behind every measurement, with the result files the chapters cite (`code/**/results/`) |
| `figures/` | every figure: its script or TikZ source, the data it is drawn from, and the built SVG in `_out/` |
| `tools/` | the figure build |

Every measured number in the book names the result file it comes from. The measurements were made on one
machine: an Intel Core i5-12600K and one NVIDIA RTX 4090 (23,028 MiB usable).

## Running the code

`code/README.md` (recognition, chapters 1 to 9), `code/tts/README.md` (synthesis, chapters 10 to 14) and
`code/serve/README.md` (serving and the cascade, chapters 15 to 18) describe the environments, the data sets and
models to fetch, and the exact command of every measurement.

## Building the figures

```bash
tools/export_figure_data.sh                   # figure data from the result files, then every plotted figure
```

The diagrams drawn in TikZ (`figures/chNN/*.tikz`) are included as built SVGs; the script rebuilds only the
plotted figures.

## Licence

The code is licensed under the Apache License 2.0 (`LICENSE`). The figures are licensed under the Creative Commons
Attribution 4.0 International licence (`LICENSE-text`). Third-party material and its terms are listed in `NOTICE`;
the licences of the toolkits, models and data sets the book uses are in Appendix C of the book.
