"""Shared helpers for the TTS hands-on code (chapters 10-14).

Modules:
    audio      - first audible sample, dBFS, WAV header stripping for streamed bytes
    stream     - per-request stream records: TTFA, RTF, inter-arrival, playback simulation
    textstream - sentence splitting and simulated LLM text streaming at a fixed rate
    errorrate  - WER / CER with the text normalization used here
    runinfo    - provenance (command line, package versions, git commit) for result JSON
"""
