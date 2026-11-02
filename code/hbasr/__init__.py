"""Shared code for the handbook's ASR chapters (1-9).

Modules:
  audio      seconds/samples/bytes arithmetic, PCM layout, resampling, chunking
  features   a numpy log-mel front end and streaming-safe normalization
  alignment  CTC and transducer forward algorithms in numpy, CTC greedy decoding
  textnorm   stated text normalizers for English WER and Mandarin CER
  metrics    edit distance, WER/CER, partial stability (UPWR/UPSR)
  latency    a streaming-latency harness (simulated or real-time pacing)
  data       fixed, documented evaluation subsets of LibriSpeech and AISHELL-1
  provenance versions, commands and machine facts written next to every result
"""
