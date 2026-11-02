"""The Hugging Face Hub revisions the TTS chapters were measured with, and loaders that use them.

Every model is loaded at the commit below, never at the repository's moving `main`. The same hashes are in
the `hf download` commands of code/tts/README.md and in the result files' provenance (`hf_snapshot`).
"""
from __future__ import annotations

REVISIONS = {
    "kyutai/mimi": "89091b3e466eb6a9d11e537bf26b144f194978f7",
    "facebook/encodec_24khz": "c1dbe2ae3f1de713481a3b3e7c47f357092ee040",
    "descript/dac_24khz": "6ba020b5ba7d9d8076fb90db7e67f27e31980f6e",
    "charactr/vocos-mel-24khz": "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
    "kyutai/moshiko-pytorch-bf16": "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289",
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "hexgrad/Kokoro-82M-v1.1-zh": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "funasr/paraformer-zh": "d7811ee3ac581fbcfdeb37c98c6ba674028433dc",
}


def rev(repo: str) -> str:
    """The pinned revision of `repo`; a repository not listed here is an error, not `main`."""
    return REVISIONS[repo]


def download(repo: str, filename: str) -> str:
    """One file of `repo` at its pinned revision (from the local cache when it is there)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, revision=rev(repo))


def kokoro_pipeline(lang_code: str, repo: str):
    """A kokoro.KPipeline whose model weights and config come from `repo` at its pinned revision.

    The device choice is the pipeline's own default (CUDA when available, else CPU)."""
    import torch
    from kokoro import KModel, KPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id=repo, config=download(repo, "config.json"),
                   model=download(repo, KModel.MODEL_NAMES[repo])).to(device).eval()
    return KPipeline(lang_code=lang_code, repo_id=repo, model=model)


def kokoro_voice(repo: str, voice: str) -> str:
    """The path of a Kokoro voice file at the pinned revision; KPipeline accepts it in place of the name."""
    return download(repo, f"voices/{voice}.pt")
