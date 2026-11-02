"""Start one streaming ASR server, run the load generator at each concurrency N, stop the server.

    python ch15/sweep.py --name sherpa_cpu --streams 1 2 4 8 16 32 --duration 60 -- \
        python ch15/sherpa_server.py --port 6006 ...

Writes $BOOK_DATA/runs/<name>/N<k>/{sessions.jsonl,summary.json}, the server log, and a
combined summary at code/serve/results/ch15_<name>.json. Server CPU use is read from /proc/<pid>/stat
(user + system time of the server process and its threads) over each N's run.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.util import RESULTS, RUN_ROOT, gpu_name, run_meta, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TICK = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid: int) -> float:
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / TICK  # utime, stime (fields 14, 15)


def server_pid(pid: int) -> int:
    """The python process that serves: the launched pid itself (env execs python), or its only child."""
    exe = Path(f"/proc/{pid}/exe").resolve().name
    if exe.startswith("python"):
        return pid
    kids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    return int(kids[0]) if kids else pid


def wait_port(port: int, proc: subprocess.Popen, timeout: float) -> None:
    t_end = time.time() + timeout
    while time.time() < t_end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"port {port} not open after {timeout} s")


def gpu_mem_used() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def merge_parts(out: Path, procs: int, n: int, warmup: float, duration: float) -> None:
    from loadgen import summarise

    sessions, metas = [], []
    for i in range(procs):
        d = out / f"p{i}"
        sessions += [json.loads(l) for l in (d / "sessions.jsonl").read_text().splitlines() if l.strip()]
        metas.append(json.loads((d / "summary.json").read_text())["meta"])
    t_start = min(json.loads((out / f"p{i}" / "summary.json").read_text())["t_start"] for i in range(procs))
    cpu = json.loads((out / "p0" / "summary.json").read_text()).get("server_cpu_cores")
    with open(out / "sessions.jsonl", "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # each part dropped its own first `warmup` seconds; the merged summary applies the same rule
    summ = {"meta": metas[0], "streams": n, "chunk_ms": json.loads((out / "p0" / "summary.json").read_text())["chunk_ms"],
            **summarise(sessions, warmup, t_start, duration), "server_cpu_cores": cpu}
    write_json(out / "summary.json", summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--streams", type=int, nargs="+", required=True)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--chunk-ms", type=float, default=160)
    ap.add_argument("--lang-hint", action="store_true")
    ap.add_argument("--streams-per-proc", type=int, default=32)
    ap.add_argument("--start-timeout", type=float, default=600)
    ap.add_argument("--stop-p95-ms", type=float, default=None,
                    help="stop the sweep after the first N whose partial-latency p95 exceeds this")
    ap.add_argument("server", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.server[1:] if a.server and a.server[0] == "--" else a.server
    run_dir = RUN_ROOT / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        wait_port(a.port, proc, a.start_timeout)
        url = f"ws://127.0.0.1:{a.port}"
        lg = [sys.executable, str(HERE / "loadgen.py"), "--url", url, "--chunk-ms", str(a.chunk_ms)]
        spid = server_pid(proc.pid)
        if a.lang_hint:
            lg.append("--lang-hint")
        # warm-up pass, not recorded
        subprocess.run(lg + ["--streams", "2", "--duration", "8", "--out", str(run_dir / "warmup")], check=True)
        for n in a.streams:
            out = run_dir / f"N{n}"
            procs = max(1, -(-n // a.streams_per_proc))
            if procs == 1:
                subprocess.run(lg + ["--streams", str(n), "--duration", str(a.duration), "--warmup", str(a.warmup),
                                     "--server-pid", str(spid), "--out", str(out), "--label", f"{a.name} N={n}"], check=True)
            else:  # several client processes so the load generator is not the bottleneck
                per = [n // procs + (1 if i < n % procs else 0) for i in range(procs)]
                ps = [subprocess.Popen(lg + ["--streams", str(k), "--duration", str(a.duration),
                                             "--warmup", str(a.warmup), "--seed", str(i),
                                             *(["--server-pid", str(spid)] if i == 0 else []),
                                             "--out", str(out / f"p{i}"), "--label", f"{a.name} N={n} part {i}"])
                      for i, k in enumerate(per)]
                for p in ps:
                    if p.wait() != 0:
                        raise RuntimeError("load generator failed")
                merge_parts(out, procs, n, a.warmup, a.duration)
            s = json.loads((out / "summary.json").read_text())
            s["client_processes"] = procs
            s["gpu_after"] = gpu_mem_used()
            write_json(out / "summary.json", s)
            rows.append({k: s.get(k) for k in ("streams", "sessions", "sessions_errored", "sessions_no_final",
                                               "audio_s", "partial_latency_ms", "final_latency_ms",
                                               "slope_ms_per_s_median", "send_lag_max_ms", "server_cpu_cores",
                                               "gpu_after", "error_rate")})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            p95 = (s["partial_latency_ms"] or {}).get("p95")
            if a.stop_p95_ms is not None and (p95 is None or p95 > a.stop_p95_ms):
                print(f"stopping sweep: partial p95 {p95} ms > {a.stop_p95_ms} ms at N={n}", flush=True)
                break
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    combined = {
        "name": a.name,
        "server_cmd": cmd,
        "meta": run_meta(["sherpa-onnx", "onnxruntime", "websockets", "nemo-toolkit", "torch"],
                         {"gpu": gpu_name()}),
        "raw_dir": str(run_dir),
        "duration_s_per_N": a.duration,
        "chunk_ms": a.chunk_ms,
        "rows": rows,
    }
    write_json(RESULTS / f"ch15_{a.name}.json", combined)
    print("wrote", RESULTS / f"ch15_{a.name}.json")


if __name__ == "__main__":
    main()
