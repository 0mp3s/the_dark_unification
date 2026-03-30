"""
runner/app.py
=============
FastAPI server for the pipeline runner.

Start with:
    py runner/app.py

Then open http://localhost:8501
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from watchfiles import awatch

REPO_ROOT  = Path(__file__).resolve().parent.parent
RUNNER_DIR = Path(__file__).resolve().parent
CSV_PATH   = REPO_ROOT / "docs" / "execution_pipeline.csv"

app = FastAPI()

# ── state ──────────────────────────────────────────────────────────────────
_run_state: Dict[str, Any] = {
    "running":      False,
    "paused":       False,
    "stop":         False,
    "current_order": None,
}
_clients: list[WebSocket] = []


# ── helpers ─────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


async def _stream_process(proc: asyncio.subprocess.Process, order: int) -> int:
    """Stream stdout+stderr of proc line-by-line to all WS clients."""
    async def _read(stream, is_stderr=False):
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            await _broadcast({"type": "output", "order": order,
                               "text": text, "err": is_stderr})

    await asyncio.gather(
        _read(proc.stdout),
        _read(proc.stderr, is_stderr=True),
    )
    return await proc.wait()


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (RUNNER_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/pipeline")
async def get_pipeline():
    from pipeline import load_pipeline
    steps = load_pipeline()
    return {"steps": [s.to_dict() for s in steps]}


@app.get("/api/image")
async def get_image(path: str):
    """Serve a repo-relative image file as base64 (avoids static path issues)."""
    full = REPO_ROOT / path
    if not full.exists() or full.suffix.lower() not in (".png", ".pdf", ".jpg"):
        return Response(status_code=404)
    data = full.read_bytes()
    b64  = base64.b64encode(data).decode()
    mime = "image/png" if full.suffix == ".png" else "application/pdf"
    return {"data": f"data:{mime};base64,{b64}"}


@app.get("/api/outputs/{order}")
async def get_outputs(order: int):
    from pipeline import load_pipeline, get_output_images, get_output_csvs
    steps = load_pipeline()
    step  = next((s for s in steps if s.order == order), None)
    if not step:
        return {"images": [], "csvs": []}
    return {
        "images": get_output_images(step),
        "csvs":   get_output_csvs(step),
    }


@app.get("/api/state")
async def get_state():
    return _run_state


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    # send current state on connect
    await ws.send_text(json.dumps({"type": "state", **_run_state}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_ws_cmd(msg)
    except WebSocketDisconnect:
        if ws in _clients:
            _clients.remove(ws)


async def _handle_ws_cmd(msg: dict) -> None:
    cmd = msg.get("cmd")

    if cmd == "run":
        orders = msg.get("orders", [])
        if not _run_state["running"] and orders:
            asyncio.create_task(_run_pipeline(orders))

    elif cmd == "pause":
        _run_state["paused"] = True
        await _broadcast({"type": "state", **_run_state})

    elif cmd == "resume":
        _run_state["paused"] = False
        await _broadcast({"type": "state", **_run_state})

    elif cmd == "stop":
        _run_state["stop"]    = True
        _run_state["paused"]  = False
        await _broadcast({"type": "state", **_run_state})

    elif cmd == "reload_pipeline":
        from pipeline import load_pipeline
        steps = load_pipeline()
        await _broadcast({"type": "pipeline",
                          "steps": [s.to_dict() for s in steps]})


# ── pipeline runner ──────────────────────────────────────────────────────────

async def _run_pipeline(orders: list[int]) -> None:
    from pipeline import load_pipeline, get_output_images, get_output_csvs

    _run_state["running"]       = True
    _run_state["paused"]        = False
    _run_state["stop"]          = False
    _run_state["current_order"] = None
    await _broadcast({"type": "state", **_run_state})

    steps = {s.order: s for s in load_pipeline()}

    for order in sorted(orders):
        if _run_state["stop"]:
            break

        # wait while paused
        while _run_state["paused"]:
            await asyncio.sleep(0.3)
            if _run_state["stop"]:
                break
        if _run_state["stop"]:
            break

        step = steps.get(order)
        if not step:
            continue

        _run_state["current_order"] = order
        await _broadcast({"type": "state", **_run_state})
        await _broadcast({"type": "step_start", "order": order,
                          "label": step.label, "stage": step.stage,
                          "missing": step.missing_inputs})

        py_exe = sys.executable   # the same Python that runs app.py
        script_path = str(REPO_ROOT / step.script)

        try:
            proc = await asyncio.create_subprocess_exec(
                py_exe, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(REPO_ROOT),
            )
            rc = await _stream_process(proc, order)
        except Exception as exc:
            rc = -1
            await _broadcast({"type": "output", "order": order,
                               "text": f"[runner error] {exc}", "err": True})

        images = get_output_images(step)
        csvs   = get_output_csvs(step)

        await _broadcast({
            "type":    "step_done",
            "order":   order,
            "rc":      rc,
            "images":  images,
            "csvs":    csvs,
        })

        if rc != 0:
            # failure — pause and wait for user decision
            _run_state["paused"] = True
            await _broadcast({"type": "state", **_run_state})
            await _broadcast({"type": "failure", "order": order, "rc": rc})
            while _run_state["paused"] and not _run_state["stop"]:
                await asyncio.sleep(0.3)
            if _run_state["stop"]:
                break

    _run_state["running"]       = False
    _run_state["current_order"] = None
    await _broadcast({"type": "state", **_run_state})
    await _broadcast({"type": "done"})


# ── CSV file watcher (pushes reload when CSV changes) ────────────────────────

async def _watch_csv() -> None:
    async for _ in awatch(str(CSV_PATH)):
        from pipeline import load_pipeline
        steps = load_pipeline()
        await _broadcast({"type": "pipeline",
                          "steps": [s.to_dict() for s in steps]})


@app.on_event("startup")
async def startup():
    asyncio.create_task(_watch_csv())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # add repo root to path so pipeline.py can import from core/
    sys.path.insert(0, str(RUNNER_DIR))
    sys.path.insert(0, str(REPO_ROOT / "core"))
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8501,
        reload=False,
        app_dir=str(RUNNER_DIR),
    )


if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify("\u2705 app done!")
    except Exception:
        pass
