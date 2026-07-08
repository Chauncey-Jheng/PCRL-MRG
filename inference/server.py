"""FastAPI app for standalone deployment of the PCRL-MRG brain CT report generator.

Run with (cwd must be this `inference/` directory so the flat `engine`/`demo_sample`
imports resolve):

    uvicorn server:app --host 0.0.0.0 --port 8700

See run_server.sh for a wrapper that sets the right env vars, and
../../central-control/child-apps/pcrl-mrg/server.js for how central-control supervises
this process.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from engine import VISUAL_TOKEN_COUNT, ReportGenerationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pcrl-mrg-inference")

INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_DIR.parent
STATIC_DIR = INFERENCE_DIR / "static"
DEMO_SAMPLE_DIR = STATIC_DIR / "sample"
DEMO_META_FILE = DEMO_SAMPLE_DIR / "meta.json"

# Hardcoded defaults match this project's known-good deployment machine (see
# PCRL-MRG/CLAUDE.md's "Hardcoded Paths" section — every entry point in this repo
# follows the same per-environment-hardcoded-path convention). Override via env vars
# for a different machine.
MODEL_NAME = os.environ.get("PCRL_MODEL_NAME", "/home/bjutcv/data/zcx/models/Meta-Llama-3-8B-Instruct")
CLIP_MODEL_PATH = os.environ.get("PCRL_CLIP_MODEL", "/home/bjutcv/data/zcx/models/clip-vit-large-patch14")
CHECKPOINT_DIR = os.environ.get(
    "PCRL_CHECKPOINT_DIR", str(PROJECT_ROOT / "results" / "baseline_LLM_MRG_v0.0.1")
)
PEFT_MODEL_NAME = os.environ.get("PCRL_PEFT_MODEL_NAME", "peft_model_best")

app = FastAPI(title="PCRL-MRG 脑CT报告生成推理服务")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

engine: Optional[ReportGenerationEngine] = None
load_error: Optional[str] = None


class GenerateResponse(BaseModel):
    report: str


@app.on_event("startup")
def load_model():
    global engine, load_error
    logger.info("Loading PCRL-MRG inference engine (LLaMA3-8B + LoRA + CLIP)...")
    try:
        engine = ReportGenerationEngine(
            model_name=MODEL_NAME,
            clip_model_path=CLIP_MODEL_PATH,
            checkpoint_dir=CHECKPOINT_DIR,
            peft_model_name=PEFT_MODEL_NAME,
        )
        logger.info("Model loaded, ready to serve.")
    except Exception as exc:  # noqa: BLE001 - report to /api/health instead of crashing the process
        load_error = str(exc)
        logger.exception("Failed to load inference engine")


@app.get("/api/health")
def health():
    if engine is not None:
        return {"status": "ready", "visual_token_count": VISUAL_TOKEN_COUNT}
    if load_error:
        return {"status": "error", "detail": load_error}
    return {"status": "loading"}


def _require_engine() -> ReportGenerationEngine:
    if engine is None:
        detail = f"模型加载失败：{load_error}" if load_error else "模型仍在加载中，请稍后重试。"
        raise HTTPException(status_code=503, detail=detail)
    return engine


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(images: List[UploadFile] = File(...)):
    eng = _require_engine()
    if len(images) != VISUAL_TOKEN_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"需要恰好上传 {VISUAL_TOKEN_COUNT} 张CT切片图像，收到 {len(images)} 张。",
        )

    pil_images = []
    for f in images:
        raw = await f.read()
        try:
            pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法解析图像文件：{f.filename}（{exc}）")

    try:
        report = eng.generate_report(pil_images)
    except Exception as exc:  # noqa: BLE001 - surface inference failures to the caller
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}")

    return GenerateResponse(report=report)


@app.get("/api/demo")
def demo_info():
    """Metadata for the one bundled demo case (see prepare_demo_sample.py) — lets the
    UI offer a "try it now" path without requiring the visitor to have real CT slices
    on hand. Not present until provisioned on the deployment host, so this degrades to
    unavailable rather than erroring.
    """
    if not DEMO_META_FILE.exists():
        return {"available": False}
    meta = json.loads(DEMO_META_FILE.read_text(encoding="utf-8"))
    images = sorted(p.name for p in DEMO_SAMPLE_DIR.glob("*.jpg"))
    if len(images) != VISUAL_TOKEN_COUNT:
        return {"available": False}
    return {
        "available": True,
        "images": [f"/sample/{name}" for name in images],
        "ground_truth": meta.get("ground_truth", {}),
    }


@app.post("/api/demo/generate", response_model=GenerateResponse)
def demo_generate():
    eng = _require_engine()
    images = sorted(DEMO_SAMPLE_DIR.glob("*.jpg"))
    if len(images) != VISUAL_TOKEN_COUNT:
        raise HTTPException(status_code=404, detail="示例CT尚未部署（未找到24张示例图像）。")

    pil_images = [Image.open(p).convert("RGB") for p in images]
    try:
        report = eng.generate_report(pil_images)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Demo inference failed")
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}")

    return GenerateResponse(report=report)


# Static frontend last, so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
