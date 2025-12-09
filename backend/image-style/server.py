"""
FastAPI service that performs simple pet photo style-transfer using
Stable Diffusion img2img. It exposes:

    POST /stylize  -> multipart form (image + style + optional strength)
    GET  /health   -> readiness probe

The heavy Stable Diffusion pipeline is loaded once and kept in memory.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn


DEFAULT_MODEL_ID = os.getenv("PET_IMAGE_MODEL_ID", "Lykon/dreamshaper-8")
DEFAULT_PORT = int(os.getenv("PET_IMAGE_PORT", "8011"))

STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "1": {"prompt": "Oil Painting, thick strokes, textured canvas", "strength": 0.65},
    "2": {"prompt": "Pixar style, disney 3d render, cute, vibrant", "strength": 0.70},
    "3": {"prompt": "Cyberpunk, neon lights, mechanical parts, sci-fi", "strength": 0.75},
    "4": {"prompt": "Pencil sketch, graphite, monochrome, rough lines", "strength": 0.55},
    "5": {"prompt": "Ghibli style, anime, vibrant colors, detailed background", "strength": 0.65},
}

NEGATIVE_PROMPT = (
    "human, person, people, man, woman, girl, boy, child, human face, human body, hands, feet, "
    "worst quality, low quality, normal quality, lowres, blurry, bad anatomy, disfigured, "
    "text, watermark, ugly, painting frame, extra limbs"
)
DEFAULT_STRENGTH = 0.75
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEPS = 30


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, (value // multiple) * multiple)


def _prepare_image(image: Image.Image) -> Image.Image:
    """
    Normalize the uploaded image to 512x512 RGB for Stable Diffusion 1.5.
    """
    image = image.convert("RGB")
    return image.resize((512, 512), Image.LANCZOS)


@dataclass
class StyleEngine:
    model_id: str = DEFAULT_MODEL_ID

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline: StableDiffusionImg2ImgPipeline | None = None

    def ensure_loaded(self) -> StableDiffusionImg2ImgPipeline:
        if self.pipeline is not None:
            return self.pipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kwargs = {"torch_dtype": dtype}
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            kwargs["use_auth_token"] = token

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            safety_checker=None,
            **kwargs,
        )
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        # Align with doc.md: use DPMSolver++ scheduler for better speed/sharpness
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )
        self.pipeline = pipe
        return pipe

    def generate(
        self,
        image: Image.Image,
        style_prompt: str,
        species: str,
        strength: Optional[float] = None,
        guidance: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> Image.Image:
        """
        Run img2img using the Colab prompt recipe with strong species conditioning.
        """
        if not style_prompt.strip():
            raise ValueError("style_prompt cannot be empty")
        if not species.strip():
            raise ValueError("species cannot be empty")

        full_prompt = (
            f"masterpiece, best quality, highres, (A cute {species.strip()}:1.5), "
            f"(animal only:1.2), {style_prompt.strip()}, 8k, extremely detailed, cinematic lighting"
        )

        pipe = self.ensure_loaded()
        result = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            strength=strength or DEFAULT_STRENGTH,
            guidance_scale=guidance or DEFAULT_GUIDANCE,
            num_inference_steps=steps or DEFAULT_STEPS,
        ).images[0]
        return result


engine = StyleEngine()

app = FastAPI(title="Pawty Pet Image API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.state.engine = engine


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _resolve_style(style_id: Optional[str], custom_prompt: Optional[str]) -> tuple[str, Optional[float]]:
    """
    Resolve style prompt and strength:
    - If custom_prompt provided, use it directly.
    - Else look up style_id (1-5) from Colab presets.
    """
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip(), None
    if style_id and style_id in STYLE_PRESETS:
        preset = STYLE_PRESETS[style_id]
        return preset["prompt"], preset.get("strength")  # type: ignore[arg-type]
    raise HTTPException(status_code=400, detail="Invalid style_id; must be 1-5 or provide style_prompt")


@app.post("/stylize")
async def stylize_endpoint(
    image: UploadFile = File(...),
    species: str = Form(...),
    style_id: str | None = Form(default=None),
    style_prompt: str | None = Form(default=None),
    strength: float | None = Form(default=None),
    guidance: float | None = Form(default=None),
    steps: int | None = Form(default=None),
) -> JSONResponse:
    if not species.strip():
        raise HTTPException(status_code=400, detail="species is required")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Image payload is empty")

    try:
        init_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # pillow raises various errors for invalid file
        raise HTTPException(status_code=400, detail="Unable to read image") from exc

    prepared = _prepare_image(init_image)

    resolved_prompt, preset_strength = _resolve_style(style_id, style_prompt)
    strength_to_use = strength if strength is not None else preset_strength

    try:
        stylized: Image.Image = await run_in_threadpool(
            app.state.engine.generate,
            prepared,
            resolved_prompt,
            species,
            strength_to_use,
            guidance,
            steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # model/runtime failure
        raise HTTPException(status_code=500, detail="Style engine failure") from exc

    buf = io.BytesIO()
    stylized.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(
        {
            "style_prompt": resolved_prompt,
            "species": species,
            "image_base64": encoded,
            "width": stylized.width,
            "height": stylized.height,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pawty pet image FastAPI service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face repo id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Delay model loading until first request",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app.state.engine.model_id = args.model
    if not args.lazy:
        # warm up in a worker thread so the main thread stays responsive
        torch.set_grad_enabled(False)
        app.state.engine.ensure_loaded()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

