from contextlib import asynccontextmanager
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uvicorn
import base64
from io import BytesIO


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg_scale: float = 7.5
    seed: int = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SDService:
    def __init__(self):
        self.pipe = None
        self.device = get_device()

    async def initialize(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._optimize_pipeline()

    def _optimize_pipeline(self):
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()

    async def cleanup(self):
        if self.pipe is not None:
            # 移到 CPU
            self.pipe.to("cpu")
            # 删除 pipeline
            del self.pipe
            self.pipe = None

            # 强制垃圾回收
            gc.collect()

            # 如果使用 CUDA，清理 CUDA 缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # 如果使用 MPS (Mac M1/M2)，清理 MPS 缓存
            elif self.device == "mps":
                torch.mps.empty_cache()

    async def generate(self, params) -> tuple[str, int]:
        generator = None
        if params.seed is not None:
            generator = torch.Generator(self.device).manual_seed(params.seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale,
                generator=generator,
            )

        # 获取使用的种子
        seed = generator.initial_seed() if generator else None

        # 转换图像为 base64
        image = result.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str, seed


def check_device():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {get_device()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_device()
    await sd_service.initialize()

    yield

    if hasattr(app.state, "sd_service"):
        await app.state.sd_service.cleanup()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    try:
        generator = None
        if request.seed is not None:
            generator = torch.Generator("cuda").manual_seed(request.seed)

        images = sd_service.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            generator=generator,
        ).images

        # 转换图像为 base64
        buffered = BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    sd_service = SDService()
    uvicorn.run(app, host="0.0.0.0", port=8000)
