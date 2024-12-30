from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uvicorn
import base64
from io import BytesIO

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg_scale: float = 7.5
    seed: int = None

class SDService:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")
        self._optimize_pipeline()
    
    def _optimize_pipeline(self):
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()

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
            generator=generator
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
