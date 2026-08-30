"""Optional Week 4 stack: BentoML wrap (~1 day, only if chasing enterprise JDs).

    uv pip install bentoml
    bentoml serve serving.bentoml.service:OCRService   # :3001
    bentoml build && bentoml containerize ocr_service  # the actual selling point

Reference: https://github.com/bentoml/BentoVLLM (OpenAI-compatible vLLM bentos).
"""
import bentoml


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 300, "concurrency": 16},
)
class OCRService:
    def __init__(self) -> None:
        # TODO(week4, optional): vllm AsyncLLMEngine here, mirror serve_app.py engine args.
        self.engine = None

    @bentoml.api
    async def ocr(self, image_b64: str) -> str:
        # TODO(week4, optional): multimodal prompt -> engine -> text.
        # BentoML adaptive batching handles the microbatch story for the blog.
        raise NotImplementedError
