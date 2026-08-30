"""Ray Serve deployment of the OCR model (Week 2). Two apps in one file:

  app        — production path: ray.serve.llm -> OpenAI-compatible API, vLLM engine,
               autoscaling replicas, fractional GPU. `serve run serving.ray.serve_app:app`
  app_manual — educational path: hand-rolled deployment showing @serve.batch dynamic
               batching + fractional-GPU placement without the serve.llm sugar.

NOTE: ray.serve.llm APIs moved during 2.4x — verify against the pinned Ray version
(https://docs.ray.io/en/latest/serve/llm/serving-llms.html) on Week 2 Day 1.
"""
import os

MODEL_ID = os.environ.get("OCR_MODEL_ID", "PaddlePaddle/PaddleOCR-VL")

# --------------------------------------------------------------------------- production
from ray.serve.llm import LLMConfig, build_openai_app  # noqa: E402

llm_config = LLMConfig(
    model_loading_config={
        "model_id": MODEL_ID,          # name clients use in /v1/chat/completions
        "model_source": MODEL_ID,      # HF repo
    },
    deployment_config={
        "autoscaling_config": {
            # THE README STORY: watch replicas go 1 -> 2 under harness load (c=16),
            # both replicas sharing the single 3060 via fractional GPU below.
            "min_replicas": 1,
            "max_replicas": 2,
            "target_ongoing_requests": 8,
            "upscale_delay_s": 10,
            "downscale_delay_s": 60,
        },
        "max_ongoing_requests": 16,
        # Fractional GPU: two replicas of the 0.9B model fit on one 12GB card.
        "ray_actor_options": {"num_gpus": 0.5},
    },
    engine_kwargs={
        # 2 replicas x 0.42 ≈ 0.84 of VRAM; single-replica runs can raise this.
        "gpu_memory_utilization": 0.42,
        "max_model_len": 8192,
        "trust_remote_code": True,
    },
)

app = build_openai_app({"llm_configs": [llm_config]})

# --------------------------------------------------------------------------- educational
from ray import serve  # noqa: E402


@serve.deployment(
    ray_actor_options={"num_gpus": 0.5},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2, "target_ongoing_requests": 8},
    max_ongoing_requests=16,
)
class ManualOCRDeployment:
    """Same idea without serve.llm: you own the engine and the batching."""

    def __init__(self) -> None:
        # TODO(week2, optional): vllm.AsyncLLMEngine.from_engine_args(...) here,
        # gpu_memory_utilization=0.42 to coexist with a second replica.
        self.engine = None

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def infer_batch(self, image_b64_list: list[str]) -> list[str]:
        # Dynamic microbatching (planflow-2 'Dynamic Microbatcher'): Serve collects
        # concurrent requests into one engine call.
        # TODO(week2, optional): build multimodal prompts, call self.engine.generate.
        return ["" for _ in image_b64_list]

    async def __call__(self, request) -> dict:
        payload = await request.json()
        text = await self.infer_batch(payload["image_b64"])
        return {"text": text}


app_manual = ManualOCRDeployment.bind()
