# Altaica Π+P serverless worker — BAKED variant (scrambled checkpoint COPY'd into the image).
# FROM the official vLLM image: torch + vLLM 0.23.0 + flashinfer are PREBUILT, so the build does
# NOT download multi-GB wheels (avoids the flashinfer/torch pip timeouts on a home connection) —
# only the tiny `runpod` package is installed. The vLLM base layers already live on Docker Hub, so
# `docker push` skips them and uploads only our model layer (~30GB) + handler.
#
# Build context MUST be shard_proto/ (see .dockerignore: ships ONLY perm_scrambled_hf/ + handler;
# perm_keys.pt and the .pt bundles are excluded -> can never enter the image).
#
# One command does build+push+endpoint:  python runpod_deploy.py --user <you> --bake

FROM vllm/vllm-openai:v0.23.0

ENV PIP_DEFAULT_TIMEOUT=1000 PYTHONUNBUFFERED=1
RUN python3 -m pip install --no-cache-dir --retries 10 --timeout 1000 runpod

ENV VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    CKPT=/models/perm_scrambled_hf MAXLEN=4096 GPU_UTIL=0.92

# model layer (big, rarely changes) — scrambled weights ONLY
COPY perm_scrambled_hf/ /models/perm_scrambled_hf/
# handler layer (tiny, changes often)
COPY runpod_handler.py /handler.py

# override the vLLM OpenAI-server entrypoint -> run our blind handler instead
ENTRYPOINT []
CMD ["python3", "/handler.py"]
