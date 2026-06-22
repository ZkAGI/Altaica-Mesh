# Altaica vertical-shard NODE-C (untrusted, blind) — runs scrambled TAIL layer block [K2:L] of
# Moonlight-16B as a persistent FastAPI server, in a REMOTE RunPod region. Receives a P-framed
# hidden per token, returns the hidden after its block. NO embedding, NO lm_head, NO tokenizer,
# NO keys -> sees only covariant-scrambled noise (moonlight_node_b.py is layer-range-agnostic).
#
# Built with nodec.Dockerfile.dockerignore: ships ONLY nodeC3_bundle.pt + moonlight_node_b.py.
FROM vllm/vllm-openai:v0.23.0

ENV PIP_DEFAULT_TIMEOUT=1000 PYTHONUNBUFFERED=1
RUN python3 -m pip install --no-cache-dir --retries 10 --timeout 1000 fastapi "uvicorn[standard]" pydantic

ENV NODE_BUNDLE=/models/nodeC3_bundle.pt PORT=8010

COPY nodeC3_bundle.pt /models/nodeC3_bundle.pt
COPY moonlight_node_b.py /node_b.py

ENTRYPOINT []
CMD ["python3", "/node_b.py"]
