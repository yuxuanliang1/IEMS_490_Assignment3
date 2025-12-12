
---

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip && \
    pip install \
      transformers \
      datasets \
      accelerate \
      numpy \
      pandas \
      matplotlib \
      tqdm \
      safetensors \
      openaip

COPY . /workspace

CMD ["bash"]
