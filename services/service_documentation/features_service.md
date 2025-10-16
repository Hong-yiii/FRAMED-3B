---

# Cursor Prompt — “OpenCLIP Labeling Service (Framed)”

**Goal:** Build a small, reliable microservice that runs **OpenCLIP** zero-shot labeling. Input: 
    {
      "photo_id": "abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea",
      "original_uri": "./data/input/30ED3B7D-090E-485E-A3B7-A3A04F816B2E.jpg",
      "ranking_uri": "./data/rankingInput/abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea.jpg",
      "exif": {
        "camera": "Apple iPhone 11 Pro Max",
        "lens": "Unknown",
        "iso": 125,
        "aperture": "f/2.4",
        "shutter_speed": "1/100",
        "focal_length": "1mm",
        "datetime": "2024:12:06 18:51:37",
        "gps": null
      },
      "format": ".jpg"
    },

Output: **top-k labels** with scores. Optimize for **RTX 2070 (8 GB VRAM)** and low latency. No training. This service will be called by other components later.

  "batch_id": "batch_2025-01-15_test",
  "features":
    {
      "photo_id": "e7c4f5e1ebbad7203b18f8c840da314144d27f639e1962e5d87a06c63d4df672",
      "tech": {
        "sharpness": 0.5824664430670188,
        "exposure": 0.7642725303263522,
        "noise": 0.15978705996824694,
        "horizon_deg": 0.6372531198368598
      },
      "clip_labels": [
        "photography",
        "landscape",
        "nature",
        "outdoor",
        "scenic"
      ]
    },

## Hard Requirements

* **Framework:** Python 3.10+, **FastAPI** + **Uvicorn**
* **Model lib:** **open_clip_torch**
* **Primary checkpoint:** `hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` (good accuracy, fits on 8 GB)
* **Device:** prefer CUDA if available; else CPU fallback (It should check for Mac and if so use the GPU as well)
* **Precision:** FP16 on GPU
* **Image size:** default 336px; allow `?size=224|336` query param
* **Batch size:** 1 (single image per request; simple + low VRAM)
* **Labels:** read from `config/labels.json`
* **Prompt templates:** 3–6 per label; average text embeddings per label
* **Return:** `top_k` labels (default 5) with softmax probabilities and raw cosine scores
* **Cache:** cache label embeddings in memory; recompute only when labels/config change
* **Thread-safety:** model/embeddings loaded once on startup (global)
* **Timeouts:** 10s request timeout default
* **Logging:** structured logs (JSON) for request id, latency, model, device, image size, top1 label, top1 prob
* **Health endpoints:** `/healthz` (model and device ready), `/labels` (current label list), `/info` (model id, device)
* **Security (basic):** max payload 8 MB, reject non-image mimetypes, cap `top_k` ≤ 10

## API

### `POST /classify`

* **Query:** `top_k: int=5`, `size: int in {224,336}=336`, `temperature: float=0.01`, `threshold: float=0.0`
* **Body:** multipart/form-data, field `file` (image)
* **Response:**

```json
{
  "model": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
  "device": "cuda:0",
  "size": 336,
  "top_k": 5,
  "results": [
    {"label":"Eiffel Tower","prob":0.71,"score":0.83},
    {"label":"Louvre Museum","prob":0.12,"score":0.46}
  ]
}
```

### `GET /labels`

* Returns current label list and templates used.

### `POST /reload-labels`

* Reload `config/labels.json` without restarting (rebuild label embedding bank).
* Returns counts + checksum.

### `GET /healthz`

* `{ "status":"ok", "device":"cuda", "fp16":true }`

### `GET /info`

* Model id, device, precision, image norm used, service version.

## Files to Generate

```
openclip_service/
  app.py
  inference.py
  clip_runtime.py
  config/
    labels.json
    templates.json
  requirements.txt
  Dockerfile
  README.md
  scripts/
    smoke_test.sh
    bench_local.py
```

### `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.5
open_clip_torch
torch>=2.1
torchvision
pillow
numpy
orjson
python-multipart
```

### `config/labels.json` (starter)

```json
{
  "labels": [
    "Eiffel Tower",
    "Louvre Museum",
    "Notre-Dame Cathedral",
    "Arc de Triomphe",
    "Tokyo Tower",
    "Shibuya Crossing",
    "Mount Fuji",
    "Sagrada Família",
    "Big Ben",
    "Colosseum"
  ]
}
```

### `config/templates.json` (starter)

```json
{
  "templates": [
    "a photo of {}",
    "a landmark: {}",
    "an outdoor scene featuring {}",
    "a travel photograph of {}",
    "a wide shot of {}"
  ]
}
```

## Implementation Details

1. **Model Load**

   * Use `open_clip.create_model_and_transforms(MODEL_ID, device=DEVICE)`.
   * `model.eval()` and `model.half()` on GPU.
   * Keep **transform** returned by `create_model_and_transforms` (handles model-specific normalization, e.g., `[-1,1]` for some L/14 variants).

2. **Text Bank Build**

   * For each label, build prompts via templates, tokenize, encode with `model.encode_text`, **L2 normalize**, then **mean-pool** across templates to a single vector.
   * Stack vectors → `LABEL_EMB` shape `[N, D]` (L2 normalized).
   * Persist in memory; expose `/reload-labels` to rebuild.

3. **Image Encode**

   * Preprocess (transform), add batch dim, move to device, cast to `half()` when CUDA.
   * `model.encode_image(x)` → L2 normalize.

4. **Scoring**

   * Cosine similarity: `sims = v @ LABEL_EMB.T` → shape `[N]`.
   * Return both:

     * `score = sims[i]` (raw cosine)
     * `prob = softmax(sims / temperature)[i]` (use default **temperature=0.01** for peaky distribution).
   * Apply `threshold` (optional) → if `max(prob) < threshold`, return `"unsure"` pseudo-label.

5. **Validation**

   * Validate content type, size, and decodable image.
   * Clamp `top_k` to `[1..min(10, N)]`.

6. **Performance Notes (RTX 2070)**

   * Default **size=336**, **bs=1**, FP16 → typical < 100 ms/image.
   * If OOM: auto-fallback to **224** and log a warning.
   * Use `@torch.inference_mode()`.

7. **Logging**

   * On each request, log `{rid, latency_ms, device, size, top1_label, top1_prob, topk_labels}` in JSON.

## `inference.py` (sketch)

```python
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from typing import List, Dict

class CLIPService:
    def __init__(self, model_id: str, device: str = "cuda", image_size: int = 336):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_id = model_id
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()
        self.label_texts = []
        self.label_names = []
        self.label_emb = None
        self.image_size = image_size

    @torch.inference_mode()
    def build_label_bank(self, labels: List[str], templates: List[str]):
        tok = open_clip.get_tokenizer(self.model_id)
        self.label_names = labels
        embs = []
        for label in labels:
            prompts = [t.format(label) for t in templates]
            tokens = tok(prompts).to(self.device)
            txt = self.model.encode_text(tokens)
            txt = F.normalize(txt, dim=-1)
            embs.append(txt.mean(dim=0, keepdim=True))
        self.label_emb = F.normalize(torch.cat(embs, dim=0), dim=-1)  # [N,D]

    @torch.inference_mode()
    def classify_pil(self, img: Image.Image, top_k: int = 5, temperature: float = 0.01):
        x = self.preprocess(img).unsqueeze(0)
        if self.device == "cuda":
            x = x.to(self.device).half()
        else:
            x = x.to(self.device)
        v = self.model.encode_image(x)
        v = F.normalize(v, dim=-1)  # [1,D]
        sims = (v @ self.label_emb.T).squeeze(0)  # [N]
        probs = F.softmax(sims / temperature, dim=0)
        k = min(top_k, len(self.label_names))
        probs_k, idx_k = torch.topk(probs, k)
        results = []
        for p, i in zip(probs_k.tolist(), idx_k.tolist()):
            results.append({
                "label": self.label_names[i],
                "prob": float(p),
                "score": float(sims[i])
            })
        return results
```

## `app.py` (sketch)

* Load config files on startup.
* Instantiate `CLIPService`.
* Expose endpoints defined above.
* Enforce size, mimetype, and `top_k` limits.

## Dockerfile

* Base: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`
* Install deps, copy code, expose 8000, set `CMD` to run uvicorn.

## Smoke & Bench

* `scripts/smoke_test.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/labels
curl -s -F "file=@sample.jpg" "http://localhost:8000/classify?top_k=5"
```

* `scripts/bench_local.py`:

  * Loop over a folder of images, measure p95 latency, log to stdout.

## Acceptance Criteria

* ✅ `/healthz` returns `ok` and reports device `cuda` on a GPU machine.
* ✅ `/labels` returns the label list from config.
* ✅ `/classify` returns **top-k** results with `prob` and `score` for a valid JPEG.
* ✅ RTX 2070 with **size=336** and FP16: **median latency ≤ 120 ms** on a 1080p photo (bs=1).
* ✅ OOM fallback: when VRAM is tight, service retries once at **224** and logs `"oom_fallback": true`.
* ✅ `/reload-labels` rebuilds the text bank within 2s for ≤ 200 labels.
* ✅ Bad input (wrong mimetype, >8 MB) rejected with 400; logs include error code & reason.

## Nice-to-Haves (later)

* Optional **multilingual** text tower (XLM-RoBERTa variant) behind a flag.
* Optional **region crops** pass (saliency or detector) → classify patches → merge.
* Prometheus metrics endpoint and basic rate limiting.

---

**Now generate all files and working code as specified. Prioritize correctness, readability, and robust FP16 CUDA handling.**
