---
title: GPT-2 Fine-Tuning Lab API
emoji: ✳
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# GPT-2 Fine-Tuning Lab API

FastAPI service that fine-tunes GPT-2 on a small text dataset and compares base and fine-tuned generations. It backs the GPT-2 Fine-Tuning Lab dashboard hosted on Netlify.

Endpoints: `GET /health`, `POST /train`, `GET /train/status`, `POST /generate`. Interactive docs are at `/docs`.

Fine-tuned checkpoints are saved to `/data/fine_tuned_gpt2` when persistent storage is attached, and to `/tmp` otherwise, where they are lost when the Space restarts.
