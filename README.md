---
title: AquaVision ML Service
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# AquaVision ML Service
This is the dedicated GPU inference service for AquaVision.
It provides a FastAPI endpoint to enhance underwater images using a PyTorch U-Net model.

## Endpoints
- `GET /health` : Health check and model status
- `POST /enhance` : Upload an image and get the enhanced output
