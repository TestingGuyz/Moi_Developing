# Moi — Self-Learning Vision + Generative AI Assistant

An AI assistant that can see objects via your webcam, describe what it sees, chat like ChatGPT, auto-gather knowledge about detected objects, store long-term memory, and generate images on request.

## Features
- Vision captioning via NVIDIA Vision API or BLIP (Hugging Face)
- Object detection via YOLOv5 (if available)
- Knowledge database (JSON-backed) with CRUD API and auto-enrichment
- Chat via Groq models (e.g., `openai/gpt-oss-120b`)
- Image generation via Stability API or Hugging Face Inference
- Web UI with camera feed, analyze button, and knowledge editing

## Quickstart
1) Create and fill `.env` (see `.env.example`).
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Run server:
```bash
python app.py
```
Open `http://localhost:5000`.

## Environment
See `.env.example` for available variables:
- `GROQ_API_KEY` (required)
- `TAVILY_API_KEY` (optional; web search)
- `NVIDIA_API_KEY` (optional; NVIDIA vision)
- `HUGGINGFACE_TOKEN` (optional; BLIP + image gen fallback)
- `STABILITY_API_KEY` (optional; image generation)
- `DEFAULT_TEMPERATURE` (optional, default 0.7)
- `AUTO_ENRICH` (true/false; default true)

## Endpoints
- `POST /chat`
- `POST /start-vision`, `POST /stop-vision`, `GET /video-feed`, `POST /analyze-frame`, `GET /vision-status`
- `POST /improve-prompt`, `POST /search-object`, `POST /upload`
- Knowledge DB: `GET /knowledge`, `GET /knowledge/search`, `GET|PUT|PATCH|DELETE /knowledge/:name`, `POST /knowledge/:name/notes`
- Image gen: `POST /generate-image`

## Notes
- First run may download models (BLIP/YOLO). If YOLO load fails, detection will be disabled but captioning still works.
- Knowledge DB stored at `data/knowledge_db.json`.
