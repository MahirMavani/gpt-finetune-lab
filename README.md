# Custom-Style GPT-2 Fine-Tuning Lab — Netlify + Python API

This project is split into:
- `frontend/`: static dashboard published by Netlify.
- `backend/`: FastAPI + PyTorch + Hugging Face Transformers service, suitable for a Docker host such as Hugging Face Spaces.
- `data/synthetic_text_dataset.csv`: synthetic examples for the assignment.

## Deployment overview

1. Deploy `backend/` as a Docker Space or another persistent Python host.
2. Confirm the backend's `/health` endpoint works.
3. Push this whole project to GitHub.
4. Import the GitHub repository into Netlify. `netlify.toml` publishes the `frontend/` folder.
5. Open the Netlify website. The first API call will ask for your backend URL; enter the backend base URL (for example `https://YOUR-SPACE-NAME.hf.space`) and it will be saved in that browser's local storage.
6. Use Dataset Lab, Training Studio, and Generation Lab.

## Deploy the backend to a Hugging Face Space

1. Go to https://huggingface.co/new-space, pick a name, choose **Docker** as the SDK (blank template), and keep the free **CPU basic** hardware.
2. Upload the four files in `backend/` (`README.md`, `Dockerfile`, `app.py`, `requirements.txt`) to the root of the Space, either through **Files → Add file → Upload files** or with git:
   ```bash
   git clone https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
   cp backend/* YOUR-SPACE-NAME/
   cd YOUR-SPACE-NAME && git add . && git commit -m "Add API" && git push
   ```
3. Wait for the Space to show **Running** (the first build takes several minutes), then open `https://YOUR-USERNAME-YOUR-SPACE-NAME.hf.space/health`.
4. Open the Netlify site, click the connection status pill, and paste that base URL (without `/health`).

Without persistent storage, the fine-tuned checkpoint is lost whenever the Space restarts; just train again.

## Backend hosting note

The GPT-2 training service requires Python, PyTorch, memory, and model downloads. Netlify hosts the static frontend, not this training process. A Docker-based Hugging Face Space can host FastAPI, but check the current account/hardware requirements and persistence settings before relying on it for a free, always-on service. GPU hardware may incur cost. For a classroom demo, you can also train on a GPU notebook and use a separate inference service, but a notebook session is not a reliable always-on backend.

## Run the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Local frontend
Open `frontend/index.html` in a browser for layout testing, but browser API calls need the backend URL and correct CORS configuration. For production use Netlify deployment.

## Netlify
The root `netlify.toml` sets the publish directory to `frontend`. No Node build step is required.
