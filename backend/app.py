import os, threading, traceback
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

BASE_MODEL = os.getenv("BASE_MODEL", "openai-community/gpt2")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/fine_tuned_gpt2"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/gpt2_training_output"))
try:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # /data only exists when persistent storage is attached (e.g. on a Hugging Face Space);
    # fall back to ephemeral storage so the API still starts without it.
    MODEL_DIR = Path("/tmp/fine_tuned_gpt2")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="GPT-2 Fine-Tuning Lab API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

state = {"status":"idle","message":"Backend ready","losses":[],"error":None,"fine_tuned_available":(MODEL_DIR/"config.json").exists()}
train_lock = threading.Lock()
base_cache = {}

class TrainRequest(BaseModel):
    texts: List[str] = Field(min_length=3, max_length=1000)
    epochs: int = Field(default=1, ge=1, le=3)
    learning_rate: float = Field(default=2e-5, gt=0, le=1e-3)
    batch_size: int = Field(default=1, ge=1, le=4)
    max_length: int = Field(default=96, ge=32, le=256)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=55, ge=1, le=120)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.92, gt=0, le=1)

class LossCallback(TrainerCallback):
    def on_log(self, args, state_obj, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                state["losses"].append(float(logs["loss"]))
                state["message"] = f"Training step {state_obj.global_step}: loss {float(logs['loss']):.4f}"
            except Exception:
                pass

def get_base():
    if "tokenizer" not in base_cache:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model.config.pad_token_id = tokenizer.pad_token_id
        base_cache["tokenizer"] = tokenizer
        base_cache["model"] = model
    return base_cache["tokenizer"], base_cache["model"]

def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def do_training(req: TrainRequest):
    try:
        state.update({"status":"running","message":"Loading GPT-2 and preparing dataset","losses":[],"error":None})
        tokenizer, _base_model = get_base()
        # Train a separate model instance so the cached base GPT-2 remains unchanged
        # for a fair before/after comparison in the generation endpoint.
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model.config.pad_token_id = tokenizer.pad_token_id
        clean = list(dict.fromkeys(t.strip() for t in req.texts if t and t.strip()))
        if len(clean) < 3:
            raise ValueError("At least three non-empty unique text samples are required.")
        tokenized = tokenizer(clean, truncation=True, max_length=req.max_length, padding=False)
        dataset = Dataset.from_dict(tokenized)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR), num_train_epochs=req.epochs,
            learning_rate=req.learning_rate, per_device_train_batch_size=req.batch_size,
            logging_strategy="steps", logging_steps=1, save_strategy="no",
            report_to="none", fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(), disable_tqdm=True,
            use_cpu=not torch.cuda.is_available()
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          data_collator=collator, callbacks=[LossCallback()])
        result = trainer.train()
        trainer.save_model(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        state.update({"status":"completed","message":f"Training complete. Final loss: {result.training_loss:.4f}","fine_tuned_available":True})
    except Exception as exc:
        state.update({"status":"failed","message":"Training failed","error":str(exc)})
        traceback.print_exc()
    finally:
        train_lock.release()

@app.get("/")
def root():
    return {"message":"GPT-2 Fine-Tuning Lab API","docs":"/docs","health":"/health"}

@app.get("/health")
def health():
    state["fine_tuned_available"] = (MODEL_DIR/"config.json").exists()
    return {"status":"ok","base_model":BASE_MODEL,"fine_tuned_available":state["fine_tuned_available"]}

@app.post("/train")
def train(req: TrainRequest):
    if state["status"] == "running" or not train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A training run is already in progress.")
    thread = threading.Thread(target=do_training, args=(req,), daemon=True)
    thread.start()
    return {"status":"running","message":"Training started in the backend. You can monitor progress in the Training Studio."}

@app.get("/train/status")
def train_status():
    state["fine_tuned_available"] = (MODEL_DIR/"config.json").exists()
    return state

@app.post("/generate")
def generate_comparison(req: GenerateRequest):
    try:
        tokenizer, base_model = get_base()
        base_text = generate(base_model, tokenizer, req.prompt, req.max_new_tokens, req.temperature, req.top_p)
        tuned_available = (MODEL_DIR/"config.json").exists()
        if tuned_available:
            tuned_model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
            tuned_model.config.pad_token_id = tokenizer.pad_token_id
            tuned_text = generate(tuned_model, tokenizer, req.prompt, req.max_new_tokens, req.temperature, req.top_p)
            del tuned_model
        else:
            tuned_text = "No fine-tuned checkpoint is available yet. Start a training run first."
        return {"base_text":base_text,"fine_tuned_text":tuned_text,"fine_tuned_available":tuned_available}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
