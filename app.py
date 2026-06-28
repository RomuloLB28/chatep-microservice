from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import torch
import librosa
import tempfile

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

# ============================
# APP
# ============================

app = FastAPI()

# ============================
# WHISPER MODEL
# ============================

WHISPER_MODEL_PATH = "./whisper-finetuned"

print("Loading Whisper model...")
whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_PATH,
    torch_dtype=torch.float32
)
whisper_model.eval()
print("Whisper loaded.")

# ============================
# TUTOR MODEL (TEXT CORRECTION)
# ============================

TUTOR_MODEL_PATH = "./TUTOR02-Large-v02"

print("Loading Tutor model...")
tutor_tokenizer = AutoTokenizer.from_pretrained(TUTOR_MODEL_PATH)
tutor_model = AutoModelForSeq2SeqLM.from_pretrained(TUTOR_MODEL_PATH)
tutor_model.eval()
print("Tutor loaded.")

# ============================
# CONVERSADOR MODEL (MISTRAL FINE-TUNED)
# ============================

CONVERSADOR_MODEL_PATH = "./Conversador02-v02"

print("Loading Conversador model...")
conversador_tokenizer = AutoTokenizer.from_pretrained(CONVERSADOR_MODEL_PATH)
conversador_model = AutoModelForCausalLM.from_pretrained(
    CONVERSADOR_MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cpu")

conversador_model.eval()
print("Conversador loaded.")

# ============================
# DTOs
# ============================

class TextRequest(BaseModel):
    text: str
    useCustomPrompt: Optional[bool] = False         # Flag ativadora Maker
    customSystemPrompt: Optional[str] = None       # Prompt injetado pelo usuário

# ============================
# ROUTES
# ============================

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    audio, sr = librosa.load(temp_path, sr=16000)

    inputs = whisper_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = whisper_model.generate(
            inputs["input_features"],
            task="transcribe",
            language="en",
            max_new_tokens=225
        )

    text = whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return {"text": text}

@app.post("/correct-text")
async def correct_text(request: TextRequest):
    inputs = tutor_tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = tutor_model.generate(
            inputs["input_ids"],
            max_length=128
        )

    corrected_text = tutor_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "originalText": request.text,
        "correctedText": corrected_text,
        "feedback": "Texto corrigido automaticamente pelo modelo."
    }

@app.post("/chat")
async def chat(request: TextRequest):
    # 🧠 Lógica do Injetor de System Prompt (Modo Maker Ativo vs Padrão)
    if request.useCustomPrompt and request.customSystemPrompt:
        system_instruction = request.customSystemPrompt.strip()
    else:
        system_instruction = (
            "You are an English tutor that helps users practice English.\n"
            "if the User greets you with \"'hi','hello'\" or something like that, Always respond to him first with a greets."
        )

    # Montagem dinâmica do Prompt preservando a estrutura aceita pelo Conversador02
    prompt = f"""{system_instruction}
User: {request.text}
Tutor:"""

    inputs = conversador_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256  # Aumentado levemente para acomodar instruções customizadas maiores
    )

    with torch.no_grad():
        outputs = conversador_model.generate(
            inputs["input_ids"],
            max_new_tokens=60,
            do_sample=False,
            eos_token_id=conversador_tokenizer.eos_token_id,
            pad_token_id=conversador_tokenizer.eos_token_id
        )

    response = conversador_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove o prompt estruturado original de dentro da resposta do modelo
    assistant_response = response[len(prompt):].strip()

    # Corta loops ou gerações que tentem simular um novo turno do usuário
    stop_tokens = ["User:", "<|user|>", "\nUser"]
    for stop_token in stop_tokens:
        if stop_token in assistant_response:
            assistant_response = assistant_response.split(stop_token)[0]

    return {
        "userInput": request.text,
        "response": assistant_response.strip()
    }
