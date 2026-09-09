import json
import tempfile
from threading import Thread
from typing import Optional

import librosa
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
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
    WHISPER_MODEL_PATH, torch_dtype=torch.float32
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
    CONVERSADOR_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cpu")

conversador_model.eval()
print("Conversador loaded.")

# ============================
# DTOs
# ============================


class TextRequest(BaseModel):
    text: str
    useCustomPrompt: Optional[bool] = False  # Flag ativadora Maker
    customSystemPrompt: Optional[str] = None  # Prompt injetado pelo usuário


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
        audio, sampling_rate=16000, return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = whisper_model.generate(
            inputs["input_features"],
            task="transcribe",
            language="en",
            max_new_tokens=225,
        )

    text = whisper_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return {"text": text}


@app.post("/correct-text")
async def correct_text(request: TextRequest):
    inputs = tutor_tokenizer(
        request.text, return_tensors="pt", truncation=True
    )

    with torch.no_grad():
        outputs = tutor_model.generate(inputs["input_ids"], max_length=128)

    corrected_text = tutor_tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    return {
        "originalText": request.text,
        "correctedText": corrected_text,
        "feedback": "Texto corrigido automaticamente pelo modelo.",
    }


@app.post("/chat")
async def chat(request: TextRequest):
    # 🧠 1. Lógica do Injetor de System Prompt (Modo Maker Ativo vs Padrão)
    if request.useCustomPrompt and request.customSystemPrompt:
        system_instruction = request.customSystemPrompt.strip()
    else:
        system_instruction = (
            "You are an English tutor that helps users practice English.\n"
            'if the User greets you with "\'hi\',\'hello\'" or something like that, Always respond to him first with a greets.'
        )

    # 2. Montagem dinâmica do Prompt preservando a estrutura aceita pelo Conversador02
    prompt = f"""{system_instruction}
User: {request.text}
Tutor:"""

    inputs = conversador_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    )

    # 3. Streamer do Hugging Face para interceptar tokens gerados
    streamer = TextIteratorStreamer(
        conversador_tokenizer,
        skip_prompt=True,  # Descarta o trecho do prompt original na resposta
        skip_special_tokens=True,  # Descarta tokens de controle como <eos>, <pad>
    )

    # 4. Parâmetros passados para o gerador
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=60,
        do_sample=False,
        eos_token_id=conversador_tokenizer.eos_token_id,
        pad_token_id=conversador_tokenizer.eos_token_id,
    )

    # 5. Executa a geração do modelo em background thread para liberar o streaming
    thread = Thread(
        target=conversador_model.generate, kwargs=generation_kwargs
    )
    thread.start()

    # 6. Generator dos eventos SSE que formata cada chunk de texto
    def generate_events():
        stop_tokens = ["User:", "<|user|>", "\nUser"]
        accumulated_text = ""

        for new_text in streamer:
            accumulated_text += new_text

            # Verifica se o modelo começou a simular o próximo turno do usuário
            should_stop = False
            for stop in stop_tokens:
                if stop in accumulated_text:
                    should_stop = True
                    break

            if should_stop:
                break

            # Formatação SSE padrão
            payload = json.dumps({"chunk": new_text})
            yield f"data: {payload}\n\n"

    # 7. Retorno via StreamingResponse com content-type event-stream
    return StreamingResponse(generate_events(), media_type="text/event-stream")
