# 🚀 ChatEP Microservice — API de Serviços

Microservice desenvolvido em **FastAPI** para o sistema **ChatEP**, especializado no processamento de linguagem natural, transcrição de áudio via Whisper e geração de conversas tutoradas em tempo real com streaming de eventos (SSE).

---

## 🛠️ Tecnologias e Stack

* **[FastAPI](https://fastapi.tiangolo.com/):** Framework web moderno e de alto desempenho para construção de APIs.
* **[PyTorch](https://pytorch.org/):** Framework de Deep Learning para execução dos modelos de IA.
* **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index):** Carregamento e inferência dos modelos pré-treinados e fine-tuned.
* **[Librosa](https://librosa.org/):** Processamento e manipulação de arquivos de áudio.
* **[Docker & Docker Compose](https://www.docker.com/):** Containerização e isolamento do ambiente de serviço.
* **[Nginx](https://www.nginx.com/):** Servidor proxy reverso, gerenciamento de timeouts e limite de payload para uploads.

---

## 📂 Estrutura do Projeto

```text
├── app.py              # Aplicação principal FastAPI (Rotas, DTOs e Carregamento de Modelos)
├── Dockerfile          # Configuração de build da imagem Python/PyTorch
├── docker-compose.yml  # Orquestração dos containers (API + Nginx)
├── nginx.conf          # Configuração do proxy reverso e regras de timeout
└── requirements.txt    # Dependências do projeto Python
```
---

## ⚙️ Pré-requisitos

Certifique-se de ter instalado em sua máquina ou servidor:

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

Além disso, os diretórios/pesos locais dos modelos de IA devem estar presentes na raiz do projeto conforme configurado no código:

* `./whisper-finetuned`
* `./TUTOR02-Large-v02`
* `./Conversador02-v02`

---

## 🚀 Como Executar o Projeto

1. Clone o repositório e acesse a pasta raiz do back-end.
2. Certifique-se de que os pesos dos modelos estão posicionados corretamente.
3. Suba os containers utilizando o Docker Compose em modo *detached*:

```bash
docker compose up --build -d

```

O Nginx estará escutando na porta **80** da sua máquina/VPS, redirecionando as requisições para a aplicação FastAPI rodando internamente na porta `8000`.

---

## 🔌 Endpoints da API

### 1. Transcrição de Áudio (`/transcribe`)

* **Método:** `POST`
* **Content-Type:** `multipart/form-data`
* **Parâmetro:** `file` (Arquivo de áudio `.webm` ou compatível)
* **Retorno:** `{ "text": "Texto transcrito pelo Whisper" }`

### 2. Correção de Texto (`/correct-text`)

* **Método:** `POST`
* **Content-Type:** `application/json`
* **Payload:**
```json
{
  "text": "Your text to correct here",
  "useCustomPrompt": false,
  "customSystemPrompt": null
}

```


* **Retorno:** `{ "originalText": "...", "correctedText": "...", "feedback": "..." }`

### 3. Chat Inteligente com Streaming (`/chat`)

* **Método:** `POST`
* **Content-Type:** `application/json`
* **Payload:**
```json
{
  "text": "Hello, can you help me?",
  "useCustomPrompt": false,
  "customSystemPrompt": "Optional custom system instruction"
}

```


* **Retorno:** `text/event-stream` (SSE — Server-Sent Events) contendo chunks de texto em tempo real gerados pelo modelo `Conversador02-v02`.

---

## 🔒 Limitações e Configurações de Produção

* **Memória RAM:** O container da API (`api_tcc`) possui um limite configurado de **7GB** no `docker-compose.yml` para evitar estouros de memória na VPS.
* **Timeouts:** O Nginx está configurado com `proxy_read_timeout` e `proxy_connect_timeout` de **300 segundos** para garantir que requisições longas de IA e streaming não sejam interrompidas prematuramente.
* **Uploads:** O limite máximo de tamanho de arquivo suportado pelo Nginx (`client_max_body_size`) é de **20MB**.

## Link para os modelos no Hugging Face:
https://huggingface.co/RomuloLB28
