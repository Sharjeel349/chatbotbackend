# University AI Advisor

FastAPI backend for student records, offline voice recognition, local Ollama
advice, optional RAG, text-to-speech, and EEG tone hints.

## Run

1. Copy `FYP2/.env.example` to `FYP2/.env` and set the local database values.
2. Start Ollama and ensure `gemma2:2b`, `mistral`, and
   `nomic-embed-text` are installed.
3. Activate the project virtual environment.
4. Index the knowledge files:

   `python -m FYP2.scripts.rebuild_knowledge_base`

5. Start the API from the repository root:

   `python -m uvicorn FYP2.main:app --host 0.0.0.0 --port 8000 --reload`

The API uses the cached OpenAI Whisper `base` model. The model is loaded once
during startup and reused by subsequent requests. `faster-whisper` is not used
on this Intel Mac because its native runtime conflicts with another OpenMP
library in the application process.

## Live voice response

`POST /api/{session_id}/voice` accepts multipart fields:

- `audio_file`: recorded audio
- `emotion`: optional EEG state
- `stream`: set to `true` for newline-delimited JSON events

Streaming event order:

1. `status`
2. `transcript`
3. `reply_start`
4. one or more `token` events
5. `reply_complete`
6. `complete`

Clients that omit `stream=true` continue receiving the original single JSON
response.

Urdu speech detected by Whisper receives a Roman-Urdu response written only in
Latin letters using Mistral. English speech uses the faster Gemma 2B model.

## Tests

`python -m unittest discover -s tests -v`

## Security

The local `.env`, generated audio, and Chroma database are ignored by Git.
Rotate any database password that was previously committed because removing it
from the current source does not remove it from Git history.
