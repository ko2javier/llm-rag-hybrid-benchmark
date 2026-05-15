# README — Vast.ai Setup (Blackwell / sin Docker)

Basado en el informe `informe_20260512_blackwell_embedding_batching.md`.
Máquina de referencia: RTX PRO 5000 Blackwell 48 GB, CUDA 13.1, SM 12.0.

> **Regla inamovible:** el embed server arranca ANTES que vLLM.
> vLLM calcula su `gpu-memory-utilization` contra la VRAM que encuentra libre.
> Si arranca primero, no deja margen para los ~3.1 GB de bge-m3.

---

## Paso 1 — Preparar directorios

```bash
# Disco raíz del container: ~16 GB. bge-m3 va a /workspace, LLM a /dev/shm.
mkdir -p /workspace/hf_cache    # cache bge-m3 (~1.2 GB)
mkdir -p /dev/shm/hf_cache      # cache modelos LLM (~31 GB RAM tmpfs disponible)

df -h / /dev/shm
```

---

## Paso 2 — Instalar dependencias del sistema

```bash
apt-get update -qq
apt-get install -y postgresql postgresql-16-pgvector
# Si no existe postgresql-16-pgvector:
# apt-get install -y postgresql-pgvector
```

---

## Paso 3 — Instalar dependencias Python

```bash
# Torch primero (sin caché — disco raíz solo 16 GB, pip cachea antes de instalar)
pip install torch --no-cache-dir

# vLLM (sin caché — falla con "No space left on device" si se cachea)
pip install vllm --no-cache-dir

# Dependencias del proyecto
pip install --no-cache-dir -r /workspace/Rag_Fase2/requirements.txt

# Limpiar caché pip (libera 2-4 GB)
pip cache purge
```

> Si ya hay una versión de torch instalada por NVIDIA incompatible con vLLM (ABI mismatch),
> desinstalarla primero: `pip uninstall torch -y` antes del `pip install torch`.

---

## Paso 4 — Copiar el proyecto

```bash
# Clonar o subir el repo al workspace
cd /workspace
git clone <repo_url> Rag_Fase2
cd Rag_Fase2
```

`scripts/embed_server_batching.py` ya está en el repositorio en `scripts/`.

---

## Paso 5 — Arrancar el embed server (PRIMERO)

```bash
hf auth login
# Introduce tu token HF cuando lo pida — nunca se guarda en el repo
```

```bash
HF_HOME=/workspace/hf_cache \
MODEL_ID=BAAI/bge-m3 \
PORT=8083 \
BATCH_WINDOW_MS=20 \
python3 scripts/embed_server_batching.py \
  > /tmp/embed_server.log 2>&1 &

# Esperar carga del modelo (~20-30s con cache, ~60-90s sin cache)
sleep 30
curl http://localhost:8083/health
# Respuesta esperada: {"status":"ok"}

# Test de embedding
curl -s http://localhost:8083/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["test"]}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'dim: {len(data[0])}')
"
# Respuesta esperada: dim: 1024
```

---

## Paso 6 — Configurar PostgreSQL

```bash
service postgresql start
sleep 3

sudo -u postgres psql <<'SQL'
ALTER USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE nexuspay_rag OWNER postgres;
\c nexuspay_rag
CREATE EXTENSION IF NOT EXISTS vector;
SQL

# Verificar extensión
sudo -u postgres psql -d nexuspay_rag \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

> Usar siempre `sudo -u postgres psql`. El acceso directo con `-U postgres`
> falla por autenticación peer en las instancias Vast.ai.

---

## Paso 7 — Chunking e ingestión

```bash
cd /workspace/Rag_Fase2

# Chunk de documentos Markdown (docs/ → output/chunks.json)
python3 scripts/chunker.py

# Verificar
python3 -c "import json; d=json.load(open('output/chunks.json')); print(f'{len(d)} chunks')"

# Ingestar en PostgreSQL con embeddings (batch_size=32 via TEI)
python3 scripts/ingest.py

# Verificar
sudo -u postgres psql -d nexuspay_rag \
  -c "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;"
```

---

## Paso 8 — Arrancar vLLM (DESPUÉS del embed server)

```bash
# Los modelos LLM van a /dev/shm para no saturar el disco raíz
export HF_HOME=/dev/shm/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat /workspace/.hf_home/token 2>/dev/null)

# Verificar VRAM antes (embed server debe ocupar ~3.1 GB)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# Arrancar modelo (ajustar MODEL_ID y gpu-memory-utilization según modelo)
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-12b-it \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --host 0.0.0.0 --port 8081 \
  --trust-remote-code \
  --hf-token $HF_TOKEN \
  > /tmp/vllm.log 2>&1 &

# Esperar healthcheck
until curl -sf http://localhost:8081/health > /dev/null 2>&1; do
  echo -n "."; sleep 10
done
echo " vLLM listo"

# Verificar modelo cargado
curl -s http://localhost:8081/v1/models | python3 -m json.tool
```

**Configuraciones validadas en Blackwell 48 GB:**

| Modelo | `--gpu-memory-utilization` | VRAM total (con embed) |
|---|---|---|
| Gemma 3 12B AWQ | 0.90 | ~45.7 GB / 48.9 GB |
| Qwen 2.5 14B AWQ | 0.90 | ~44.7 GB / 48.9 GB |

---

## Paso 9 — Ejecutar evaluación

Verificar configuración en `scripts/evaluator.py` antes de correr:

```python
MODEL_URL       = "http://localhost:8081/v1"
MODEL_NAME      = "google/gemma-3-12b-it"   # debe coincidir con el modelo arrancado
EMBEDDING_URL   = "http://localhost:8083/embed"
EMBEDDING_MODEL = "BAAI/bge-m3"
OUTPUT_FILE     = "output/eval_results.csv"
```

```bash
cd /workspace/Rag_Fase2

# Evaluación estándar (RAG cosine similarity → CSV)
python3 scripts/evaluator.py
# Output: output/eval_results.csv

# Evaluación HyDE (opcional)
python3 scripts/hyde.py
# Output: output/hyde_results.json

# Test del router (solo queries deterministas)
python3 scripts/test_router.py
# Output: output/router_test_results.json
```

---

## Paso 10 — Cambio de modelo (limpieza entre runs)

```bash
# Matar vLLM
kill $(pgrep -f "vllm.entrypoints") 2>/dev/null
sleep 6
ps aux | grep -E "vllm|EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Verificar VRAM liberada (deben quedar solo ~3.1 GB del embed server)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
sleep 10

# Arrancar siguiente modelo — volver al Paso 8
```

> El embed server NO se reinicia entre modelos. Permanece activo toda la sesión.

---

## Referencia de puertos

| Servicio | Puerto | Variable en script |
|---|---|---|
| Embed server (bge-m3) | 8083 | `EMBEDDING_URL` en evaluator/ingest/hyde |
| vLLM | 8081 | `MODEL_URL` en evaluator/hyde |
| PostgreSQL | 5432 | `DB_PORT` en ingest |

---

## Orden de arranque (resumen)

```
1. apt-get (postgresql + pgvector)
2. pip install torch --no-cache-dir
3. pip install vllm --no-cache-dir
4. pip install --no-cache-dir -r requirements.txt
5. service postgresql start
6. python3 scripts/embed_server_batching.py  ← SIEMPRE PRIMERO
7. python3 scripts/chunker.py
8. python3 scripts/ingest.py
9. python3 -m vllm.entrypoints.openai.api_server ...  ← DESPUÉS del embed
10. python3 scripts/evaluator.py
```
