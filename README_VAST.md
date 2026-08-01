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

# Cargar schema + seed de api_facts (necesario para el router determinista — Exp D)
python3 scripts/ingest_facts.py --file sql/setup_api_facts.sql

# Verificar
sudo -u postgres psql -d nexuspay_rag -c "SELECT COUNT(*) FROM api_facts;"
# Esperado: 32
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

`evaluator.py` y `hyde.py` aceptan el modelo, el CSV de salida y el precio de GPU por CLI —
no hace falta editar el script para cambiar de modelo.

```bash
cd /workspace/Rag_Fase2

# Evaluación estándar (RAG cosine similarity → CSV con coste)
python3 scripts/evaluator.py \
  --model google/gemma-3-12b-it \
  --output results/gemma-3-12b_selfhosted.csv \
  --gpu-hourly-rate 0.35
# Columnas de coste incluidas: cost_per_query_usd, gpu_hourly_rate_usd, equivalent_api_cost_usd
# --api-input-cost-per-m / --api-output-cost-per-m opcionales para poblar equivalent_api_cost_usd
# contra un modelo API de referencia (Kimi K3, GLM-5.2, etc. — ver FASE2_LEY.md sección 7)

# Evaluación HyDE (opcional — mismo modelo genera hipótesis y respuesta final)
python3 scripts/hyde.py \
  --model google/gemma-3-12b-it \
  --output results/gemma-3-12b_hyde.csv \
  --gpu-hourly-rate 0.35
# Output: results/gemma-3-12b_hyde.csv

# Test del router (solo queries deterministas, offline — no consulta api_facts)
python3 scripts/test_router.py
# Output: output/router_test_results.json
```

---

## Paso 10 — Cambio de modelo (limpieza entre runs)

```bash
# Matar vLLM — usar pkill/kill por PID directo, no un pipeline con awk/xargs
# (esa combinación falló de forma intermitente en la práctica; matar por PID exacto
# obtenido con pgrep -af es más fiable)
pkill -9 -f "vllm.entrypoints"
sleep 6
pkill -9 -f "EngineCore"
sleep 5
pgrep -af "vllm|EngineCore"   # debe salir vacío

# Verificar VRAM liberada (deben quedar solo ~3.1 GB del embed server)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# CRÍTICO — borrar el cache del modelo anterior en /dev/shm ANTES de cargar el siguiente.
# /dev/shm son 31GB en total; un modelo ~30B en AWQ pesa ~15-20GB, así que dos modelos
# cacheados a la vez llenan el tmpfs y el siguiente `hf download`/vLLM falla con
# "OSError: No space left on device" a mitad de la descarga.
rm -rf /dev/shm/hf_cache/hub
df -h /dev/shm   # confirmar que vuelve a ~31G libres

# Arrancar siguiente modelo — volver al Paso 8
```

> El embed server NO se reinicia entre modelos. Permanece activo toda la sesión — su cache
> vive en `/workspace/hf_cache` (disco raíz), NO en `/dev/shm`, así que no se ve afectado
> por el `rm -rf` de arriba.

---

## Paso 11 — Scoring RAGAS (juez de tercera familia)

```bash
# Parche obligatorio tras cada pip install fresco: ragas==0.4.3 importa
# ChatVertexAI/VertexAI desde un submódulo que langchain-community ya no tiene
# (movido a paquete aparte en su sunset). Import top-level, no perezoso — rompe
# ragas_eval.py antes de ejecutar una fila. Solo se usa en un isinstance()
# irrelevante si el juez es ChatOpenAI (nuestro caso), así que un try/except
# con clases dummy es seguro.
python3 - <<'PYEOF'
path = "/venv/main/lib/python3.12/site-packages/ragas/llms/base.py"
old = "from langchain_community.chat_models.vertexai import ChatVertexAI\nfrom langchain_community.llms import VertexAI\n"
new = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "    from langchain_community.llms import VertexAI\n"
    "except ImportError:\n"
    "    class ChatVertexAI:\n        pass\n"
    "    class VertexAI:\n        pass\n"
)
content = open(path, encoding="utf-8").read()
assert old in content, "import block not found — ragas version may have changed"
open(path, "w", encoding="utf-8").write(content.replace(old, new))
print("patched")
PYEOF

# Cargar el juez (tercera familia, ni Gemma ni Qwen) — volver al Paso 8 con este modelo.
# meta-llama/Llama-3.1-8B-Instruct esta gateado (requiere aceptar licencia manual en HF);
# mistralai/Mistral-7B-Instruct-v0.3 es la alternativa ungated usada en la práctica.
python3 -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --max-num-seqs 64 \
  --host 0.0.0.0 --port 8081 --trust-remote-code --hf-token $HF_TOKEN &

# Por cada CSV de evaluator.py/hyde.py ya generado:
python3 scripts/ragas_eval.py \
  --input results/<modelo>_selfhosted.csv \
  --output results/<modelo>_ragas.csv \
  --judge-model mistralai/Mistral-7B-Instruct-v0.3
```

> `ragas_eval.py` ya instancia `OpenAIEmbeddings(..., tiktoken_enabled=False)` —
> sin ese flag, `langchain_openai` pre-tokeniza el input a IDs enteros vía `tiktoken`
> (mimetizando la API de OpenAI real) y `embed_server_batching.py` lo rechaza con 422
> porque solo acepta `str`/`list[str]`. No hace falta tocar nada si se usa este script tal cual.

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
9. python3 scripts/ingest_facts.py --file sql/setup_api_facts.sql
10. python3 -m vllm.entrypoints.openai.api_server ...  ← DESPUÉS del embed
11. python3 scripts/evaluator.py --model <MODEL_ID> --output results/<modelo>.csv
```
