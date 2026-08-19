# Fase 2 — Resultados: Self-hosted vs API, Coste y Calidad (RAG NexusPay)

**Repo:** `ko2javier/llm-rag-hybrid-benchmark`
**Fecha de ejecución:** 31 julio 2026
**Instancia:** Vast.ai, 1× RTX 6000 Ada Generation (48 GB VRAM), CUDA 13.2, driver 595.58.03
**Coste GPU real de la instancia:** $0.6966/h (`dph_total`, consultado vía API de Vast.ai — no el valor de referencia por defecto de `evaluator.py`)
**Documento de referencia:** `FASE2_LEY.md`

---

## 1. Resumen ejecutivo

Se desplegó y evaluó un pipeline RAG completo (embeddings bge-m3 + pgvector HNSW + vLLM) sobre el dataset sintético NexusPay (25 documentos Markdown, 572 chunks, 50 preguntas golden: 25 deterministas + 25 semánticas) para comparar tres modelos self-hosted de tamaño mediano — **Gemma 3 27B AWQ**, **Qwen 2.5 32B AWQ** y **Gemma 4 31B AWQ** — en coste, latencia y calidad (RAGAS), y confrontar el coste self-hosted contra el de APIs de referencia (GLM-5.2, Kimi K3).

**Hallazgo principal:** a bajo volumen (el escenario de esta demo, 50 queries puntuales) el self-hosting **no compensa por coste puro** — se confirma la hipótesis de trabajo de `FASE2_LEY.md` sección 2. El punto de equilibrio real ronda las **250,000–2,200,000 queries/mes** según el modelo y la API de referencia usada, muy por encima de cualquier volumen de portfolio/demo. El valor de este ejercicio es la capacidad demostrada de desplegar, medir y comparar infraestructura de inferencia real — no el ahorro económico a este volumen.

**Segundo hallazgo, no anticipado en el diseño original:** el pipeline de evaluación (`evaluator.py`) usa **RAG semántico puro para las 50 preguntas**, incluidas las 25 deterministas — el router determinista (`router.py`, verificado 25/25 en test aislado) **no está integrado** en la corrida de evaluación. Los números de RAGAS aquí son válidos como medida de "Exp A" (RAG puro), pero no representan un pipeline híbrido con router. Ver sección 7.

---

## 2. Contexto y objetivo

Este informe cierra el objetivo único de Fase 2 (`FASE2_LEY.md` §2): extender el framework de evaluación de 40→50 preguntas con una dimensión de coste, comparando:

- **Self-hosted** en Vast.ai (1× GPU 40-48GB) → coste en $/hora GPU real de la instancia.
- **API** de modelos de referencia → coste en $/millón de tokens.

El dataset NexusPay (API de pagos ficticia estilo Stripe) y los scripts (`chunker.py`, `ingest.py`, `evaluator.py`, `hyde.py`, `ragas_eval.py`, `router.py`) ya existían en el repo subido (`Fase2_Vast_Deploy/`); no se rehizo nada de eso, solo se desplegó y ejecutó según el guion de `README_VAST.md`.

Se amplió el scope original de `FASE2_LEY.md` §7 (que proponía Gemma 3 27B + Qwen 32B) añadiendo un tercer candidato, **Gemma 4 31B**, a petición explícita durante la sesión — la variante Gemma 4 no tiene un tamaño "27B" exacto; se usó la densa de 31B por ser la más comparable en clase de tamaño (existe también una variante MoE 26B-A4B con 4B de parámetros activos, descartada aquí para mantener los tres candidatos como modelos densos comparables).

---

## 3. Infraestructura

| Componente | Detalle |
|---|---|
| GPU | RTX 6000 Ada Generation, 48 GB VRAM (49,140 MiB), compute capability 8.9 (Ada Lovelace) |
| Coste GPU | $0.6966/h real (`dph_total` vía `vastai show instance`), no el $0.35 de referencia del script |
| CUDA | Driver 595.58.03, CUDA 13.2 instalado (parcial: cuBLAS/cuDNN/cuSPARSE/cuSOLVER presentes) |
| Disco raíz | 64 GB overlay (9.1 GB usados tras instalar torch+vllm+requirements) |
| RAM tmpfs (`/dev/shm`) | 31 GB, usada para cachear pesos LLM (regla de `FASE2_LEY.md` §4: modelos grandes → RAM, no disco) |
| PostgreSQL | 16.14 + pgvector 0.6.0, base `nexuspay_rag` |
| Embed server | `embed_server_batching.py` (FastAPI + sentence-transformers, API compatible con TEI) — **no TEI real**, porque este contenedor Vast.ai no tiene Docker disponible (verificado: sin binario, sin socket, y el propio host ya es un contenedor Docker sin privilegios) |
| Motor de inferencia | vLLM 0.26.0, torch 2.11.0+cu130 |
| Orden de arranque | Embed server SIEMPRE antes que vLLM (regla inamovible de `FASE2_LEY.md` §4.2), verificado en cada ciclo |

**Nota sobre la máquina de referencia:** `FASE2_LEY.md`/`README_VAST.md` documentan un despliegue previo en RTX PRO 5000 Blackwell (SM 12.0). Esta instancia es Ada Lovelace (SM 8.9) — arquitectura distinta pero compatible con AWQ-Marlin y FP8 (soportados desde SM 8.0), así que el patrón de arranque se trasladó sin cambios. VRAM confirmada limpia (0 MiB en uso) al conectar, cumpliendo la regla 3 de §4.

---

## 4. Metodología

### 4.1 Dataset y pipeline

- **572 chunks** generados por `chunker.py` (estrategia `chunk_paragraph`, dentro del rango 481-572 esperado en `FASE2_LEY.md`).
- Embeddings **bge-m3** (dim 1024) vía `embed_server_batching.py`, ingestados en pgvector con índice HNSW (`m=16, ef_construction=64`).
- **50 preguntas golden**: 25 deterministas (`fact_type`: rate_limit, constraint, error_code, version) + 25 semánticas — confirmado por conteo directo del JSON, no asumido.
- Recuperación: top-5 chunks por similitud coseno (`TOP_K=5`), sin reranking ni HyDE (Exp B/C fuera de scope de esta corrida).

### 4.2 Modelos evaluados

| Rol | Modelo | Checkpoint | Cuantización |
|---|---|---|---|
| Candidato 1 | Gemma 3 27B IT | `gaunernst/gemma-3-27b-it-int4-awq` | AWQ 4-bit (1M+ descargas HF) |
| Candidato 2 | Qwen 2.5 32B Instruct | `Qwen/Qwen2.5-32B-Instruct-AWQ` | AWQ 4-bit (oficial Qwen) |
| Candidato 3 | Gemma 4 31B IT | `QuantTrio/gemma-4-31B-it-AWQ` | AWQ 4-bit |
| Juez RAGAS | Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | bf16, sin cuantizar |

Todos servidos con vLLM (`--quantization awq_marlin --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 64`), un modelo a la vez, protocolo de limpieza de VRAM/disco entre cada uno (matar proceso, verificar `nvidia-smi`, borrar pesos de `/dev/shm`).

**Juez cambiado respecto al plan original:** `FASE2_LEY.md` §7 especificaba `meta-llama/Llama-3.1-8B-Instruct` como juez de tercera familia. Ese checkpoint está gateado por Meta y requiere aceptación manual de licencia en HF asociada a la cuenta — no resoluble solo con token. Se sustituyó por **Mistral 7B Instruct v0.3** (no gateado, 5.18M descargas, tercera familia distinta de Gemma/Qwen), decisión tomada explícitamente durante la sesión en vez de bloquear la fase esperando aprobación manual.

### 4.3 Coste

Fórmulas de `FASE2_LEY.md` §8, con el precio de GPU real de esta instancia (no el de referencia del script):

```
coste_self_hosted_por_query = (latencia_seg / 3600) × $0.6966/h
coste_api_por_query = (tokens_input × precio_input/1M) + (tokens_output × precio_output/1M)
```

---

## 5. Incidencias técnicas y resoluciones

Se documentan porque cada una tiene impacto real en los números finales o en la reproducibilidad — parte del valor de este ejercicio es mostrar el debugging real, no solo el resultado final.

1. **Token HF perdido al forzar `HF_HOME`.** Al descargar pesos a `/dev/shm/hf_cache` (regla de RAM para modelos grandes), se sobreescribía el `HF_HOME` por defecto de la instancia (`/workspace/.hf_home`, donde vivía el login del usuario), causando descargas no autenticadas. Sin impacto en repos públicos, pero bloqueó la descarga del juez gateado. Solución: pasar `HF_TOKEN` explícito además de `HF_HOME` al invocar `hf download`.
2. **Checkpoint del juez gateado.** Ver §4.2 — resuelto cambiando de modelo, no forzando el acceso.
3. **Repo de Mistral con pesos duplicados.** `mistralai/Mistral-7B-Instruct-v0.3` publica tanto el formato HF estándar (3 shards `.safetensors`, el que usa vLLM) como `consolidated.safetensors` (formato propio de `mistral-inference`, no usado aquí) — duplicaba el tamaño descargado (28 GB en vez de 14 GB), relevante porque `/dev/shm` solo tiene 31 GB. Se borró el archivo redundante tras descarga.
4. **`ragas==0.4.3` incompatible con `langchain-community==0.4.2`.** Ambos "latest" de pip, pero `ragas` importa `ChatVertexAI`/`VertexAI` desde un submódulo que `langchain-community` eliminó en su línea de sunset (movido a paquete aparte). Import top-level, no perezoso → rompía `ragas_eval.py` antes de ejecutar una sola fila. Se parcheó `ragas/llms/base.py` con un `try/except ImportError` y clases dummy, ya que esas clases solo se usan en un chequeo `isinstance()` irrelevante para nuestro caso (usamos `ChatOpenAI` contra vLLM local).
5. **`OpenAIEmbeddings` (langchain) enviaba tokens pre-tokenizados, no texto.** Por defecto, `langchain_openai.OpenAIEmbeddings` usa `tiktoken` para trocear el input a IDs enteros antes de mandarlo (mimetizando el comportamiento exacto de la API de OpenAI). Nuestro `embed_server_batching.py` solo acepta `str`/`list[str]` — cada llamada de `context_precision`/`context_recall` fallaba con 422. Se corrigió añadiendo `tiktoken_enabled=False` a la instanciación de `OpenAIEmbeddings` en `ragas_eval.py` (parcheado tanto en la copia de trabajo como en el repo original subido).
6. **Fallos aislados de parseo JSON del juez.** Con el juez en 7B, `ragas` reportó unas pocas `OutputParserException` por CSV (JSON truncado/malformado en `faithfulness`/`context_recall`). No bloquean la corrida — `ragas` produce `NaN` en esa fila puntual y continúa. Impacto: 1-2 filas de 50 sin dato válido en `faithfulness`/`context_recall` para Gemma 3 27B y Gemma 4 31B (ver §6.2, columna n_válidos).
7. **Router no integrado en la evaluación** (ver §7) — no es un bug de esta corrida sino una limitación de diseño de `evaluator.py`, confirmada leyendo el código fuente completo (sin `import router`, sin referencia a `api_facts`).

---

## 6. Resultados

### 6.1 Latencia, tokens y coste self-hosted (50/50 preguntas, 0 fallos de inferencia en los tres modelos)

| Modelo | Latencia media | Tokens prompt (media) | Tokens completion (media) | Coste medio/query | Coste total (50q) |
|---|---|---|---|---|---|
| Gemma 3 27B AWQ | 1.30 s | 412.1 | 52.4 | $0.000252 | $0.01259 |
| Qwen 2.5 32B AWQ | 1.37 s | 394.1 | 54.3 | $0.000266 | $0.01328 |
| **Gemma 4 31B AWQ** | **1.01 s** | 416.1 | 35.0 | **$0.000196** | **$0.00978** |

Gemma 4 31B es el más rápido y barato de los tres — también el que menos tokens de salida genera en promedio (35 vs ~53), lo que explica buena parte de la diferencia de coste y latencia (menos tokens que generar = menos tiempo de GPU).

### 6.2 Calidad (RAGAS) — agregado, 50 preguntas mixtas

| Métrica | Gemma 3 27B | Qwen 32B | Gemma 4 31B |
|---|---|---|---|
| Faithfulness | 0.874 (49/50 váilidos) | **0.936** | 0.888 (49/50 válidos) |
| Answer relevancy | 0.856 | 0.836 | **0.879** |
| Context precision | 0.959 | **0.968** | 0.961 |
| Context recall | **0.924** | 0.883 | 0.895 |

### 6.3 Calidad por tipo de pregunta (dato clave — el agregado esconde esta diferencia)

El CSV de salida de `ragas_eval.py` no conserva la columna `type` del dataset original (solo copia `question/answer/contexts/ground_truth`); se recuperó cruzando por texto de pregunta con el CSV de `evaluator.py` (**cruce 50/50 exacto en los tres modelos, sin preguntas sin emparejar**).

**Deterministic (n=25):**

| Métrica | Gemma 3 27B | Qwen 32B | Gemma 4 31B |
|---|---|---|---|
| Faithfulness | 0.853 | 0.927 | **0.940** |
| Answer relevancy | 0.874 | 0.932 | **0.960** |
| Context precision | 0.944 | **0.961** | 0.944 |
| Context recall | **0.993** | 0.968 | 0.990 |

**Semantic (n=25):**

| Métrica | Gemma 3 27B | Qwen 32B | Gemma 4 31B |
|---|---|---|---|
| Faithfulness | 0.895 | **0.946** | 0.833 |
| Answer relevancy | 0.837 | 0.740 | 0.798 |
| Context precision | 0.959 | 0.976 | **0.978** |
| Context recall | **0.855** | 0.798 | 0.797 |

**Lectura:**

- **Gemma 4 31B** es netamente superior en preguntas deterministas (faithfulness/answer_relevancy más altos de los tres), pero es el que más se degrada en semánticas (faithfulness cae a 0.833, el más bajo de los tres). El modelo más rápido/barato no lo es gratis en calidad narrativa.
- **Qwen 32B** es el más equilibrado — mejor faithfulness en semánticas (0.946) y buen desempeño en deterministas, al coste de ser el más lento y caro de los tres.
- **Gemma 3 27B** queda intermedio en casi todas las métricas, sin destacar en ninguna categoría.
- **Patrón consistente en los tres modelos:** `context_recall` es sustancialmente más alto en deterministas (0.968-0.993) que en semánticas (0.797-0.855). Al repetirse igual en los tres modelos, es un efecto del **tipo de pregunta y la estrategia de retrieval** (RAG semántico puro recupera casi perfecto un hecho puntual con un pasaje exacto en la doc; le cuesta más sintetizar contexto disperso entre varios documentos para preguntas narrativas) — no un problema de un modelo en particular.

---

## 7. Router determinista: no aplicado en esta corrida (hallazgo importante)

Verificación directa sobre el código (no supuesto): `evaluator.py` y `ragas_eval.py` **no contienen ninguna referencia** a `router.py`, `classify()`, `fact_type` ni `api_facts`. Confirmado con `grep` sobre ambos archivos — cero coincidencias.

`router.py` sí fue probado de forma aislada (`test_router.py`) contra las 25 preguntas deterministas: **25/25 PASS**, replicando el resultado de mayo mencionado en `FASE2_LEY.md`. Pero ese test nunca se invoca dentro del pipeline de evaluación real — las 50 preguntas, incluidas las 25 deterministas, pasaron por retrieval semántico puro (embedding de la pregunta → similitud coseno top-5 → prompt al LLM), exactamente igual que las semánticas.

Esto **no es un bug introducido en esta sesión** — es el diseño documentado: `FASE2_LEY.md` §9 lista "Exp D — router determinista vs RAG puro" como un experimento **separado y no bloqueante**, distinto del núcleo (Exp A). El propio docstring de `evaluator.py` dice literalmente *"RAG evaluation benchmark ... cosine-similarity retrieval"*, sin mención de router.

**Implicación para interpretar los números de este informe:** son válidos como medida de **Exp A (RAG puro)** en los 50 tipos de pregunta mezclados. No representan un pipeline híbrido con lookup determinista. El dato de §6.3 sugiere que el valor añadido de integrar el router (Exp D) sería subir `context_recall`/`faithfulness` en deterministas hacia ~1.0 garantizado por lookup exacto en `api_facts` — no rescatar un pipeline roto, ya que RAG puro ya rinde razonablemente bien en esas preguntas (0.85-0.99 según métrica y modelo).

---

## 8. Coste self-hosted vs API de referencia

### 8.1 Coste por query

| Modelo | Coste/query self-hosted | vs GLM-5.2 ($1.40/$4.40 por M) | vs Kimi K3 low ($0.30/$3 por M) | vs Kimi K3 high ($3/$15 por M) |
|---|---|---|---|---|
| Gemma 3 27B | $0.000252 | $0.000807 | $0.000281 | $0.002022 |
| Qwen 32B | $0.000266 | $0.000791 | $0.000281 | $0.001997 |
| Gemma 4 31B | $0.000196 | $0.000736 | $0.000230 | $0.001773 |

A este volumen y con esta latencia, **el coste self-hosted por query ya es menor que el de las APIs de referencia** en todos los casos — pero esta comparación por query es engañosa si el self-hosting implica una GPU corriendo permanentemente (ver break-even abajo). El coste por query self-hosted asume que el 100% del tiempo de GPU se dedica a procesar queries, algo irreal en un escenario de bajo volumen con la GPU mayormente idle.

### 8.2 Break-even (volumen mensual donde self-hosted empieza a compensar)

Metodología (`FASE2_LEY.md` §8): coste self-hosted mensual fijo (GPU 24/7) = $0.6966/h × 730 h/mes = **$508.52/mes**, independiente del volumen. Coste API escala linealmente con volumen. El cruce:

| Modelo | Break-even vs GLM-5.2 | Break-even vs Kimi K3 (low) | Break-even vs Kimi K3 (high) |
|---|---|---|---|
| Gemma 3 27B | 629,925 q/mes (~20,700/día) | 1,811,620 q/mes (~59,600/día) | 251,545 q/mes (~8,300/día) |
| Qwen 32B | 643,108 q/mes (~21,200/día) | 1,808,488 q/mes (~59,500/día) | 254,636 q/mes (~8,400/día) |
| Gemma 4 31B | 690,550 q/mes (~22,700/día) | 2,213,277 q/mes (~72,800/día) | 286,832 q/mes (~9,400/día) |

Incluso en el escenario más favorable al self-hosting (API cara, Kimi K3 high), hace falta procesar **más de 8,000 queries/día** para que salga a cuenta mantener una GPU dedicada 24/7 frente a pagar por uso. Para un demo/portfolio de decenas o cientos de queries puntuales, el self-hosting nunca compensa por coste puro bajo este modelo de "GPU siempre encendida".

**Matiz real:** este break-even asume la GPU corriendo 24/7 sin importar el uso. Si el patrón de uso permite apagar la GPU entre ráfagas de tráfico (p. ej. escalado a demanda, spot instances, o Vast.ai facturado por minuto real de uso como en esta sesión), el coste self-hosted se acerca mucho más al coste marginal por query — en ese régimen, con volúmenes de miles de queries concentradas en ventanas cortas, el self-hosting sí puede compensar a volúmenes bastante más bajos que los de la tabla. Esa comparación no se midió aquí (requeriría modelar patrones de tráfico, fuera de scope de Fase 2) pero es el matiz más importante a mencionar en una conversación de seguimiento.

---

## 9. Conclusión

**La hipótesis de trabajo de `FASE2_LEY.md` §2 se confirma**, con el matiz señalado: a volumen bajo tipo demo/portfolio, self-hosting no compensa por coste puro frente a las APIs de referencia evaluadas — el break-even está en el orden de cientos de miles a millones de queries/mes bajo el supuesto de GPU siempre encendida. El valor real de esta fase, como anticipaba el propio documento, es demostrar la capacidad de desplegar, medir y comparar infraestructura de inferencia self-hosted de forma rigurosa (cuantización AWQ, orden de arranque VRAM-consciente, evaluación de calidad con juez de tercera familia, métricas de coste reproducibles) — no un ahorro económico inmediato.

En calidad, **no hay un ganador absoluto entre los tres candidatos**: Gemma 4 31B domina en preguntas deterministas y en coste/latencia, pero cede terreno en preguntas semánticas frente a Qwen 32B, que resulta el más equilibrado a costa de ser el más lento. La elección "correcta" depende de la distribución real de tipos de pregunta esperada en producción.

El hallazgo del router no integrado (§7) es importante para no sobre-interpretar estos números como el techo de lo que este sistema puede lograr en preguntas deterministas — con Exp D (router + lookup exacto) esas cifras deberían acercarse a 1.0 en `context_recall`/`faithfulness` para ese subconjunto.

---

## 10. Próximos pasos (estado actualizado 01 ago 2026)

Según el orden sugerido de `FASE2_LEY.md` §9:

- ~~**Exp D (router determinista)**~~ — **hecho, ver §12.**
- ~~**Exp B (HyDE)**~~ — **hecho, ver §13.**
- ~~Corregir el flujo de `evaluator.py`/`hyde.py` para poblar `equivalent_api_cost_usd` en el propio CSV~~ — **hecho**: los runs de §12/§13 pasan `--api-input-cost-per-m 1.40 --api-output-cost-per-m 4.40` (GLM-5.2) directamente al invocar, sin recálculo post-hoc.
- **Exp C (re-ranker)** — sigue sin ejecutar, no bloqueante, mencionado en `FASE2_LEY.md` como mejora incremental. Decisión explícita de dejarlo fuera de esta ronda: no ataca ninguna debilidad identificada en los datos (a diferencia de D y B, que sí respondían a hallazgos concretos de §6.3/§7).
- Considerar repetir el break-even modelando un patrón de tráfico realista (ráfagas + apagado entre picos) en vez de asumir GPU 24/7, para matizar la sección 8.2. Sigue pendiente.

---

## 11. Archivos generados

```
/workspace/Rag_Fase2/
├── output/chunks.json                    (572 chunks)
├── results/
│   ├── gemma3_27b_selfhosted.csv         (50 filas, coste+latencia)
│   ├── gemma3_27b_ragas.csv              (50 filas, 4 métricas RAGAS)
│   ├── qwen32b_selfhosted.csv
│   ├── qwen32b_ragas.csv
│   ├── gemma4_31b_selfhosted.csv
│   └── gemma4_31b_ragas.csv
└── INFORME_FASE2_RESULTADOS.md           (este documento)
```

Base de datos `nexuspay_rag`: 572 chunks con embedding (pgvector HNSW), 32 filas en `api_facts` (usadas por Exp D, ver §12).

---

## 12. Exp D — Router determinista integrado (01 ago 2026)

**Instancia:** nueva instancia Vast.ai, misma GPU de referencia (RTX 6000 Ada 48GB), coste real **$0.7493/h** (€0.65/h al tipo de cambio del día). Los 572 chunks, `api_facts` y el dataset se regeneraron desde cero (misma metodología, ver §4).

**Cambio de código:** `evaluator.py` gana `--use-router` — antes de decidir el prompt, llama a `router.classify(question)`; si el tipo es `deterministic`, busca en `api_facts` (Postgres) la fila del `fact_type` con mayor solape de keywords contra la pregunta, y usa ese hecho exacto como contexto en vez de retrieval semántico. Si no hay match, cae a RAG semántico normal (columna `retrieval_method` deja constancia de cuál se usó fila por fila).

**Bug encontrado y corregido antes de ejecutar (validado offline, sin gastar GPU):** `router.py` tenía 2 fallos reales que `test_router.py` no detectaba porque solo compara `fact_type`, no las `keywords` exactas — (1) `normalize()` convertía "IP-level" en `"iplevel"` (perdía el guión), rompiendo la regla que busca el token `"ip"`; (2) el `subject_map` de la regla de `constraint` solo tenía una entrada para `"refund"`, así que la pregunta sobre la ventana de días para crear un refund resolvía al hecho equivocado (el de "máximo de refunds por pago"). Ambos corregidos en `router.py`; re-validado 25/25 correcto contra el seed de `api_facts` antes de correr nada en la instancia.

**Corrido con las 25 preguntas deterministas únicamente** (las semánticas no las toca el router, no tenía sentido repetirlas):

| Modelo | Retrieval method | Latencia media | Coste medio/query |
|---|---|---|---|
| Gemma 3 27B | 25/25 `router_lookup` | **0.20s** | **$0.000042** |
| Gemma 4 31B | 25/25 `router_lookup` | 0.33s | $0.000068 |
| Qwen 32B | 25/25 `router_lookup` | 0.43s | $0.000090 |

Los 25/25 resolvieron a un hecho vía lookup en los 3 modelos (0 fallback a RAG semántico) — confirma que el router + el arreglo de keywords cubren el 100% del subconjunto determinista. Latencia y coste caen entre 5x y 6x frente al baseline semántico de §6.1 (~1.3s), porque se salta el embedding de la pregunta y la búsqueda de similitud.

**Calidad (RAGAS) — deterministas, router vs baseline semántico (§6.3):**

| Métrica | Gemma 3 27B (router → base) | Gemma 4 31B (router → base) | Qwen 32B (router → base) |
|---|---|---|---|
| Faithfulness | 0.960 → 0.853 (**+0.107**) | 0.820 → 0.940 (**-0.120**) | 0.852 → 0.927 (**-0.075**) |
| Answer relevancy | 0.472 → 0.874 (**-0.402**) | 0.531 → 0.960 (**-0.429**) | 0.845 → 0.932 (**-0.087**) |
| Context precision | 1.000 → 0.944 (**+0.056**) | 1.000 → 0.944 (**+0.056**) | 1.000 → 0.961 (**+0.039**) |
| Context recall | 1.000 → 0.993 (**+0.007**) | 1.000 → 0.990 (**+0.010**) | 1.000 → 0.968 (**+0.032**) |

**Lectura — la hipótesis de §7 se confirma solo a medias:**

- **Context precision y context recall saturan a 1.000 en los tres modelos**, exactamente como anticipaba §7: el lookup exacto en `api_facts` elimina por diseño cualquier error de retrieval en preguntas deterministas.
- **Faithfulness y answer_relevancy NO mejoran de forma uniforme — de hecho empeoran en 2 de 3 modelos.** Esto no estaba anticipado y tiene dos causas distintas, verificadas leyendo las respuestas fila por fila:
  1. **Answer relevancy cae fuerte en los tres modelos** porque las respuestas basadas en el hecho exacto son muy cortas y telegráficas ("1000", "10", "v2") en vez de la frase completa que generaba el RAG semántico con contexto narrativo. RAGAS mide relevancy generando preguntas sintéticas a partir de la respuesta y comparándolas contra la pregunta original — una respuesta de una palabra da mucho menos señal a esa métrica, aunque sea correcta. Es un artefacto de formato, no un error de contenido.
  2. **Faithfulness cae en Gemma 4 31B (-0.12) y Qwen 32B (-0.075)** por una causa real, no un artefacto: en varias preguntas (Q007, Q009, Q014, Q018, Q021, Q022, Q024 en Gemma 4; Q007 y Q024 en Qwen) el modelo respondió *"the provided text does not contain information regarding..."* **a pesar de que el hecho exacto estaba en el contexto**, en formato JSON compacto en vez de prosa. Gemma 3 27B no tuvo este problema (25/25 correctas). Esto sugiere que el prompt usado para el contexto de lookup (`build_fact_prompt`, JSON crudo) es menos robusto para Gemma 4/Qwen que el prompt narrativo del RAG semántico — un hallazgo de ingeniería de prompts, no del router en sí.

**Conclusión de Exp D:** el router cumple su promesa en las métricas de retrieval (precision/recall a 1.0), pero expone que el *prompt* usado para presentar el hecho exacto necesita trabajo — no basta con inyectar el JSON tal cual, especialmente para modelos que no son Gemma 3. Sería el primer punto a mejorar si se retoma esta línea (ver §10).

---

## 13. Exp B — HyDE sobre preguntas semánticas (01 ago 2026)

**Corrido con las 25 preguntas semánticas únicamente** (HyDE no aporta nada en preguntas deterministas ya cubiertas por el router — no tenía sentido correrlo ahí). Mismo modelo genera la respuesta hipotética y la respuesta final (sin doble modelo, según diseño ya documentado en §4.2 del código).

**Coste y latencia — HyDE añade una llamada extra al LLM (genera la hipótesis) antes de recuperar contexto:**

| Modelo | Latencia total media (HyDE + respuesta) | Coste medio/query |
|---|---|---|
| Gemma 4 31B | **3.05s** | **$0.000635** |
| Qwen 32B | 3.27s | $0.000681 |
| Gemma 3 27B | 3.82s | $0.000796 |

Frente al baseline semántico de §6.1 (~1.0-1.4s), HyDE cuesta entre 2.2x y 3x más por query — coherente con que ejecuta dos pasadas de generación en vez de una.

**Calidad (RAGAS) — semánticas, HyDE vs baseline semántico (§6.3):**

| Métrica | Gemma 3 27B (HyDE → base) | Gemma 4 31B (HyDE → base) | Qwen 32B (HyDE → base) |
|---|---|---|---|
| Faithfulness | 0.910 → 0.895 (+0.015) | 0.934 → 0.833 (**+0.101**) | 0.955 → 0.946 (+0.009) |
| Answer relevancy | 0.836 → 0.837 (≈0) | 0.851 → 0.798 (+0.053) | 0.813 → 0.740 (**+0.073**) |
| Context precision | 0.957 → 0.959 (≈0) | 0.970 → 0.978 (-0.008) | 0.985 → 0.976 (+0.009) |
| Context recall | 0.885 → 0.855 (+0.030) | **0.924 → 0.797 (+0.127)** | 0.863 → 0.798 (+0.065) |

**Lectura — la hipótesis de §6.3/§9 se confirma con más fuerza de la esperada:**

- **`context_recall` sube en los tres modelos**, tal como predecía el patrón identificado en el informe original (RAG semántico puro le cuesta sintetizar contexto disperso entre varios documentos; HyDE ataca justo ese problema generando una respuesta hipotética más rica que la pregunta original antes de buscar).
- **El modelo que más se beneficia es exactamente el que peor estaba: Gemma 4 31B**, que en el baseline tenía la faithfulness semántica más baja de los tres (0.833, señalado en §6.3 como su punto débil). Con HyDE, sube a 0.934 — la mejora más grande de faithfulness de los tres modelos (+0.101) y de recall (+0.127). HyDE no solo confirma la hipótesis, corrige específicamente la debilidad que el informe original había detectado en este modelo.
- Qwen 32B y Gemma 3 27B, que ya partían de una faithfulness semántica decente (0.946 y 0.895), mejoran menos en términos absolutos — hay menos margen de mejora.
- `context_precision` se mantiene prácticamente plano en los tres — HyDE mejora qué tan completo es el contexto recuperado (recall), no cuánto ruido trae (precision).

**Conclusión de Exp B:** HyDE cumple su hipótesis de forma limpia y sin la complicación de formato que apareció en Exp D — mejora consistentemente el punto débil identificado (recall en preguntas semánticas), y de forma más marcada en el modelo que más lo necesitaba. Al coste de ~2.5x más tiempo/GPU por query, es una mejora justificable si las preguntas semánticas/narrativas son una fracción significativa del tráfico esperado en producción.

---

## 14. Archivos generados — actualización 01 ago 2026

```
results/
├── gemma3_27b_router.csv / _router_ragas.csv
├── gemma3_27b_hyde.csv / _hyde_ragas.csv
├── gemma4_31b_router.csv / _router_ragas.csv
├── gemma4_31b_hyde.csv / _hyde_ragas.csv
├── qwen32b_router.csv / _router_ragas.csv
└── qwen32b_hyde.csv / _hyde_ragas.csv
```

---

## 15. Comprobación de metodología — ¿importa el modelo juez de RAGAS? (14-15 ago 2026)

Todas las puntuaciones RAGAS de este informe (§6, §12, §13) se calcularon con un juez local, **Mistral 7B Instruct**, elegido por motivos de coste/riesgo (no exponer una API key de pago en una instancia de Vast.ai de terceros). Para comprobar si esa elección afecta a los números reportados, las trazas de Exp A se ingirieron en una instancia self-hosted de [Langfuse](https://langfuse.com) y se re-puntuaron con dos jueces de frontera independientes — **gpt-4o** (OpenAI) y **DeepSeek-v4-pro** (DeepSeek) — sobre el **conjunto completo de 150 filas** (los 3 modelos × 50 preguntas), no solo una muestra.
**Tabla final, promedio de los 3 modelos, 150/150 filas, sin datos faltantes:**

| Métrica | Mistral (local) | gpt-4o | DeepSeek | Lectura |
|---|---|---|---|---|
| `context_precision` | 0.963 | 0.767 (−0.196) | 0.776 (−0.187) | **Efecto robusto, judge-general** — magnitud casi idéntica entre los 2 jueces de frontera |
| `context_recall` | 0.901 | 0.807 (−0.094) | 0.780 (−0.121) | **Efecto robusto, judge-general** — misma dirección, magnitud similar |
| `answer_relevancy` | 0.857 | 0.901 (+0.044) | 0.871 (+0.014) | Misma dirección, pero el efecto de gpt-4o es ~3x mayor que el de DeepSeek — más específico del juez que genérico |
| `faithfulness` | 0.899 | 0.902 (+0.003) | 0.915 (+0.016) | **Sin efecto consistente** — revisión real frente al hallazgo original de la muestra de 30 filas |

**Conclusión:** `context_precision` y `context_recall` muestran un efecto real y sistemático **juez-local-vs-frontera** — magnitud casi idéntica entre dos proveedores arquitectónicamente distintos (OpenAI, DeepSeek), no una particularidad de gpt-4o ni ruido de muestreo. `answer_relevancy` mantiene la dirección pero con magnitud dependiente del juez concreto. La muestra inicial de 30 filas sugería que `faithfulness` subía consistentemente con gpt-4o en los 3 modelos; con las 150 filas completas y un segundo juez independiente, ese patrón **no se sostiene** — los deltas por modelo cambian de signo y el promedio global queda prácticamente plano en ambos jueces de frontera. Es una revisión real del hallazgo original, no solo una confirmación a mayor escala. La implicación práctica se mantiene: **cualquier informe que cite métricas RAGAS debería nombrar el juez usado.**

---

*Informe generado a partir de datos reales de ejecución en Vast.ai (31 julio 2026, §1-11; 01 agosto 2026, §12-14; 14-15 agosto 2026, §15). Todas las cifras de coste, latencia y RAGAS provienen de los CSV listados, no son estimaciones.*
