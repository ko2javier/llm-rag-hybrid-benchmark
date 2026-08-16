# Postmortem: observabilidad (Langfuse) y comparación de juez RAGAS

Documento de trabajo, no documentación del producto: recoge lo que salió mal (y lo que salió bien
y vale la pena citar) al montar observabilidad sobre los resultados ya producidos por este
proyecto y al comparar el juez RAGAS local contra uno de pago. Mismo formato que el POSTMORTEM del
proyecto hermano (`llm-agent-mcp-eval`): se distingue entre **Error** (llegó a ejecutarse mal) y
**Riesgo detectado a tiempo** (se vio antes de romper nada), y entre ambos y un **Hallazgo** (no es
un error de nadie, es un resultado real que vale la pena dejar escrito).

---

# Parte 1 — Langfuse self-hosted + ingesta histórica + comparación de juez (14/08/2026)

Primera vez que este proyecto tiene observabilidad centralizada. Hasta hoy, cada fase (Exp A, Exp
D, Exp B) vivía en su propio CSV/JSON en `results/`, sin forma de comparar entre fases o modelos
salvo abriendo archivos a mano. Sesión 100% local, sin GPU — Langfuse y RAGAS no necesitan modelo
propio corriendo (solo RAGAS con juez LLM-based necesita *algún* modelo vivo en el momento de
puntuar, no necesariamente el modelo evaluado), así que no se rentó ninguna instancia.

**Alcance:** ingesta de las 3 fases de este repo (Exp A completo, más las CSVs de RAGAS ya
calculadas) junto con las de los dos proyectos hermanos (`llm-agent-mcp-eval`: golden-set de
tool-calling y piloto multi-turno de personas) en un único Langfuse local — 1021 trazas en total.
Decisión de Jabier: ingerir todo el histórico, no solo lo de hoy, para tener un panel único que
cubra las 4 fases del portfolio (RAG → tool-calling → root-cause de seguridad → evaluación
multi-turno).

## E1. Mismatch de credenciales Postgres al levantar el stack de Langfuse

**Qué pasó.** `docker compose up` con un `.env` que fijaba `POSTGRES_USER`/`POSTGRES_PASSWORD`
propios hizo que `langfuse-web` fallara al arrancar: `P1000: Authentication failed... for
'postgres'`.

**Causa.** El `docker-compose.yml` oficial de Langfuse tiene `DATABASE_URL` como variable
**independiente**, con su propio valor por defecto (`postgresql://postgres:postgres@postgres:5432/postgres`)
— no se deriva automáticamente de `POSTGRES_USER`/`POSTGRES_PASSWORD`. Al fijar solo estas
últimas, el contenedor de Postgres se inicializó con un usuario nuevo, pero `langfuse-web` seguía
intentando conectar con las credenciales por defecto sin sobrescribir.

**Corrección.** Quitar el override de `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` del `.env`
y dejar que ambas partes (Postgres y `DATABASE_URL`) usen el mismo par por defecto
(`postgres`/`postgres`) — aceptable para infraestructura puramente local/efímera, no expuesta.
Requirió `docker compose down -v` (el volumen de Postgres ya se había inicializado con las
credenciales viejas) antes de volver a levantar.

**Impacto.** Bajo — un ciclo de `down -v && up` de menos de un minuto, detectado en el primer
intento de conexión, no tras publicar ningún dato.

**Lección.** Cuando un `docker-compose.yml` de terceros tiene dos variables que *parecen* la misma
cosa (credenciales de Postgres vs. la URL de conexión completa), verificar si una deriva de la
otra automáticamente o si son independientes **antes** de sobrescribir solo una — el mismo patrón
de "dos piezas que parecen acopladas pero no lo están" que ya costó tiempo en otras partes de este
portfolio (ver R1 del POSTMORTEM del proyecto hermano, puerto del embed server vs. vLLM).

## E2. El parche de `ChatVertexAI` para RAGAS, ya documentado en `README_VAST.md`, volvió a hacer falta — con una técnica distinta

**Qué pasó.** Al importar `ragas` en el nuevo entorno local (`langfuse_local/.venv`),
`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` — el mismo
síntoma que ya está documentado en `README_VAST.md` (líneas ~259-282) como "parche obligatorio
tras cada pip install fresco".

**Causa, confirmada de nuevo hoy.** `ragas==0.4.3` importa `ChatVertexAI`/`VertexAI` de forma
incondicional en `ragas/llms/base.py`, pero `langchain-community` (a partir de cierta versión)
movió/eliminó ese submódulo al separar los conectores de Google Cloud en un paquete aparte. No
hace falta Vertex AI para nada de este proyecto (juez es Mistral local o gpt-4o vía OpenAI) — el
import solo se usa para un `isinstance()` interno irrelevante en nuestro caso.

**Diferencia con el parche ya documentado.** `README_VAST.md` parchea el **archivo fuente de
ragas** (`ragas/llms/base.py`), reemplazando el bloque de import por un `try/except` con clases
dummy — funciona, pero depende de que el texto exacto del import no haya cambiado entre versiones
de ragas, y hay que reaplicarlo cada vez que se reinstala el paquete. La técnica usada hoy en
`langfuse_local/` fue distinta: crear un **módulo stub** en
`langchain_community/chat_models/vertexai.py` con una clase `ChatVertexAI` dummy, sin tocar ningún
archivo de `ragas`. Ambas resuelven el mismo síntoma; el stub tiene la ventaja de sobrevivir a un
`pip install --upgrade ragas` sin tener que volver a parchear (mientras la ruta del módulo de
`langchain_community` no cambie), a costa de depender de la estructura de paquetes de
`langchain_community` en vez de la de `ragas`.

**Impacto.** Bajo — ya estaba documentado el síntoma y la causa exacta; solo tomó confirmar que
seguía aplicando y escribir el fix una vez más, en vez de re-diagnosticar desde cero.

**Lección.** Un gotcha ya documentado en un runbook (`README_VAST.md`) de un proyecto no se
propaga automáticamente a un entorno nuevo (aquí, un venv separado para Langfuse, no el venv de
Vast.ai) — hay que releer los runbooks existentes del *dominio* (aquí, "instalar ragas") no solo
del *repo* donde se originó el problema. Vale la pena centralizar este tipo de gotcha en un lugar
único (ej. este mismo POSTMORTEM) en vez de dejarlo solo en el runbook de instalación de Vast.ai
de un proyecto hermano.

## R1. Scores `NaN` de RAGAS rechazados por la API de Langfuse (riesgo detectado a tiempo)

**Qué pasó.** El primer intento de ingesta de las CSVs de RAGAS (`gemma3_27b`/`gemma4_31b`, 3
filas combinadas) devolvió `Bad Request` de la API de Langfuse al crear el score.

**Causa.** 3 de las 150 filas del Exp A original tienen `NaN` en alguna métrica RAGAS — RAGAS
genuinamente no pudo calcular esa métrica para esas filas (no es un artefacto de la ingesta de
hoy, ya estaba así en los CSVs de la corrida original del 31 de julio). `NaN` no es JSON válido, y
el script de ingesta lo mandaba tal cual.

**Corrección.** Filtrar `NaN` antes de crear el score (`f != f` como chequeo de NaN sin depender
de `numpy`/`math`), omitiendo esa métrica puntual para esa fila en vez de fallar toda la ingesta.

**Impacto.** Ninguno en datos — se detectó antes de que ninguna traza se reportara como completa
con datos corruptos; las 150 filas terminaron ingeridas correctamente tras el fix.

**Nota aparte, no perseguida hoy:** por qué RAGAS dejó esas 3 métricas en `NaN` en la corrida
original (¿fallo del juez local, contexto vacío, timeout?) no se investigó — queda como una
pregunta abierta si se vuelve a tocar la metodología de Fase 2.

## H1. La elección de juez cambia las puntuaciones RAGAS de forma sistemática, no solo por ruido

**Qué se hizo.** Se re-puntuó una muestra acotada (10 filas de cada uno de los 3 modelos
evaluados en Exp A — 30 en total, no las 150 completas, para controlar el gasto real de API) con
**gpt-4o vía la API de OpenAI** como juez, y se comparó contra las puntuaciones ya existentes del
juez local (**Mistral-7B-Instruct**, la elección original de Fase 2 — ver
`project-fase2-vast-deploy` en memoria: elegido explícitamente para evitar gasto/riesgo de una API
de pago en una instancia de Vast.ai de terceros).

**Resultado — la diferencia es consistente en dirección, no aleatoria, en los 3 modelos:**

| Métrica | gpt-4o vs. Mistral local (promedio de las 3 corridas) |
|---|---|
| `faithfulness` | gpt-4o puntúa **más alto** (+0.03 a +0.16) |
| `answer_relevancy` | gpt-4o puntúa **más alto** (+0.02 a +0.07) |
| `context_precision` | gpt-4o puntúa **más bajo** (−0.14 a −0.18) |
| `context_recall` | gpt-4o puntúa **más bajo** (−0.08 a −0.09) |

**Por qué esto importa.** No es solo "los números son distintos" — es que **la dirección del
sesgo es la misma para los 3 modelos evaluados** (Gemma 3 27B, Gemma 4 31B, Qwen 32B), lo que
sugiere un sesgo sistemático del juez (gpt-4o es más generoso juzgando si una respuesta es fiel al
contexto y relevante a la pregunta; más estricto juzgando si el contexto recuperado es preciso y
completo) y no ruido de muestreo. Con solo 10 filas por modelo no alcanza para probarlo
estadísticamente, pero la consistencia de dirección en las 3 corridas independientes es una señal
más fuerte que un solo delta aislado.

**Relevancia para la decisión original de Fase 2.** La elección de Jabier de usar un juez local
(Mistral) en vez de una API de pago fue tomada por costo/riesgo, no por convicción de que el juez
elegido no importa. Este resultado confirma que **sí importa, y de forma sistemática**: reportar
un número de RAGAS sin nombrar qué juez lo calculó es una comparación incompleta. Vale la pena
dejar esto explícito en cualquier informe que cite las métricas de Fase 2 de ahora en más.

**Actualización 2026-08-15 — extendido a las 150 filas completas y a un tercer juez
(DeepSeek-v4-pro), con datos ya sin huecos.** El re-puntuado completo con gpt-4o sufrió pérdida de
datos por rate-limit de OpenAI (hasta 34% de celdas en `context_recall` para qwen32b) — se
verificó que el hueco **no era aleatorio**: las filas perdidas tenían un contexto recuperado
~40-50% más largo de media que las filas con dato, consistente en los 3 modelos, lo que habría
sesgado cualquier media calculada solo con las filas disponibles hacia las preguntas de contexto
corto. Se rellenaron las ~108 celdas faltantes con un segundo paso (`ragas_gpt4o_fill_missing.py`,
concurrencia baja — `max_workers=2` vs. el 16 por defecto — para respetar el límite de tokens/min)
hasta cobertura 150/150. Se añadió además **DeepSeek-v4-pro** como tercer juez, independiente de
OpenAI y de Anthropic, para aislar si el efecto era "gpt-4o específicamente" o "cualquier juez de
frontera" — pregunta que el hallazgo original dejaba abierta.

**Tabla final, promedio de los 3 modelos, 150/150 filas, sin datos faltantes:**

| Métrica | Mistral (local) | gpt-4o | DeepSeek | Lectura |
|---|---|---|---|---|
| `context_precision` | 0.963 | 0.767 (−0.196) | 0.776 (−0.187) | **Efecto robusto, judge-general** — magnitud casi idéntica entre los 2 jueces de frontera |
| `context_recall` | 0.901 | 0.807 (−0.094) | 0.780 (−0.121) | **Efecto robusto, judge-general** — misma dirección, magnitud similar |
| `answer_relevancy` | 0.857 | 0.901 (+0.044) | 0.871 (+0.014) | Misma dirección, pero el efecto de gpt-4o es ~3x mayor que el de DeepSeek — más específico del juez que genérico |
| `faithfulness` | 0.899 | 0.902 (+0.003) | 0.915 (+0.016) | **Sin efecto consistente** — revisión real frente al hallazgo original |

**Esto revisa, no solo confirma, la tabla original de la muestra de 30 filas.** El hallazgo inicial
decía que `faithfulness` subía consistentemente con gpt-4o (+0.03 a +0.16) en los 3 modelos. Con
las 150 filas completas y un segundo juez independiente, `faithfulness` **no muestra ningún patrón
judge-general** — por modelo el signo cambia (gemma3 −0.030, gemma4 +0.072, qwen32b −0.034 con
gpt-4o) y la media global queda prácticamente plana en ambos jueces de frontera (+0.003 y +0.016).
`answer_relevancy` sí mantiene la dirección pero con magnitud claramente dependiente del juez
concreto (gpt-4o +0.044 vs. DeepSeek +0.014) — tampoco es un efecto genérico y uniforme como se
pensaba inicialmente.

**Lo que sí queda confirmado y reforzado:** `context_precision` y `context_recall` son un efecto
real, sistemático y **genérico de juez-local-vs-frontier** — no una particularidad de gpt-4o, no
ruido de muestra, y no un artefacto de datos incompletos. Es la parte del hallazgo original que se
sostiene sin matices tras la verificación completa.

## Estado final

Langfuse corriendo local (`D:\LLM_Testing\langfuse_local\`, Docker Compose, sin exponer a
internet), 1021 trazas del histórico completo del portfolio (Fase 2 RAG + tool-calling/T08 +
piloto de personas), con las puntuaciones ya calculadas de cada fase adjuntas como scores. Script
de ingesta (`ingest.py`) y de comparación de juez (`ragas_gpt4o_compare.py`) versionables si se
decide llevarlos a un repo — no se hizo hoy, viven sueltos en `langfuse_local/`.
