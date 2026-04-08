# IndicScriptureQA — OpenEnv Environment

**Semantic structure and factual grounding evaluation for low-resource Indic languages.**

Most LLM benchmarks for Hindi, Sanskrit, and other Indic languages test surface-level factual recall — did the model get the right answer? This environment goes further. It evaluates whether an agent can produce answers that are not only **factually correct** but also **semantically well-formed**: logically ordered, terminologically precise, structurally complete, and coherently written — the qualities that separate a genuinely useful answer from one that merely contains the right words in the wrong shape.

The domain is Indic scriptural knowledge (Vedas, Upanishads, Ramayana, Mahabharata, Bhagavad Gita, Puranas), chosen because it stresses every axis at once: factual precision matters (misattributing a verse to the wrong text is a hallucination), but so does structural literacy — knowing that an explanation of Rta should distinguish its natural-law and moral-law dimensions, that the Samudra Manthan narrative has a specific dramatic arc, or that "nishkama karma" is the correct term, not the English gloss "selfless action."

## The problem with low-resource language evaluation

LLMs fail on low-resource languages in ways that pure accuracy metrics miss:

**Terminology collapse.** Models substitute English glosses for domain-specific terms — writing "cosmic order" instead of "Rta", "meditation" instead of "dhyana", "duty" instead of "svadharma." This strips cultural and semantic precision even when the underlying fact is technically correct.

**Structural incoherence.** Answers about complex topics arrive as bags of loosely related facts instead of logically sequenced arguments. An explanation of the six Darshanas that jumbles founders with commentators, or a Dashavatara account that breaks chronological ordering, fails structurally even if every individual claim is true.

**Completeness gaps.** Models cover one dimension of a multi-faceted concept and call it done — describing dharma only as "duty" without addressing its subtlety (sukshma), context-dependence, or the rajadharma/apaddharma/moksha-dharma triad that the Mahabharata actually teaches.

**Misconception propagation.** Some errors are so common in training data that models reproduce them confidently — Shankaracharya "founding" Vedanta (he was a commentator, not the founder), or Indra "maintaining" Rta (that's Varuna). These need active detection and penalisation, not just factual comparison.

This environment provides a structured RL benchmark for training and evaluating agents that address **all four failure modes simultaneously**.

---

## How it works

An agent receives a question and a pre-generated answer that may be flawed along any combination of axes — factual errors, poor structure, missing terminology, wrong ordering, incomplete coverage. The agent interacts with the environment through a fixed action space to iteratively improve the answer before finalising it.

## Action Space

| Action        | Payload                                   | Effect                                                |
|---------------|-------------------------------------------|-------------------------------------------------------|
| `RETRIEVE`    | Optional query string                     | Surfaces the next available source passage             |
| `EDIT`        | New answer text                           | Rewrites to fix factual errors and improve content     |
| `RESTRUCTURE` | Reorganised answer text                   | Reorganises flow, ordering, and terminology without changing facts |
| `CITE`        | Citation string (e.g. `"Bhagavad Gita 2.47"`) | Attaches a citation                               |
| `ACCEPT`      | —                                         | Accepts answer as final (terminal)                     |
| `REJECT`      | —                                         | Rejects the answer entirely (terminal)                 |

The distinction between `EDIT` and `RESTRUCTURE` is deliberate. EDIT changes what the answer says. RESTRUCTURE changes how it says it — reordering paragraphs, inserting transitions, swapping an English gloss for the correct Sanskrit term, expanding a single sentence into the three conceptual aspects the topic requires. The grader scores them differently: RESTRUCTURE is penalised if it destroys factual content, and EDIT is measured on both factual and structural improvement.

## Observation Space

| Field                | Type         | Description                                                 |
|----------------------|--------------|-------------------------------------------------------------|
| `question`           | `str`        | The question being answered                                  |
| `current_answer`     | `str`        | Current (possibly flawed) answer                             |
| `retrieved_passages` | `list[str]`  | Source passages retrieved so far                              |
| `current_citations`  | `list[str]`  | Citations attached so far                                     |
| `steps_remaining`    | `int`        | Steps left in the episode                                     |
| `task_name`          | `str`        | Active task identifier                                        |
| `feedback`           | `str?`       | Feedback from the last action (includes structural breakdown) |
| `structural_hints`   | `list[str]`  | Non-spoiler hints about expected answer structure             |

`structural_hints` are the agent's window into what the grader expects structurally — things like "Use the Sanskrit term for selfless action", "Cover scriptural, ritual, AND mathematical dimensions", or "Follow narrative arc: setup → churning → crisis → treasures → resolution." They don't reveal the answer but guide the agent toward well-formed output.

## Tasks

| Task                  | Difficulty | Max Steps | Focus |
|-----------------------|------------|-----------|-------|
| `verify-factual`      | Easy       | 5         | Can the agent distinguish a correct answer from a wrong one, accounting for both factual accuracy and structural adequacy? |
| `correct-and-cite`    | Medium     | 8         | Given a partially correct answer with missing citations and poor structure, can the agent retrieve sources, fix gaps, add terminology, and reorganise? |
| `fix-hallucination`   | Hard       | 12        | Can the agent detect subtle hallucinations woven into plausible text while simultaneously fixing structural problems: wrong concept ordering, banned misconception terms, incomplete aspect coverage? |

Each task has 5 scenarios covering the Vedas, Upanishads, Ramayana, Mahabharata, Bhagavad Gita, and Puranas. Every scenario carries both factual ground truth and a `StructuralMeta` specification defining required terms, required sections, expected ordering, and banned misconception markers.

## Reward Structure

The final score blends **factual quality** and **structural quality** into **[0.0, 1.0]**.

### Terminal reward (on ACCEPT)

| Component           | Max   | What it measures                                     |
|---------------------|-------|------------------------------------------------------|
| Factual similarity  | 0.90  | Token-F1 between final answer and ground truth       |
| Citation recall     | 0.30  | Fraction of expected citations matched               |
| Structural quality  | 0.70  | Composite of 4 axes (see below)                      |
| Efficiency bonus    | 0.20  | Reward for finishing in fewer steps                   |

### Structural quality composite (0.70 max)

| Axis             | Weight | What it catches                                                |
|------------------|--------|----------------------------------------------------------------|
| **Terminology**  | 0.30   | Are the correct Sanskrit/domain terms present? Are banned misconception markers absent? |
| **Completeness** | 0.25   | Does the answer cover all required conceptual aspects of the topic? |
| **Ordering**     | 0.25   | Do concepts appear in the expected logical/narrative sequence?  |
| **Coherence**    | 0.20   | Transition quality, sentence-structure uniformity, multi-sentence flow |

All four axes are computed without ML dependencies — token matching, keyword heuristics, positional analysis, and discourse marker detection — so the environment runs on minimal hardware (2 vCPU, 8 GB RAM).

### Per-step shaping

| Action       | Good outcome          | Bad outcome                    |
|--------------|-----------------------|--------------------------------|
| `RETRIEVE`   | +0.05 (useful)        | −0.15 (redundant, >3 times)    |
| `EDIT`       | +0.20 + quality delta | −0.20 (degradation)            |
| `RESTRUCTURE`| +0.25 + struct delta  | −0.25 (destroyed facts)        |
| `CITE`       | +0.15 (correct)       | −0.05 (wrong)                  |

Step-level rewards blend factual and structural deltas (60/40 for EDIT, structure-dominant for RESTRUCTURE), giving the agent continuous signal throughout the episode rather than only at termination.

---

## Setup

### Server (Docker)

```bash
docker build -t indic-scripture-qa .
docker run -p 8000:8000 indic-scripture-qa
```

Verify: `curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{}'`

### Inference

```bash
pip install -r requirements.txt

export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export PING_URL="http://localhost:8000"

python inference.py
```

### Validate

```bash
pip install openenv-core
openenv validate
```

## Baseline Scores

| Task                 | Score  |
|----------------------|--------|
| `verify-factual`     | ~0.40  |
| `correct-and-cite`   | ~0.30  |
| `fix-hallucination`  | ~0.22  |
| **Average**          | ~0.31  |

*(Qwen2.5-72B-Instruct, temperature=0.4, scenario 0, structural eval enabled)*

## Project Structure

```
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Server container
├── main.py                 # FastAPI server (reset/step/state)
├── environment.py          # Core env logic
├── models.py               # Typed Pydantic models + StructuralMeta
├── tasks.py                # Task definitions, scenarios, structural metadata
├── rewards.py              # Factual + structural reward computation
├── inference.py            # Baseline inference script
├── requirements.txt        # Client deps
└── requirements-server.txt # Server deps
```

## License

MIT