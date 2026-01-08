# nl2spec

**nl2spec** is a research-oriented pipeline for transforming **natural language
descriptions of API usage rules** into **executable Runtime Verification (RV)
specifications** using Large Language Models (LLMs).

The project focuses on **prompt engineering, generation, validation, and
evaluation** of Intermediate Representations (IRs), enabling controlled
experiments such as **zero-shot, one-shot, and few-shot prompting**.

---

## 🔒 Architecture Status (Important)

⚠️ **The prompting and generation architecture is frozen.**

From this point forward:

- The **codebase is stable**
- Experimental variations must be performed **only via configuration**
- Structural changes require explicit justification

This freeze enables **reproducible batch experiments**, **ablation studies**,
and **sound empirical evaluation**.

---

## 🧠 Supported IR Types

The pipeline supports the following Intermediate Representation (IR) types:

- **FSM** — Finite State Machine specifications  
- **ERE** — Event-Response Expressions  
- **EVENT** — Event-based rules  
- **LTL** — Linear Temporal Logic specifications *(experimental)*

The IR type is **automatically inferred** from each scenario’s metadata.
No manual selection is required.

---

## 📂 Repository Structure

```text
nl2spec/
├── core/               # Core validation and LLM abstractions
│   ├── inspection/     # IR schema validation
│   ├── handlers/       # Few-shot loaders
│   └── llms/           # LLM backends (mock, real)
│
├── prompts/            # Prompt templates (FSM / ERE / EVENT / LTL)
│
├── pipeline/           # Orchestration (generation, logging, batch runs)
│
├── datasets/
│   ├── nl_scenarios.json   # Natural language scenarios
│   └── fewshot/            # Few-shot examples (fsm/, ere/, event/, ltl/)
│
├── outputs/            # Experimental results (CSV, tables)
│
├── tests/              # Unit and integration tests
│
├── config.yaml         # Experimental configuration (DO NOT hardcode)
└── README.md
