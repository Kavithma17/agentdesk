# AgentDesk

**A multilingual AI customer-support agent for small businesses. It answers in Sinhala, Tamil, English and Singlish, takes real actions, and hands off to a human when it isn't sure.**

> 🚧 **Status:** In active development. See the [Roadmap](#roadmap) for progress.

---

## The Problem

Small and medium businesses in Sri Lanka (online shops, clinics, tuition classes, retailers) get a constant flow of repetitive customer questions on WhatsApp, Messenger and their websites:

- *"Mage order eka koheda?"* (Where is my order?)
- *"Delivery charge eka kiyada Kandy walata?"* (What's the delivery charge to Kandy?)
- *"Can I return this item?"*

Answering them takes staff time, replies are slow outside working hours, and existing chatbots fail on two fronts:

1. **Language:** customers mix Sinhala, Tamil and English, often typed in Latin script (Singlish), which most bots can't handle.
2. **Usefulness:** FAQ bots can only recite text. They can't check an order, book an appointment or open a ticket, and when they don't know an answer they make one up.

## The Solution

AgentDesk is an AI agent that a business connects to its own knowledge and systems:

- 📚 **Grounded answers (RAG):** answers come from the business's own FAQs, policies and product catalog, with **citations** to the source document.
- 🛠️ **Tool use:** the agent takes actions such as checking order status, checking stock, booking appointments and creating support tickets.
- 🌐 **Multilingual:** it understands and replies in Sinhala, Tamil, English and Singlish.
- 🙋 **Human handoff:** when confidence is low, or the customer is upset, it escalates to a human with a summary of the conversation instead of guessing.
- 🛡️ **Guardrails:** PII masking, prompt-injection defense, and refusals for out-of-scope requests.
- 📊 **Measured, not assumed:** an evaluation suite tracks accuracy, hallucination rate, tool-call correctness, latency and cost.

## Architecture

```
                ┌──────────────────────────────┐
 Customer ───►  │  API (FastAPI) / Chat Widget │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼───────────────┐
                │        Agent (router)        │
                │  intent · language · confidence
                └───┬──────────┬───────────┬───┘
                    │          │           │
          ┌─────────▼──┐  ┌────▼──────┐  ┌─▼──────────────┐
          │ RAG        │  │ Tools     │  │ Human handoff  │
          │ retrieve + │  │ orders,   │  │ ticket +       │
          │ cite       │  │ stock,    │  │ summary        │
          └─────┬──────┘  │ booking   │  └────────────────┘
                │         └────┬──────┘
          ┌─────▼──────┐  ┌────▼──────┐
          │ Vector DB  │  │ Business  │
          │ (docs)     │  │ DB / APIs │
          └────────────┘  └───────────┘

     Guardrails · Logging/Tracing · Cost tracking  (app/core)
```

## Project Structure

```
app/
  agents/    # Agent orchestration: routing, reasoning loop, handoff logic
  api/       # FastAPI endpoints (chat, ingestion, health)
  core/      # Config, guardrails, logging, tracing
  rag/       # Document ingestion, chunking, embeddings, retrieval
  tools/     # Actions the agent can take (order lookup, tickets, booking)
data/
  docs/      # Sample business knowledge base (FAQs, policies, catalog)
evals/       # Evaluation datasets and scoring scripts
tests/       # Unit and integration tests
```

## Evaluation

AgentDesk is built eval-first. The test set covers questions in all supported languages and scores:

| Metric | What it measures | Result |
|---|---|---|
| Answer accuracy | Correct, grounded answers | _TBD_ |
| Hallucination rate | Claims not supported by sources | _TBD_ |
| Citation correctness | Citations point to the right source | _TBD_ |
| Tool-call accuracy | Right tool, right arguments | _TBD_ |
| Handoff precision | Escalates when it should, and only then | _TBD_ |
| Latency (p50 / p95) | Response time | _TBD_ |
| Cost per conversation | LLM + embedding spend | _TBD_ |

Results will be published here as each phase is completed.

## Roadmap

- [ ] **Phase 1 – Core RAG:** document ingestion, embeddings, vector store, cited answers
- [ ] **Phase 2 – Agent & tools:** tool calling against a mock business database, routing
- [ ] **Phase 3 – Evals:** multilingual test set, automated scoring (LLM-as-judge plus exact checks)
- [ ] **Phase 4 – API & UI:** FastAPI backend and an embeddable chat widget
- [ ] **Phase 5 – Production readiness:** guardrails, human handoff, observability, Docker, deployment, demo

## Tech Stack (planned)

- **Language:** Python 3.11+
- **API:** FastAPI
- **LLM:** Claude (Anthropic API)
- **Vector store:** ChromaDB (local) → pgvector (production)
- **Evals:** custom harness, LLM-as-judge
- **Deployment:** Docker

## Getting Started

_Setup instructions will be added in Phase 1._

## License

MIT
