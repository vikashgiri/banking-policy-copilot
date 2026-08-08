# Architecture Case Study: Banking Policy Copilot

## 1. Executive Summary
Banking Policy Copilot is an enterprise-style RAG system that answers policy questions with grounded citations.

It supports:
- Simple/Hybrid/Reranked RAG pipeline
- Agentic RAG mode using LangGraph
- Evaluation and basic observability
- Modular, loosely coupled design for provider and storage replacement

## 2. Problem
Bank staff need reliable answers from policy documents.
Generic LLM answers are risky due to hallucination and missing source references.

## 3. Goals
- Answer only from approved policy content
- Provide citations (source/page)
- Support both pipeline and agentic modes
- Keep architecture replaceable (LLM, embeddings, vector DB)
- Measure quality with offline evaluation

## 4. High-Level Architecture

```text
Client
  -> FastAPI (/ask, /health)
      -> Mode switch
         A) Pipeline RAG
            hybrid retrieve -> rerank -> generate
         B) Agentic RAG (LangGraph)
            rewrite -> retrieve/rerank -> answer
  -> Postgres/pgvector (knowledge)
  -> Ollama (LLM + embeddings)
  -> Logs (+ optional LangSmith)
```

## 5. Component Design

### 5.1 API Layer (src/api)
- FastAPI service endpoints
- Request validation
- Mode selection (pipeline / agentic)

### 5.2 RAG Layer (src/rag)
- PDF loading
- Chunking
- Embedding provider abstraction
- Hybrid retrieval (dense + keyword + RRF)
- Reranking abstraction
- Ingestion pipeline
- Answer pipeline

### 5.3 Agent Layer (src/agent)
- LangGraph state and nodes
- rewrite -> retrieve -> answer flow

### 5.4 LLM Layer (src/llm)
- Chat provider interface
- Ollama implementation
- Factory-based selection

### 5.5 Evaluation (src/eval)
- Golden dataset
- Offline runner
- Keyword pass/fail scoring and report

## 6. Key Design Decisions

### Decision 1: Interface + Factory pattern
Embeddings, chat model, vector store, and reranker use abstractions.
Why:
- change provider with minimal code impact
- follow DIP/OCP

### Decision 2: Hybrid retrieval
Dense search for semantic match + keyword search for exact terms, merged by RRF.
Why:
- better recall on policy language and exact phrases

### Decision 3: Reranking stage
Retrieve broader set, then keep top evidence.
Why:
- improve precision of context sent to LLM

### Decision 4: Dual mode (pipeline and agentic)
Why:
- pipeline for low-latency standard questions
- agentic for query rewrite and controlled multi-step flow

### Decision 5: Grounded prompting
Model must use context only and can say "I don't know".
Why:
- reduce hallucination risk in banking context

## 7. Data Design

### documents table (pgvector)
- content
- embedding (vector 768)
- source
- page_number
- chunk_index
- metadata

## 8. Runtime Flows

### Ingestion
PDF -> load pages -> chunk -> embed -> store in pgvector

### Pipeline ask
question -> hybrid search -> rerank -> prompt -> LLM -> answer

### Agentic ask
question -> rewrite -> hybrid search/rerank -> grounded answer

## 9. Quality and Operability

### Evaluation
- golden questions in data/eval/questions.json
- offline report with pass rate

### Observability
- structured logs for retrieval context
- LangSmith env configuration support

## 10. Security and Production Considerations
- secrets in .env (not in git)
- grounded responses with citations
- mode control at API
- future: auth gateway, RBAC, audit trail

## 11. Trade-offs
- Local simple reranker is lightweight but weaker than cross-encoder/Cohere
- Ollama is cost-effective for development; production may use managed LLM
- Basic keyword evaluation is useful but not full semantic judging

## 12. Future Roadmap
1. Redis caching
2. Cross-encoder/Cohere reranker
3. Full LangSmith instrumentation
4. Spring Boot enterprise gateway
5. Stronger evaluation (LLM-as-judge)
6. Dockerized deployment

## 13. Outcome
This project demonstrates practical enterprise RAG architecture skills:
- retrieval quality techniques
- agentic orchestration
- modular design
- evaluation mindset
- API-first delivery
