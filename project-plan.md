# SAP BTP Buddy Project Plan

Status: active implementation plan. Measured results are reported only where saved artifacts or checks exist; otherwise the status is `NOT MEASURED`, `BLOCKED`, or `PROPOSED`.

This plan translates the Week 2 assignment into a build and submission plan for SAP BTP Buddy. It uses the Python LangChain + LangGraph track and a proposed SAP documentation corpus. The settings below are proposals, not measured results.

## 1. Proposed project scope

SAP BTP Buddy will help SAP BTP administrators and developers answer supported platform-administration and onboarding questions from a pinned, curated snapshot of public SAP documentation. The first proposed corpus is `SAP-docs/btp-cloud-platform`; `SAP-docs/btp-best-practices-guide` is an optional second source after the baseline works.

Surface: a reproducible Python pipeline documented in Jupyter, with an optional local Streamlit interface.

Proposed answer behavior: retrieve evidence, generate a cited answer, validate citations, and either ask for clarification or abstain when the corpus does not support an answer.

Out of scope: SAP tenant inspection or mutation, execution of BTP commands, private SAP Notes, personalized account information, and claims beyond the pinned documentation snapshot.

## 2. Assignment traceability

| ID | Assignment requirement | Planned artifact | Check / evidence | Status |
|---|---|---|---|---|
| R01 | One-line RAG primer | Submission notebook §1 | Names user, question type, specific corpus, surface, and quality target | PENDING |
| R02 | Complete every framework field in 1–2 sentences | Submission notebook §2 | Use case, corpus, cleaning, freshness, chunking/embedding, and retrieval are filled with actual final choices | PENDING |
| R03 | Pick a use case and build track | Project plan + submission notebook §1–2 | Custom SAP documentation Q&A and LangChain + LangGraph are explicitly selected | PENDING |
| R04 | Ingest the corpus | Ingestion code + corpus manifest | Pinned repository and allowlist load reproducibly; counts and hashes are recorded | PENDING |
| R05 | Chunk, embed, store, retrieve, and generate cited answers | Python source package + notebook §§5–8 | End-to-end smoke run produces an answer with source evidence | PENDING |
| R06 | Enterprise Q&A evaluation pattern | Evaluation dataset and report | At least 15 questions, including ambiguous, cross-document, and unanswerable cases | PENDING |
| R07 | Explain datasets, prompts/agent instructions, iterations, and learnings | Submission notebook §§3, 8, 10–12 | Actual prompts, decisions, changes, observations, and limitations are recorded | PENDING |
| R08 | Project documentation as a Google Doc | Google Doc submission copy | Document is complete, readable, and accessible from a reviewer context | PENDING |
| R09 | Video demo of five minutes or less | Demo recording and link | Shows the application, build explanation, AI coding workflow, and final result live | PENDING |
| R10 | Code or workflow assets | GitHub repository or ZIP | Fresh checkout and documented checks reproduce the project without secrets | PENDING |
| R11 | Submission form | Completed assignment form | Google Doc, video, and repository links are checked before submission | PENDING |

## 3. Proposed framework

These are planning assumptions. Replace them with measured, final values before submission.

| Framework field | Proposed answer | Status |
|---|---|---|
| Primer | My RAG app helps SAP BTP administrators and developers answer platform administration and onboarding questions from a pinned SAP-docs Markdown corpus through a local chat surface with cited answers and an explicit evidence-failure path. | PROPOSED |
| Use case | Documentation-grounded SAP BTP Q&A for supported concepts and procedures, with citations and clarification/abstention behavior. | PROPOSED |
| Corpus | Start with a curated English Markdown snapshot of `SAP-docs/btp-cloud-platform`; record actual repository commit, selected paths, file count, language, ownership, and license. | PROPOSED |
| Ingestion + cleaning | Load only an explicit path allowlist; preserve headings, lists, warnings, tables, commands, links, and source spans while removing repeated publishing boilerplate. | PROPOSED |
| Ingestion + freshness | Evaluate against an immutable pinned snapshot; define a manual refresh procedure and verify changed, added, and deleted documents before rebuilding the index. | PROPOSED |
| Chunking + embedding | Compare fixed-size chunks with Markdown-section-aware chunks; record the selected embedding model, revision, tokenizer limits, and configuration after verification. | PROPOSED |
| Retrieve | Compare dense retrieval with hybrid dense + BM25 retrieval; use a bounded top-k configuration and preserve chunk/source IDs for citation validation. | PROPOSED |

## 4. Proposed LangChain and LangGraph architecture

1. Repository snapshot and deterministic file allowlist.
2. Markdown parsing, cleaning, metadata, and provenance capture.
3. Baseline and section-aware chunking comparison.
4. Cached embeddings and persistent vector storage.
5. Lexical index using the same chunk IDs and text.
6. LangChain retrieval and generation components.
7. LangGraph state machine with retrieve, evidence-sufficiency, answer, citation-validation, clarification, abstention, and error paths.
8. Evaluation runner that saves raw per-query results before computing aggregates.
9. Notebook report and optional Streamlit interface using the same tested backend.

This is workflow orchestration, not GraphRAG. A knowledge graph is not required for this selected scope.

## 5. Required artifacts and checks

| Artifact | Contents | Check | Status |
|---|---|---|---|
| `project-plan.md` | Scope, traceability, proposed framework, requirements, enhancements, and gates | Review against the handout and mark unresolved items `PENDING` | COMPLETE |
| `notebooks/SAP_BTP_Buddy_Submission.ipynb` | Authoritative report structure and evidence placeholders | Opens as valid notebook; every evidence section remains clearly marked | COMPLETE |
| Source package | Loaders, cleaning, chunking, indexes, graph, generation, evaluation | Unit tests and an end-to-end smoke test | INDEX/EMBEDDING COMPLETE; 38 tests pass |
| Corpus manifest | Repository URL, commit, paths, counts, hashes, license, retrieval date | Rebuild produces the same manifest from the pinned snapshot | COMPLETE |
| Evaluation set | At least 15 labeled questions with expected evidence and answerability | Includes all required edge-case categories | DATASET COMPLETE; reviewed scores NOT MEASURED |
| Evaluation report | Per-query results, metrics, failure analysis, and limitations | Metrics have denominators and raw results are retained | PENDING |
| Google Doc | Transfer of the executed notebook report | Reviewer-access check and visual review | PENDING |
| Demo video | ≤5-minute walkthrough | Duration and link/access check | PENDING |
| GitHub/ZIP assets | Reproducible code and project files | Fresh-checkout smoke test; no secrets or private data | PENDING |
| Form submission | Final links and confirmation | Submission confirmation retained | PENDING |

## 6. Requirements versus enhancements

### Required

- Custom SAP documentation Q&A scope and selected LangChain + LangGraph track.
- All handout framework fields completed with final decisions.
- Working ingestion, chunking, embedding, storage, retrieval, and cited generation pipeline.
- At least 15 evaluation questions for the enterprise Q&A pattern, including ambiguous, cross-document, and unanswerable questions.
- Documentation covering overview, datasets, prompts/agent instructions, iterations, learnings, and observations.
- Google Doc, video no longer than five minutes, and GitHub/ZIP assets.

### Suggested enhancements

- Expand the evaluation set to 20 questions.
- Compare fixed-size and Markdown-section-aware chunking.
- Add hybrid dense + BM25 retrieval and optional reranking only if measured benefit justifies latency.
- Add citation validation, explicit abstention, refresh/delete fixture tests, and cold/warm timing.
- Add a local Streamlit chatbot UI.
- Add the second SAP-docs repository after the first-source baseline.
- Track Recall@5, citation precision, claim faithfulness, latency, and cost.

Enhancements must not be presented as official assignment thresholds. Their results remain `PENDING` until actually run.

## 7. Submission-format and deadline notes

The handout explicitly asks for project documentation in a Google Doc. This repository also uses Jupyter. The executed Jupyter notebook should remain the complete technical record, and a readable Google Doc should be created as the required submission copy. A notebook alone may not satisfy the handout.

The printed deadline is Sunday, October 18, 2026 at 11:59 p.m. Pacific Time, equivalent to Monday, October 19, 2026 at 1:59 a.m. Central Time. The handout permits late submission through the same form but says late work is not considered for Builder of the Week.

## 8. Build gates

| Gate | Exit condition | Status |
|---|---|---|
| G1 Plan | Scope, corpus choice, track, requirements, and artifacts are recorded | COMPLETE |
| G2 Environment | Python environment and imports are reproducible | COMPLETE |
| G3 Corpus | Pinned source and manifest are verified | COMPLETE |
| G4 Retrieval | Baseline retrieval and selected configuration have evidence | INDEX COMPLETE; experiment metrics NOT MEASURED |
| G5 Answers | Cited answers, clarification, abstention, and error paths work | PENDING |
| G6 Evaluation | Required questions, raw results, scores, and failure analysis exist | PENDING |
| G7 Submission package | Notebook, Google Doc, video, assets, links, and form are checked | PENDING |

## 9. Non-goals for this planning step

No source code, corpus, index, model call, evaluation result, Google Doc, video, GitHub publication, or assignment submission has been created by this plan.
