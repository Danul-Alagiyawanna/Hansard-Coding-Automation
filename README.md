# Multimodal Large Language Models for Trilingual Parliamentary Hansard Ingestion and Procedural Classification: A Systems-Oriented Study of the Sri Lankan Case

**Abstract.** Parliamentary Hansards are semi-structured, multilingual, and visually complex documents. For small jurisdictions with under-resourced language technology, manual coding of speeches into analytic schemas is slow and inconsistent. This paper describes an end-to-end pipeline that converts Hansard PDFs to page images, extracts speaker-attributed discourse with a vision-language model, normalizes content through translation and light structural repair, assigns thematic labels, and maps each utterance to a fine-grained procedural taxonomy aligned with Sri Lankan standing orders and section headings. The design emphasizes cross-page contextual classification, deterministic post-processing (merging continuations, filtering spurious rows, question–answer linkage), and operational observability via token-level cost accounting and tracing. We discuss strengths and failure modes of large-model-driven extraction for low-density layouts, mixed scripts, and long procedural prompts, and outline evaluation strategies suitable for legislative text. The contribution is primarily architectural and methodological: a reproducible pattern for **multilingual Hansard analytics** that combines multimodal LLMs with schema-first validation rather than a single monolithic prompt.

**Keywords:** Hansard; multilingual NLP; vision-language models; parliamentary procedure; Sri Lanka; document AI; Gemini; structured extraction

---

## 1. Introduction

Legislative open data initiatives increasingly treat verbatim records of debate as first-class corpora for transparency, research, and policy analysis. The Sri Lankan Parliamentary Hansard is **trilingual** (Sinhala, Sri Lankan Tamil, and English) and published in PDF form with typographic conventions that reflect print tradition rather than machine-readable markup. Classical OCR plus rule-based parsers struggle with speaker line breaks, marginal notes, cross-page continuations, and section-sensitive interpretation (e.g., oral answers vs. private notice questions).

Recent **multimodal large language models (MLLMs)** can, in principle, “read” page images and emit structured JSON, reducing the engineering cost of bespoke layout parsers. However, unconstrained generation yields hallucinated speakers, dropped clauses, and taxonomy drift. A practical system must therefore combine **LLM flexibility** with **schema validation**, **retries**, **cross-page state**, and **post-hoc deterministic transforms**.

This paper presents such a system as implemented in the `testGem.py` research prototype: a **sequential multi-agent pipeline** built on Google’s Gemini family (with emphasis on `gemini-3-flash-preview` for latency/cost), optional LangSmith tracing, and Pydantic models for intermediate artifacts. While we do not report new benchmark numbers on a held-out gold set (that would require domain expert annotation), we articulate the **design rationale**, **threats to validity**, and **evaluation protocols** appropriate for parliamentary corpora.

---

## 2. Related Work

**Parliamentary text as NLP corpora.** Prior work on the UK, EU, and Nordic parliaments exploited cleaner HTML or XML feeds; many techniques (topic models, sentiment, stance) assume tokenized monolingual streams. Trilingual Hansards complicate tokenization, code-switching, and alignment across language editions.

**Document AI and OCR.** Classical pipelines use layout analysis (e.g., detecting columns and blocks) before text recognition. MLLMs collapse some of these stages by conditioning directly on rasterized pages, at the cost of opacity and variable fidelity on faint scans or unusual fonts.

**Instruction-tuned JSON output.** Schema-guided generation and tool use (including code execution sandboxes) are now common patterns for reducing format errors. Parliamentary applications benefit from **strict output contracts** (arrays of speeches with required keys) validated before downstream storage.

**Procedural classification.** Beyond topic labels, legislatures require codes for question types, bill stages, points of order, privilege, adjournment motions, and written answers. Such taxonomies are **long-tailed** and **context-sensitive**; the correct label for a ministerial reply may depend on the **section heading** and the **immediately preceding question**, including when a question ends on one page and the answer begins on the next.

Our work sits at the intersection: **multimodal ingestion + multilingual normalization + heading-aware procedural IDs + cross-page discourse context.**

---

## 3. Problem Formulation

**Input.** A Hansard PDF (possibly hundreds of pages).

**Output.** A structured list of records, each describing one speech-like unit, with fields such as speaker, section heading, timestamps (original bracketed lines and normalized English clock readings where available), translated primary text, thematic classification, and a procedural **input_id** / **input_type** drawn from a fixed inventory.

**Hard constraints.**

1. **Linguistic fidelity:** Original language content must not be silently dropped; translation is auxiliary for analytics, not a substitute for archival fidelity.
2. **Procedural consistency:** IDs must respect section headings (e.g., ID 2 vs 18 for ministerial replies under different headings).
3. **Discourse continuity:** Units split across pages or interrupted by procedural interjections should be merged where appropriate.
4. **Operability:** Runs should be traceable, costed, and recoverable from transient API failures.

---

## 4. System Architecture

The implementation follows a **page-sequential** outer loop. Each page undergoes four LLM-mediated stages, then results accumulate for cross-page context. After all pages, additional **global** post-processing operates on the full corpus.

### 4.1 Stage 1 — Speaker and speech extraction (vision)

The PDF is rasterized (e.g., 200 DPI via `pdf2image`) to balance OCR-like clarity against payload size. A vision-capable Gemini model receives the page image (and derived prompts) with instructions to:

- Detect **all-caps English section headings** and associate speeches with the correct heading.
- Extract **speaker names** and **speech blocks**, including bracketed **speech start time** lines where present.
- Preserve content faithfully and exclude non-debate boilerplate where instructed.

**Rationale.** Rasterization sidesteps brittle PDF text-layer extraction when the publisher’s encoding does not match visual order. The main risk is **long pages** hitting model context or latency limits; the codebase may split or resize images where needed.

### 4.2 Stage 2 — Translation and structural analysis

A text model consumes the raw extracted speeches and produces **English analytical text** while preserving critical formatting and heading associations. This stage supports downstream classifiers that are easier to prompt in English and enables cross-lingual search for researchers who do not read Sinhala or Tamil.

**Rationale.** Translation as a separate stage isolates failure modes: extraction errors are not conflated with classification errors, and the pipeline can log which stage diverged from expectations.

### 4.3 Stage 3 — Thematic content classification

A dedicated prompt assigns **theme** metadata (main and sub categories) suitable for research dashboards (e.g., policy domains). This is distinct from procedural IDs: a speech about health might still be a “written question — follow-up” procedurally.

### 4.4 Stage 4 — Procedural input ID classification

The longest and most rule-intensive prompt maps each speech to **input_id** values keyed to sections such as:

- Oral answers to questions (written questions, replies, follow-ups, written responses printed in the appendix).
- Bills presented (administrative introduction vs. debate contributions).
- Cross-cutting oral contributions, points of order, privilege, motions, adjournment-time debates, addresses, and related categories.

**Cross-page context.** The classifier receives a **read-only window** of the last few speeches from the previous page (slimmed to save tokens). This addresses patterns such as:

- Question/answer pairing when the question appears on page *n* and the reply on page *n*+1.
- Continuing points of order or privilege declarations.
- Distinguishing **Motion at Adjournment Time** (live debate) from **forthcoming business** adjournment motions, including tie-breaking guidance using normalized clock readings **without** letting time override clear headings or ongoing debate context.

**Rationale.** Page boundaries are artifacts of printing, not of parliamentary discourse; ignoring prior-page context systematically mislabels boundary-spanning exchanges.

### 4.5 Global post-processing

After per-page accumulation, the system applies deterministic steps including:

- Merging **continuation** speeches and combining units that span pages.
- **Short speech filtering** with allowlists for procedurally short but meaningful IDs (e.g., formal motions).
- Merging consecutive segments with **similar speaker** strings to heal fragmentation.
- Marking whether **questions** received answers in the structured sense of the schema.
- Removing **address** rows (e.g., input_id 30) when those are metadata-only for the analytic corpus.

**Rationale.** LLMs are strong at semantics but inconsistent at counting lines or enforcing global invariants; Python post-processing stabilizes analytics.

---

## 5. Engineering and Operations

**Schema validation.** Pydantic models (`StructureFlags`, `Theme`, `ClassifiedSpeech`, page containers) validate shapes early and reduce garbage-in-garbage-out in later stages.

**Retries and backoff.** Transient API failures trigger bounded retries with delays, improving robustness on long jobs.

**Cost tracking.** Token usage metadata is aggregated per model with configurable price tables, printing per-call and run-level summaries—essential when processing full sittings.

**Tracing.** LangSmith-compatible wrappers annotate Gemini calls for debugging prompt regressions and comparing model versions.

**Exports.** Results are written as JSON and flattened to CSV/XLSX for analysts using spreadsheet tools.

---

## 6. Evaluation Considerations

Rigorous evaluation requires **expert-labeled gold data**. Recommended metrics:

1. **Extraction F1** at the speech level (speaker + span overlap), possibly with normalized speaker names.
2. **Translation adequacy** via BLEU/chrF (weak signals) plus **human** grading on a stratified sample.
3. **Theme accuracy** against a closed label set.
4. **Procedural ID accuracy**, stratified by section heading and by cross-page vs. within-page cases—this is expected to be the hardest task.

**Ablation studies** of interest: with vs. without cross-page context; single mega-prompt vs. staged agents; DPI and image tiling choices; model tier (Flash vs. Pro) vs. error rate and cost.

---

## 7. Limitations and Risks

- **Hallucination** in extraction remains possible; validation and spot checks against source PDFs are mandatory for high-stakes use.
- **Prompt length** for procedural rules creates **attention dilution**; maintenance may require modular rules or retrieval-augmented prompting.
- **Bias and framing** in translation/classification can skew downstream research; transparency about model version and prompt revision history is necessary.
- **Legal and ethical** use of parliamentary text is generally permitted for research, but republication of translated content may implicate third-party rights depending on jurisdiction—users should consult official guidance.

---

## 8. Future Work

- Fine-tuned or adapter-based **smaller models** for ID classification once enough labeled data exist.
- **Layout-aware** baselines (e.g., OCR + transformers) to quantify when MLLMs justify their cost.
- **Alignment** across Sinhala/Tamil/English editions of the “same” sitting if the parliament publishes parallel versions.
- **Active learning** interfaces for clerks to correct labels and feed improved prompts or training sets.

---

## 9. Conclusion

We described a **multimodal, multi-stage LLM pipeline** for trilingual Sri Lankan Hansard PDFs that treats extraction, translation, thematic labeling, and procedural coding as **separate contracts**, strengthened by **cross-page context** and **deterministic corpus-level repair**. The approach is a pragmatic response to the absence of clean digital markup: it trades some opacity in the vision stage for implementation speed, while using schema validation and post-processing to recover structure suitable for quantitative parliamentary research. Empirical benchmarking remains essential future work; the present contribution is a **reference architecture** and a checklist of **design forces** peculiar to multilingual legislative records.

---

## References (illustrative — verify and extend for formal submission)

1. Google AI. Gemini API documentation and model cards (multimodal capabilities, safety). https://ai.google.dev/
2. Bird, S. et al. *Natural Language Processing with Python* (O’Reilly) — classical NLP baselines for comparison.
3. Parliamentary copyright and reuse policies — Parliament of Sri Lanka official publications (consult current terms).
4. LangChain / LangSmith documentation — observability patterns for LLM applications. https://docs.smith.langchain.com/

---

## Appendix A — Mapping to the `testGem.py` implementation

| Paper concept | Implementation anchor |
|---------------|------------------------|
| PDF → images | `convert_pdf_to_images`, DPI defaults |
| Extraction agent | `async_speaker_extraction_agent` |
| Translation / structure | `async_translation_and_structure_agent` |
| Theme classification | `async_content_classification_agent` |
| Procedural IDs + cross-page context | `generate_input_id_prompt`, `async_input_id_classification_agent` |
| Schema | Pydantic models (`ClassifiedSpeech`, etc.) |
| Cost / tracing | `_track_cost`, `gemini_generate_content_wrapped`, LangSmith `@traceable` |
| Global merge / filter | `combine_speeches_across_pages`, `filter_short_speeches`, `mark_answered_questions`, etc. |

---

*Document version: 1.0 (April 2026). Authored to describe the `hansardGemini` / `testGem.py` research prototype; not peer-reviewed.*
