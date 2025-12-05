## Intent Expansion Pipeline

This pipeline classifies customer support messages for a skincare/haircare brand using a Gemini LLM backed by rule-based fallbacks and parallel batch processing for throughput.

### Approach and reasoning
- Use a rich system prompt with domain-specific taxonomy to anchor the LLM to expected classes and avoid drifting outputs.
- Keep temperature low and enforce strict JSON parsing to maximize determinism.
- Add bounded retries plus structured fallbacks so a single bad LLM response does not break processing.
- Support thread-pooled batch classification to process multiple messages concurrently without changing model semantics.

### Workflow architecture
- `IntentClassifier.classify`: builds the full prompt (system + conversation history + current message), calls Gemini, parses JSON, and validates required fields.
- Retry loop with capped attempts (`max_retries`) on parsing or API failures.
- Fallback path invokes a lightweight keyword matcher before defaulting to a safe bucket.
- `classify_batch`: optional thread pool (`parallel=True`, `max_workers`) dispatches per-message classification while preserving result order; each task has its own fallback handling.
- Configuration: `GOOGLE_API_KEY`, model name (default `gemini-2.5-flash`), `max_retries`, `fallback_enabled`, `max_workers`.

### Findings (new intents + why)
- Expanded taxonomy to better reflect real support flows and reduce overloading of generic buckets:
  - `Basic Interactions`: `language_preference` to explicitly capture language requests.
  - `About Company`: `company_name` to separate “who are you/brand name” asks.
  - `About Product`: `product_usage_sequence`, `results_timeline`, `safety_concerns` to cover sequencing, timelines, and safety questions.
  - `Recommendation`: (unchanged, but benefits from clearer product sub-intents).
  - `Logistics`: `order_confirmation`, `delivery_delay`, `delivery_contact`, `account_management` to distinguish order state, delays, delivery partner contact asks, and account changes; `service_complaint` split from generic complaint; `product_complaint` split to isolate product-performance issues.
  - `Special Categories`: `gibberish`, `out_of_scope` to catch noise and unrelated asks.
- Added practical keyword coverage to the fallback for frequent Logistics intents where users often phrase status asks casually:
  - `("Logistics", "order_status")`: "where is my order", "track", "tracking", "order status", "not received", "not delivered", "pending", "kaha hai".
  - Included additional Logistics basics (delivered-not-received, delivery_delay, wrong_order) and Basic Interactions (greetings, acknowledgment, language preference) to reduce misclassification during LLM outages.

### Failure cases, limitations, and handling
- LLM returns non-JSON or missing keys: retried up to `max_retries`; on exhaustion, rule-based fallback triggers.
- API/network errors: logged, retried, then routed to fallback so the pipeline still yields a structured result.
- Rule-based coverage is intentionally narrow; off-pattern phrasing may fall through to `Special Categories/out_of_scope`.
- Parallel batch uses threads; extremely large batches should tune `max_workers` to match available resources and rate limits.

### Guardrails and fallback strategy
- Deterministic prompt with explicit output schema and examples.
- Strict JSON parsing; errors do not propagate unhandled—bounded retries first.
- Rule-based matcher for high-frequency intents to preserve useful signals during LLM failures.
- Final safety net: returns `Special Categories/out_of_scope` with reasoning, never raises when `fallback_enabled` is true (default).
