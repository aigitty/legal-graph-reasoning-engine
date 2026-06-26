# How It Works — Following One Query Through the Code

This is the simple version. No jargon. We take **one real question**, drop it into
the system, and follow it step by step — *which file runs first, where it goes next,
and what each file actually does.*

> **Our example question:**
> *"I was fired without notice after 3 years at a private company in Karnataka."*

Keep that sentence in mind. We'll watch it travel through the whole system.

---

## The big picture in one breath

Think of the system as an **assembly line**. The question is a product on a conveyor
belt. It stops at 8 stations. Each station does one small job and passes it along.
At the end, a finished answer comes out.

```
  YOUR QUESTION
       │
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │   THE ASSEMBLY LINE  (8 stations, runs top to bottom)          │
  │                                                                │
  │   1. Extraction      → understand the question                 │
  │   2. Grounding       → match it to known legal topics          │
  │   3. Traversal       → pull matching law sections from the DB   │
  │   4. Sufficiency     → "is this enough to answer?"             │
  │        │                                                       │
  │        ├── not enough → 5. Expansion → back to step 3 (loop)    │
  │        │                                                       │
  │        └── enough     → 6. Synthesis → write the answer         │
  │                              │                                 │
  │                        7. Guardrail   → fact-check the answer    │
  │                              │                                 │
  │                        8. Final Response → format it nicely      │
  └──────────────────────────────────────────────────────────────┘
       │
       ▼
  FINISHED ANSWER
```

Three of those stations (1, 4, 6) ask the **AI (Gemini)** for help.
The other five are **plain code** — no AI, just rules and database lookups.

---

## Who starts everything? → `graph_agent.py`

This is the **front door**. When you run `python graph_agent.py`, this file:

1. **Asks who you are** (a quick login): `[1] Normal Citizen` or
   `[2] Lawyer / Judge / Advocate`. This choice — the **persona** — rides along
   with your question and decides *how the final answer is written* (plain and
   reassuring for a citizen; technical and section-by-section for a lawyer). It
   never changes *which* law is found — only the wording at the end.
2. **Builds the assembly line** (connects the 8 stations in order) — done once.
3. **Drops your question onto the belt** with `run("...your question...", persona)`.

```
   You type:  python graph_agent.py
                    │
                    ▼
   graph_agent.py  ──►  build_graph()   ← wires the 8 stations together (once)
                    │
                    └──►  run(question)  ← puts your question on the belt
```

The "belt" itself is a shared **clipboard** that travels with the question. Every
station reads from it and writes its results back onto it. That clipboard is one
object called `LegalQueryState` (defined in `agents/graph_state.py`). Picture it as
a form that starts blank and gets filled in box by box as it moves down the line.

```
   THE CLIPBOARD (LegalQueryState) — starts almost empty:

   ┌─────────────────────────────────────────┐
   │ raw_query:  "I was fired without notice…" │ ← filled at start
   │ persona:    "citizen" / "lawyer"          │ ← also filled at start (login)
   │ extraction:        (empty)                │
   │ grounded_concepts: (empty)                │
   │ retrieval:         (empty)                │
   │ sufficiency:       (empty)                │
   │ cited_section_ids: (empty)                │
   │ confidence:        (empty)                │
   │ final_answer:      (empty)                │
   └─────────────────────────────────────────┘
```

Now let's follow the question through each station.

---

## Station 1 — Extraction → `agents/nodes/extraction_node.py`

**Job: read the messy human sentence and pull out the useful bits.** This is the
**first AI call.**

It hands your sentence to Gemini and asks: *"What legal topics are in here? What
state? Is this even an employment-law question?"*

```
   IN:   "I was fired without notice after 3 years at a private company in Karnataka"
              │
              ▼  (asks Gemini)
   OUT:  legal_concepts = ["wrongful termination", "notice period", "retrenchment"]
         jurisdiction    = "Karnataka"
         in_domain       = true        ← yes, this IS employment law
```

It writes these onto the clipboard (the `extraction` box). The AI here is **only
reading and labeling** — it does not answer the question yet.

➡️ Passes to Station 2.

---

## Station 2 — Grounding → `agents/nodes/grounding_node.py`

**Job: match the AI's free-text topics to the system's official list of 25 legal
concepts.** No AI here — just dictionary lookups.

The system has a fixed vocabulary of concepts it actually knows about (stored in
`data/ontology/concept_map.json`). The AI might say *"wrongful termination"* but the
official concept might be spelled `wrongful_termination`. This station does that
translation, using a helper called `ground_query()` (in `agents/ontology.py`).

```
   AI said:                      Official concept it maps to:
   "wrongful termination"   ──►  wrongful_termination   ✓
   "notice period"          ──►  notice_and_pay          ✓
   "retrenchment"           ──►  retrenchment            ✓
   "divorce" (hypothetical) ──►  (no match — ignored)    ✗
```

**Why this matters:** the next station searches the database using *these exact
concept names*. If a topic isn't in the official list, the system honestly admits it
doesn't cover it instead of making something up.

It writes the matched names into `grounded_concepts`. ➡️ Passes to Station 3.

---

## Station 3 — Traversal → `agents/nodes/retrieval_node.py` (+ the `graph/` folder)

**Job: go into the Neo4j database and pull out the actual law sections connected to
those concepts.** No AI — this is pure database work, and it's the heart of the whole
project.

This station is a thin wrapper. The real database work lives in the `graph/` folder:

| File | What it does |
|---|---|
| `graph/db_connection.py` | Opens the connection to the Neo4j database |
| `graph/queries.py` | Holds the actual database search commands (Cypher) |
| `graph/traversal.py` | The "walking" logic — starts at a concept and walks outward |

### What "the graph" looks like

The database isn't tables of rows. It's a **web of connected dots**. Picture it:

```
        (Concept)                    (Concept)
     "wrongful_termination"        "retrenchment"
            ▲                            ▲
            │ APPLIES_TO                 │ APPLIES_TO
            │                            │
        (Section 25F) ──CITES──► (Section 25N) ──CITES──► (Section 2A)
            │                            │
            │ belongs to                 │ belongs to
            ▼                            ▼
         (Act: Industrial Disputes Act, 1947)
```

- **Concept** = a topic (like "wrongful termination")
- **Section** = one actual piece of law (like Section 25F)
- **Act** = the full law book a section belongs to
- The **lines** are relationships: `APPLIES_TO` (section is about this topic),
  `CITES` (this section references that section), `HAS_SECTION` (act contains section).

### How traversal "walks" the web

```
   START at the concepts from Station 2
        │
        ▼
   Step 1: "Which sections APPLY_TO 'wrongful_termination'?"
            → finds Section 25F  (these are the "anchor" sections)
        │
        ▼
   Step 2: "What do those sections CITE?"   (walk one hop outward)
            → 25F cites 25N, 25N cites 2A
        │
        ▼
   Collect everything found (up to a limit), with the law text for each.
```

This walking-outward is why it's called **graph traversal**. It starts at solid
anchor points and follows the connections, like clicking related links on Wikipedia
but in a controlled, limited way (max 2 hops by default).

```
   OUT (written to the clipboard's `retrieval` box):
        primary_sections = [25F]          ← direct hits
        all sections     = [25F, 25N, 2A] ← hits + what they cite
        + the full legal text of each one
```

**The key promise of the project lives here:** every section that ends up in the
answer was *physically pulled from the database through these connections*. Nothing
is invented.

➡️ Passes to Station 4.

---

## Station 4 — Sufficiency → `agents/nodes/sufficiency_node.py`

**Job: ask "do we have enough law here to actually answer the person?"** This is the
**second AI call.**

It shows Gemini the question + a preview of the sections found, and asks: *"Is this
enough, or are we missing something?"*

```
   IN:   question + the sections found in Station 3
              │
              ▼  (asks Gemini)
   OUT:  sufficient = true / false
         missing     = ["maybe need the gratuity section too"]  (if false)
```

This station is also a **fork in the road**. After it, a small piece of plain code
called `_route_after_sufficiency` (back in `graph_agent.py`) decides where to go:

```
                    Station 4 done
                          │
              _route_after_sufficiency decides:
                          │
        ┌─────────────────┴──────────────────┐
        │                                     │
   "not enough yet"                      "good enough"
   AND we haven't looped too many times   (OR we've looped
   AND there's something to expand        the max number of
        │                                  times — give up
        ▼                                  and answer anyway)
   go to Station 5 (Expansion)                  │
                                                ▼
                                          go to Station 6 (Synthesis)
```

**Important:** the AI only gives an *opinion* ("sufficient: true/false"). The actual
decision to loop or stop is made by **plain code counting how many times we've
looped** — never by the AI. This prevents the AI from looping forever.

---

## Station 5 — Expansion → `agents/nodes/expansion_node.py` (the loop-back)

**Job: if Station 4 said "not enough," widen the search and try again.** No AI.

It does one simple thing: increases how far traversal is allowed to walk (`max_hops`
goes from 2 to 3, up to a hard cap of 4), then sends the question **back to Station 3**
to search again — this time reaching further out into the web.

```
   Station 5: "search wider"  (max_hops + 1)
        │
        └──────────────► back to Station 3 (Traversal) ──► Station 4 again
                         (this is the only loop in the system)
```

This loop can only happen a limited number of times (`MAX_RETRIEVAL_ITERATIONS = 2`).
After that, the fork in Station 4 forces "good enough" and moves on — so it can never
spin forever.

For our example, let's say Station 4 was happy the first time. So we skip Station 5
and go straight to Station 6.

---

## Station 6 — Synthesis → `agents/nodes/synthesis_node.py`

**Job: write the actual human-readable answer, using ONLY the sections we found.**
This is the **third and final AI call.**

It hands Gemini the law text from Station 3 and says: *"Write a clear answer to the
person's question. You may ONLY use these sections. Cite them by their IDs. Do not
add any law I didn't give you."*

```
   IN:   the question + full text of [25F, 25N, 2A]
              │
              ▼  (asks Gemini)
   OUT:  draft_answer = "Since you were employed for 3 years, Section 25F
                         requires the employer to give one month's notice
                         and compensation before termination… [IDA_1947_S25F]"
         cited_section_ids = ["IDA_1947_S25F", "IDA_1947_S25N"]
```

The AI writes prose, but it's on a tight leash: it can only talk about the sections
it was handed. (There's also a shortcut here: if Station 3 found *nothing*, this
station skips the AI entirely and just says "no relevant law found" — saving a
wasted AI call.)

**This is where the persona matters.** The instructions handed to the AI come in
two parts: a shared rulebook (`synthesis_base.txt` — "only use these sections,
always tag them") that is *identical* for everyone, plus a style sheet picked from
your login: `synthesis_citizen.txt` (warm, plain language, "here's what this means
for you and what to do next") or `synthesis_lawyer.txt` (technical, structured,
section-by-section). Same law, same citations — just written for the right reader.

➡️ Passes to Station 7.

---

## Station 7 — Guardrail → `agents/nodes/output_guardrail_node.py`

**Job: fact-check the AI's answer before anyone sees it.** No AI — this is the
**trust checkpoint.**

The AI just wrote an answer and claimed it cited certain sections. This station does
**not** trust that blindly. For every section the AI claims to cite, it checks two
things:

```
   For each cited section (e.g. "IDA_1947_S25F"):

   Check 1 — PROVENANCE: "Was this section actually in what we retrieved
             in Station 3?"   (did we really show this to the AI?)

   Check 2 — EXISTENCE:  "Does this section really exist in the Neo4j
             database right now?"   (re-checks the live database)

   If a citation fails either check → it gets STRIPPED OUT and a warning added.
```

Then it computes a **confidence score** (0 to 1) using a fixed formula — how well did
we cover the concepts, how strong were the anchors, did the AI's citations check out,
etc. If confidence is too low (below 0.4), it honestly downgrades the answer's status
to "insufficient evidence."

Finally it attaches the **legal disclaimer** (always, in code — never left to the AI).

```
   OUT (written to clipboard):
        verified_section_ids = ["IDA_1947_S25F", "IDA_1947_S25N"]  ← survived both checks
        confidence           = 0.82
        status               = "ok"
        disclaimer           = "This is not legal advice…"
```

➡️ Passes to Station 8.

---

## Station 8 — Final Response → `agents/nodes/final_response_node.py`

**Job: assemble the final, polished answer the user actually reads.** No AI, no
database. Pure formatting. **This station is built to never crash** — it's the safety
net at the very end.

It looks at the `status` and picks the right kind of response:

```
   status = "ok"                   → show the answer + citations + confidence + disclaimer
   status = "insufficient_evidence"→ "I don't have enough to answer confidently…"
   status = "out_of_domain"        → "This isn't something I cover (employment law only)…"
   status = "rejected"             → "I can't help with that request."  (safety)
   status = "error"                → a calm honest error message
```

There's a **safety pecking order** here: even if the AI happily wrote an answer
upstream, if the question was out-of-domain or flagged unsafe, this station throws
that answer away and shows the honest "I can't help with that" message instead. Safety
wins over a polished answer.

It also strips any leftover `[SECTION_ID]` tags that weren't verified, so the final
text only references checked sections.

The little summary at the bottom of the answer is also tailored to your login: a
**lawyer** sees `Verified citations: …` and the raw confidence number with its
factor breakdown; a **citizen** sees a friendlier `The law behind this answer: …`
and a plain `How confident is this answer: high/moderate/low.` — no jargon.

```
   OUT:  final_answer = the complete, formatted text the person sees
```

➡️ This is the end of the line (`END`). The filled-in clipboard goes back to
`graph_agent.py`, which prints the `final_answer`.

---

## The whole trip, on one page

```
 graph_agent.py            ← front door: builds line, drops question on belt
      │
      ▼
 1. extraction_node.py     [AI #1]  understand the question
      │                             → concepts, state, in_domain?
      ▼
 2. grounding_node.py      [code]   match to the 25 official concepts
      │                             → grounded_concepts
      ▼
 3. retrieval_node.py      [code + Neo4j]  walk the graph, pull law sections
      │  (uses graph/db_connection.py, queries.py, traversal.py)
      │                             → sections + their legal text
      ▼
 4. sufficiency_node.py    [AI #2]  "enough to answer?"  → true/false
      │
      │   ┌─ not enough → 5. expansion_node.py [code] search wider ─┐
      │   │                                                          │
      │   │              (loops back up to Station 3, max 2 times)   │
      │   └──────────────────────────────────────────────────◄──────┘
      │
      ▼ enough
 6. synthesis_node.py      [AI #3]  write the answer using ONLY found sections
      │                             → draft_answer + cited_section_ids
      ▼
 7. output_guardrail_node.py [code + Neo4j]  fact-check every citation,
      │                             score confidence, add disclaimer
      ▼
 8. final_response_node.py [code]   format final answer, enforce safety
      │                             → final_answer
      ▼
 graph_agent.py prints the final answer
```

---

## The two simplest things to remember

1. **AI is used in only 3 spots** — to *understand* the question (1), to *judge if we
   have enough* (4), and to *write* the answer (6). Everything else — finding the law,
   checking citations, deciding to loop, scoring confidence, adding disclaimers — is
   **plain, predictable code.** The AI is never trusted to invent law.

2. **One clipboard travels the whole line.** It starts with just your question and
   ends completely filled in. Each station only reads what it needs and writes its one
   result. That's the entire mechanism — no station talks to another directly; they
   only talk *through the clipboard.*

---

## Where each file lives (cheat sheet)

| Station | File | AI? | One-line job |
|---|---|---|---|
| Front door | `graph_agent.py` | – | Login (persona), build the line, run the question |
| The clipboard | `agents/graph_state.py` | – | The form that travels station to station |
| Persona | `agents/persona.py` | – | "citizen" vs "lawyer" — who the answer is written for |
| 1 | `agents/nodes/extraction_node.py` | ✅ | Understand the question |
| 2 | `agents/nodes/grounding_node.py` | – | Match to 25 official concepts |
| 3 | `agents/nodes/retrieval_node.py` | – | Pull law from the database |
| — | `graph/db_connection.py` | – | Connect to Neo4j |
| — | `graph/queries.py` | – | The database search commands |
| — | `graph/traversal.py` | – | Walk the web of sections |
| 4 | `agents/nodes/sufficiency_node.py` | ✅ | "Enough to answer?" |
| 5 | `agents/nodes/expansion_node.py` | – | Search wider, loop back |
| 6 | `agents/nodes/synthesis_node.py` | ✅ | Write the answer |
| 7 | `agents/nodes/output_guardrail_node.py` | – | Fact-check citations + confidence |
| 8 | `agents/nodes/final_response_node.py` | – | Format final answer, enforce safety |
| Settings | `config.py` | – | All the knobs (limits, temperatures, weights) |
