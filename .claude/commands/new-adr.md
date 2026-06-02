Write an ADR for: $ARGUMENTS

1. Determine the next ADR number by reading docs/adr/README.md

2. Create the file `docs/adr/{NNN}-{short-slug}.md` using this template:

```markdown
# ADR-{NNN}: {Short descriptive title}

**Date:** {today}
**Status:** Accepted

## Context
What situation prompted this decision?
What constraints exist? What is in tension?
2-4 sentences. Be specific about the actual problem.

## Decision
What was decided. State it directly and clearly.
One paragraph maximum.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| ...    | ...  | ...  | ...         |

## Consequences

**Positive:** what this enables, what becomes easier
**Negative:** what this constrains, what becomes harder
**Neutral:** things that change but aren't clearly better/worse

## Implementation notes
Specific files, patterns, or code that implements this decision.
```

3. **Update docs/adr/README.md** — add a row to the index table.

4. Tell me the ADR number and title when done.
