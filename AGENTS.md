# Project instructions

## Read before working

1. Read [progress](docs/PROGRESS.md), [project context](docs/PROJECT_CONTEXT.md), and [decisions](docs/DECISIONS.md).
2. For review or revision work, read [final audit](docs/FINAL_AUDIT.md) and [revision plan](docs/REVISION_PLAN.md), including acceptance criteria and rejected recommendations.
3. Inspect `git status`, HEAD, available refs, and relevant implementation. The local checkout and cached `origin/main` currently describe different stages. A cached ref does not prove the live remote state. Do not automatically pull, overwrite, or discard local work.
4. Investigate documentation/code conflicts before deciding which is correct. Identify the commit, data version, and scope of each claim.

## Scope and evidence

- Current product scope is retrospective detected-state transition research on eligible grids (DEC-12). Patrol Top-K/allocation work is obsolete (REV-04); operational deployment is outside scope. Do not reactivate it without a new scope decision.
- NASA standard-product missing dates do not establish observed zero activity. Model/data correction is conditional on the quality evidence and REV-03 branch; do not automatically refit or declare no refit necessary.
- Preserve active decisions. Reopen a rejected approach only with new evidence or an explicit scope change; record the reason and superseding decision.
- Keep verified facts, historical observations, reported results, decisions, assumptions, and unfinished work distinct. Do not invent missing chat history, source files, test results, or approvals.
- Preserve original ZIP/reporting evidence, hashes, frozen configuration, and one-shot evaluation guards. Never remove guards to replay the historical test. Use separate revision outputs if implementation is later requested.
- Existing binary/0.1-degree notebooks and the local batch builder are historical work, not the final 0.5-degree three-state pipeline. Check the applicable snapshot before transferring a bug report.
- No figure or numeric claim without source, period, and traceable identifier. Use `AUD-*`, `REV-*`, and `DEC-*` identifiers for continuing findings, work, and decisions.

## Knowledge maintenance protocol

1. Start with the documents above and verify facts relevant to the task against the repository. Record material drift, missing inputs, and verification limits.
2. Use the revision plan's dependencies and acceptance criteria when implementing an authorized revision. A plan entry alone is not a claim of completion or permission to expand scope.
3. Update context for material architecture/data changes, audit for changed evidence, decisions for changed rationale, and plan/progress for changed work status. Avoid edits that only restate unchanged information.
4. After code changes, run relevant tests and regression checks. Protect historical artifacts; test calendar transitions, cohort rules, and provenance where affected. Documentation-only work requires link/consistency checks and confirmation that source and evidence bytes remain unchanged.
5. Before finishing, record actual checks, failures, remaining uncertainty, and the next step in progress. Mark work complete only with implementation and verification evidence. Date checks and distinguish newly executed checks from historical results.
6. Keep credentials, tokens, private account information, and secret-bearing URLs out of documentation. Dataset identifiers/checksums are appropriate; do not copy unnecessary creator-machine absolute paths.

## User and repository conventions

- Ask before destructive actions, deleting files, changing secrets, pushing to GitHub, or installing major dependencies. Preserve unrelated edits. Do not send messages to teammates without an explicit instruction.
- Prefer `rg` for search. Use Context7 when current framework/library/API behavior matters. Use Playwright for relevant frontend inspection/testing when available; record unavailable tools without inventing a run.
- If frontend work is explicitly requested, preserve the user's preference for polished, responsive, accessible interfaces: consistent typography/spacing/states, realistic content, and useful subtle motion. Prefer Tailwind, appropriate shadcn/ui and lucide-react; use shadcn/Magic UI tools when relevant and available. Verify desktop/mobile and runtime behavior. This research revision does not require creating a frontend.
- Retain course integrity requirements documented in README: traceable provenance, each member's own contribution/commits, no prohibited duplicate submission, and disclosure plus human verification of AI assistance. Do not fabricate authorship, results, or approvals; keep publisher PDFs out of Git.

## Documentation ownership

The five linked documents are the maintained knowledge base. Earlier audit/source-request documents are dated evidence, not current status. Preserve their historical content and the original files under `docs/attachments/final_artifacts/`. Detailed evidence and readiness criteria belong in the linked documents, not duplicated here.
