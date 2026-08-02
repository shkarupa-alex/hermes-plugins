# Clean-room specification review

## Facts & Constraints (White Hat)

- Disposable manifest mutation is feasible because both Hatch projects derive metadata from `plugin.yaml` and force-include it in wheels.
- GitHub Actions outputs require an identified resolver step, explicit job outputs, and direct `needs: release-version` wherever outputs are consumed.
- Every reusable release-gate checkout must use the canonical tag; otherwise manual dispatch can test default-branch code while publishing tag code.
- The helper must be standard-library-only if it runs before `uv sync`.
- SourceTree behavior depends on both configured prefix and entered text; empty prefix plus bare `X.Y.Z` will not trigger `tags: ["v*"]`.

## Risks & Failure Modes (Black Hat)

- Sequential `Path.replace()` calls are per-file atomic, not transactionally atomic across both manifests; the guarantee must be stated honestly.
- Artifact verification must inspect embedded `plugin.yaml` in wheel and sdist, not only filenames and core metadata.
- Project discovery must be shared by mutation and verification so orphan manifests/projects cannot drift.
- Short tag refs can be branch-ambiguous; use `refs/tags/vX.Y.Z`.
- Manual recovery of old tags that predate the helper needs an explicit support boundary.

## Strengths & Benefits (Yellow Hat)

- One canonical resolver eliminates duplicated prefix logic.
- Rejecting malformed `vv` tags is safer than silently legitimizing them.
- Disposable mutation avoids release-only commits while keeping package and manifest metadata aligned.

## Alternatives & Creative Ideas (Green Hat)

- Define one `Project` discovery model shared by apply and artifact verification.
- Emit a fully qualified ref as a third resolver output.
- Document the complete SourceTree prefix/input matrix.

## Completeness & Process (Blue Hat)

Specify exact Actions wiring, tagged checkouts for every gate, per-file atomic semantics, verifier interfaces and archive layouts, and direct release-gates dispatch behavior.

## Traceability

The ledger maps to the body, but artifact verification and SourceTree operations require stronger contracts.

## Decomposition Readiness

The work decomposes into resolver/discovery, mutation, artifact verification, workflow wiring, tests, and documentation after the shared contracts are explicit.

## Weak-Model Executability

Normalization is executable as written. Mutation, artifact inspection, and Actions wiring need exact signatures and expressions.

## Contract Completeness

Add the verifier signature, archive schemas, exact project enumeration, output/ref wiring, and honest crash semantics.

---REVIEW-META---
approval_score: 5
would_adopt: false
