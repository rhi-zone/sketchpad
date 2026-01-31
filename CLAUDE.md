# CLAUDE.md

Behavioral rules for Claude Code working in this repository.

## Project Goal

Parity with stable-diffusion-burn and stable-diffusion-xl-burn, but:
- Pure Rust (no Python for weight conversion - use safetensors crate)
- Support both SD 1.x and SDXL

## Core Rule

ALWAYS NOTE THINGS DOWN. When you discover something important, write it immediately:
- Bugs/issues → fix them or add to TODO.md
- Environment issues → TODO.md
- Design decisions → docs/ or code comments
- Future work → TODO.md
- Conventions → this file

**Triggers to document immediately:**
- User corrects you → write down what you learned before fixing
- Trial-and-error (2+ failed attempts) → document what actually works
- Framework/library quirk discovered → add to relevant docs/ file
- "I'll remember this" thought → you won't, write it down now

## Workflow

**Batch cargo commands** to minimize round-trips:
```bash
cargo clippy --all-targets --all-features -- -D warnings && cargo test
```
After editing multiple files, run the full check once — not after each edit. Formatting is handled automatically by the pre-commit hook (`cargo fmt`).

**When making the same change across multiple crates**, edit all files first, then build once.

**Use `normalize view` for structural exploration:**
```bash
~/git/rhizone/normalize/target/debug/normalize view <file>    # outline with line numbers
~/git/rhizone/normalize/target/debug/normalize view <dir>     # directory structure
```

## Commit Convention

Use conventional commits: `type(scope): message`

Types:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code change that neither fixes a bug nor adds a feature
- `docs` - Documentation only
- `chore` - Maintenance (deps, CI, etc.)
- `test` - Adding or updating tests

Scope is optional but recommended for multi-crate repos.

## Negative Constraints

Do not:
- Announce actions with "I will now..." - just do them
- Write preamble or summary in generated content
- Catch generic errors - catch specific error types
- Leave work uncommitted
- Create special cases - design to avoid them; if stuck, ask user rather than special-casing
- Deprecate things - no users, just remove
- **Return tuples from functions** - use structs with named fields. Tuples obscure meaning and cause ordering bugs. Only use tuples when names would be pure ceremony (e.g., `(x, y)` coordinates).
- **Replace content when editing lists** - when adding to TODO.md or similar, extend existing content, don't replace sections.
- **Mark as done prematurely** - if work is incomplete, note what remains in TODO.md.
- Use path dependencies in Cargo.toml - causes clippy to stash changes across repos
- Use `--no-verify` - fix the issue or fix the hook
- Assume tools are missing - check if `nix develop` is available for the right environment

## Design Principles

**Unify, don't multiply.** Fewer concepts = less mental load.
- One interface that handles multiple cases > separate interfaces per case
- Extend existing abstractions > create parallel ones

**Simplicity over cleverness.**
- If proposing a new dependency, ask: can stdlib/existing code do this?
- "Going in circles" = signal to simplify, not add complexity.

**Explicit over implicit.**
- Convenience = zero-config. Hiding information = pretending everything is okay.
- Log when skipping something - user should know why.

**When stuck (2+ failed attempts):**
- Step back and reconsider the problem itself, not just try more solutions.
- Ask: "Am I solving the right problem?"

## Working Style

Agentic by default - continue through tasks unless:
- Genuinely blocked and need clarification
- Decision has significant irreversible consequences
- User explicitly asked to be consulted

When you say "do X first" or "then we can Y" - add it to TODO.md immediately. Don't just say it, track it.

Bail out early if stuck in a loop rather than burning tokens.

## Commits

Commit consistently. Each commit = one logical change.
Avoid tiny commits - batch related changes unless they're truly independent logical units.
