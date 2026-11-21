# Blast Radius selector

Entry for the RWX "Blast Radius" track: for each patch against sqlglot at
`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`, pick the tests that can catch it.

## How it works

1. **Maps from the pinned commit** (`data/map.json.gz`, built by `tools/`):
   per-test line coverage of `sqlglot/` and `tests/` (collected with
   `sys.monitoring`, process pools run in-process so their work is attributed),
   per-query coverage for fixture-driven tests, which fixture files each test
   reads, which dialect classes each test instantiates, lines that only run on
   first use, and each test's CPU cost in the runner. Nothing in the maps knows
   about any patch.
2. **Planning** (`br/main.py`): the patch is applied in memory and diffed at the
   AST level. Comment-only changes are no-ops. A change inside a function body
   maps to the tests covering that statement; import-time changes (class
   attributes, signatures, module constants) map to the tests that use the
   module/class (instantiation, name references, import graph). Fixture edits
   map to the tests that read the fixture, down to the changed entries.
3. **Running** (`br/runner.py`): cheap candidates are run in forked children
   against the patched code (served from memory by an import hook, fixtures by an
   `open` redirect), and only the tests that fail are kept. Expensive candidates
   are selected without running when that is cheaper.
4. **Packing**: the selection is split over 1–8 workers (longest first) to
   minimise busiest worker + 2 s per worker.

## Local check

```sh
make sample   # needs the sample kit's sample/ folder next to the Makefile
```
