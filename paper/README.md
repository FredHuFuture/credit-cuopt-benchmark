# Paper source and release build

**Credit Portfolio Optimization with cuOpt: A Reproducible Benchmark of Risk,
Constraints, and Solver Endpoints**, Fred Hu. This is a technical working
paper. A GitHub release is not an arXiv/SSRN submission or peer review.

## Build from the repository root

Install the CPU dependencies in `../README.md` and activate that environment.
For Ubuntu/Debian, install the typesetting and inspection tools:

```bash
sudo apt install texlive-latex-base texlive-latex-recommended \
    texlive-fonts-recommended texlive-science lmodern poppler-utils
bash paper/build.sh
bash paper/make_arxiv.sh
```

Outputs are `paper/main.pdf` and `paper/arxiv.tar.gz`. Published releases
attach the PDF and verified source archive, so reading requires no toolchain.
`PYTHON=/absolute/path/to/python` selects a different interpreter. A local
`.venv/bin/python` is otherwise preferred to `python` on PATH.

`build.sh --draft` permits missing measurements and warns on PDF quality
issues. Draft output is not for release. TeX and BibTeX failures remain fatal.
Strict builds require stable references, resolved citations, no rendered
placeholders or missing figures, no overfull boxes, and no Type 3 fonts.
Missing or failed Poppler inspectors are errors, not silent passes.

## Source files

| File | Purpose |
|---|---|
| `main.tex` | Claims, equations, model parameters and measurement macros |
| `references.bib` | Bibliography, including source-verification comments |
| `numbers.tex` | Generated, ignored macro values with CSV-cell provenance |
| `build.sh` | Regenerate numbers, run LaTeX/BibTeX, inspect final PDF |
| `make_arxiv.sh` | Build and verify a minimal independent source package |

The current figure is `results/m9_tail_sweep.pdf`, produced by
`scripts/plot_tail_sweep.py` from its committed CSV. Older figures remain
archived but are not automatically included in the paper.

## Measurement policy

`python scripts/make_paper_numbers.py` derives measurements, ratios and
rounding from archived result files. Each macro records the cells and
arithmetic actually used. Model parameters and experimental settings are
stated explicitly in the manuscript; they are not fabricated measurements.

Strict generation fails for missing required measurements. DNF, OOM, solver
errors, harness memory refusals and excluded contaminated timings are
printable recorded non-results. A complete CSV does not itself establish
scientific validity, continuous GPU idleness or matched solution quality.

Both P0 SOCP files support feasibility, not comparative timing. The reference
timing macros now emit an exclusion reason because the file flags contention.
QP ratios use matching raw endpoints and independent feasibility; repaired
rows are reported separately and are not evidence of equal optimality.

Historical result files are retained. Rerun into a new filename or
`results/tmp/`, never over a released CSV. See `ERRATA.md` for corrections.

## Current revision candidate

Appendix B integrates the October single-request timing and hypothetical
bond-lot pilots. The `p1_*.csv` sources and quantity sidecars accompany the
code. All three GPU smoke rows retain failed acceptance gates; they support
no timing ratio. The quantity repair does not enforce sigma, and its capped
feasible incumbent is not certified optimal. These are supplemental results,
not an expanded production-risk or performance claim.

The PDF uses a fixed October 8, 2026 candidate date and descriptive metadata.
Building replaces the local `main.pdf` and prepares a new `arxiv.tar.gz`;
published v1.0.0 assets remain intact. Fred performs final publication.

## Independent source package

`make_arxiv.sh` stages only `main.tex`, generated `numbers.tex`, `main.bbl`
and the figure PDFs named by the manuscript. It compiles that staged tree
with `pdflatex`, without Python or BibTeX and without access to parent result
files. The same strict PDF checks run on this independent build.

The archive contains source inputs, not build logs or private documents.
An existing `arxiv.tar.gz` is preserved as an ignored numbered backup before
replacement. The package is suitable for submission preparation; building it
does not submit anything or guarantee platform acceptance.

Use the released SHA-256 manifest to identify the distributed binaries.
Rebuilding on a different TeX version or date need not produce byte-identical
PDFs. The reproducibility claim concerns inputs, measurements and successful
typesetting, not identical PDF metadata across machines.
