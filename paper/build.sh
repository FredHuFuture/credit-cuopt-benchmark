#!/usr/bin/env bash
# Build the preprint: regenerate numbers from CSVs, run TeX/BibTeX, gate PDF.
#
# ./build.sh          strict: all measurements and PDF quality checks required
# ./build.sh --draft  allow missing measurements; PDF quality issues and missing
#                    Poppler inspectors become warnings. Never submit a draft.
# TeX/BibTeX failures and absent output remain fatal in both modes.
# Requires pdflatex, bibtex and (strict mode) Poppler's pdftotext/pdffonts.
# PYTHON can select an interpreter, including an external WSL virtualenv.
#
# make_arxiv.sh sources this file to apply the SAME gate to its independent
# staged build, without regenerating numbers or compiling the working paper.

paper_check_pdf() {
    local directory=$1 draft=$2
    local log="$directory/main.log" pdf="$directory/main.pdf"
    local text fonts matches compact issue
    local missing_figure_pattern='\[figure[^]]*unavailable:[^]]*\]'
    local -a issues=()

    if [[ ! -s "$log" || ! -s "$pdf" ]]; then
        echo "FAIL: missing or empty main.log/main.pdf in $directory" >&2
        return 1
    fi
    # Inspect only the last pass: early undefined references are normal.
    if matches=$(grep -Ei '((Reference|Citation).*undefined|There were undefined (references|citations)|Undefined control sequence|Label\(s\) may have changed|Rerun to get cross-references right)' "$log"); then
        issues+=("unresolved references/citations or unstable labels in main.log: $matches")
    fi
    if matches=$(grep -E 'Overfull[[:space:]]+\\[hv]box' "$log"); then
        issues+=("overfull boxes in main.log: $matches")
    fi

    # Check rendered occurrences, not unused ?? definitions in numbers.tex.
    if ! command -v pdftotext >/dev/null 2>&1; then
        issues+=("pdftotext unavailable; install poppler-utils to verify PDF text")
    elif ! text=$(pdftotext "$pdf" -); then
        issues+=("pdftotext failed; PDF text was not verified")
    else
        if [[ "$text" == *'??'* ]]; then
            issues+=("the PDF prints a ?? placeholder")
        fi
        compact=$(printf '%s' "$text" | tr -d '[:space:]')
        # MaybeFigure's current bracketed placeholder can wrap in PDF text.
        # Retain detection of its earlier wording, with the same draft policy.
        if [[ "$compact" =~ $missing_figure_pattern || "$compact" == *notbuiltyet* ]]; then
            issues+=("a figure did not resolve in the PDF (missing-figure placeholder)")
        fi
    fi
    if ! command -v pdffonts >/dev/null 2>&1; then
        issues+=("pdffonts unavailable; install poppler-utils to verify PDF fonts")
    elif ! fonts=$(pdffonts "$pdf"); then
        issues+=("pdffonts failed; PDF fonts were not verified")
    elif matches=$(grep -E 'Type[[:space:]]*3([[:space:]]|$)' <<< "$fonts"); then
        issues+=("Type 3 fonts embedded in the PDF: $matches")
    fi

    for issue in "${issues[@]}"; do
        if [[ "$draft" == 1 ]]; then
            printf 'WARNING (draft): %s\n' "$issue" >&2
        else
            printf 'FAIL: %s\n' "$issue" >&2
        fi
    done
    if [[ ${#issues[@]} -gt 0 && "$draft" != 1 ]]; then
        return 1
    fi
}

paper_build() {
    local draft=0 py
    case "${1:-}" in
        --draft) draft=1 ;;
        --help|-h)
            echo "Usage: $0 [--draft]"
            echo "Strict requires measured numbers, resolved references/citations, no ??,"
            echo "no overfull boxes, stable labels, no Type 3 fonts, and working Poppler."
            echo "Rendered missing-figure placeholders also fail strict checks."
            echo "--draft allows missing numbers and warns on PDF quality/inspection issues."
            echo "TeX/BibTeX errors remain fatal. Draft PDFs are not for submission."
            return 0 ;;
        '') ;;
        *) echo "Unknown option: $1; use --help" >&2; return 2 ;;
    esac
    if [[ $# -gt 1 ]]; then
        echo "Usage: $0 [--draft]" >&2
        return 2
    fi
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    py=${PYTHON:-python}
    if [[ -z ${PYTHON:-} && -x ../.venv/bin/python ]]; then
        py=../.venv/bin/python
    fi

    echo "== 1/3  regenerating numbers.tex"
    if [[ "$draft" == 1 ]]; then
        "$py" ../scripts/make_paper_numbers.py --allow-missing
    elif ! "$py" ../scripts/make_paper_numbers.py; then
        echo "FAIL: numbers.tex was not regenerated; resolve missing measurements" >&2
        echo "or use ./build.sh --draft to keep drafting." >&2
        return 1
    fi

    echo "== 2/3  running LaTeX"
    if ! command -v pdflatex >/dev/null 2>&1; then
        echo "pdflatex not found; install a TeX distribution." >&2
        return 2
    fi
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    if ! bibtex main; then
        echo "FAIL: bibtex failed; check references.bib and main.blg" >&2
        return 1
    fi
    if [[ ! -s main.blg ]] || grep -qi 'error message' main.blg; then
        echo "FAIL: bibtex reported errors or produced no main.blg" >&2
        return 1
    fi
    # Four passes allow page references to settle after bibliography insertion.
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex

    echo "== 3/3  checking the final PDF"
    paper_check_pdf . "$draft"
    if [[ "$draft" == 1 ]]; then
        echo "built main.pdf (DRAFT; do not submit)"
    else
        echo "built main.pdf (strict checks passed)"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    paper_build "$@"
fi
