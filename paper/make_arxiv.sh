#!/usr/bin/env bash
# Assemble arxiv.tar.gz from current main.tex, numbers.tex, main.bbl and the
# literal PDF basenames in \MaybeFigure{...}. Run build.sh first for fresh data.
#
# ./make_arxiv.sh          independently compile and strictly check staged sources
# ./make_arxiv.sh --draft  downgrade PDF quality/inspection failures to warnings;
#                         draft archives are NOT suitable for submission
# Rendered missing-figure placeholders fail strict checks and warn in draft.
# Missing/unsafe source files and TeX errors remain fatal in both modes.
# Poppler inspectors are required in strict mode. Uses GNU coreutils/tar (WSL).
# Sources are built in a private mktemp directory without BibTeX or a generator.
# Only allowlisted sources enter the archive, never build debris or private data.
# Publish after verification; retain previous archives as arxiv.tar.gz.~N~.
# Existing arxiv_stage and original source/build files are never cleaned up.

set -euo pipefail
cd -- "$(dirname -- "$0")"
PAPER_DIR=$(pwd -P)
REPO_DIR=$(cd .. && pwd -P)
# Definitions only: sourcing does not regenerate or build the working paper.
source "$PAPER_DIR/build.sh"

DRAFT=0
case "${1:-}" in
    --draft) DRAFT=1 ;;
    --help|-h)
        echo "Usage: $0 [--draft]"
        echo "Strict independently checks staged references/citations, ??, figures, boxes and fonts."
        echo "--draft warns on PDF quality/inspection issues; do not submit its archive."
        echo "Unsafe/missing source files and TeX failures stay fatal."
        echo "Previous archives are preserved as arxiv.tar.gz.~N~. Requires GNU tools."
        exit 0 ;;
    '') ;;
    *) echo "Unknown option: $1; use --help" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--draft]" >&2
    exit 2
fi

TARBALL="$PAPER_DIR/arxiv.tar.gz"
if [[ -L "$TARBALL" || ( -e "$TARBALL" && ! -f "$TARBALL" ) ]]; then
    echo "FAIL: archive destination must be a regular file, not a symlink/directory" >&2
    exit 1
fi

require_plain_file() {
    if [[ ! -f "$1" || -L "$1" || $(realpath -e -- "$1") != "$1" ]]; then
        echo "FAIL: missing, symlinked or non-regular source: $1" >&2
        return 1
    fi
}
FILES=(main.tex numbers.tex main.bbl)
for file in "${FILES[@]}"; do
    require_plain_file "$PAPER_DIR/$file"
done

umask 077
TEMP_BASE=$(cd -- "${TMPDIR:-/tmp}" && pwd -P)
WORK=
ARCHIVE_TMP=
valid_work_dir() {
    [[ -n "$WORK" && "$WORK" == "$TEMP_BASE/"* &&
       "${WORK%/*}" == "$TEMP_BASE" &&
       "${WORK##*/}" =~ ^cuopt-arxiv\.[[:alnum:]]{10}$ &&
       -d "$WORK" && ! -L "$WORK" && -O "$WORK" &&
       $(realpath -e -- "$WORK") == "$WORK" ]]
}
valid_archive_temp() {
    [[ -n "$ARCHIVE_TMP" && "${ARCHIVE_TMP%/*}" == "$PAPER_DIR" &&
       "${ARCHIVE_TMP##*/}" =~ ^\.arxiv\.tar\.gz\.[[:alnum:]]{10}$ &&
       -f "$ARCHIVE_TMP" && ! -L "$ARCHIVE_TMP" && -O "$ARCHIVE_TMP" &&
       $(realpath -e -- "$ARCHIVE_TMP") == "$ARCHIVE_TMP" ]]
}
cleanup() {
    local status=$?
    if [[ -n "$ARCHIVE_TMP" ]]; then
        if valid_archive_temp; then
            rm -f -- "$ARCHIVE_TMP" || status=1
        else
            echo "FAIL: refusing cleanup of unvalidated archive temp: $ARCHIVE_TMP" >&2
            status=1
        fi
    fi
    if [[ -n "$WORK" ]]; then
        if valid_work_dir; then
            # Exact owned mktemp directory, validated again immediately before removal.
            rm -rf --one-file-system -- "$WORK" || status=1
        else
            echo "FAIL: refusing cleanup of unvalidated staging temp: $WORK" >&2
            status=1
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
WORK=$(mktemp -d "$TEMP_BASE/cuopt-arxiv.XXXXXXXXXX")
valid_work_dir || { echo "FAIL: invalid staging directory" >&2; exit 1; }
STAGE="$WORK/source"
mkdir -- "$STAGE"
cp -- "${FILES[@]}" "$STAGE/"

# Parse the copied source, so the manifest and compiled main.tex agree. The
# paper uses literal, single-line MaybeFigure calls; reject paths and options.
declare -A SEEN=()
FIGURE_PATTERN='\\MaybeFigure\{([^}]*)\}'
while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line%%\%*}
    while [[ "$line" =~ $FIGURE_PATTERN ]]; do
        match=${BASH_REMATCH[0]}
        fig=${BASH_REMATCH[1]}
        if [[ ! "$fig" =~ ^[[:alnum:]][[:alnum:]_.-]*\.pdf$ ||
              "${fig,,}" == nul.* || "$fig" == main.pdf ]]; then
            echo "FAIL: unsafe or reserved figure basename: $fig" >&2
            exit 1
        fi
        require_plain_file "$REPO_DIR/results/$fig"
        if [[ -z ${SEEN[$fig]:-} ]]; then
            cp -- "$REPO_DIR/results/$fig" "$STAGE/$fig"
            FILES+=("$fig")
            SEEN[$fig]=1
        fi
        line=${line#*"$match"}
    done
done < "$STAGE/main.tex"

echo "== independently verifying staged sources (pdflatex only)"
(
    cd -- "$STAGE"
    for _ in 1 2 3 4; do
        # Do not inherit repo-specific search paths or permit TeX shell commands
        # and parent-path reads/writes. Standard system TeX packages still work.
        if ! env -u TEXINPUTS -u BIBINPUTS -u BSTINPUTS -u TEXMFOUTPUT \
            openin_any=p openout_any=p \
            pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex \
            > "$WORK/pdflatex.stdout" 2>&1; then
            cat -- "$WORK/pdflatex.stdout" >&2
            echo "FAIL: staged pdflatex failed; archive not replaced" >&2
            exit 1
        fi
    done
)
paper_check_pdf "$STAGE" "$DRAFT"

# An explicit manifest excludes .env, private/, nul and compiler-created files.
# Create the temporary archive beside its destination for atomic publication.
ARCHIVE_TMP=$(mktemp "$PAPER_DIR/.arxiv.tar.gz.XXXXXXXXXX")
valid_archive_temp || { echo "FAIL: invalid archive temp" >&2; exit 1; }
# Inherited tar/gzip options must not change the manifest or run hook commands.
env -u TAR_OPTIONS -u GZIP tar -czf "$ARCHIVE_TMP" -C "$STAGE" -- "${FILES[@]}"
env -u TAR_OPTIONS -u GZIP tar -tzf "$ARCHIVE_TMP"
mv -T --backup=numbered -- "$ARCHIVE_TMP" "$TARBALL"
ARCHIVE_TMP=
if [[ "$DRAFT" == 1 ]]; then
    echo "wrote $TARBALL (DRAFT; do not submit)"
else
    echo "wrote $TARBALL (strict staged checks passed)"
fi
