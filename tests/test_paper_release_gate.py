"""Exercise release scripts with deterministic TeX/Poppler command fixtures.

Run under Linux/WSL. Real shell, filesystem, cleanup and tar commands are used;
only the expensive compiler, number generator and PDF inspectors are replaced.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="run Bash tests in WSL")
SCRIPTS = ("build.sh", "make_arxiv.sh")
DEFECTS = [
    ("FIXTURE_LOG", "LaTeX Warning: Reference `missing' on page 1 undefined.\n"),
    ("FIXTURE_LOG", "Package natbib Warning: Citation `missing' on page 1 undefined.\n"),
    ("FIXTURE_LOG", "LaTeX Warning: There were undefined references.\n"),
    ("FIXTURE_LOG", "Package natbib Warning: There were undefined citations.\n"),
    ("FIXTURE_LOG", "Overfull \\hbox (1.0pt too wide) in paragraph at lines 1--2\n"),
    ("FIXTURE_LOG", "Overfull \\vbox (1.0pt too high) detected at line 2\n"),
    ("FIXTURE_LOG", "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"),
    ("FIXTURE_TEXT", "An unresolved value is ?? in the PDF.\n"),
    ("FIXTURE_FONTS", "ABCDEF+bitmap Type 3 Custom yes no no 4 0\n"),
]


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "repo with spaces"
    paper = root / "paper"
    paper.mkdir(parents=True)
    (root / "results").mkdir()
    (root / "scripts").mkdir()
    for script in SCRIPTS:
        shutil.copyfile(REPO / "paper" / script, paper / script)
    (paper / "main.tex").write_text("\\MaybeFigure{plot.pdf}{missing figure}\n")
    (paper / "numbers.tex").write_text("unused macro: \\textbf{??}\n")
    (paper / "main.bbl").write_text("fixture bibliography\n")
    (paper / "main.log").write_text("stale clean log\n")
    (paper / "main.pdf").write_text("stale PDF\n")
    (root / "results" / "plot.pdf").write_text("fixture figure\n")
    scratch = tmp_path / "scratch with spaces"
    scratch.mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    # Keep PATH hermetic so installed Poppler cannot hide missing-tool bugs.
    for command in (
        "bash", "cat", "cp", "dirname", "env", "find", "grep", "gzip",
        "mkdir", "mktemp", "mv", "pwd", "realpath", "rm", "rmdir", "sed",
        "tar", "tr",
    ):
        commands.joinpath(command).symlink_to(shutil.which(command))
    driver = tmp_path / "fixture-command"
    driver.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "p = pathlib.Path\n"
        "name = p(sys.argv[0]).name\n"
        "with open(os.environ['FIXTURE_TRACE'], 'a') as trace:\n"
        "    trace.write(json.dumps({'name': name, 'cwd': os.getcwd(), 'args': sys.argv[1:]}) + '\\n')\n"
        "if os.environ.get('FIXTURE_FAIL') == name:\n"
        "    sys.exit(7)\n"
        "if name == 'python':\n"
        "    p('numbers.tex').write_text('unused macro: \\\\textbf{??}\\n')\n"
        "elif name == 'pdflatex':\n"
        "    count = int(p('.passes').read_text()) + 1 if p('.passes').exists() else 1\n"
        "    p('.passes').write_text(str(count))\n"
        "    log = os.environ.get('FIXTURE_LOG', 'No warnings.\\n')\n"
        "    if count == 1 and os.environ.get('FIXTURE_EARLY_WARNING'):\n"
        "        log = 'LaTeX Warning: There were undefined references.\\n'\n"
        "    p('main.log').write_text(log)\n"
        "    p('main.pdf').write_text('fixture PDF\\n')\n"
        "    p('main.aux').write_text('fixture aux\\n')\n"
        "    p('nul').write_text('compiler side product\\n')\n"
        "    p('.env').mkdir(exist_ok=True)\n"
        "    p('private').mkdir(exist_ok=True)\n"
        "elif name == 'bibtex':\n"
        "    p('main.bbl').write_text('fixture bibliography\\n')\n"
        "    p('main.blg').write_text(os.environ.get('FIXTURE_BLG', 'No errors.\\n'))\n"
        "elif name == 'pdftotext':\n"
        "    if not p(sys.argv[1]).is_file(): sys.exit(8)\n"
        "    print(os.environ.get('FIXTURE_TEXT', 'A measured result.'))\n"
        "elif name == 'pdffonts':\n"
        "    if not p(sys.argv[1]).is_file(): sys.exit(8)\n"
        "    print('name type encoding emb sub uni object ID')\n"
        "    print('----------------------------------------')\n"
        "    print(os.environ.get('FIXTURE_FONTS', 'ABCDEF+LMRoman Type 1 Custom yes yes yes 4 0'))\n"
    )
    driver.chmod(0o755)
    for command in ("python", "pdflatex", "bibtex", "pdftotext", "pdffonts"):
        commands.joinpath(command).symlink_to(driver)
    env = dict(os.environ, PATH=str(commands), PYTHON=str(commands / "python"),
               TMPDIR=str(scratch), FIXTURE_TRACE=str(tmp_path / "trace.jsonl"))

    def run(script, *args, **overrides):
        return subprocess.run(
            [str(commands / "bash"), str(paper / script), *args], cwd=root,
            env=dict(env, **overrides), capture_output=True, text=True, timeout=20,
        )

    return paper, scratch, commands, run, Path(env["FIXTURE_TRACE"])


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("variable,value", DEFECTS)
def test_strict_rejects_pdf_defects(release, script, variable, value):
    paper, scratch, _, run, _ = release
    result = run(script, **{variable: value})
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (paper / "arxiv.tar.gz").exists()
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("variable,value", DEFECTS)
def test_draft_warns_but_allows_quality_defects(release, script, variable, value):
    _, scratch, _, run, _ = release
    result = run(script, "--draft", **{variable: value})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING" in result.stdout + result.stderr
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("tool", ("pdftotext", "pdffonts"))
@pytest.mark.parametrize("missing", (False, True))
def test_strict_requires_working_inspectors(release, script, tool, missing):
    _, _, commands, run, _ = release
    overrides = {}
    if missing:
        (commands / tool).unlink()
    else:
        overrides["FIXTURE_FAIL"] = tool
    result = run(script, **overrides)
    assert result.returncode != 0, result.stdout + result.stderr
    assert tool in result.stdout + result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_clean_final_pass_succeeds_despite_early_warnings_and_unused_macros(release, script):
    _, _, _, run, _ = release
    result = run(script, FIXTURE_EARLY_WARNING="1")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("draft", (False, True))
def test_compiler_failure_is_always_fatal(release, script, draft):
    _, scratch, _, run, _ = release
    result = run(script, *(["--draft"] if draft else []), FIXTURE_FAIL="pdflatex")
    assert result.returncode != 0
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("draft", (False, True))
def test_bibtex_errors_are_always_fatal(release, draft):
    _, _, _, run, _ = release
    result = run("build.sh", *(["--draft"] if draft else []),
                 FIXTURE_BLG="(There was 1 error message)\n")
    assert result.returncode != 0


def test_package_is_isolated_allowlisted_and_preserves_originals(release):
    paper, scratch, _, run, trace = release
    old_stage = paper / "arxiv_stage"
    old_stage.mkdir()
    original = old_stage / "keep.txt"
    original.write_text("do not delete\n")
    archive = paper / "arxiv.tar.gz"
    archive.write_bytes(b"previous release")
    result = run("make_arxiv.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert original.read_text() == "do not delete\n"
    assert (paper / "main.pdf").read_text() == "stale PDF\n"
    assert any(p.read_bytes() == b"previous release" for p in paper.glob("arxiv.tar.gz.*"))
    with tarfile.open(archive) as bundle:
        assert {m.name.removeprefix("./") for m in bundle.getmembers() if m.isfile()} == {
            "main.tex", "numbers.tex", "main.bbl", "plot.pdf",
        }
        assert all(m.isfile() for m in bundle.getmembers())
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    compiles = [event for event in events if event["name"] == "pdflatex"]
    assert compiles
    for event in compiles:
        assert Path(event["cwd"]).is_relative_to(scratch)
        assert "-no-shell-escape" in event["args"]
    assert not any(event["name"] in {"python", "bibtex"} for event in events)
    assert not list(scratch.iterdir())
    assert not list(paper.glob(".arxiv.tar.gz.*"))


@pytest.mark.parametrize("failure", ("quality", "compiler", "archive"))
def test_failed_package_preserves_previous_archive_and_cleans_temps(release, failure):
    paper, scratch, commands, run, _ = release
    archive = paper / "arxiv.tar.gz"
    archive.write_bytes(b"previous release")
    overrides = {"FIXTURE_TEXT": "??"} if failure == "quality" else {"FIXTURE_FAIL": "pdflatex"}
    if failure == "archive":
        (commands / "tar").unlink()
        (commands / "tar").write_text("#!/bin/bash\nexit 9\n")
        (commands / "tar").chmod(0o755)
        overrides = {}
    result = run("make_arxiv.sh", **overrides)
    assert result.returncode != 0
    assert archive.read_bytes() == b"previous release"
    assert not list(scratch.iterdir())
    assert not list(paper.glob(".arxiv.tar.gz.*"))


@pytest.mark.parametrize("figure", ("../private/leak.pdf", "../.env", "nul", "NUL.pdf", "-option.pdf", "main.pdf"))
def test_package_rejects_unsafe_figure_names(release, figure):
    paper, _, _, run, _ = release
    (paper / "main.tex").write_text(f"\\MaybeFigure{{{figure}}}{{missing}}\n")
    # A traversal target exists so rejection cannot merely be 'file missing'.
    private = paper.parent / "private"
    private.mkdir()
    (private / "leak.pdf").write_text("private fixture\n")
    if "/" not in figure:
        (paper.parent / "results" / figure).write_text("unsafe fixture\n")
    result = run("make_arxiv.sh")
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (paper / "arxiv.tar.gz").exists()


@pytest.mark.parametrize("target", ("main.tex", "numbers.tex", "main.bbl", "figure", "results"))
def test_package_rejects_symlinked_inputs(release, target):
    paper, _, _, run, _ = release
    source = paper.parent / "results" / "plot.pdf" if target == "figure" else paper / target
    if target == "results":
        source = paper.parent / "results"
    moved = source.with_name(source.name + ".original")
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=moved.is_dir())
    result = run("make_arxiv.sh")
    assert result.returncode != 0, result.stdout + result.stderr
    assert moved.exists()
    assert not (paper / "arxiv.tar.gz").exists()


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("draft", (False, True))
@pytest.mark.parametrize("placeholder", (
    "[figure m9_tail_sweep.pdf unavailable: run scripts/plot_tail_sweep.py]",
    "[figure m9_tail_sweep.pdf\nunavailable:\nrun scripts/plot_tail_sweep.py]",
    "plot.pdf not built\n yet",
))
def test_rendered_missing_figure_follows_strict_and_draft_policy(release, script, draft, placeholder):
    paper, scratch, _, run, _ = release
    result = run(script, *(["--draft"] if draft else []), FIXTURE_TEXT=placeholder)
    if draft:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARNING" in result.stderr and "figure" in result.stderr
        if script == "make_arxiv.sh":
            assert (paper / "arxiv.tar.gz").exists()
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        assert "FAIL" in result.stderr and "figure" in result.stderr
        assert not (paper / "arxiv.tar.gz").exists()
    assert not list(scratch.iterdir())


def test_package_without_figures_is_valid(release):
    paper, _, _, run, _ = release
    (paper / "main.tex").write_text("A paper without figures.\n")
    result = run("make_arxiv.sh")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_unknown_options_fail_before_building(release, script):
    _, _, _, run, trace = release
    result = run(script, "--darft")
    assert result.returncode != 0
    assert not trace.exists()


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("tool", ("pdftotext", "pdffonts"))
def test_draft_explicitly_warns_when_inspection_is_unavailable(release, script, tool):
    _, _, commands, run, _ = release
    (commands / tool).unlink()
    result = run(script, "--draft")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING" in result.stderr and tool in result.stderr


def test_cleanup_refuses_unexpected_mktemp_directory(release):
    paper, scratch, commands, run, trace = release
    sentinel = paper / "keep.txt"
    sentinel.write_text("original\n")
    (commands / "mktemp").unlink()
    (commands / "mktemp").write_text('#!/bin/bash\nprintf "%s\\n" "$FIXTURE_TEMP_TARGET"\n')
    (commands / "mktemp").chmod(0o755)
    result = run("make_arxiv.sh", FIXTURE_TEMP_TARGET=str(paper.parent))
    assert result.returncode != 0
    assert "refusing cleanup" in result.stderr
    assert sentinel.read_text() == "original\n"
    assert not trace.exists()
    assert not list(scratch.iterdir())


def test_cleanup_refuses_unexpected_archive_temp(release):
    paper, scratch, commands, run, _ = release
    sentinel = paper / "keep.txt"
    sentinel.write_text("original\n")
    (commands / "mktemp").unlink()
    (commands / "mktemp").write_text(
        '#!/bin/bash\nif [[ "$1" == -d ]]; then\n'
        f'    exec {shutil.which("mktemp")} "$@"\n'
        'fi\nprintf "%s\\n" "$FIXTURE_TEMP_TARGET"\n'
    )
    (commands / "mktemp").chmod(0o755)
    result = run("make_arxiv.sh", FIXTURE_TEMP_TARGET=str(sentinel))
    assert result.returncode != 0
    assert "refusing cleanup" in result.stderr
    assert sentinel.read_text() == "original\n"
    assert not list(scratch.iterdir())


def test_package_refuses_symlinked_archive_destination(release):
    paper, _, _, run, trace = release
    sentinel = paper / "keep.txt"
    sentinel.write_text("original\n")
    (paper / "arxiv.tar.gz").symlink_to(sentinel)
    result = run("make_arxiv.sh")
    assert result.returncode != 0
    assert sentinel.read_text() == "original\n"
    assert (paper / "arxiv.tar.gz").is_symlink()
    assert not trace.exists()


def test_tar_environment_cannot_change_manifest(release):
    paper, _, _, run, _ = release
    result = run("make_arxiv.sh", TAR_OPTIONS="--exclude=main.tex")
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(paper / "arxiv.tar.gz") as bundle:
        assert set(bundle.getnames()) == {"main.tex", "numbers.tex", "main.bbl", "plot.pdf"}


@pytest.mark.parametrize("external_input", (False, True))
def test_real_tex_staged_build_is_self_contained(release, external_input):
    paper, scratch, commands, run, _ = release
    for tool in ("pdflatex", "pdftotext", "pdffonts"):
        real_tool = shutil.which(tool)
        if not real_tool:
            pytest.skip(f"real integration requires {tool}")
        (commands / tool).unlink()
        (commands / tool).symlink_to(real_tool)
    # Use an actual PDF figure to exercise the paper's flat-path fallback too.
    figures = paper.parent / "results"
    (figures / "plot.tex").write_text(
        "\\documentclass{article}\\usepackage{lmodern}"
        "\\begin{document}A figure.\\end{document}\n"
    )
    figure_build = subprocess.run(
        [str(commands / "pdflatex"), "-no-shell-escape", "-interaction=nonstopmode",
         "-halt-on-error", "plot.tex"], cwd=figures, capture_output=True, text=True,
        timeout=20,
    )
    assert figure_build.returncode == 0, figure_build.stdout + figure_build.stderr
    (paper / "numbers.tex").write_text("\\newcommand{\\Measured}{42}\n")
    (paper / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\n\\bibitem{known} Known reference.\n"
        "\\end{thebibliography}\n"
    )
    (paper / "hidden.tex").write_text("Repository-only content.\n")
    extra = "\\input{hidden}\n" if external_input else ""
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\usepackage{lmodern,graphicx}\n"
        "\\graphicspath{{../results/}{./}}\n"
        "\\newcommand{\\MaybeFigure}[2]{%\n"
        "\\IfFileExists{../results/#1}{\\includegraphics[height=1cm]{#1}}{%\n"
        "\\IfFileExists{#1}{\\includegraphics[height=1cm]{#1}}{not built yet}}}\n"
        "\\input{numbers}\n\\begin{document}\n"
        "Measured: \\Measured. See Section~\\ref{sec:ok} and \\cite{known}.\n"
        "\\section{Result}\\label{sec:ok}\n\\MaybeFigure{plot.pdf}{missing}\n" + extra +
        "\\bibliography{references}\n\\end{document}\n"
    )
    result = run("make_arxiv.sh", TEXINPUTS=f"{paper}:")
    if external_input:
        assert result.returncode != 0
        assert "hidden" in result.stderr
        assert not (paper / "arxiv.tar.gz").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (paper / "arxiv.tar.gz").exists()
    assert not list(scratch.iterdir())
