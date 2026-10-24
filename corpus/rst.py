"""reStructuredText -> Markdown, aimed at retrieval rather than rendering.

Both corpus sources ship RST, and neither ships Markdown. Running the real
Sphinx toolchain would mean building two whole documentation projects with
their extensions (OpenDSA's `avembed`/`codeinclude`, CPython's `c:function`
roles) just to throw the HTML away again, so this converts directly.

The goal is not faithful rendering. It is text that embeds well:

  * directives that carry no prose (interactive visualisations, index
    entries, raw HTML passthrough) are dropped with their bodies, because a
    chunk full of `.. avembed:: Searching/BinarySearchCON.html` is a chunk
    that wins a cosine match and then tells the critic nothing
  * literal blocks become fenced code blocks, so the chunker can see them
    and keep them intact
  * inline roles collapse to their display text -- `:term:`binary search``
    reads as "binary search", not as markup the embedder has to look past
  * section adornments become ATX headings, so the chunker's heading trail
    works

Anything this drops is recorded on the result, so `build_corpus` can report
what was discarded instead of discovering it later as a silent gap.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Directive policy
# ---------------------------------------------------------------------------

# Dropped with their bodies: no prose a reader would miss.
DROP_DIRECTIVES = frozenset({
    # OpenDSA interactive machinery
    "avembed", "inlineav", "avmetadata", "odsascript", "odsalink",
    "extrtoolembed", "showhidecontent", "odsafig", "avsumm",
    # Sphinx structural / presentational
    "index", "only", "raw", "toctree", "tabularcolumns", "highlight",
    "image", "figure", "contents", "sectionauthor", "moduleauthor",
    "centered", "hlist", "graphviz", "youtube", "replace", "unicode",
    "meta", "include", "literalinclude", "csv-table",
    # Sphinx autodoc and doctest scaffolding: either an empty stub whose
    # real content is generated from Python source at build time, or setup
    # code that exists to make a doctest pass.
    "automodule", "autoclass", "autofunction", "automethod",
    "autoattribute", "autodata", "autoexception", "currentmodule",
    "module", "testsetup", "testcleanup", "doctest-skip",
    # Source inclusion: resolved separately, see sources.py. OpenDSA's
    # reference implementations are mostly Java/C++, and injecting Java into
    # a Python tutor's retrieval context invites the critic to cite syntax
    # the student cannot run.
    "codeinclude",
    # Editorial scaffolding
    "todo", "TODO", "comment",
    # Block math. Inline :math: is cleaned up and kept (see _clean_math),
    # but a display-math body is typically a `\begin{array}` table of growth
    # rates. There is no honest text rendering of that, and passing the raw
    # LaTeX through puts `\hline` and `\textbf` into the embedding.
    "math",
})

# Body is prose worth keeping; the directive wrapper itself is noise.
UNWRAP_DIRECTIVES = frozenset({
    "note", "warning", "important", "caution", "danger", "attention",
    "tip", "hint", "admonition", "topic", "sidebar", "epigraph",
    "rubric", "seealso", "deprecated", "versionadded", "versionchanged",
    "centered-text", "compound", "container", "pull-quote",
    "blockquote", "glossary", "describe", "default-role",
    # Presentational wrappers with prose inside.
    "line-block", "rst-class", "class", "role",
    # SQLAlchemy's own admonition for superseded APIs.
    "legacy",
    # Hand-written API descriptions (as opposed to the auto* family above):
    # the body is documentation prose worth keeping.
    "method", "function", "attribute", "data", "exception", "classmethod",
    "staticmethod", "property", "decorator", "py:method", "py:function",
    "py:class", "py:attribute", "py:data",
})

# Body is code; fence it with the given language.
CODE_DIRECTIVES = {
    "code-block": None,      # language comes from the directive argument
    "code": None,
    "sourcecode": None,
    "doctest": "python",
    "testcode": "python",
    "parsed-literal": "text",
}

_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([A-Za-z0-9_:+-]+)::\s*(.*)$")
_COMMENT = re.compile(r"^(\s*)\.\.(\s+.*)?$")
_TARGET = re.compile(r"^\s*\.\.\s+_[^:]+:\s*$")
_OPTION = re.compile(r"^\s*:[A-Za-z0-9_-]+:\s*.*$")
_FIELD = re.compile(r"^:[A-Za-z0-9_-]+:\s*.*$")
_ADORNMENT = re.compile(r'^([=\-`:\'"~^_*+#<>])\1{1,}\s*$')
_BULLET = re.compile(r"^(\s*)[*+-]\s+(.*)$")
_FOOTNOTE_DEF = re.compile(r"^\s*\.\.\s+\[[^\]]+\]\s")


@dataclass
class Conversion:
  """The converted document plus what was thrown away getting there."""

  markdown: str
  title: str = ""
  dropped: Counter[str] = field(default_factory=Counter)
  code_blocks: int = 0
  headings: int = 0

  @property
  def char_count(self) -> int:
    return len(self.markdown)


# ---------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------

# :role:`display <target>`  ->  display
# :role:`content`           ->  content
_ROLE_TARGETED = re.compile(r":[A-Za-z0-9_:+-]+:`([^`<]*?)\s*<[^`>]*>`")
_ROLE_PLAIN = re.compile(r":[A-Za-z0-9_:+-]+:`([^`]*)`")
# `display <url>`_ / `display <url>`__  ->  display
_LINK = re.compile(r"`([^`<]*?)\s*<[^`>]*>`__?")
# `single-backtick role-less text`_  ->  text
_BARE_REF = re.compile(r"`([^`]+)`__?")
_LITERAL = re.compile(r"``([^`]+)``")
_MATH = re.compile(r":math:`([^`]*)`")
_FOOTNOTE_REF = re.compile(r"\[[#*]?[A-Za-z0-9_-]*\]_")
_SUBSTITUTION = re.compile(r"\|([A-Za-z0-9_ -]+)\|")
_TRAILING_UNDERSCORE = re.compile(r"(?<=\w)_(?=\s|$|[.,;:)])")
# SQLAlchemy's `pycon+sql` lexer uses inline markers to switch highlighting
# mid-block. They are directives to a Sphinx lexer, not content, and left in
# place they appear in a retrieved passage as `{printsql}SELECT ...`.
_LEXER_MARKER = re.compile(r"\{(printsql|opensql|execsql|stop|sql)\}")
# Cross-reference targets in SQLAlchemy docs carry a leading underscore
# (`:class:`_sql.Select``) that is namespace bookkeeping, not part of the
# name a reader should see.
_XREF_UNDERSCORE = re.compile(r"\b_(?=[a-z]+\.[A-Z])")


_MATH_WORDS = {
    r"\log": "log", r"\lg": "lg", r"\ln": "ln", r"\exp": "exp",
    r"\sin": "sin", r"\cos": "cos", r"\max": "max", r"\min": "min",
    r"\Theta": "Theta", r"\Omega": "Omega", r"\theta": "theta",
    r"\alpha": "alpha", r"\beta": "beta", r"\lambda": "lambda",
    r"\le": "<=", r"\leq": "<=", r"\ge": ">=", r"\geq": ">=",
    r"\ne": "!=", r"\neq": "!=", r"\times": "*", r"\cdot": "*",
    r"\ldots": "...", r"\dots": "...", r"\cdots": "...",
    r"\approx": "~=", r"\equiv": "==", r"\pm": "+/-", r"\div": "/",
    r"\lfloor": "floor(", r"\rfloor": ")",
    r"\lceil": "ceil(", r"\rceil": ")",
    r"\left": "", r"\right": "", r"\,": " ", r"\;": " ", r"\!": "",
}


def _clean_math(expr: str) -> str:
  """Render a LaTeX fragment as something a reader and an embedder can use.

  OpenDSA is heavy on `:math:` -- an untouched `k_{mid}` or `O(\\log n)`
  puts LaTeX control sequences into the embedding, where they are noise, and
  in front of the student, where they are worse than noise.
  """
  out = expr
  for token, replacement in _MATH_WORDS.items():
    out = out.replace(token, replacement)
  # \frac{a}{b} -> a/b
  out = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", out)
  # \sqrt{n} -> sqrt(n)
  out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", out)
  # Subscripts and superscripts: k_{mid} -> k_mid, n^{2} -> n^2
  out = re.sub(r"_\{([^{}]*)\}", r"_\1", out)
  out = re.sub(r"\^\{([^{}]*)\}", r"^\1", out)
  # Any surviving control sequence: keep the name, drop the backslash.
  out = re.sub(r"\\([A-Za-z]+)", r"\1", out)
  out = out.replace("{", "").replace("}", "").replace("$", "")
  return re.sub(r"\s+", " ", out).strip()


def _inline(text: str) -> str:
  """Collapse inline RST markup to its display text.

  Order matters: math before the generic role patterns (it needs its own
  cleanup), targeted roles before plain roles (otherwise the plain pattern
  eats the `<target>` as content), and double-backtick literals before
  anything that touches single backticks.
  """
  # Protect double-backtick literals first; they may contain anything.
  literals: list[str] = []

  def _stash(match: re.Match[str]) -> str:
    literals.append(match.group(1))
    return f"\x00{len(literals) - 1}\x00"

  text = _LITERAL.sub(_stash, text)

  text = _MATH.sub(lambda m: _clean_math(m.group(1)), text)
  # A leading `!` in a cross-reference role suppresses link generation in
  # Sphinx. It is markup, so `:keyword:`!return`` must read as "return".
  text = _ROLE_TARGETED.sub(lambda m: m.group(1).lstrip("!~"), text)
  text = _ROLE_PLAIN.sub(lambda m: m.group(1).lstrip("!~"), text)
  text = _LINK.sub(r"\1", text)
  text = _BARE_REF.sub(r"\1", text)
  text = _FOOTNOTE_REF.sub("", text)
  text = _SUBSTITUTION.sub(r"\1", text)
  text = _TRAILING_UNDERSCORE.sub("", text)
  text = _XREF_UNDERSCORE.sub("", text)
  text = _LEXER_MARKER.sub("", text)

  # Restore literals as single-backtick code spans.
  def _restore(match: re.Match[str]) -> str:
    body = literals[int(match.group(1))]
    # A literal containing a backtick cannot round-trip through a single
    # backtick span; leave it bare rather than emit broken markdown.
    return f"`{body}`" if "`" not in body else body

  text = re.sub(r"\x00(\d+)\x00", _restore, text)
  return text


# ---------------------------------------------------------------------------
# Block structure
# ---------------------------------------------------------------------------


def _indent_of(line: str) -> int:
  return len(line) - len(line.lstrip())


def _is_blank(line: str) -> bool:
  return not line.strip()


def _dedent(lines: list[str]) -> list[str]:
  """Remove the common leading indentation from a directive body."""
  widths = [_indent_of(l) for l in lines if l.strip()]
  if not widths:
    return []
  cut = min(widths)
  return [l[cut:] if l.strip() else "" for l in lines]


def _is_adornment(line: str) -> bool:
  return bool(_ADORNMENT.match(line)) and len(line.strip()) >= 2


# Sphinx projects define their own composite lexers. `pycon+sql` is
# SQLAlchemy's interleaved Python-console-and-SQL highlighter; written into
# a fence it is not a language any reader or renderer knows.
_LANG_ALIASES = {
    "pycon+sql": "python", "python+sql": "python", "pycon": "python",
    "py": "python", "python3": "python", "ipython": "python",
    "sql+python": "sql", "postgresql": "sql", "psql": "sql", "mysql": "sql",
    "console": "shell", "bash": "shell", "sh": "shell", "shell-session":
        "shell", "winbatch": "shell", "doscon": "shell",
    "none": "text", "default": "text", "guess": "text", "": "text",
}


def _normalise_lang(lang: str) -> str:
  cleaned = lang.strip().lstrip(":").lower()
  # A composite like `pycon+sql` not in the table still should not reach the
  # output with a `+` in it.
  if cleaned not in _LANG_ALIASES and "+" in cleaned:
    cleaned = cleaned.split("+", 1)[0]
  return _LANG_ALIASES.get(cleaned, cleaned or "text")


class _Converter:

  def __init__(self, text: str, *, source_name: str = "") -> None:
    self.lines = text.replace("\r\n", "\n").expandtabs(4).split("\n")
    self.source_name = source_name
    self.out: list[str] = []
    self.dropped: Counter[str] = Counter()
    self.code_blocks = 0
    self.headings = 0
    self.title = ""
    # Adornment character -> heading level, assigned in order of appearance.
    self.levels: dict[str, int] = {}
    self.i = 0

  # -- helpers ------------------------------------------------------------

  def _peek(self, offset: int = 0) -> str:
    idx = self.i + offset
    return self.lines[idx] if 0 <= idx < len(self.lines) else ""

  def _emit(self, line: str = "") -> None:
    # Collapse runs of blank lines; the chunker splits on them and a run of
    # four produces phantom empty blocks.
    if not line.strip() and self.out and not self.out[-1].strip():
      return
    self.out.append(line)

  def _level_for(self, char: str) -> int:
    if char not in self.levels:
      self.levels[char] = len(self.levels) + 1
    return min(self.levels[char], 6)

  def _consume_body(self, base_indent: int) -> list[str]:
    """Take every line belonging to a block opened at `base_indent`."""
    body: list[str] = []
    while self.i < len(self.lines):
      line = self._peek()
      if _is_blank(line):
        # A blank line only ends the body if what follows is dedented.
        nxt = self.i + 1
        while nxt < len(self.lines) and _is_blank(self.lines[nxt]):
          nxt += 1
        if nxt >= len(self.lines) or _indent_of(self.lines[nxt]) <= base_indent:
          break
        body.append("")
        self.i += 1
        continue
      if _indent_of(line) <= base_indent:
        break
      body.append(line)
      self.i += 1
    while body and not body[-1].strip():
      body.pop()
    return body

  # -- block handlers -----------------------------------------------------

  def _handle_directive(self, match: re.Match[str]) -> None:
    indent, name, argument = match.group(1), match.group(2), match.group(3)
    base = len(indent)
    self.i += 1

    # Options belong to the directive, never to its body.
    options: dict[str, str] = {}
    while self.i < len(self.lines):
      line = self._peek()
      if _is_blank(line):
        break
      if _indent_of(line) > base and _OPTION.match(line):
        key, _, value = line.strip()[1:].partition(":")
        options[key.strip()] = value.strip()
        self.i += 1
        continue
      break

    body = self._consume_body(base)
    key = name.split(":")[-1]

    if key in DROP_DIRECTIVES:
      self.dropped[key] += 1
      return

    if key in CODE_DIRECTIVES:
      lang = CODE_DIRECTIVES[key] or (argument.strip().split()[0]
                                      if argument.strip() else "text")
      self._emit_code(_dedent(body), lang)
      return

    if key in UNWRAP_DIRECTIVES:
      self._emit()
      label = argument.strip() or key.capitalize()
      if key in {"note", "warning", "important", "caution", "tip", "hint",
                 "seealso", "attention", "danger"}:
        self._emit(f"**{key.capitalize()}:** {_inline(label)}"
                   if argument.strip() else f"**{key.capitalize()}:**")
      elif argument.strip():
        self._emit(_inline(label))
      self._emit()
      for line in self._subconvert(_dedent(body)):
        self._emit(line)
      self._emit()
      return

    # Unknown directive: keep the body, note the name. Better to carry
    # unexpected prose than to silently lose a section because OpenDSA
    # added a directive after this was written.
    self.dropped[f"unknown:{key}"] += 1
    if body:
      for line in self._subconvert(_dedent(body)):
        self._emit(line)
      self._emit()

  def _subconvert(self, body: list[str]) -> list[str]:
    """Convert a nested body with a fresh converter, sharing counters."""
    if not body:
      return []
    sub = _Converter("\n".join(body), source_name=self.source_name)
    sub.levels = self.levels
    sub.run()
    self.dropped.update(sub.dropped)
    self.code_blocks += sub.code_blocks
    self.headings += sub.headings
    return sub.out

  def _emit_code(self, body: list[str], lang: str) -> None:
    if not any(l.strip() for l in body):
      return
    lang = _normalise_lang(lang)
    self._emit()
    self._emit(f"```{lang}")
    for line in body:
      # Lexer-switch markers appear inside code bodies, where _inline never
      # runs. Strip them here or they end up quoted to the student.
      self._emit_raw(_LEXER_MARKER.sub("", line))
    self._emit("```")
    self._emit()
    self.code_blocks += 1

  def _emit_raw(self, line: str) -> None:
    """Append without blank-run collapsing; code keeps its own blank lines."""
    self.out.append(line)

  def _handle_heading(self) -> bool:
    """Emit an ATX heading if the cursor is on a section title."""
    line = self._peek()

    # Overline form: adornment / title / matching adornment.
    if _is_adornment(line) and self._peek(1).strip() and _is_adornment(
        self._peek(2)) and line.strip()[0] == self._peek(2).strip()[0]:
      char = line.strip()[0]
      title = _inline(self._peek(1).strip())
      self.i += 3
      self._write_heading(char, title, overline=True)
      return True

    # Underline form: title / adornment.
    nxt = self._peek(1)
    if (line.strip() and not _is_adornment(line) and _is_adornment(nxt)
        and len(nxt.strip()) >= len(line.strip()) - 2
        and not _DIRECTIVE.match(line) and not _COMMENT.match(line)):
      char = nxt.strip()[0]
      title = _inline(line.strip())
      self.i += 2
      self._write_heading(char, title)
      return True

    return False

  def _write_heading(self, char: str, title: str, *,
                     overline: bool = False) -> None:
    if not title:
      return
    # An overlined section outranks an underlined one using the same
    # character; CPython's top-level titles are overlined with `*`.
    level = self._level_for(("over:" if overline else "") + char)
    if not self.title:
      self.title = title
    self._emit()
    self._emit(f"{'#' * level} {title}")
    self._emit()
    self.headings += 1

  def _handle_literal_block(self) -> bool:
    """Handle `paragraph::` followed by an indented block."""
    line = self._peek()
    stripped = line.rstrip()
    if not stripped.endswith("::"):
      return False

    base = _indent_of(line)
    lead = stripped[:-2].rstrip()
    self.i += 1

    # The paragraph text keeps its sentence; `::` alone is just a marker.
    if lead:
      self._emit(_inline(lead) + ("" if lead.endswith(":") else ":"))

    while self.i < len(self.lines) and _is_blank(self._peek()):
      self.i += 1

    if self.i >= len(self.lines) or _indent_of(self._peek()) <= base:
      return True  # `::` with no block after it; nothing more to do.

    body = _dedent(self._consume_body(base))
    lang = "python" if self._looks_like_python(body) else "text"
    self._emit_code(body, lang)
    return True

  @staticmethod
  def _looks_like_python(body: list[str]) -> bool:
    text = "\n".join(body)
    if ">>>" in text:
      return True
    markers = ("def ", "class ", "import ", "return ", "print(", "lambda ",
               "elif ", "self.", "None", "True", "False")
    hits = sum(1 for m in markers if m in text)
    # Pseudocode in OpenDSA uses `then`/`do` and 1-based arrays; real Python
    # does not. Requiring two markers keeps pseudocode out of ```python.
    pseudo = any(m in text for m in (" then", " do\n", "endfor", "endwhile"))
    return hits >= 2 and not pseudo

  def _handle_doctest_block(self) -> bool:
    """A bare block of `>>>` lines, outside any literal block."""
    if not self._peek().lstrip().startswith(">>>"):
      return False
    base = _indent_of(self._peek())
    body: list[str] = []
    while self.i < len(self.lines):
      line = self._peek()
      if _is_blank(line):
        nxt = self.i + 1
        if nxt < len(self.lines) and self.lines[nxt].lstrip().startswith(">>>"):
          body.append("")
          self.i += 1
          continue
        break
      if _indent_of(line) < base:
        break
      body.append(line)
      self.i += 1
    self._emit_code(_dedent(body), "python")
    return True

  # -- driver -------------------------------------------------------------

  def run(self) -> None:
    saw_content = False

    while self.i < len(self.lines):
      line = self._peek()

      if _is_blank(line):
        self._emit()
        self.i += 1
        continue

      if self._handle_heading():
        saw_content = True
        continue

      directive = _DIRECTIVE.match(line)
      if directive:
        self._handle_directive(directive)
        continue

      if _TARGET.match(line) or _FOOTNOTE_DEF.match(line):
        # Hyperlink targets and footnote definitions: drop the line and any
        # indented continuation.
        base = _indent_of(line)
        self.i += 1
        self._consume_body(base)
        continue

      if _COMMENT.match(line) and not line.strip().startswith(".. ["):
        # RST comment: `..` plus text, no `::`. Drop with its body.
        base = _indent_of(line)
        self.i += 1
        self._consume_body(base)
        self.dropped["comment"] += 1
        continue

      # Docinfo field list, but only before any real content -- `:param x:`
      # inside a body is prose we want.
      if not saw_content and _FIELD.match(line):
        self.i += 1
        continue

      if self._handle_doctest_block():
        saw_content = True
        continue

      if self._handle_literal_block():
        saw_content = True
        continue

      bullet = _BULLET.match(line)
      if bullet:
        self._emit(f"{bullet.group(1)}- {_inline(bullet.group(2))}")
        self.i += 1
        saw_content = True
        continue

      # Ordinary text. Dedent so stray 3-space RST indentation does not read
      # as a markdown code block, and so definition-list bodies stay prose.
      self._emit(_inline(line.strip()))
      self.i += 1
      saw_content = True


def _postprocess(markdown: str) -> str:
  """Unwrap hard-wrapped prose and drop repeated headings.

  RST sources are hard-wrapped at ~70 columns, which leaves every paragraph
  as a column of short lines. Harmless for the embedder, but it makes the
  retrieved passage ugly when the critic quotes it back to a student. Code
  fences are left exactly as they are.

  OpenDSA files also open with a document title and then an identically
  named first section, which would give the chunker a heading trail of
  `X > X`.
  """
  lines = markdown.split("\n")
  out: list[str] = []
  paragraph: list[str] = []
  in_fence = False
  last_heading = ""

  def flush() -> None:
    if paragraph:
      out.append(" ".join(paragraph))
      paragraph.clear()

  for line in lines:
    if line.lstrip().startswith("```"):
      flush()
      in_fence = not in_fence
      out.append(line)
      continue
    if in_fence:
      out.append(line)
      continue

    stripped = line.strip()
    if not stripped:
      flush()
      if out and out[-1].strip():
        out.append("")
      continue

    if stripped.startswith("#"):
      flush()
      title = stripped.lstrip("#").strip()
      if title and title == last_heading:
        continue  # title immediately restated as a section
      last_heading = title
      out.append(stripped)
      continue

    # List items and definition-ish lines stay on their own line.
    if stripped.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+[.)]\s", stripped):
      flush()
      out.append(stripped)
      continue

    paragraph.append(stripped)

  flush()
  return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def rst_to_markdown(text: str, *, source_name: str = "") -> Conversion:
  """Convert one RST document to retrieval-ready Markdown."""
  conv = _Converter(text, source_name=source_name)
  conv.run()

  body = _postprocess("\n".join(conv.out))

  return Conversion(
      markdown=body,
      title=conv.title,
      dropped=conv.dropped,
      code_blocks=conv.code_blocks,
      headings=conv.headings,
  )
