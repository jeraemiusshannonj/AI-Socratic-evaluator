"""Structure-aware chunking for the unstructured corpus.

Fixed-width character splitting is the obvious approach and the wrong one for
CS prose: it cuts code blocks in half and strands explanations away from the
heading that says what they explain. Retrieval quality then looks like a
model problem when it is a chunking problem.

This splitter:
  * keeps fenced code blocks intact, never splitting inside one
  * splits on heading boundaries first, paragraph boundaries second
  * prefixes each chunk with its heading trail, so an embedded chunk carries
    the context a reader would have had from scrolling up
  * overlaps consecutive chunks so a concept spanning a boundary is
    retrievable from either side
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from . import config

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```")


@dataclass
class Block:
  """A paragraph, a code fence, or a heading -- an atom we never split."""

  text: str
  kind: str  # "heading" | "code" | "prose"
  level: int = 0
  headings: tuple[str, ...] = ()

  def __len__(self) -> int:
    return len(self.text)


@dataclass
class Chunk:
  text: str
  heading_trail: tuple[str, ...]
  block_span: tuple[int, int]
  has_code: bool = False
  char_count: int = field(default=0)

  def __post_init__(self) -> None:
    self.char_count = len(self.text)


def _iter_blocks(markdown: str) -> Iterator[Block]:
  """Split into atoms, tracking the heading stack as we go."""
  lines = markdown.splitlines()
  stack: list[str] = []
  buffer: list[str] = []
  in_fence = False
  i = 0

  def flush(kind: str) -> Iterator[Block]:
    nonlocal buffer
    body = "\n".join(buffer).strip()
    buffer = []
    if body:
      yield Block(text=body, kind=kind, headings=tuple(stack))

  while i < len(lines):
    line = lines[i]

    if _FENCE.match(line.strip()):
      if not in_fence:
        yield from flush("prose")
        in_fence = True
        buffer.append(line)
      else:
        buffer.append(line)
        in_fence = False
        yield from flush("code")
      i += 1
      continue

    if in_fence:
      buffer.append(line)
      i += 1
      continue

    heading = _HEADING.match(line)
    if heading:
      yield from flush("prose")
      level = len(heading.group(1))
      title = heading.group(2).strip()
      # Pop deeper-or-equal headings, then push this one.
      del stack[level - 1 :]
      stack.append(title)
      yield Block(text=line.strip(), kind="heading", level=level,
                  headings=tuple(stack))
      i += 1
      continue

    if not line.strip():
      yield from flush("prose")
      i += 1
      continue

    buffer.append(line)
    i += 1

  if in_fence:
    # Unterminated fence in the source document. Close it ourselves: emit
    # the tail rather than dropping it, but not with a dangling opener that
    # would make every downstream chunk look like malformed markdown.
    buffer.append("```")
    yield from flush("code")
  else:
    yield from flush("prose")


def _context_prefix(blocks: list[Block]) -> tuple[str, tuple[str, ...]]:
  """Build the heading context for a chunk that may span several sections.

  Using only the first block's trail understates a chunk that runs across a
  heading boundary -- the embedding then claims less coverage than the text
  actually has. Instead: the deepest trail common to every block, plus the
  distinct sections the chunk reaches into.
  """
  trails = [b.headings for b in blocks if b.headings]
  if not trails:
    return "", ()

  common: list[str] = []
  for parts in zip(*trails):
    if len(set(parts)) == 1:
      common.append(parts[0])
    else:
      break
  common_trail = tuple(common)

  extras: list[str] = []
  for trail in trails:
    leaf = trail[len(common_trail) :]
    if leaf and leaf[-1] not in extras:
      extras.append(leaf[-1])

  prefix = " > ".join(common_trail)
  if extras:
    joined = "; ".join(extras)
    prefix = f"{prefix} > {joined}" if prefix else joined
  return prefix, common_trail + tuple(extras)


def _split_code_block(block: Block, max_chars: int) -> list[Block]:
  """Split a code block too large to embed, keeping each piece valid code.

  A naive character split leaves one piece with an opening fence and no
  close, and the next with a stray ``` and no open. Instead we unwrap,
  split on line boundaries, and re-fence every piece with the original
  language tag so each one still reads as runnable code.
  """
  lines = block.text.splitlines()
  has_opener = bool(lines) and lines[0].lstrip().startswith("```")
  opener = lines[0].strip() if has_opener else "```"
  body = lines[1:] if has_opener else lines
  if body and body[-1].strip().startswith("```"):
    body = body[:-1]

  # Budget for the fences we re-add around every piece.
  budget = max(200, max_chars - len(opener) - 8)

  pieces: list[Block] = []
  current: list[str] = []
  size = 0
  for line in body:
    if current and size + len(line) + 1 > budget:
      pieces.append(current)
      current, size = [], 0
    current.append(line)
    size += len(line) + 1
  if current:
    pieces.append(current)

  total = len(pieces)
  out: list[Block] = []
  for i, piece in enumerate(pieces):
    marker = "" if total == 1 else f"  # (part {i + 1} of {total})"
    text = f"{opener}{marker}\n" + "\n".join(piece) + "\n```"
    out.append(
        Block(text=text, kind="code", level=block.level, headings=block.headings)
    )
  return out


def _split_oversize(blocks: list[Block], max_chars: int) -> list[Block]:
  """Hard-split any single block that exceeds the embedding input ceiling.

  Preference order for the cut point: sentence end, then line end, then a
  raw character cut. A raw cut is ugly but it is strictly better than the
  alternative, which is the embedding API quietly dropping the tail.
  """
  out: list[Block] = []
  for block in blocks:
    if len(block) <= max_chars:
      out.append(block)
      continue

    if block.kind == "code":
      out.extend(_split_code_block(block, max_chars))
      continue

    remaining = block.text
    while len(remaining) > max_chars:
      window = remaining[:max_chars]
      cut = max(
          window.rfind(". "),
          window.rfind("\n"),
      )
      if cut < max_chars // 2:
        cut = max_chars
      else:
        cut += 1
      out.append(Block(text=remaining[:cut].strip(), kind=block.kind,
                       level=block.level, headings=block.headings))
      remaining = remaining[cut:].lstrip()
    if remaining.strip():
      out.append(Block(text=remaining.strip(), kind=block.kind,
                       level=block.level, headings=block.headings))
  return out


def chunk_markdown(
    markdown: str,
    *,
    target_chars: int = config.CHUNK_TARGET_CHARS,
    overlap_chars: int = config.CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
  """Split a document into retrievable chunks."""
  blocks = [b for b in _iter_blocks(markdown) if b.kind != "heading"]
  if not blocks:
    return []

  chunks: list[Chunk] = []
  current: list[Block] = []
  current_len = 0
  start_idx = 0

  def emit(end_idx: int) -> None:
    nonlocal current, current_len, start_idx
    if not current:
      return
    prefix, trail = _context_prefix(current)
    body = "\n\n".join(b.text for b in current)
    text = f"{prefix}\n\n{body}" if prefix else body
    chunks.append(
        Chunk(
            text=text,
            heading_trail=trail,
            block_span=(start_idx, end_idx),
            has_code=any(b.kind == "code" for b in current),
        )
    )

  blocks = _split_oversize(blocks, config.MAX_CHUNK_CHARS)

  for idx, block in enumerate(blocks):
    block_len = len(block)

    # A single block larger than the target is kept whole. Splitting a long
    # code example does more damage than an oversized chunk.
    if current and current_len + block_len > target_chars:
      emit(idx - 1)
      carry: list[Block] = []
      carried = 0
      for prev in reversed(current):
        # Check the budget BEFORE taking the block, not after. Checking
        # after lets a single block larger than the whole overlap budget
        # get carried in full, which then compounds into a chunk that
        # exceeds the embedding input limit.
        if prev.kind == "code" or carried + len(prev) > overlap_chars:
          break
        carry.insert(0, prev)
        carried += len(prev)
      current = carry
      current_len = carried
      start_idx = idx - len(carry)

    current.append(block)
    current_len += block_len

  emit(len(blocks) - 1)
  return _merge_runts(chunks, config.MIN_CHUNK_CHARS, config.MAX_CHUNK_CHARS)


def _merge_runts(
    chunks: list[Chunk], min_chars: int, max_chars: int
) -> list[Chunk]:
  """Fold undersized chunks into their neighbour where that fits."""
  if len(chunks) < 2:
    return chunks

  merged: list[Chunk] = []
  for chunk in chunks:
    if (
        merged
        and chunk.char_count < min_chars
        and merged[-1].char_count + chunk.char_count <= max_chars
    ):
      prev = merged[-1]
      # Drop the runt's heading prefix line; the merged chunk keeps the
      # preceding chunk's context line and gains the runt's headings.
      body = chunk.text
      if chunk.heading_trail:
        body = body.split("\n\n", 1)[-1]
      combined_trail = tuple(
          dict.fromkeys(prev.heading_trail + chunk.heading_trail)
      )
      merged[-1] = Chunk(
          text=f"{prev.text}\n\n{body}",
          heading_trail=combined_trail,
          block_span=(prev.block_span[0], chunk.block_span[1]),
          has_code=prev.has_code or chunk.has_code,
      )
    else:
      merged.append(chunk)
  return merged
