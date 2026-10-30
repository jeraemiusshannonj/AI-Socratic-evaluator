"""Caps on the endpoints that cost money.

A public URL that calls Gemini is a way for a stranger to spend the
project's budget, and the budget alert notifies rather than stops. Two
limits, both backed by Mongo rather than process memory, because Cloud Run
scales to zero and back and an in-process counter resets every cold start
-- which is exactly when it would matter.

  per IP     a sliding window, default 30 requests an hour
  per day    a total across everyone, default 500

Only the endpoints that make a model call are metered. Health checks and
the concept list are free, so a monitor polling `/api/health` cannot
exhaust anyone's allowance.

Failure is open, deliberately. If Mongo is unreachable the request is
allowed: the limiter exists to bound cost, not to be a security control,
and a database blip should not take the whole service down. The budget
alert is the backstop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pymongo import ASCENDING

from . import config, db

log = logging.getLogger(__name__)

COLLECTION = "rate_limits"
_indexes_ready = False


@dataclass
class Decision:
  allowed: bool
  reason: str = ""
  retry_after: int = 0
  remaining: int | None = None

  def headers(self) -> dict[str, str]:
    out: dict[str, str] = {}
    if self.remaining is not None:
      out["X-RateLimit-Remaining"] = str(max(0, self.remaining))
    if not self.allowed and self.retry_after:
      out["Retry-After"] = str(self.retry_after)
    return out


def _collection():
  global _indexes_ready
  coll = db.db()[COLLECTION]
  if not _indexes_ready:
    try:
      # TTL on the window end, so old counters delete themselves rather
      # than accumulating in a 512MB free-tier cluster.
      coll.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0,
                        name="ttl")
      coll.create_index([("key", ASCENDING)], name="key", unique=True)
      _indexes_ready = True
    except Exception as exc:  # noqa: BLE001
      log.warning("Could not ensure rate-limit indexes: %s", exc)
  return coll


def client_ip(headers: dict, fallback: str = "unknown") -> str:
  """The caller's address, taken from X-Forwarded-For.

  Known weakness: this takes the leftmost entry, which is correct when the
  client sends no X-Forwarded-For of its own and wrong when it does. A
  proxy that appends to the header keeps whatever the client supplied on
  the left, so a caller can choose its own key and get a fresh per-IP
  allowance per request. The daily total in `check()` is keyed on nothing
  the caller controls, so total spend stays bounded either way.

  The fix is to count from the right, skipping the entries added by known
  proxies. That needs the exact header shape Cloud Run delivers, confirmed
  from one logged request, rather than assumed; a wrong guess would put
  every user in one bucket.
  """
  forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
  if forwarded:
    first = forwarded.split(",")[0].strip()
    if first:
      return first
  return headers.get("x-real-ip") or fallback


def _bump(key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
  """Increment a counter, returning (allowed, remaining, retry_after)."""
  now = time.time()
  coll = _collection()

  doc = coll.find_one_and_update(
      {"key": key},
      {
          "$inc": {"count": 1},
          "$setOnInsert": {
              "started_at": now,
              "expires_at": datetime.fromtimestamp(
                  now + window_seconds, tz=timezone.utc),
          },
      },
      upsert=True,
      return_document=True,
  )

  count = doc.get("count", 1)
  started = doc.get("started_at", now)
  retry_after = max(1, int(started + window_seconds - now))
  return count <= limit, limit - count, retry_after


def check(path: str, ip: str) -> Decision:
  """Should this request be allowed?"""
  if not any(path.startswith(p) for p in config.METERED_PATHS):
    return Decision(allowed=True)

  try:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ok_day, remaining_day, retry_day = _bump(
        f"day:{today}", config.RATE_LIMIT_DAILY_TOTAL, 86_400)
    if not ok_day:
      log.warning("Daily request cap reached (%s)",
                  config.RATE_LIMIT_DAILY_TOTAL)
      return Decision(
          allowed=False,
          reason=(
              f"This demo accepts {config.RATE_LIMIT_DAILY_TOTAL} model "
              f"requests a day and has reached that. Try again tomorrow."
          ),
          retry_after=retry_day,
          remaining=0,
      )

    ok_ip, remaining_ip, retry_ip = _bump(
        f"ip:{ip}:{today}", config.RATE_LIMIT_PER_IP,
        config.RATE_LIMIT_WINDOW_SECONDS)
    if not ok_ip:
      return Decision(
          allowed=False,
          reason=(
              f"Rate limit: {config.RATE_LIMIT_PER_IP} requests per "
              f"{config.RATE_LIMIT_WINDOW_SECONDS // 60} minutes."
          ),
          retry_after=retry_ip,
          remaining=0,
      )

    return Decision(allowed=True, remaining=min(remaining_ip, remaining_day))

  except Exception as exc:  # noqa: BLE001
    # Fail open. This bounds spending; it is not an access control, and a
    # Mongo hiccup should not take the service down with it.
    log.warning("Rate limiter unavailable, allowing request: %s", exc)
    return Decision(allowed=True)


def usage_today() -> dict:
  """Current counters, for /api/health."""
  try:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = _collection().find_one({"key": f"day:{today}"}) or {}
    used = doc.get("count", 0)
    return {
        "requests_today": used,
        "daily_cap": config.RATE_LIMIT_DAILY_TOTAL,
        "remaining_today": max(0, config.RATE_LIMIT_DAILY_TOTAL - used),
    }
  except Exception as exc:  # noqa: BLE001
    return {"error": str(exc)}
