"""
locustfile_worldcup.py  —  WorldCup98 traffic replay for Locust.

Environment variables (set before running):
    NUM_USERS          — must match the -u flag passed to locust (default: 100)
    TEST_DURATION_SEC  — real-wall-clock duration of the test (default: 3600)

How the math works:
    Each Locust user independently calls wait_time() to get the gap between
    requests. Total RPS = NUM_USERS / wait_time_per_user. To hit a target of
    T RPS with N users, each user must fire at T/N req/s, so wait = N/T.

    The profile rps value is the TOTAL target — always divide by NUM_USERS.
    Do NOT use wait = 1/profile_rps — that treats profile_rps as a per-user
    rate and overshoots by a factor of NUM_USERS.

Example:
    NUM_USERS=500 TEST_DURATION_SEC=1800 \\
    python -m locust -f locustfile_worldcup.py \\
        --headless -u 500 -r 500 \\
        --host http://localhost:8080 \\
        --run-time 30m
"""

import csv
import os
import threading
import time

from locust import HttpUser, task

# ── Profile loading ───────────────────────────────────────────────────────────
_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "traffic_profile_test.csv")
_profile: list[tuple[int, float]] = []

with open(_PROFILE_PATH, newline="") as f:
    for row in csv.DictReader(f):
        _profile.append((int(row["second_offset"]), float(row["rps"])))

_TOTAL_SEC = _profile[-1][0] if _profile else 3600

# ── Configuration from environment ───────────────────────────────────────────
NUM_USERS         = int(os.environ.get("NUM_USERS",         100))
TEST_DURATION_SEC = int(os.environ.get("TEST_DURATION_SEC", 3600))

TIME_COMPRESSION_FACTOR = _TOTAL_SEC / max(TEST_DURATION_SEC, 1)

print("[locust] Profile loaded:")
print(f"  Total dataset seconds : {_TOTAL_SEC}")
print(f"  Target test duration  : {TEST_DURATION_SEC}s")
print(f"  Time compression      : {TIME_COMPRESSION_FACTOR:.2f}x")
print(f"  Profile buckets       : {len(_profile)}")
print(f"  Peak RPS in profile   : {max(rps for _, rps in _profile):.1f}")
print(f"  NUM_USERS             : {NUM_USERS}  (must match -u)")

# ── Lazy clock start ──────────────────────────────────────────────────────────
# _T0 is set on the first request, not at module load time.
# Locust loads the module before the ramp starts, so using time.time() at
# import would offset the profile timeline by several seconds before any
# traffic actually flows.
_T0: float | None = None
_T0_lock = threading.Lock()

def _get_t0() -> float:
    global _T0
    if _T0 is None:
        with _T0_lock:
            if _T0 is None:
                _T0 = time.time()
    return _T0


# ── RPS lookup (cached once per second) ──────────────────────────────────────
_cached_rps:       float = 0.1
_cache_updated_at: float = 0.0

def _current_target_rps() -> float:
    global _cached_rps, _cache_updated_at
    now = time.time()

    if now - _cache_updated_at < 1.0:
        return _cached_rps

    elapsed_real = now - _get_t0()
    elapsed_sim  = (elapsed_real * TIME_COMPRESSION_FACTOR) % max(_TOTAL_SEC, 1)

    lo, hi = 0, len(_profile) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _profile[mid][0] <= elapsed_sim:
            lo = mid
        else:
            hi = mid - 1

    _cached_rps       = max(_profile[lo][1], 0.1)
    _cache_updated_at = now
    return _cached_rps


# ── Locust user ───────────────────────────────────────────────────────────────

class WorldCupReplayUser(HttpUser):

    @task
    def hit_api(self):
        self.client.get("/api")

    def wait_time(self):
        target      = _current_target_rps()   # total RPS across all users
        user_target = target / NUM_USERS       # this user's share
        wait        = 1.0 / max(user_target, 0.01)
        return max(wait, 0.0)