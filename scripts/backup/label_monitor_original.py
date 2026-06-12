#!/usr/bin/env python3
"""
label_monitor.py — Sidecar for PUSCH capture self-labeling.

Monitors the OAI AMF container logs and gNB nrRRC_stats.log to build a
real-time RNTI→IMSI mapping, writing label_map.json continuously during
a capture session.

Correlation chain
-----------------
  AMF log  (docker logs -f oai-amf):
    "[amf_n1] [info] UE (IMSI X, GUTI ..., current RAN ID Y, current AMF ID Z)
     has been registered"
        → IMSI X owns ran_ue_ngap_id Y  (fires on every registration)

  gNB nrRRC_stats.log  (polled every --poll-interval seconds):
    "UE N  CU UE ID Y  DU UE ID ...  RNTI Z  ..."
        → cu_ue_id Y is on air as RNTI Z

  Join key: ran_ue_ngap_id == cu_ue_id   (both are the gNB-assigned CU UE ID)

Timestamps
----------
  The PUSCH plugin timestamps captures with CLOCK_MONOTONIC.
  This script records the wall↔monotonic offset at startup so that
  post_label.py can map capture timestamps into session time windows.

Usage
-----
  python3 label_monitor.py                          # defaults
  python3 label_monitor.py --output /tmp/map.json --poll-interval 0.5
  python3 label_monitor.py --amf-container oai-amf --stats-file /path/to/nrRRC_stats.log
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_AMF_CONTAINER = "oai-amf"
DEFAULT_STATS_FILE = "/home/user/openairinterface5g/nrRRC_stats.log"
DEFAULT_OUTPUT = str(
    Path(__file__).parent.parent / "data" / "label_map.json"
)
DEFAULT_POLL_INTERVAL = 1.0

# ── regex ─────────────────────────────────────────────────────────────────────

# AMF info line — fires on every NAS registration regardless of GUTI state
_RE_AMF = re.compile(
    r'\[amf_n1\]\s+\[info\]\s+UE\s+\('
    r'IMSI\s+(\d+),\s+GUTI\s+\S+,\s+'
    r'current RAN ID\s+(\d+),\s+current AMF ID\s+(\d+)'
    r'\)\s+has been registered'
)

# gNB RRC stats — one line per connected UE
_RE_RRC = re.compile(
    r'UE\s+\d+\s+CU UE ID\s+(\d+)\s+DU UE ID\s+\d+\s+RNTI\s+([0-9a-fA-F]+)'
)


class LabelMonitor:
    def __init__(self, amf_container, stats_file, output_path, poll_interval):
        self.amf_container = amf_container
        self.stats_file    = Path(stats_file)
        self.output_path   = Path(output_path)
        self.poll_interval = poll_interval

        # Snapshot wall↔monotonic offset at startup.
        # Any CLOCK_MONOTONIC value from this machine can be converted to
        # wall time as:  wall_ns = mono_ns + mono_to_wall_offset_ns
        self.mono_to_wall_offset = time.time_ns() - time.monotonic_ns()

        self._lock      = threading.Lock()
        self._stop      = threading.Event()

        # ran_ue_ngap_id → {imsi, amf_ue_ngap_id, t_registered_wall_ns}
        # AMF registration seen but RNTI not yet resolved from stats file.
        self._pending: dict = {}

        # rnti → {imsi, amf_ue_ngap_id, cu_ue_id,
        #          t_start_mono_ns, t_start_wall_ns}
        self._active: dict = {}

        # list of closed session dicts
        self._completed: list = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        tag = {"info": "INFO ", "warning": "WARN ", "error": "ERROR"}[level]
        print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)

    # ── AMF log thread ────────────────────────────────────────────────────────

    def _amf_thread(self):
        """
        Tail docker AMF logs.  On each registration event, record
        IMSI → ran_ue_ngap_id in _pending so the stats poller can
        resolve it to a RNTI.
        """
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(
                    ["docker", "logs", "-f", "--tail", "0", self.amf_container],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    if self._stop.is_set():
                        break
                    m = _RE_AMF.search(line)
                    if not m:
                        continue
                    imsi   = m.group(1)
                    ran_id = int(m.group(2))
                    amf_id = int(m.group(3))
                    with self._lock:
                        # A re-registration from a device already in _active:
                        # update ran_id (RNTI may have changed after handover/
                        # restart) by moving it back to pending so the stats
                        # poller can re-resolve the RNTI.
                        existing_rnti = next(
                            (r for r, v in self._active.items()
                             if v["imsi"] == imsi), None
                        )
                        if existing_rnti is not None:
                            old = self._active.pop(existing_rnti)
                            self._completed.append({
                                **old,
                                "rnti":          existing_rnti,
                                "t_end_mono_ns": time.monotonic_ns(),
                                "t_end_wall_ns": time.time_ns(),
                                "status":        "completed",
                            })
                        # Purge stale pending entries for the same IMSI
                        # (e.g. rapid re-registrations before RNTI is resolved)
                        for k in [k for k, v in self._pending.items()
                                  if v["imsi"] == imsi]:
                            del self._pending[k]
                        # If a different IMSI already holds this ran_id in
                        # pending (CU UE ID slot recycled by gNB), close it
                        # as a short/incomplete session so it isn't silently lost.
                        displaced = self._pending.get(ran_id)
                        if displaced is not None and displaced["imsi"] != imsi:
                            self._log(
                                f"DISP  ran_id={ran_id} was pending for "
                                f"IMSI={displaced['imsi']}, displaced by {imsi}",
                                "warning",
                            )
                        self._pending[ran_id] = {
                            "imsi":           imsi,
                            "amf_ue_ngap_id": amf_id,
                            "t_registered_wall_ns": time.time_ns(),
                        }
                    self._log(
                        f"AMF  IMSI={imsi}  ran_id={ran_id}  amf_id={amf_id}"
                    )
                proc.wait()
            except Exception as exc:
                self._log(f"AMF thread error: {exc}", "warning")
            if not self._stop.is_set():
                time.sleep(2)   # back-off before reconnecting

    # ── RRC stats poll thread ─────────────────────────────────────────────────

    def _stats_thread(self):
        """
        Poll nrRRC_stats.log on a fixed interval.

        Resolves pending entries (IMSI known, RNTI unknown) to active once
        the stats file shows the UE.  Detects disconnections when a CU UE ID
        disappears from the stats file.
        """
        while not self._stop.is_set():
            try:
                text = self.stats_file.read_text(errors="replace")
                # cu_ue_id → rnti for every UE currently in the stats file
                current: dict[int, int] = {
                    int(m.group(1)): int(m.group(2), 16)
                    for m in _RE_RRC.finditer(text)
                }

                now_mono = time.monotonic_ns()
                now_wall = time.time_ns()

                with self._lock:
                    # Promote pending → active for any CU UE ID now visible
                    for cu_ue_id, rnti in current.items():
                        if cu_ue_id not in self._pending:
                            continue
                        info = self._pending.pop(cu_ue_id)
                        # If this RNTI is already active for a different IMSI
                        # (gNB recycled the slot), close the old session first.
                        if rnti in self._active and self._active[rnti]["imsi"] != info["imsi"]:
                            old = self._active.pop(rnti)
                            self._completed.append({
                                **old,
                                "rnti":          rnti,
                                "t_end_mono_ns": now_mono,
                                "t_end_wall_ns": now_wall,
                                "status":        "completed",
                            })
                            self._log(
                                f"RNTI RECYCLE  0x{rnti:04x} was "
                                f"IMSI={old['imsi']}, now {info['imsi']}",
                                "warning",
                            )
                        self._active[rnti] = {
                            "imsi":            info["imsi"],
                            "amf_ue_ngap_id":  info["amf_ue_ngap_id"],
                            "cu_ue_id":        cu_ue_id,
                            "t_start_mono_ns": now_mono,
                            "t_start_wall_ns": now_wall,
                        }
                        self._log(
                            f"LINK  RNTI=0x{rnti:04x}  IMSI={info['imsi']}  "
                            f"cu_ue_id={cu_ue_id}"
                        )

                    # Close sessions for UEs no longer in the stats file
                    active_cu_ids = {v["cu_ue_id"] for v in self._active.values()}
                    gone_cu_ids   = active_cu_ids - set(current.keys())
                    for rnti, info in list(self._active.items()):
                        if info["cu_ue_id"] in gone_cu_ids:
                            self._completed.append({
                                **info,
                                "rnti":          rnti,
                                "t_end_mono_ns": now_mono,
                                "t_end_wall_ns": now_wall,
                                "status":        "completed",
                            })
                            del self._active[rnti]
                            self._log(
                                f"GONE  RNTI=0x{rnti:04x}  IMSI={info['imsi']}"
                            )

            except FileNotFoundError:
                pass   # gNB not running yet, wait quietly
            except Exception as exc:
                self._log(f"Stats poll error: {exc}", "warning")

            self._stop.wait(self.poll_interval)

    # ── output ────────────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        """Build a serialisable snapshot of current state."""
        sessions = list(self._completed)
        for rnti, info in self._active.items():
            sessions.append({
                **info,
                "rnti":          rnti,
                "t_end_mono_ns": None,
                "t_end_wall_ns": None,
                "status":        "active",
            })
        return {
            "mono_to_wall_offset_ns": self.mono_to_wall_offset,
            "generated_at_wall_ns":   time.time_ns(),
            "sessions":               sessions,
        }

    def _write(self):
        """Atomically write label_map.json via a temp file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix(".tmp")
        with self._lock:
            data = self._snapshot()
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.output_path)

    def _write_thread(self):
        while not self._stop.is_set():
            try:
                self._write()
            except Exception as exc:
                self._log(f"Write error: {exc}", "warning")
            self._stop.wait(self.poll_interval * 2)

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._log(f"AMF container : {self.amf_container}")
        self._log(f"RRC stats file: {self.stats_file}")
        self._log(f"Output        : {self.output_path}")
        self._log(f"Poll interval : {self.poll_interval}s")
        self._log(f"mono_to_wall_offset = {self.mono_to_wall_offset} ns")

        for name, target in [
            ("amf",   self._amf_thread),
            ("stats", self._stats_thread),
            ("write", self._write_thread),
        ]:
            threading.Thread(target=target, daemon=True, name=name).start()

        try:
            while True:
                time.sleep(5)
                with self._lock:
                    self._log(
                        f"State  pending={len(self._pending)}  "
                        f"active={len(self._active)}  "
                        f"completed={len(self._completed)}"
                    )
        except KeyboardInterrupt:
            self._log("Stopping — writing final label map…")
            self._stop.set()
            self._write()
            self._log(f"Done. {self.output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sidecar label monitor for PUSCH capture experiments."
    )
    ap.add_argument(
        "--amf-container", default=DEFAULT_AMF_CONTAINER,
        help=f"Docker container name for the AMF (default: {DEFAULT_AMF_CONTAINER})"
    )
    ap.add_argument(
        "--stats-file", default=DEFAULT_STATS_FILE,
        help="Path to nrRRC_stats.log (default: OAI build directory)"
    )
    ap.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output label_map.json path (default: {DEFAULT_OUTPUT})"
    )
    ap.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
        help=f"RRC stats poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})"
    )
    args = ap.parse_args()

    LabelMonitor(
        amf_container  = args.amf_container,
        stats_file     = args.stats_file,
        output_path    = args.output,
        poll_interval  = args.poll_interval,
    ).run()


if __name__ == "__main__":
    main()
