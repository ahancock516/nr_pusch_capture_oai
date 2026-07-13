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
import socket
import subprocess
import threading
import time
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_AMF_CONTAINER = "oai-amf"
DEFAULT_GNB_CONTAINER = "oai-gnb"
DEFAULT_STATS_FILE = "/opt/oai-gnb/nrRRC_stats.log"
DEFAULT_OUTPUT = str(
    Path(__file__).parent.parent / "data" / "label_map.json"
)
DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_SOCKET_PATH   = "/tmp/pusch_label.sock"

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

# Service Request — two consecutive lines in [amf_n1] context
# Phase 1: AMF assigns new ran_ue_ngap_id to existing UE
_RE_SR_NGAP = re.compile(
    r'\[amf_n1\].*\bamf_ue_ngap_id (\d+), ran_ue_ngap_id (\d+)\b'
)
# Phase 2: SUPI resolved from existing context
_RE_SR_SUPI = re.compile(
    r'\[amf_n1\].*Key for PDU Session context: SUPI imsi-(\d+)'
)

# gNB MAC-layer RA failure signatures — a UE that never reached RRC_CONNECTED,
# so it will never appear in the AMF log or nrRRC_stats.log. These RNTIs are
# confirmed dead ("ghosts": false PRACH detections or genuinely failed
# attaches) and their pending PUSCH captures can never be labeled.
_RE_RA_FAIL = [
    re.compile(r'UE (?:0x)?([0-9a-fA-F]+) RA failed at state WAIT_Msg3'),
    re.compile(r'RA Contention Resolution timer expired for UE (?:0x)?([0-9a-fA-F]+)'),
    re.compile(r'No UE found with C-RNTI ([0-9a-fA-F]+), ignoring Msg3'),
    re.compile(r'TC-RNTI ([0-9a-fA-F]+): exceeded RA window, cannot schedule Msg2'),
]


class LabelMonitor:
    def __init__(self, amf_container, gnb_container, stats_file, output_path,
                 poll_interval, socket_path):
        self.amf_container = amf_container
        self.gnb_container = gnb_container
        self.stats_file    = Path(stats_file)
        self.output_path   = Path(output_path)
        self.poll_interval = poll_interval
        self.socket_path   = socket_path

        # Snapshot wall↔monotonic offset at startup.
        # Any CLOCK_MONOTONIC value from this machine can be converted to
        # wall time as:  wall_ns = mono_ns + mono_to_wall_offset_ns
        self.mono_to_wall_offset = time.time_ns() - time.monotonic_ns()

        self._lock         = threading.Lock()
        self._stop         = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: list = []   # connected plugin socket file objects

        # ran_ue_ngap_id → {imsi, amf_ue_ngap_id, t_registered_wall_ns}
        # AMF registration seen but RNTI not yet resolved from stats file.
        self._pending: dict = {}

        # rnti → {imsi, amf_ue_ngap_id, cu_ue_id,
        #          t_start_mono_ns, t_start_wall_ns}
        self._active: dict = {}

        # list of closed session dicts
        self._completed: list = []

        # rnti (int) -> True, RNTIs confirmed dead via gNB MAC RA-failure logs.
        # Used to dedupe "D" discard broadcasts and to avoid ever promoting
        # a confirmed-dead RNTI to active/pending.
        self._dead_rntis: dict = {}
        self._ghosts_discarded = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        tag = {"info": "INFO ", "warning": "WARN ", "error": "ERROR"}[level]
        print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)

    def _add_pending(self, imsi, ran_id, amf_id):
        """Add or update an IMSI→ran_id mapping in _pending (must be called unlocked)."""
        with self._lock:
            # If this IMSI is already active, move it to completed so the stats
            # poller can re-resolve its new RNTI.
            existing_rnti = next(
                (r for r, v in self._active.items() if v["imsi"] == imsi), None
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
            # Purge stale pending entries for the same IMSI.
            for k in [k for k, v in self._pending.items() if v["imsi"] == imsi]:
                del self._pending[k]
            # Warn if a different IMSI already holds this ran_id (CU UE ID recycled).
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

    # ── AMF log thread ────────────────────────────────────────────────────────

    def _amf_thread(self):
        """
        Tail docker AMF logs.  Detects both full NAS registrations and
        Service Requests (where the UE reuses an existing context) so that
        RNTI changes after idle-mode reconnects are captured.
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
                # Service Request state: set when phase-1 line is seen,
                # consumed when phase-2 (SUPI) line is seen.
                sr_ran_id = None
                sr_amf_id = None
                for line in proc.stdout:
                    if self._stop.is_set():
                        break

                    # Full NAS registration
                    m = _RE_AMF.search(line)
                    if m:
                        imsi   = m.group(1)
                        ran_id = int(m.group(2))
                        amf_id = int(m.group(3))
                        sr_ran_id = None  # registration supersedes any pending SR
                        self._add_pending(imsi, ran_id, amf_id)
                        self._log(f"AMF  IMSI={imsi}  ran_id={ran_id}  amf_id={amf_id}")
                        continue

                    # Service Request phase 1 — new ran_ue_ngap_id assigned
                    m = _RE_SR_NGAP.search(line)
                    if m:
                        sr_amf_id = int(m.group(1))
                        sr_ran_id = int(m.group(2))
                        continue

                    # Service Request phase 2 — SUPI resolved from existing context
                    if sr_ran_id is not None:
                        m = _RE_SR_SUPI.search(line)
                        if m:
                            imsi = m.group(1)
                            ran_id, amf_id = sr_ran_id, sr_amf_id
                            sr_ran_id = None
                            self._add_pending(imsi, ran_id, amf_id)
                            self._log(f"SR   IMSI={imsi}  ran_id={ran_id}  amf_id={amf_id}")

                proc.wait()
            except Exception as exc:
                self._log(f"AMF thread error: {exc}", "warning")
            if not self._stop.is_set():
                time.sleep(2)   # back-off before reconnecting

    # ── gNB MAC RA-failure thread ─────────────────────────────────────────────

    def _gnb_thread(self):
        """
        Tail the gNB's own container log for RA-failure signatures. These
        RNTIs never reach RRC_CONNECTED, so they'll never show up in the AMF
        log or nrRRC_stats.log — they're permanently un-labelable. Broadcast
        a "D <rnti_hex>" message so the plugin can discard their pending
        captures immediately instead of writing them unlabeled at shutdown.
        """
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(
                    ["docker", "logs", "-f", "--tail", "0", self.gnb_container],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    if self._stop.is_set():
                        break

                    for pattern in _RE_RA_FAIL:
                        m = pattern.search(line)
                        if not m:
                            continue
                        try:
                            rnti = int(m.group(1), 16)
                        except ValueError:
                            continue
                        self._mark_dead(rnti)
                        break

                proc.wait()
            except Exception as exc:
                self._log(f"gNB RA-fail thread error: {exc}", "warning")
            if not self._stop.is_set():
                time.sleep(2)   # back-off before reconnecting

    def _mark_dead(self, rnti: int):
        """Record a confirmed-dead RNTI and broadcast a discard to connected plugins."""
        with self._lock:
            if rnti in self._dead_rntis:
                return
            self._dead_rntis[rnti] = True
            # A dead RNTI can never be a real session — drop any stray active
            # entry that might exist for it (defensive; _active is keyed by
            # RNTI, unlike _pending which is keyed by ran_ue_ngap_id).
            self._active.pop(rnti, None)
            self._ghosts_discarded += 1
        self._log(f"GHOST RNTI=0x{rnti:04x}  confirmed dead (RA never completed)")
        self._broadcast(f"D {rnti:04x}\n")

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

                # Collect socket messages to broadcast after releasing _lock
                add_msgs:    list = []
                remove_msgs: list = []

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
                            remove_msgs.append(f"R {rnti:04x}\n")
                        # Back-date t_start by one poll interval so captures
                        # that arrived before this poll are still within window.
                        grace_ns = int(self.poll_interval * 1e9)
                        self._active[rnti] = {
                            "imsi":            info["imsi"],
                            "amf_ue_ngap_id":  info["amf_ue_ngap_id"],
                            "cu_ue_id":        cu_ue_id,
                            "t_start_mono_ns": now_mono - grace_ns,
                            "t_start_wall_ns": now_wall - grace_ns,
                        }
                        self._log(
                            f"LINK  RNTI=0x{rnti:04x}  IMSI={info['imsi']}  "
                            f"cu_ue_id={cu_ue_id}"
                        )
                        add_msgs.append(f"A {rnti:04x} {info['imsi']}\n")

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
                            remove_msgs.append(f"R {rnti:04x}\n")

                # Broadcast outside _lock to avoid nesting with _clients_lock
                for msg in remove_msgs + add_msgs:
                    self._broadcast(msg)

            except FileNotFoundError:
                pass   # gNB not running yet, wait quietly
            except Exception as exc:
                self._log(f"Stats poll error: {exc}", "warning")

            self._stop.wait(self.poll_interval)

    # ── socket server ─────────────────────────────────────────────────────────

    def _broadcast(self, msg: str):
        """Send msg to all connected plugin clients, dropping dead connections."""
        data = msg.encode()
        with self._clients_lock:
            dead = []
            for conn in self._clients:
                try:
                    conn.sendall(data)
                except Exception:
                    dead.append(conn)
            for conn in dead:
                self._clients.remove(conn)
                try:
                    conn.close()
                except Exception:
                    pass

    def _socket_server_thread(self):
        """
        Unix domain socket server.  On each new connection send a snapshot of
        the current RNTI→IMSI table, then stream add/remove events in real time.

        Protocol (line-delimited text):
          S\\n                    — snapshot start
          A <rnti_hex> <imsi>\\n  — add / update mapping
          R <rnti_hex>\\n         — remove mapping
        """
        sock_path = Path(self.socket_path)
        sock_path.unlink(missing_ok=True)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(4)
        srv.settimeout(1.0)
        self._log(f"Label socket listening at {self.socket_path}")

        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except Exception as exc:
                    self._log(f"Socket accept error: {exc}", "warning")
                    continue

                # Build and send snapshot while holding the main lock
                with self._lock:
                    lines = ["S\n"]
                    for rnti, info in self._active.items():
                        lines.append(f"A {rnti:04x} {info['imsi']}\n")

                try:
                    for line in lines:
                        conn.sendall(line.encode())
                except Exception:
                    conn.close()
                    continue

                with self._clients_lock:
                    self._clients.append(conn)
                self._log(
                    f"Plugin connected — sent snapshot of "
                    f"{len(lines) - 1} active entries"
                )
        finally:
            srv.close()
            sock_path.unlink(missing_ok=True)

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
        self._log(f"gNB container : {self.gnb_container}")
        self._log(f"RRC stats file: {self.stats_file}")
        self._log(f"Output        : {self.output_path}")
        self._log(f"Poll interval : {self.poll_interval}s")
        self._log(f"Socket path   : {self.socket_path}")
        self._log(f"mono_to_wall_offset = {self.mono_to_wall_offset} ns")

        for name, target in [
            ("amf",    self._amf_thread),
            ("gnb",    self._gnb_thread),
            ("stats",  self._stats_thread),
            ("write",  self._write_thread),
            ("socket", self._socket_server_thread),
        ]:
            threading.Thread(target=target, daemon=True, name=name).start()

        try:
            while True:
                time.sleep(5)
                with self._lock:
                    self._log(
                        f"State  pending={len(self._pending)}  "
                        f"active={len(self._active)}  "
                        f"completed={len(self._completed)}  "
                        f"ghosts_discarded={self._ghosts_discarded}"
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
        "--gnb-container", default=DEFAULT_GNB_CONTAINER,
        help=f"Docker container name for the gNB, tailed for RA-failure "
             f"signatures (default: {DEFAULT_GNB_CONTAINER})"
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
    ap.add_argument(
        "--socket-path", default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path for real-time RNTI→IMSI push (default: {DEFAULT_SOCKET_PATH})"
    )
    args = ap.parse_args()

    LabelMonitor(
        amf_container  = args.amf_container,
        gnb_container  = args.gnb_container,
        stats_file     = args.stats_file,
        output_path    = args.output,
        poll_interval  = args.poll_interval,
        socket_path    = args.socket_path,
    ).run()


if __name__ == "__main__":
    main()
