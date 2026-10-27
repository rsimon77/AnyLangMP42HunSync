"""Futó feldolgozás nyilvántartása a munkamappában: egyszerre egy futás, és kívülről leállítható."""

import os
import signal
import subprocess
from pathlib import Path

LOCK_NAME = "run.pid"


class AlreadyRunning(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return _is_ours(pid)


def _is_ours(pid: int) -> bool:
    """A PID tényleg egy mp4translator-futásé? (Egy elárvult zárfájl PID-jét a rendszer másnak is kiadhatja.)"""
    try:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False)
    except OSError:
        return True  # nem tudjuk ellenőrizni: inkább futónak tekintjük
    return "mp4translator" in cmd.stdout


def running_pid(workdir: Path) -> int | None:
    """A munkamappán dolgozó élő folyamat PID-je; az elárvult zárfájlt eltávolítja."""
    lock = workdir / LOCK_NAME
    try:
        pid = int(lock.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if pid != os.getpid() and not _alive(pid):
        lock.unlink(missing_ok=True)
        return None
    return pid


def acquire(workdir: Path) -> None:
    lock = workdir / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomi: két egyszerre induló futásból egy nyer
        except FileExistsError:
            pid = running_pid(workdir)  # az elárvult zárfájlt eltávolítja
            if pid == os.getpid():
                return
            if pid is not None:
                raise AlreadyRunning(f"Ezen a munkamappán már fut egy feldolgozás (PID {pid}): {workdir}") from None
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return
    raise AlreadyRunning(f"Nem sikerült zárolni a munkamappát: {workdir}")


def release(workdir: Path) -> None:
    lock = workdir / LOCK_NAME
    if running_pid(workdir) == os.getpid():
        lock.unlink(missing_ok=True)


def request_stop(workdir: Path) -> int | None:
    """Szelíd leállítást kér; visszaadja a PID-et, vagy None-t, ha nincs futás.

    SIGTERM-et küld, mert a háttérben (&, nohup) indított folyamat a SIGINT-et figyelmen kívül hagyja;
    a futás a SIGTERM-et Ctrl+C-ként kezeli.
    """
    pid = running_pid(workdir)
    if pid is None:
        return None
    os.kill(pid, signal.SIGTERM)
    return pid
