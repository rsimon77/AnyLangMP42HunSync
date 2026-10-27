"""Futó feldolgozás nyilvántartása a munkamappában: egyszerre egy futás, és kívülről leállítható."""

import os
import signal
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
    return True


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
    pid = running_pid(workdir)
    if pid is not None and pid != os.getpid():
        raise AlreadyRunning(f"Ezen a munkamappán már fut egy feldolgozás (PID {pid}): {workdir}")
    (workdir / LOCK_NAME).write_text(str(os.getpid()))


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
