"""A webes felület munkái: egy munka = egy munkamappa (work/<név>/), a futás a CLI külön folyamatban.

A futás külön folyamat, így a felület nem tartja a memóriában a modelleket, a leállítás és a folytatás pedig
pontosan úgy működik, mint parancssorból (run.pid, SIGTERM, manifest).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from .. import runlock
from ..config import Config
from ..manifest import Manifest
from ..pipeline import STEP_NAMES

JOB_FILE = "job.json"
JOB_CONFIG = "config.yaml"
LOG_FILE = "run.log"
VIDEO_TYPES = (".mp4", ".mov", ".m4v", ".mkv", ".webm")


@dataclass
class Job:
    name: str
    workdir: Path
    source: Path
    output: Path

    @property
    def config_path(self) -> Path:
        return self.workdir / JOB_CONFIG

    @property
    def log_path(self) -> Path:
        return self.workdir / LOG_FILE

    def manifest(self) -> Manifest | None:
        p = self.workdir / "manifest.json"
        return Manifest.load(p) if p.exists() else None


class Workspace:
    """A projektmappa (config.yaml, work/, output/) és az onnan indított futások."""

    def __init__(self, root: Path):
        self.root = root
        self.work = root / "work"
        self.outputs = root / "output"
        self._procs: dict[str, subprocess.Popen] = {}  # a felületről indított, még élő folyamatok

    # --- munkák ---

    def jobs(self) -> list[Job]:
        if not self.work.exists():
            return []
        found = []
        for d in sorted(p for p in self.work.iterdir() if p.is_dir()):
            job = self.load(d.name)
            if job is not None:
                found.append(job)
        return found

    def load(self, name: str) -> Job | None:
        wd = self.work / name
        meta = wd / JOB_FILE
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return Job(name, wd, Path(data["source"]), Path(data["output"]))
        if (wd / "manifest.json").exists():  # parancssorból indított munka
            m = Manifest.load(wd / "manifest.json")
            return Job(name, wd, Path(m.source), self.outputs / f"{name}_hu.mp4")
        return None

    def create(self, upload: Path) -> Job:
        """A feltöltött videóból új munka; a videó a munkamappába másolódik."""
        if upload.suffix.lower() not in VIDEO_TYPES:
            raise ValueError(f"Nem támogatott fájltípus: {upload.suffix}")
        name = unique_name(slugify(upload.stem), {p.name for p in self.work.glob("*")} if self.work.exists() else set())
        wd = self.work / name
        wd.mkdir(parents=True)
        source = wd / f"source{upload.suffix.lower()}"
        shutil.copyfile(upload, source)
        job = Job(name, wd, source, self.outputs / f"{name}_hu.mp4")
        (wd / JOB_FILE).write_text(
            json.dumps({"source": str(source), "output": str(job.output), "original_name": upload.name}),
            encoding="utf-8",
        )
        return job

    # --- beállítások ---

    def base_config(self) -> dict:
        p = self.root / "config.yaml"
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}

    def job_config_default(self) -> Config:
        return Config.load(self.root / "config.yaml")

    def job_config(self, job: Job) -> Config:
        """A munka beállításai: a saját config.yaml, ha van, különben a projekté."""
        return Config.load(job.config_path if job.config_path.exists() else self.root / "config.yaml")

    def save_config(self, job: Job, values: dict) -> Config:
        """A projekt config.yaml-ja + az űrlap értékei → a munka config.yaml-ja (érvénytelen értéknél hiba)."""
        data = self.base_config()
        if job.config_path.exists():
            data.update(yaml.safe_load(job.config_path.read_text(encoding="utf-8")) or {})
        data.update(values)
        cfg = Config.model_validate(data)
        job.config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return cfg

    # --- futtatás ---

    def running(self) -> Job | None:
        """Az éppen futó munka (egyszerre csak egy futhat: 16 GB-on két futás nem fér el)."""
        for name, proc in list(self._procs.items()):
            if proc.poll() is None:
                return self.load(name)
            del self._procs[name]
        for job in self.jobs():
            if runlock.running_pid(job.workdir) is not None:
                return job
        return None

    def command(self, job: Job, from_step: str | None = None) -> list[str]:
        if from_step is not None and from_step not in STEP_NAMES:
            raise ValueError(f"Ismeretlen lépés: {from_step}")
        cmd = [sys.executable, "-m", "mp4translator.cli", "dub", str(job.source), "-o", str(job.output)]
        cmd += ["--workdir", str(job.workdir)]
        if job.config_path.exists():
            cmd += ["-c", str(job.config_path)]
        elif (self.root / "config.yaml").exists():
            cmd += ["-c", str(self.root / "config.yaml")]
        if from_step:
            cmd += ["--from-step", from_step]
        return cmd

    def start(self, job: Job, from_step: str | None = None) -> None:
        busy = self.running()
        if busy is not None:
            raise RuntimeError(f"Már fut egy feldolgozás: {busy.name}")
        job.output.parent.mkdir(parents=True, exist_ok=True)
        log = job.log_path.open("a", encoding="utf-8")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "120", "NO_COLOR": "1", "TERM": "dumb"}
        self._procs[job.name] = subprocess.Popen(
            self.command(job, from_step), cwd=self.root, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,  # a felület leállítása ne vigye magával a futást
        )
        log.close()

    def stop(self, job: Job) -> bool:
        return runlock.request_stop(job.workdir) is not None


# --- állapot ---


def step_states(m: Manifest | None, running: bool) -> list[tuple[str, str]]:
    """Lépésenként: done | running | paused | pending."""
    out = []
    for name in STEP_NAMES:
        if m is not None and name in m.steps_done:
            state = "done"
        elif m is not None and name == m.in_progress:
            state = "running" if running else "paused"
        else:
            state = "pending"
        out.append((name, state))
    return out


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return ""
    text = ANSI.sub("", path.read_bytes().decode("utf-8", errors="replace"))  # read_text a \\r-t sortörésre cserélné
    # a folyamatjelzők \r-rel írják felül magukat: soronként csak az utolsó állapot kell
    rows = [ln.rsplit("\r", 1)[-1] for ln in text.split("\n")]
    rows = [BAR.sub("", ln) for ln in rows]
    keep = [ln for ln in rows if ln.strip(" |") and not NOISE.search(ln)]
    return "\n".join(keep[-lines:])


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
BAR = re.compile(r"[█▏▎▍▌▋▊▉#]{2,}\s*")  # folyamatjelző sávok
# A külső könyvtárak figyelmeztetései és folyamatjelzői nem érdeklik a felhasználót
NOISE = re.compile(r"Warning|warnings\.warn|_VF\.stft|Loading weights|^\s*(txt|for match) |transformers\]|Redirects are")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:60] or "video"


def unique_name(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}_{k}" in taken:
        k += 1
    return f"{name}_{k}"


def ollama_ok(host: str) -> bool:
    try:
        return httpx.get(f"{host.rstrip('/')}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def start_ollama(root: Path) -> None:
    """Az Ollama elindítása a projekt ajánlott beállításaival (kisebb memória), a felülettől függetlenül."""
    env = {**os.environ, "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    log = (root / "work").joinpath("ollama.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        subprocess.Popen(["ollama", "serve"], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
