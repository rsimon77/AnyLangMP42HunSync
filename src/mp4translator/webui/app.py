"""Helyi webes felület (Gradio): feltöltés, beállítások, futtatás élő naplóval, eredmény, beszélők."""

from pathlib import Path

import gradio as gr
import numpy as np

from .. import runlock
from ..models import DEMUCS, LLMS, WHISPERS, ModelChoice, ollama_tag
from ..ollama import installed_models, is_installed, pull
from ..pipeline import STEP_NAMES
from ..steps.synthesize import is_synthesized
from . import segments as seged
from .i18n import LANGS, pick_lang, t
from .jobs import Job, Workspace, ollama_ok, restart_ollama, start_ollama, step_states, stop_ollama, tail

STATE_ICON = {"done": "✅", "running": "⏳", "paused": "⏸️", "pending": "▫️"}
KINDS = ["auto", "male", "female", "child"]
THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def model_choices(catalog: list[ModelChoice], lang: str, current: str | None) -> list[tuple[str, str]]:
    """Az ajánlott modellek (címke, érték) párjai; az aktuális, listán kívüli érték is választható marad."""
    choices = [(m.title(lang), m.id) for m in catalog]
    if current and current not in {m.id for m in catalog}:
        choices.append((current, current))
    return choices


def llm_choices(lang: str, current: str | None, installed: list[str] | None) -> list[tuple[str, str]]:
    """Az ajánlott fordítómodellek (✓ = már letöltve) és a többi letöltött Ollama-modell."""
    have = installed or []
    choices = [
        (("✓ " if is_installed(ollama_tag(m.id), have) else "⬇ ") + m.title(lang), m.id) for m in LLMS
    ]
    known = {ollama_tag(m.id) for m in LLMS}
    choices += [(f"✓ {tag}", f"ollama:{tag}") for tag in have if tag not in known]
    if current and current not in {v for _, v in choices}:
        choices.append((current, current))
    return choices


def build(ws: Workspace) -> gr.Blocks:
    host = ws.job_config_default().ollama_host
    i18n_targets: list[tuple[gr.components.Component, callable]] = []

    def tr(component, key: str, attr: str = "label"):
        """A komponens szövegét a nyelvváltáskor frissíti."""
        i18n_targets.append((component, lambda lang: gr.update(**{attr: t(lang, key)})))
        return component

    def tr_choices(component, prefix: str, values: list[str]):
        i18n_targets.append(
            (component, lambda lang: gr.update(choices=[(t(lang, f"{prefix}_{v}"), v) for v in values]))
        )
        return component

    def job_of(name: str | None) -> Job | None:
        return ws.load(name) if name else None

    def job_names() -> list[str]:
        jobs = sorted(ws.jobs(), key=lambda j: j.workdir.stat().st_mtime, reverse=True)
        return [j.name for j in jobs]

    with gr.Blocks(title="AnyLangMP42HunSync") as demo:
        lang = gr.State("en")
        was_running = gr.State(False)  # a mondatok táblázata a futás végén frissül
        with gr.Row():
            with gr.Column(scale=4):
                tr(gr.Markdown(t("en", "title")), "title", "value")
                tr(gr.Markdown(t("en", "subtitle")), "subtitle", "value")
            with gr.Column(scale=1, min_width=200):
                lang_radio = tr(
                    gr.Radio([(name, code) for code, name in LANGS.items()], value="en", label=t("en", "language")),
                    "language",
                )
                ollama_md = gr.Markdown()
                ollama_btn = tr(gr.Button(t("en", "ollama_start"), size="sm", visible=False), "ollama_start", "value")
                with gr.Row():
                    ollama_stop_btn = tr(
                        gr.Button(t("en", "ollama_stop"), size="sm", visible=False), "ollama_stop", "value"
                    )
                    ollama_restart_btn = tr(
                        gr.Button(t("en", "ollama_restart"), size="sm", visible=False), "ollama_restart", "value"
                    )

        with gr.Tabs():
            # --- 1. Szinkron ---
            with gr.Tab(t("en", "tab_jobs")) as tab_jobs:
                with gr.Row():
                    upload = tr(gr.File(label=t("en", "upload"), file_types=["video"], type="filepath"), "upload")
                    with gr.Column():
                        job_dd = tr(gr.Dropdown(choices=job_names(), label=t("en", "job"), interactive=True), "job")
                        refresh_btn = tr(gr.Button(t("en", "refresh"), size="sm"), "refresh", "value")
                with gr.Row():
                    start_btn = tr(gr.Button(t("en", "start"), variant="primary"), "start", "value")
                    stop_btn = tr(gr.Button(t("en", "stop"), variant="stop"), "stop", "value")
                    step_dd = tr(gr.Dropdown(STEP_NAMES, label=t("en", "from_step"), value=None), "from_step")
                    run_from_btn = tr(gr.Button(t("en", "run_from")), "run_from", "value")
                with gr.Row():
                    with gr.Column(scale=1):
                        status_md = gr.Markdown()
                        steps_md = gr.Markdown()
                    with gr.Column(scale=2):
                        log_box = tr(gr.Textbox(label=t("en", "log"), lines=30, max_lines=30, autoscroll=True), "log")

            # --- 2. Beállítások ---
            with gr.Tab(t("en", "tab_settings")) as tab_settings:
                tr(gr.Markdown(t("en", "settings_note")), "settings_note", "value")
                with gr.Row():
                    llm = tr(gr.Dropdown(label=t("en", "llm"), allow_custom_value=True, scale=2), "llm")
                    shorten_llm = tr(
                        gr.Dropdown(label=t("en", "shorten_llm"), allow_custom_value=True, scale=2), "shorten_llm"
                    )
                    formality = tr_choices(
                        tr(gr.Radio(label=t("en", "formality"), choices=["tegezo", "magazo"]), "formality"),
                        "formality",
                        ["tegezo", "magazo"],
                    )
                with gr.Row():
                    pull_btn = tr(gr.Button(t("en", "pull"), size="sm"), "pull", "value")
                    pull_md = gr.Markdown()
                with gr.Row():
                    whisper = tr(gr.Dropdown(label=t("en", "whisper_model"), allow_custom_value=True), "whisper_model")
                    demucs = tr(gr.Dropdown(label=t("en", "demucs_model"), allow_custom_value=True), "demucs_model")
                with gr.Row():
                    tts = tr_choices(
                        tr(gr.Radio(label=t("en", "tts"), choices=["xtts", "edge"]), "tts"), "tts", ["xtts", "edge"]
                    )
                    license_cb = tr(gr.Checkbox(label=t("en", "xtts_license")), "xtts_license")
                    background = tr_choices(
                        tr(gr.Radio(label=t("en", "background"), choices=["separate", "duck", "none"]), "background"),
                        "background",
                        ["separate", "duck", "none"],
                    )
                with gr.Row():
                    threshold = tr(
                        gr.Slider(0.2, 0.8, step=0.05, label=t("en", "speaker_threshold")), "speaker_threshold"
                    )
                    min_spk = tr(gr.Number(label=t("en", "min_speaker_seconds"), minimum=0), "min_speaker_seconds")
                    max_speedup = tr(gr.Slider(1.0, 1.6, step=0.05, label=t("en", "max_speedup")), "max_speedup")
                with gr.Row():
                    pitch = tr_choices(
                        tr(gr.Radio(label=t("en", "pitch_correction"), choices=["off", "child", "all"]), "pitch_correction"),
                        "pitch",
                        ["off", "child", "all"],
                    )
                    max_pitch = tr(gr.Slider(0, 12, step=0.5, label=t("en", "max_pitch_shift")), "max_pitch_shift")
                    animals = tr(gr.Checkbox(label=t("en", "keep_animal_sounds")), "keep_animal_sounds")
                glossary = tr(
                    gr.Dataframe(headers=["EN", "HU"], column_count=(2, "fixed"), label=t("en", "glossary"), type="array"),
                    "glossary",
                )
                save_btn = tr(gr.Button(t("en", "save"), variant="primary"), "save", "value")

            # --- 3. Eredmény ---
            with gr.Tab(t("en", "tab_result")) as tab_result:
                with gr.Row():
                    video = tr(gr.Video(label=t("en", "video"), interactive=False), "video")
                    downloads = tr(gr.File(label=t("en", "downloads"), file_count="multiple", interactive=False), "downloads")
                report_md = gr.Markdown()

            # --- 4. Mondatok (szegmensszerkesztő) ---
            with gr.Tab(t("en", "tab_segments")) as tab_segments:
                tr(gr.Markdown(t("en", "seg_note")), "seg_note", "value")
                with gr.Row():
                    seg_filter = tr_choices(
                        tr(gr.Radio(choices=["all", "problems", "pending"], value="all", label=t("en", "seg_filter")),
                           "seg_filter"),
                        "seg_filter",
                        ["all", "problems", "pending"],
                    )
                    with gr.Column():
                        seg_pending_md = gr.Markdown()
                        seg_apply_btn = tr(gr.Button(t("en", "seg_apply"), variant="primary"), "seg_apply", "value")
                seg_df = gr.Dataframe(interactive=False, wrap=True, max_height=420,
                                      column_widths=["5%", "12%", "8%", "9%", "33%", "33%"])
                with gr.Row():
                    seg_id = tr(gr.Number(label=t("en", "seg_id"), precision=0, interactive=False, scale=1), "seg_id")
                    seg_speaker = tr(gr.Dropdown(label=t("en", "seg_speaker"), interactive=True, scale=1), "seg_speaker")
                    seg_en = tr(gr.Textbox(label=t("en", "seg_en"), interactive=False, scale=4), "seg_en")
                seg_hu = tr(gr.Textbox(label=t("en", "seg_hu"), lines=2), "seg_hu")
                with gr.Row():
                    seg_orig = tr(gr.Audio(label=t("en", "seg_orig_audio"), interactive=False), "seg_orig_audio")
                    seg_dub = tr(gr.Audio(label=t("en", "seg_dub_audio"), interactive=False), "seg_dub_audio")
                with gr.Row():
                    seg_save_btn = tr(gr.Button(t("en", "seg_save"), variant="primary"), "seg_save", "value")
                    seg_resynth_btn = tr(gr.Button(t("en", "seg_resynth")), "seg_resynth", "value")

            # --- 5. Beszélők ---
            with gr.Tab(t("en", "tab_speakers")) as tab_speakers:
                speakers_df = tr(gr.Dataframe(label=t("en", "speakers_table"), interactive=False), "speakers_table")
                with gr.Row():
                    speaker_dd = tr(gr.Dropdown(label=t("en", "speaker"), interactive=True), "speaker")
                    kind_radio = tr_choices(tr(gr.Radio(label=t("en", "kind"), choices=KINDS), "kind"), "kind", KINDS)
                with gr.Row():
                    ref_audio = tr(gr.Audio(label=t("en", "ref_audio"), interactive=False), "ref_audio")
                    cal_audio = tr(gr.Audio(label=t("en", "cal_audio"), interactive=False), "cal_audio")
                apply_btn = tr(gr.Button(t("en", "apply_speakers"), variant="primary"), "apply_speakers", "value")
                with gr.Accordion(t("en", "threshold_preview"), open=False) as preview_acc:
                    preview_btn = tr(gr.Button(t("en", "preview"), size="sm"), "preview", "value")
                    preview_df = gr.Dataframe(interactive=False)

        for tab, key in [(tab_jobs, "tab_jobs"), (tab_settings, "tab_settings"), (tab_result, "tab_result"), (tab_segments, "tab_segments"), (tab_speakers, "tab_speakers")]:
            tr(tab, key)
        tr(preview_acc, "threshold_preview")

        # --- állapot ---

        def refresh(name, lang):
            job = job_of(name)
            running = ws.running()
            up = ollama_ok(host)
            ollama = (
                t(lang, "ollama_ok") if up else t(lang, "ollama_down"),
                gr.update(visible=not up),
                gr.update(visible=up),
                gr.update(visible=up),
            )
            if job is None:
                return (t(lang, "no_job"), "", "", gr.update(interactive=False), gr.update(interactive=False), *ollama)
            pid = runlock.running_pid(job.workdir)
            m = job.manifest()
            status = t(lang, "status_running", pid=pid) if pid else t(lang, "status_idle")
            if m is not None and m.segments:
                status += "\n\n" + t(
                    lang,
                    "segments",
                    total=len(m.segments),
                    translated=sum(1 for s in m.segments if s.text_hu),
                    voiced=sum(1 for s in m.segments if is_synthesized(s, job.workdir)),
                    speakers=len(m.speakers),
                )
            steps = "\n".join(
                f"- {STATE_ICON[state]} `{name}` – {t(lang, 'state_' + state)}"
                for name, state in step_states(m, pid is not None)
            )
            can_start = running is None
            return (
                status,
                steps,
                tail(job.log_path),
                gr.update(interactive=can_start),
                gr.update(interactive=pid is not None),
                *ollama,
            )

        status_outputs = [status_md, steps_md, log_box, start_btn, stop_btn, ollama_md, ollama_btn, ollama_stop_btn,
                          ollama_restart_btn]

        def on_upload(path, lang):
            if not path:
                return gr.update()
            try:
                job = ws.create(Path(path))
            except ValueError as e:
                raise gr.Error(str(e))
            gr.Info(t(lang, "created", name=job.name))
            return gr.update(choices=job_names(), value=job.name)

        def run(name, lang, from_step=None):
            job = job_of(name)
            if job is None:
                raise gr.Error(t(lang, "no_job"))
            try:
                ws.start(job, from_step)  # zárolva ellenőrzi, hogy nem fut-e már valami
            except RuntimeError:
                busy = ws.running()
                raise gr.Error(t(lang, "busy", name=busy.name if busy else "?"))
            gr.Info(t(lang, "started", name=job.name))

        def on_stop(name, lang):
            job = job_of(name)
            if job is not None and ws.stop(job):
                gr.Info(t(lang, "stopping"))
            else:
                gr.Warning(t(lang, "not_running"))

        def on_start_ollama(lang):
            start_ollama(ws.root)

        def ollama_guard(lang):
            """Futás közben ne állítsuk le: a fordítás és a rövidítés az Ollamát használja."""
            busy = ws.running()
            if busy is not None:
                raise gr.Error(t(lang, "ollama_busy", name=busy.name))

        def on_stop_ollama(lang):
            ollama_guard(lang)
            if stop_ollama(host):
                gr.Info(t(lang, "ollama_stopped"))
            else:
                gr.Warning(t(lang, "ollama_stop_failed"))

        def on_restart_ollama(lang):
            ollama_guard(lang)
            if restart_ollama(ws.root, host):
                gr.Info(t(lang, "ollama_restarted"))
            else:
                gr.Warning(t(lang, "ollama_restart_failed"))

        # --- beállítások ---

        def load_settings(name, lang):
            job = job_of(name)
            cfg = ws.job_config(job) if job else ws.job_config_default()
            installed = installed_models(host)
            choices = llm_choices(lang, cfg.llm, installed)
            return (
                gr.update(choices=choices, value=cfg.llm),
                gr.update(choices=[(t(lang, "same_model"), "")] + llm_choices(lang, cfg.shorten_llm, installed),
                          value=cfg.shorten_llm or ""),
                cfg.formality,
                cfg.tts,
                cfg.xtts_accept_license,
                cfg.background,
                cfg.speaker_threshold,
                cfg.min_speaker_seconds,
                cfg.max_speedup,
                cfg.pitch_correction,
                cfg.max_pitch_shift,
                cfg.keep_animal_sounds,
                [[en, hu] for en, hu in cfg.glossary.items()] or [["", ""]],
                gr.update(choices=model_choices(WHISPERS, lang, cfg.whisper_model), value=cfg.whisper_model),
                gr.update(choices=model_choices(DEMUCS, lang, cfg.demucs_model), value=cfg.demucs_model),
            )

        settings_fields = [llm, shorten_llm, formality, tts, license_cb, background, threshold, min_spk, max_speedup,
                           pitch, max_pitch, animals, glossary, whisper, demucs]

        def save_settings(name, lang, llm_v, shorten_v, formality_v, tts_v, license_v, background_v, threshold_v,
                          min_spk_v, max_speedup_v, pitch_v, max_pitch_v, animals_v, glossary_v, whisper_v, demucs_v):
            job = job_of(name)
            if job is None:
                raise gr.Error(t(lang, "no_job"))
            rows = glossary_v.values.tolist() if hasattr(glossary_v, "values") else (glossary_v or [])
            values = {
                "llm": llm_v,
                "shorten_llm": shorten_v or None,
                "formality": formality_v,
                "tts": tts_v,
                "xtts_accept_license": bool(license_v),
                "background": background_v,
                "speaker_threshold": float(threshold_v),
                "min_speaker_seconds": float(min_spk_v),
                "max_speedup": float(max_speedup_v),
                "pitch_correction": pitch_v,
                "max_pitch_shift": float(max_pitch_v),
                "keep_animal_sounds": bool(animals_v),
                "glossary": {str(r[0]).strip(): str(r[1]).strip() for r in rows if r and str(r[0]).strip() and str(r[1]).strip()},
                "whisper_model": whisper_v,
                "demucs_model": demucs_v,
            }
            try:
                ws.save_config(job, values)
            except Exception as e:  # pydantic validációs hiba
                raise gr.Error(t(lang, "invalid", error=e))
            gr.Info(t(lang, "saved"))

        def on_pull(llm_v, lang):
            """Letölti a kiválasztott fordítómodellt az Ollamába, közben mutatja a haladást."""
            if not llm_v or not llm_v.startswith("ollama:"):
                raise gr.Error(t(lang, "pull_pick"))
            tag = ollama_tag(llm_v)
            installed = installed_models(host)
            if installed is None:
                raise gr.Error(t(lang, "ollama_down"))
            if is_installed(tag, installed):
                yield t(lang, "pull_have", model=tag)
                return
            try:
                for status, frac in pull(host, tag):
                    yield t(lang, "pull_progress", model=tag, status=status) + (f" – {frac:.0%}" if frac is not None else "")
            except Exception as e:  # hálózati vagy Ollama-hiba
                raise gr.Error(t(lang, "pull_failed", error=e))
            yield t(lang, "pull_done", model=tag)

        # --- eredmény ---

        def load_result(name, lang):
            job = job_of(name)
            if job is None or not job.output.exists():
                return None, None, t(lang, "no_result")
            files = [p for p in [job.output, job.output.with_suffix(".hu.srt"), job.workdir / "report.md"] if p.exists()]
            report = (job.workdir / "report.md").read_text(encoding="utf-8") if (job.workdir / "report.md").exists() else ""
            return str(job.output), [str(p) for p in files], report

        # --- beszélők ---

        def speakers_table(job: Job | None, lang):
            m = job.manifest() if job else None
            if m is None or not m.speakers:
                return None
            cols = ["col_speaker", "col_kind", "col_source", "col_f0", "col_clone_f0", "col_shift", "col_seconds"]
            rows = [
                [name, t(lang, f"kind_{sp.kind}") if sp.kind else "?", sp.kind_source or "–",
                 round(sp.f0) if sp.f0 else None, round(sp.clone_f0) if sp.clone_f0 else None,
                 sp.pitch_shift, round(sp.seconds)]
                for name, sp in m.speakers.items()
            ]
            return gr.update(value=rows, headers=[t(lang, c) for c in cols])

        def load_speakers(name, lang):
            job = job_of(name)
            table = speakers_table(job, lang)
            if table is None:
                if job is not None:
                    gr.Info(t(lang, "no_speakers"))
                return None, gr.update(choices=[], value=None)
            names = list(job.manifest().speakers)
            return table, gr.update(choices=names, value=names[0])

        def on_speaker(name, speaker):
            job = job_of(name)
            if job is None or not speaker:
                return None, None, "auto"
            ref = job.workdir / "speakers" / f"{speaker}.wav"
            cal = job.workdir / "speakers" / f"{speaker}_cal0.wav"
            kind = ws.job_config(job).speaker_kinds.get(speaker, "auto")
            return (str(ref) if ref.exists() else None), (str(cal) if cal.exists() else None), kind

        def on_kind(name, speaker, kind):
            job = job_of(name)
            if job is None or not speaker:
                return
            kinds = dict(ws.job_config(job).speaker_kinds)
            if kind == "auto":
                kinds.pop(speaker, None)
            else:
                kinds[speaker] = kind
            ws.save_config(job, {"speaker_kinds": kinds})

        def on_apply(name, lang):
            run(name, lang, "voices")
            gr.Info(t(lang, "applied"))

        def on_preview(name, lang):
            from ..steps.voices import cluster_speakers

            job = job_of(name)
            cache = job.workdir / "speakers" / "embeddings.npz" if job else None
            if cache is None or not cache.exists():
                raise gr.Error(t(lang, "no_embeddings"))
            data = np.load(cache)
            durations = list(data["spans"][:, 1] - data["spans"][:, 0])
            min_s = ws.job_config(job).min_speaker_seconds
            rows = []
            for th in THRESHOLDS:
                labels = cluster_speakers(data["embs"], durations, th, min_s)
                rows.append([th, max(labels) + 1 if labels else 0])
            return gr.update(value=rows, headers=[t(lang, "threshold"), t(lang, "count")])

        # --- mondatok ---

        def seg_headers(lang):
            return [t(lang, f"seg_col_{c}") for c in ("id", "time", "speaker", "state", "en", "hu")]

        def load_segments(name, lang, only):
            job = job_of(name)
            m = job.manifest() if job else None
            if m is None or not any(s.text_hu for s in m.segments):
                return (gr.update(value=[], headers=seg_headers(lang)), t(lang, "seg_not_ready"),
                        gr.update(interactive=False), gr.update(choices=[]))
            n = len(seged.pending(m, job.workdir))
            return (
                gr.update(value=seged.rows(m, job.workdir, only or "all"), headers=seg_headers(lang)),
                t(lang, "seg_pending", n=n),
                gr.update(interactive=n > 0 and ws.running() is None),
                gr.update(choices=list(m.speakers) or ["SPK0"]),
            )

        def load_segment(name, seg_no):
            """A kiválasztott mondat a szerkesztőbe: szövegek, beszélő, eredeti és magyar hang."""
            job = job_of(name)
            m = job.manifest() if job else None
            if m is None or seg_no is None:
                return None, "", "", None, None, None
            seg = seged.find(m, int(seg_no))
            return (seg.id, seg.text_en, seg.text_hu or "", seg.speaker,
                    seged.original_clip(job.workdir, seg), seged.dub_clip(job.workdir, seg))

        def seg_tick(name, lang, only, before):
            """Ha az előző ellenőrzés óta befejeződött egy futás, a mondatok táblázata frissül."""
            now = ws.running() is not None
            if before and not now:
                return *load_segments(name, lang, only), now
            return gr.update(), gr.update(), gr.update(), gr.update(), now

        def on_seg_select(name, evt: gr.SelectData):
            row = evt.row_value or []
            return load_segment(name, row[0] if row else None)

        def seg_change(name, lang, change):
            job = job_of(name)
            if job is None:
                raise gr.Error(t(lang, "no_job"))
            try:
                ws.edit_manifest(job, change)
            except RuntimeError:
                busy = ws.running()
                raise gr.Error(t(lang, "seg_busy", name=busy.name if busy else "?"))
            except seged.EditError as e:
                raise gr.Error(str(e))

        def on_seg_save(name, lang, seg_no, text, speaker):
            if seg_no is None:
                raise gr.Error(t(lang, "seg_pick"))
            changed = []
            seg_change(name, lang, lambda m: changed.append(seged.edit(m, int(seg_no), text or "", speaker)))
            gr.Info(t(lang, "seg_saved", id=int(seg_no)) if changed and changed[0] else t(lang, "seg_unchanged"))

        def on_seg_resynth(name, lang, seg_no):
            if seg_no is None:
                raise gr.Error(t(lang, "seg_pick"))
            seg_change(name, lang, lambda m: seged.request_resynth(m, int(seg_no)))
            gr.Info(t(lang, "seg_marked", id=int(seg_no)))

        def on_seg_apply(name, lang):
            job = job_of(name)
            m = job.manifest() if job else None
            n = len(seged.pending(m, job.workdir)) if m else 0
            if n == 0:
                raise gr.Error(t(lang, "seg_nothing"))
            try:
                ws.rerun_pending(job)
            except RuntimeError:
                busy = ws.running()
                raise gr.Error(t(lang, "seg_busy", name=busy.name if busy else "?"))
            except seged.EditError as e:
                raise gr.Error(str(e))
            gr.Info(t(lang, "seg_applied", n=n))

        # --- nyelv ---

        def relabel(lang_v):
            return [lang_v] + [make(lang_v) for _, make in i18n_targets]

        def init(request: gr.Request):
            code = pick_lang(request.headers.get("accept-language") if request else None)
            names = job_names()
            return code, gr.update(choices=names, value=names[0] if names else None)

        # --- események ---

        demo.load(init, None, [lang_radio, job_dd])
        lang_radio.change(relabel, lang_radio, [lang] + [c for c, _ in i18n_targets]).then(
            refresh, [job_dd, lang], status_outputs
        )
        gr.Timer(2.0).tick(refresh, [job_dd, lang], status_outputs)
        upload.upload(on_upload, [upload, lang], job_dd)
        refresh_btn.click(lambda: gr.update(choices=job_names()), None, job_dd)
        job_dd.change(refresh, [job_dd, lang], status_outputs).then(load_settings, [job_dd, lang], settings_fields).then(
            load_result, [job_dd, lang], [video, downloads, report_md]
        ).then(load_speakers, [job_dd, lang], [speakers_df, speaker_dd])
        start_btn.click(run, [job_dd, lang]).then(refresh, [job_dd, lang], status_outputs)
        stop_btn.click(on_stop, [job_dd, lang])
        run_from_btn.click(lambda n, lg, s: run(n, lg, s or None), [job_dd, lang, step_dd]).then(
            refresh, [job_dd, lang], status_outputs
        )
        ollama_btn.click(on_start_ollama, lang)
        ollama_stop_btn.click(on_stop_ollama, lang).then(refresh, [job_dd, lang], status_outputs)
        ollama_restart_btn.click(on_restart_ollama, lang).then(refresh, [job_dd, lang], status_outputs)
        save_btn.click(save_settings, [job_dd, lang] + settings_fields)
        tab_settings.select(load_settings, [job_dd, lang], settings_fields)
        pull_btn.click(on_pull, [llm, lang], pull_md).then(load_settings, [job_dd, lang], settings_fields)
        tab_result.select(load_result, [job_dd, lang], [video, downloads, report_md])
        tab_speakers.select(load_speakers, [job_dd, lang], [speakers_df, speaker_dd])
        speaker_dd.change(on_speaker, [job_dd, speaker_dd], [ref_audio, cal_audio, kind_radio])
        kind_radio.input(on_kind, [job_dd, speaker_dd, kind_radio])
        apply_btn.click(on_apply, [job_dd, lang]).then(refresh, [job_dd, lang], status_outputs)
        preview_btn.click(on_preview, [job_dd, lang], preview_df)
        seg_outputs = [seg_df, seg_pending_md, seg_apply_btn, seg_speaker]
        seg_editor = [seg_id, seg_en, seg_hu, seg_speaker, seg_orig, seg_dub]
        tab_segments.select(load_segments, [job_dd, lang, seg_filter], seg_outputs)
        gr.Timer(3.0).tick(seg_tick, [job_dd, lang, seg_filter, was_running], seg_outputs + [was_running])
        seg_filter.change(load_segments, [job_dd, lang, seg_filter], seg_outputs)
        seg_df.select(on_seg_select, job_dd, seg_editor)
        seg_save_btn.click(on_seg_save, [job_dd, lang, seg_id, seg_hu, seg_speaker]).then(
            load_segments, [job_dd, lang, seg_filter], seg_outputs
        )
        seg_resynth_btn.click(on_seg_resynth, [job_dd, lang, seg_id]).then(
            load_segments, [job_dd, lang, seg_filter], seg_outputs
        )
        seg_apply_btn.click(on_seg_apply, [job_dd, lang]).then(
            load_segments, [job_dd, lang, seg_filter], seg_outputs
        ).then(refresh, [job_dd, lang], status_outputs)

    return demo


def launch(root: Path, port: int = 7860, open_browser: bool = False) -> None:
    ws = Workspace(root)
    demo = build(ws)
    paths = {str(p) for d in (ws.work, ws.outputs) for p in (d, d.resolve())}
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, allowed_paths=sorted(paths))
