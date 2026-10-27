# Mp4Translator – angol → magyar AI szinkron (tervdokumentum)

## 1. Cél és korlátok

| Szempont | Követelmény |
|---|---|
| Bemenet | Angol nyelvű videó (mp4/mov/mkv), **max. 10 perc** |
| Kimenet | Ugyanaz a videó magyar szinkronhanggal (+ opcionálisan magyar felirat, eredeti hangsáv megtartása) |
| Minőség | Természetes magyar szöveg, az eredeti beszélő hangszíne (hangklónozás), időzítés a jelenetekhez, háttérzene/zajok megtartása |
| Költség | Csak ingyenes AI: helyi modellek, opcionálisan ingyenes felhős szint |
| Hardver | MacBook Air M5, 16 GB egységes memória (Apple Silicon, Metal/MPS, MLX) |

**Alapelv:** minden lépés lokálisan is futtatható. A modelleket **egymás után** töltjük be, és minden lépés után felszabadítjuk őket, így a csúcs memóriaigény egy-egy modellé (kb. 8–9 GB), nem az összegé.

## 2. Feldolgozási folyamat

```
 input.mp4
    │
    ▼
[1] Hang kinyerése (ffmpeg) ──────────────► audio.wav (44.1 kHz, sztereó)
    │
    ▼
[2] Hangforrás-szétválasztás (Demucs) ────► vocals.wav + background.wav
    │
    ▼
[3] Beszédfelismerés + időbélyegek (Whisper)► szegmensek: {start, end, text_en}
    │        (+ opcionális beszélőazonosítás)    + speaker
    ▼
[4] Fordítás EN→HU időkeret-tudatosan (LLM) ► szegmensek: text_hu
    │
    ▼
[5] Hangminta beszélőnként (vocals.wav-ból) ► speaker_X.wav (6–15 mp tiszta beszéd)
    │
    ▼
[6] Magyar TTS hangklónozással ───────────► seg_001.wav … seg_N.wav
    │
    ▼
[7] Időzítés-illesztés (rövidítés / gyorsítás / csend) ► dub_vocals.wav
    │
    ▼
[8] Keverés: dub_vocals + background, hangerő-normalizálás ► dub_mix.wav
    │
    ▼
[9] Muxolás (ffmpeg, videó újrakódolás nélkül) ► output_hu.mp4 (+ .srt)
```

A központi adatszerkezet egy **`manifest.json`**, amelyet minden lépés olvas és bővít. Minden lépés eredménye a `work/<videó-név>/` mappába kerül, így a futás **bármelyik lépéstől folytatható** (pl. csak a fordítást javítod kézzel, és onnan futtatod újra).

```json
{
  "source": "input.mp4",
  "duration": 587.3,
  "speakers": {"SPK0": {"ref_audio": "speakers/SPK0.wav"}},
  "segments": [
    {
      "id": 12, "start": 41.20, "end": 44.85, "speaker": "SPK0",
      "text_en": "So let's take a look at how this actually works.",
      "text_hu": "Nézzük meg, hogyan működik ez valójában.",
      "tts_file": "tts/seg_012.wav", "tts_duration": 3.41,
      "speed_factor": 1.0, "status": "ok"
    }
  ]
}
```

## 3. Modellválasztás lépésenként

### [2] Ének/beszéd és háttér szétválasztása
- **Demucs `htdemucs`** (Meta, MIT licenc) – két sávra bontás (`--two-stems vocals`). PyTorch MPS-en vagy CPU-n fut, ~1 GB memória.
- Ennélkül a szinkron alatt nem maradna meg a háttérzene és a zajok, vagy az angol beszéd áthallatszana.

### [3] Beszédfelismerés
- **Elsődleges: `mlx-whisper` + `whisper-large-v3-turbo`** – Apple Silicon-ra optimalizált (MLX), ~1,6 GB, kiváló angol pontosság, szó szintű időbélyegek.
- Alternatíva: `faster-whisper` (CPU, int8) vagy WhisperX.
- A felismerést a **leválasztott `vocals.wav`-on** futtatjuk (tisztább jel → kevesebb hallucináció).
- Mondathatárok szerinti szegmentálás: a Whisper szegmenseit mondatvégeknél vonjuk össze/vágjuk szét, célszerűen 2–12 mp hosszú egységekre.

### [3b] Beszélőazonosítás (opcionális)
- **`pyannote/speaker-diarization-3.1`** – ingyenes, de Hugging Face fiók és a modell feltételeinek elfogadása kell hozzá.
- Ha ki van kapcsolva: egyetlen beszélőt feltételezünk (oktatóvideóknál, előadásoknál ez tipikus).

### [4] Fordítás – a minőség kulcsa
Egyszerű gépi fordító (pl. MarianMT `opus-mt-en-hu`, NLLB) **nem elég**, mert nem tud a hosszra figyelni, és nem látja a kontextust. LLM-et használunk, két ingyenes lehetőséggel:

| Opció | Modell | Memória | Megjegyzés |
|---|---|---|---|
| **A) Helyi (alapértelmezett)** | `Gemma 3 12B` 4 bites (Ollama vagy `mlx-lm`) | ~8 GB | Jó többnyelvű tudás, magyarul is gördülékeny. Gyengébb gépen: `Gemma 3 4B` / `Qwen3 8B` |
| **B) Ingyenes felhő** | Google Gemini API (Flash modell, ingyenes kvóta) | 0 | Általában a legjobb magyar minőség; internet és API-kulcs kell, a kvóta és a feltételek változhatnak |

A fordítóprompt fő szabályai:
1. **Kontextus:** a teljes átirat összefoglalója + az előző/következő 2–3 szegmens is a promptba kerül, de csak az aktuális köteget (kb. 20 szegmens) fordíttatjuk, JSON kimenettel (`{id: text_hu}`).
2. **Időkeret (izokrónia):** minden szegmenshez megadjuk a rendelkezésre álló időt és egy cél szótagszámot (magyar beszéd ≈ 5,5–6,5 szótag/mp). A magyar szöveg tipikusan 10–20%-kal hosszabb az angolnál, ezért ez kritikus.
3. **Stílus:** élőbeszédszerű, nem szó szerinti fordítás; tegezés/magázás és szakszavak a konfigurációban rögzíthetők; egy **glosszárium** (`glossary.yaml`) biztosítja a következetes terminológiát.
4. **Számok, rövidítések kiírása** betűvel ("2025" → "kétezer-huszonöt"), mert a TTS így ejti jól.

### [5–6] Magyar beszédszintézis (TTS)
A magyar a kevésbé támogatott nyelvek közé tartozik, ezért a TTS réteg **cserélhető interfész** mögött van:

| Motor | Hangklónozás | Minőség (magyar) | Helyi? | Megjegyzés |
|---|---|---|---|---|
| **XTTS-v2** (`coqui-tts`, az idiap által karbantartott fork) | ✅ 6–15 mp mintából | Jó | ✅ ~2 GB | **Alapértelmezett.** A magyar hivatalosan támogatott. Licenc: Coqui Public Model License – **csak nem kereskedelmi célra** |
| **edge-tts** (Microsoft Neural: `hu-HU-NoemiNeural`, `hu-HU-TamasNeural`) | ❌ | Nagyon jó, stabil | ❌ online | Ingyenes, de nem hivatalos API; gyors tartalék, ha a klónozás nem kell |
| **Piper** (`hu_HU` hangok) | ❌ | Közepes | ✅ nagyon könnyű | Offline vészmegoldás |

> A TTS-modellek gyorsan fejlődnek: érdemes implementáláskor ellenőrizni, van-e jobb, magyart is tudó, hangklónozós nyílt modell (pl. F5-TTS-alapú magyar finomhangolás). Az interfész miatt ez egyetlen új osztály.

Beszélőnkénti hangminta kiválasztása: a `vocals.wav`-ból automatikusan összegyűjtjük az adott beszélő leghosszabb, tiszta (nem átfedő, megfelelő hangerejű) szegmenseit 10–15 mp összhosszig.

XTTS-specifikus minőségi trükkök:
- Szegmensenként szintetizálunk (rövid szöveg → stabilabb kimenet), `temperature≈0.65`.
- Hallucináció-ellenőrzés: ha a kimenet hossza > 1,6× a várható értéknek, vagy a visszafelé futtatott Whisper (magyar) átirat túl eltér, újragenerálás (max. 3×).
- Csend levágása a TTS-klipek elejéről/végéről.

### [7] Időzítés-illesztés (algoritmus)
Minden szegmensre, ahol `slot = end - start` (+ a következő szegmensig tartó szünet egy része kölcsönvehető):

```
ha tts_duration <= slot:            → elhelyezés a start időpontban, a maradék csend
ha tts_duration <= slot * 1.15:     → gyorsítás (time-stretch, hangmagasság-megtartó)
ha tts_duration <= slot + gap*0.8:  → belecsúszás a következő szünetbe
egyébként:                           → LLM: "rövidítsd ~N szótagra" → új TTS (max. 2 kör)
végső tartalék:                      → gyorsítás max. 1.3×-ig, figyelmeztetés a riportban
```
- Time-stretch: `pyrubberband` (Rubber Band, jobb minőség) vagy ffmpeg `atempo`.
- Az eredményről **riport** készül (mely szegmensek lettek gyorsítva/rövidítve), hogy kézzel javíthatók legyenek.

### [8] Keverés
- `dub_vocals.wav` + `background.wav` összekeverése (`pydub`/`numpy`), majd ffmpeg `loudnorm` (EBU R128, −16 LUFS).
- Mivel az eredeti beszéd le van választva, „ducking” nem kell.

### [9] Kimenet
- `ffmpeg -c:v copy` – a videó nem kódolódik újra (gyors, veszteségmentes).
- Hangsávok: 1) magyar szinkron (alapértelmezett), 2) eredeti angol (opcionális).
- Opcionálisan magyar `.srt` felirat a `text_hu` alapján (beégetve vagy külön sávként).

### Nem része az első verziónak: szájszinkron
Ajak-szinkronizálás (pl. LatentSync, Wav2Lip) 10 perces videónál 16 GB-os gépen nagyon lassú, és minősége csak közeli arcképeknél számít. Később opcionális modulként hozzáadható.

## 4. Memória és futásidő (becslés, 10 perces videó, M5 / 16 GB)

| Lépés | Csúcs memória | Becsült idő |
|---|---|---|
| Demucs | ~1–2 GB | 2–5 perc |
| Whisper large-v3-turbo (MLX) | ~2 GB | 1–3 perc |
| Fordítás Gemma 3 12B Q4 (helyi) | ~8–9 GB | 5–12 perc |
| Fordítás Gemini (felhő) | – | < 1 perc |
| XTTS-v2 (MPS/CPU) | ~2–3 GB | 10–25 perc |
| Illesztés, keverés, mux | < 1 GB | < 1 perc |
| **Összesen** | **≤ 9 GB egyszerre** | **kb. 20–45 perc** |

A számok tájékoztató jellegűek, implementáció után mérni kell. A MacBook Air passzív hűtése miatt hosszabb futásnál a teljesítmény csökkenhet.

## 5. Technológiai stack

- **Python 3.11 vagy 3.12** (a gépen lévő 3.15-öt az ML-csomagok – PyTorch, coqui-tts – még nem feltétlenül támogatják). Környezetkezelés: **`uv`**.
- Rendszerszintű: `brew install ffmpeg rubberband` (ffmpeg jelenleg nincs telepítve a gépen).
- Python csomagok: `mlx-whisper`, `demucs`, `torch`, `coqui-tts`, `edge-tts`, `ollama` (vagy `mlx-lm`), `google-genai` (opcionális), `pyannote.audio` (opcionális), `pydub`, `soundfile`, `pyrubberband`, `pydantic`, `typer`, `rich`, `gradio` (opcionális UI).

## 6. Projektstruktúra

```
Mp4Translator/
├── pyproject.toml
├── config.yaml                 # modellek, stílus, TTS motor, határértékek
├── glossary.yaml               # terminológia (EN → HU)
├── src/mp4translator/
│   ├── cli.py                  # typer: `mp4translator dub input.mp4 [--from-step translate]`
│   ├── pipeline.py             # lépések sorrendje, folytathatóság, memória-felszabadítás
│   ├── manifest.py             # pydantic modellek (Segment, Speaker, Manifest)
│   ├── config.py
│   ├── steps/
│   │   ├── extract.py          # [1] ffmpeg
│   │   ├── separate.py         # [2] Demucs
│   │   ├── transcribe.py       # [3] mlx-whisper + szegmentálás
│   │   ├── diarize.py          # [3b] pyannote (opcionális)
│   │   ├── translate.py        # [4] promptok, kötegelés, JSON-validálás
│   │   ├── voices.py           # [5] referencia-hangminták
│   │   ├── synthesize.py       # [6] TTS + minőségellenőrzés
│   │   ├── align.py            # [7] időzítés-illesztés
│   │   ├── mix.py              # [8]
│   │   └── mux.py              # [9] + SRT
│   ├── llm/                    # LLMBackend: OllamaBackend, MLXBackend, GeminiBackend
│   └── tts/                    # TTSEngine: XTTSEngine, EdgeTTSEngine, PiperEngine
├── ui/app.py                   # Gradio: feltöltés, szegmensszerkesztő, előnézet
└── tests/
```

Kulcs-interfészek:

```python
class LLMBackend(Protocol):
    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...

class TTSEngine(Protocol):
    sample_rate: int
    def prepare_speaker(self, speaker_id: str, ref_audio: Path) -> None: ...
    def synthesize(self, text: str, speaker_id: str, out: Path) -> float: ...  # visszaadja a hosszt
```

## 6b. Felhasználói felület

1. **CLI** (elsődleges): `mp4translator dub video.mp4 --tts xtts --llm ollama:gemma3:12b`
2. **Gradio webes felület** (opcionális): videó feltöltése → automatikus futás → **szegmenstábla**, ahol az angol/magyar szöveg szerkeszthető, szegmensenként újragenerálható és meghallgatható → végső export. A kézi utómunka itt a legnagyobb minőségi nyereség.

## 7. Minőségbiztosítás

- Validálás: max. 10 perc (`ffprobe`), támogatott formátum, van-e hangsáv.
- Fordítás: JSON-séma ellenőrzés, hiányzó/üres szegmensek újrakérése, glosszárium-egyezés ellenőrzése.
- TTS: hosszarány- és visszafelé-ASR ellenőrzés, újragenerálás.
- Végső riport (`report.html`/`.md`): gyorsított, rövidített, hibás szegmensek listája időbélyeggel.
- Tesztek: egységtesztek az illesztési algoritmusra és a szegmentálásra; 30 mp-es mintavideós end-to-end teszt.

## 8. Fejlesztési ütemterv

1. **MVP:** ffmpeg + mlx-whisper + Ollama fordítás + edge-tts + egyszerű illesztés + mux → működő, de nem klónozott hang.
2. Demucs háttér-megtartás + keverés.
3. XTTS-v2 hangklónozás + minőségellenőrzés.
4. Izokrón fordítás és rövidítési kör, riport.
5. Beszélőazonosítás (többszereplős videók).
6. Gradio szerkesztő felület.
7. Hangtípus felismerése (férfi / női / gyerek / állat), és a szinkron ehhez igazítása (részletek: STATUS.md).
8. (Később) szájszinkron, Gemini-háttér, batch feldolgozás.

## 9. Jogi megjegyzés

Hangklónozást csak olyan hanggal szabad végezni, amelyhez megvan a beszélő hozzájárulása, és a videó szinkronizálásához is jogosultság kell. Az XTTS-v2 licence nem engedi a kereskedelmi felhasználást. Ha ez kell, edge-tts-t vagy kereskedelmi licencű modellt kell használni.
