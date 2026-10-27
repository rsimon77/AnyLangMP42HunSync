# Állapot – 2026-10-27

## Kész
- MVP (Whisper + Gemma 3 12B + edge-tts), mintavideón tesztelve.
- 2. fázis: Demucs háttér-szétválasztás, beszélők szétválogatása (XTTS-beszélővektorok csoportosítása), XTTS-v2 hangklónozás. A mintavideón tesztelve: [samples/](samples/), illetve a scratchpadben a `sample_xtts_hu.mp4`.
- 67 egységteszt, mind átmegy.
- XTTS licenc elfogadva (`config.yaml`: `xtts_accept_license: true`).

## Barbie-videó – kész (2026-10-27)
Kimenet: `output/barbie_hu.mp4`: magyar hangsáv (alapértelmezett) + eredeti angol sáv; felirat: `output/barbie_hu.hu.srt`. Riport: `work/barbie/report.md`.

- A `--from-step synthesize` most minden szegmenst újragenerál (korábban a kész szegmenseket átugrotta); folytatáskor továbbra is csak a hiányzókat.
- Teljes futás az elejétől, az új (kétmenetes) kóddal: **20 perc 18 mp**, a swap nem nőtt. Részletek: OPTIMIZATION.md, „Eredmény” szakasz.
- A futásban 166 szegmens és 5 beszélő volt. A régi futásban 162 szegmens és 6 beszélő: az átírás kissé másképp tagolt, és ettől változott a csoportosítás is.
- Tesztek: 94, mind átmegy (új: `tests/test_synthesize.py`).
- Nyitott kérdések:
  - A mért tempó minden beszélőnél 4,0–4,6 szótag/mp. Az SPK2 a 4,0-s alsó határra lett szorítva, ezt ellenőrizni kell.
  - ~~A hossz-ellenőrzés a beszélő mért tempóját használja a 6,0 helyett~~ – javítva (93 teszt átmegy). Korábban emiatt kellett 91 újrapróbálkozás; újrafuttatva a synthesize 610 → 395 mp, az újrapróbálások 91 → 44 (OPTIMIZATION.md).
  - Meghallgatva (2026-10-27): a felhasználó szerint a magyar szinkron „egészen jó és érthető”.

Újrafuttatás:
```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &
uv run mp4translator dub "/Users/roberto/Downloads/Barbie Family Story - Chelsea Gets Lost at The Doll Supermarket.mp4" \
  -o output/barbie_hu.mp4 --workdir work/barbie --from-step extract
```

## Későbbi teendők

### Hangtípus felismerése: női / férfi / gyerek / állat (felvéve: 2026-10-27)
Cél: minden beszélő a jellegének megfelelő magyar hangot kapja, az állathangok pedig ne vesszenek el.

- **Felismerés beszélőnként** (a voices lépésben, a hangmintából):
  - nem és korcsoport osztályozása (férfi / női / gyerek), pl. az alapfrekvencia (F0) és egy kis, ingyenes kor- és nem-osztályozó modell alapján;
  - címke a manifestben: `Speaker.kind`, kézzel felülírható.
- **Állathangok** (ugatás, nyávogás stb.): ezek nem beszédek, nem kell fordítani, de meg kell őrizni őket.
  - Kockázat: a Demucs a „vocals” sávba teheti őket, a Whisper viszont nem írja át → most kimaradhatnak a kész hangsávból.
  - Megoldás: hangesemény-felismerés (pl. YAMNet/PANNs vagy AST, AudioSet-címkékkel). Az állathangnak jelölt részeket a beszédsávból vissza kell keverni a háttérbe, változatlanul.
  - Ha egy állatszereplő beszél (mesefigura), az beszédként fut, a saját klónozott hangjával.
- **Szinkron a típus szerint:**
  - XTTS: a klónozás eleve átviszi a hangszínt, de a referencia-minta legyen tiszta. Gyerekhangnál ellenőrizni kell, hogy a klón nem lesz-e felnőttes. Ha igen, tartalék megoldás: magasabb hangfekvésű hangminta vagy enyhe hangmagasság-eltolás.
  - edge-tts: a típus alapján választ hangot (férfi: Tamás, női/gyerek: Noémi + magasabb hangmagasság).
  - A fordítási prompt is kapja meg a beszélő típusát (pl. gyerekhez egyszerűbb, gyerekesebb fogalmazás).
- **Ellenőrzés:** a riportban beszélőnként a felismert típus, a `speakers` parancs kiírja a típust is.
