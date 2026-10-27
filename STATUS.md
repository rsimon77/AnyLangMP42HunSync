# Állapot – 2026-10-27

## Kész
- MVP (Whisper + Gemma 3 12B + edge-tts), mintavideón tesztelve.
- 2. fázis: Demucs háttér-szétválasztás, beszélők szétválogatása (XTTS-beszélővektorok csoportosítása), XTTS-v2 hangklónozás. A mintavideón tesztelve: [samples/](samples/), illetve a scratchpadben a `sample_xtts_hu.mp4`.
- 67 egységteszt, mind átmegy.
- XTTS licenc elfogadva (`config.yaml`: `xtts_accept_license: true`).

## Webes felület (2026-10-27)
- `uv run mp4translator ui`: Gradio, angol/magyar, a `127.0.0.1:7860`-on. A kód a `src/mp4translator/webui/` mappában van:
  - `jobs.py`: munkák kezelése, a futás külön folyamatban a CLI-n keresztül;
  - `i18n.py`: a felület szövegei;
  - `app.py`: maga a felület.
- A böngészőben kipróbálva a mintavideón:
  - indítás, élő napló, leállítás és folytatás;
  - Eredmény fül;
  - Beszélők fül: a kézi hangtípus és az újrafuttatás a voices lépéstől működik;
  - küszöb-előnézet és nyelvváltás.
- A feltöltés gombot a böngészőben nem próbáltam ki: a tesztben a munkát Pythonból hoztam létre, a `Workspace.create` egységteszttel ellenőrzött.
- Megfigyelés: a rövidítési körben a modell egyszer „AI”-t írt a „mesterséges intelligencia” helyett.

## Barbie-videó – kész (2026-10-27)
Kimenet: `output/barbie_hu.mp4`: magyar hangsáv (alapértelmezett) + eredeti angol sáv; felirat: `output/barbie_hu.hu.srt`. Riport: `work/barbie/report.md`.

- A `--from-step synthesize` most minden szegmenst újragenerál (korábban a kész szegmenseket átugrotta); folytatáskor továbbra is csak a hiányzókat.
- Teljes futás az elejétől, az új (kétmenetes) kóddal: **20 perc 18 mp**, a swap nem nőtt. Részletek: OPTIMIZATION.md, „Eredmény” szakasz.
- A futásban 166 szegmens és 5 beszélő volt. A régi futásban 162 szegmens és 6 beszélő: az átírás kissé másképp tagolt, és ettől változott a csoportosítás is.
- Tesztek: 128, mind átmegy (új: `tests/test_synthesize.py`).
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

### Hangtípus felismerése: női / férfi / gyerek / állat – megvalósítva (2026-10-27)
- **Hangtípus beszélőnként** (`voices` lépés): az alaphang mediánja alapján (`male_max_f0` 165 Hz, `child_min_f0` 300 Hz).
  - Az AST AudioSet „férfi/női/gyerek beszéd” címkéi rajzfilmhangokon semmitmondóak voltak, ezért nem azokat használjuk.
  - Kézi felülírás: `speaker_kinds: {SPK2: child}` a config.yaml-ban.
- **Hangmagasság-korrekció:** a klónozott hang alaphangját a próbamondatokon méri.
  - Gyerekhangnál (`pitch_correction: child`) az align lépés feljebb tolja, legfeljebb 7 félhanggal.
  - Barbie: SPK1 262 → 364 Hz (eredeti 408), SPK2 392 → 488 Hz (eredeti 464).
- **Állathangok** (`sounds` lépés): AST hangesemény-felismerés csak a beszédszünetekben, mert a beszéd alatt a magas hangokat tévesen „nyávogásnak” nézte. A találatok a mix lépésben változatlanul visszakerülnek. A Barbie-videóban nincs valódi állathang (0 találat), ezért valódi állathangos videón még nincs kipróbálva.
- **Fordítás:** a prompt megkapja a beszélők típusát (pl. „SPK1: child”), a gyerekek egyszerűbben beszélnek.
- **edge-tts:** hangválasztás típus szerint (férfi: Tamás, női: Noémi, gyerek: Noémi +25 Hz).
- **Riport és `status`:** beszélőnként a típus, az alaphang, a klón alaphangja és a korrekció.
- Barbie-futás a `sounds` lépéstől: 16 perc 48 mp. A felismerés: SPK0 és SPK4 női, SPK1–SPK3 gyerek. **Meghallgatásra vár:** a +6–7 félhangos eltolás lehet kissé „mókusos”.
- Nyitott: a felnőtt női klónok is kb. 3,5 félhanggal mélyebbek az eredetinél (SPK0: 284 → 229 Hz). `pitch_correction: all` mellett ezek is korrigálódnának.
