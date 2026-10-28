# Állapot – 2026-10-27

## Kész
- MVP (Whisper + Gemma 3 12B + edge-tts), mintavideón tesztelve.
- 2. fázis: Demucs háttér-szétválasztás, beszélők szétválogatása (XTTS-beszélővektorok csoportosítása), XTTS-v2 hangklónozás. A mintavideón tesztelve: [samples/](samples/), illetve a scratchpadben a `sample_xtts_hu.mp4`.
- XTTS licenc elfogadva (`config.yaml`: `xtts_accept_license: true`).

## Hangmagasság beszélőnként és mondatonként (2026-10-28)
- **Beszélők fül:** a klónozott hang hangmagassága csúszkával állítható (±12 félhang), előnézettel, vagy visszaállítható automatikusra.
  - Az érték a munka `config.yaml`-jába kerül (`speaker_pitch`), így a voices lépés újrafuttatása is megtartja.
  - A riportban a kézi érték „(kézi)” jelölést kap.
- **Mondatok fül:** egy mondat saját eltolása (`Segment.pitch_offset`), ami a beszélőéhez adódik. A táblázatban így jelenik meg: „+2♪”.
- Egyik sem generál új hangot: csak az align, a mix és a mux fut újra, a 3 perces részleten kb. 8 mp alatt.
- Az align törli az elavult `_fit.wav` fájlt, ha egy mondatnál már nem kell illesztés.
- A böngészőben kipróbálva:
  - beszélő +6 félhang: a próbamondat 223 Hz-ről 321 Hz-re ment;
  - mondat −2 félhang: összesen +3,6 félhang, a várt +4 közelében.
- 165 teszt, mind átmegy.

## Szegmensszerkesztő (2026-10-28)
- Webes felület, **Mondatok** fül. A logika a `src/mp4translator/webui/segments.py`-ban van, Gradio nélkül, tesztelve.
  - A javítás a manifestbe kerül. Az **Alkalmazás** a futást a synthesize lépés közepére állítja, így a szokásos folytatás csak a nem kész mondatokat generálja újra, utána illeszt, kever és muxol.
  - A kézzel javított szöveg `edited` jelölést kap: az LLM nem rövidíti, a túl hosszú mondat inkább gyorsítva lesz. Egy újabb fordítás (translate) törli a jelölést.
  - Az újragenerálás csak azoknak a beszélőknek a hangmintáját dolgozza fel, akiknek van újragenerálandó mondatuk: egy mondat 49 helyett 21 mp, a teljes újraépítés kb. 30 mp a 3 perces részleten.
- A böngészőben kipróbálva:
  - szöveg javítása, majd alkalmazás;
  - újragenerálás ugyanazzal a szöveggel;
  - a táblázat magától frissül a futás végén.
- 159 teszt, mind átmegy.

## Modellválasztás és kódátnézés (2026-10-27)
- Választható modellek: fordító LLM (ajánlott lista, `src/mp4translator/models.py`), Whisper, Demucs, hangmotor.
  - Parancssorban: `--llm`, `--whisper`, `--demucs`, `--tts`; a `models` parancs listáz.
  - Webes felületen: legördülő listák, és egy gomb letölti a kiválasztott Ollama-modellt, haladásjelzővel.
  - A gondolkodó modellek (qwen3, gemma4) `think: false`-t kapnak.
  - Csak a Gemma 3 12B van magyarra kipróbálva; a többi letöltése és meghallgatása még hátravan.
- **Új hangmotor: az OmniVoice kipróbálva, nem lett beépítve** (2026-10-27).
  - Összevetés az XTTS-sel ugyanazon a két hangmintán (gyerek és női hang), ugyanazzal a 4 magyar mondattal. Az érthetőséget a Whisper magyar átírása mérte (szóegyezés a célszöveggel):
    - XTTS: 0,83 és 0,91;
    - OmniVoice, 32 lépés: 0,39 és 0,52 (pl. „hitfigyett” a „hétvégét” helyett);
    - OmniVoice, 16 vagy 8 lépés: 0,16–0,23.
  - Sebesség MPS-en: az OmniVoice kb. 9–10 mp mondatonként 32 lépéssel (8 lépéssel 2,7 mp); az XTTS 1–3,6 mp. A kötegelt generálás nem gyorsít.
  - Előnyei: a gyerekhang magassága közelebb marad az eredetihez, és a `duration` paraméterrel pontos hosszra generál (3,0 mp kérve → 3,1–3,2 mp lett).
  - Meghallgatható: `work/omnivoice_test/xtts_vs_omnivoice.wav` (felváltva XTTS, majd OmniVoice).
- **A magyar F5-TTS (Maxdorger29/f5-tts-hungarian) is kipróbálva, nem lett beépítve** (2026-10-28).
  - Ugyanaz a két hangminta, 6 magyar mondat, MPS. Érthetőség (a Whisper magyar átiratának szóegyezése), gyerek / női hang:
    - XTTS: 0,78 / 0,84, kb. 2–3 mp mondatonként;
    - F5, 16 lépés, rövid (6–7 mp-es) minta: 0,77 / 0,81, kb. 10 mp;
    - F5, 32 lépés, rövid minta: 0,77 / 0,78, kb. 21 mp;
    - F5, 16 lépés, teljes (11–13 mp-es) minta: 0,73 / 0,78, kb. 21 mp.
  - A hangmagasság sem jobb: a gyerekhang (eredeti 302 Hz) az F5-nél 220 Hz, az XTTS-nél 235 Hz. A női hangot (289 Hz) az F5 jobban tartja: 299 Hz, az XTTS 238 Hz.
  - Az f5-tts csomag a projekt környezetében visszaléptetné a huggingface-hub csomagot (2.3 → 1.33). A teszt ezért külön virtuális környezetben futott.
  - Összegzés: azonos érthetőség 3–5-ször lassabban, egy 9 perces videónál kb. +20–25 perc hanggenerálás. Az XTTS marad.
  - Meghallgatható: `work/f5_test/xtts_vs_f5.wav` (mondatonként XTTS, majd F5).
- A kódátnézés javításai:
  - XTTS: nem omlik össze, ha minden próbálkozás üres hangot ad;
  - minden Ollama-hiba (500-as hiba, időtúllépés) `LLMError`, így a rövidítés helyett gyorsítás jön;
  - megszakított `--from-step synthesize` után a folytatás nem keveri a régi és az új hangokat;
  - a hangfájlok írása atomi;
  - `run.pid`: atomi zárolás, és egy újrahasznosított PID-et nem tekint futásnak, nem is állítja le;
  - webes felület: szálbiztos indítás, egy sérült munka nem viszi el a felületet, a napló végét olvassa;
  - mkv/webm hossz, gyorsított beszéd vége közös helyen (`Segment.speech_end`);
  - Whisper-memória felszabadítása, párhuzamos align;
  - determinisztikus Demucs.
- 146 teszt, mind átmegy.

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
