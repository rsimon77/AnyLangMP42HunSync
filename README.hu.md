# AnyLangMP42HunSync

[English version](README.md)

Rövid videók (max. 10 perc) magyar szinkronja ingyenes, helyben futó AI-val. 16 GB-os Apple Silicon Macen is elfut. A tervet a [DESIGN.md](DESIGN.md) írja le.

A forrásnyelv jelenleg angol: az átírás lépés `en`-re van állítva. A Whisper magától sok más nyelvet is felismer.

## Működés

Lépések:

1. **extract**: az ffmpeg kiveszi a videóból a hangot.
2. **separate**: a Demucs szétválasztja a beszédet és a hátteret (zene, effektek).
3. **transcribe**: a Whisper (`mlx-whisper`, large-v3-turbo) szó szintű időbélyegekkel átírja a beszédet.
4. **sounds**: egy AST hangesemény-felismerő (AudioSet-címkék) állathangokat keres a beszédszünetekben, hogy ne vesszenek el a magyar sávból.
5. **voices**:
   - A beszélőket az XTTS beszélővektorainak csoportosítása választja szét.
   - Ez a lépés minden beszélőnél:
     - hangmintát készít;
     - felismeri a hangtípust (férfi / női / gyerek) az alaphangból;
     - megméri a klónozott hang tempóját;
     - korrigálja a klónozott gyerekhangok hangmagasságát.
6. **translate**: a Gemma 3 12B (Ollamán) beszélt magyar nyelvre fordít.
   - Minden sor szótagkeretet kap, hogy beférjen az eredeti időzítésbe.
   - A modell megkapja a beszélők hangtípusát, így a gyerekek egyszerűen beszélnek.
7. **synthesize**: az XTTS-v2 az eredeti beszélők klónozott hangján mondja ki a magyar szöveget.
   - A túl hosszú sorokat az LLM egy kötegben rövidíti.
   - A hangmodell és az LLM soha nincs egyszerre a memóriában.
8. **align**: nyújtással és hangmagasság-eltolással minden sor az eredeti idővonalra kerül.
9. **mix**: a szinkron a háttérrel és a megtartott állathangokkal keveredik, majd hangerő-normalizálás következik.
10. **mux**: elkészül a kimeneti MP4 (magyar sáv alapértelmezettként, mellette az eredeti), az `.srt` felirat és a riport.

MacBook Air M5-ön (16 GB) egy 9 perces, 5 szereplős rajzfilm teljes feldolgozása kb. 20 perc.

## Telepítés (Apple Silicon Mac)

```bash
brew install ffmpeg uv ollama
ollama serve &                 # vagy: brew services start ollama
ollama pull gemma3:12b
uv sync
```

Az első futás letölti ezeket a modelleket:

| Modell | Méret |
|---|---|
| Whisper | ~1,6 GB |
| Demucs | ~80 MB |
| XTTS-v2 | ~1,9 GB |
| AST hangfelismerő | ~350 MB |

**Licenc:** az XTTS-v2 a Coqui Public Model License alatt érhető el, ami **csak nem kereskedelmi felhasználást** enged.
- Az elfogadást a `config.yaml`-ban az `xtts_accept_license: true` sor jelzi.
- Ha nem fogadod el, használd a `tts: edge` motort. Ez online Microsoft-hangokkal dolgozik, klónozás nélkül; a hangot a hangtípus alapján választja.

## Használat

```bash
uv run mp4translator dub video.mp4
```

Kimenet:
- `video_hu.mp4`: az 1. hangsáv magyar, a 2. az eredeti;
- `video_hu.hu.srt`: magyar felirat;
- `work/video/report.md`: beszélők, hangtípusok, megtartott hangok, és a gyorsított vagy rövidített szegmensek.

Hasznos kapcsolók:

- `--from-step <lépés>`: újrafuttatás egy lépéstől. Például ha kézzel javítottad a `work/video/manifest.json` `text_hu` vagy `speaker` mezőit, vagy ha beállítást változtattál.
- `--llm ollama:gemma3:4b`: kisebb, gyorsabb fordítómodell.
- `-c config.yaml`: beállítások, többek között:
  - tegezés vagy magázás;
  - glosszárium;
  - gyorsítási határok;
  - a beszélők szétválogatásának érzékenysége (`speaker_threshold`);
  - kézi hangtípus (`speaker_kinds: {SPK2: child}`);
  - hangmagasság-korrekció (`pitch_correction: off | child | all`);
  - állathangok megtartása (`keep_animal_sounds`).

### Beszélők vizsgálata

```bash
uv run mp4translator speakers --workdir work/video
```

Kiírja, hány beszélő jön ki az egyes küszöbértékekkel, és előnézeti mintákat is ki tud menteni.

A beszélőnkénti hangminták és próbamondatok a `work/video/speakers/` mappába kerülnek, bele lehet hallgatni.

## Leállítás és folytatás

- **Leállítás:** `Ctrl+C`, vagy egy másik terminálból `uv run mp4translator stop`. A program felszabadítja a modelleket, és kiírja a folytatás parancsát.
- **Folytatás:** ugyanaz a `dub` parancs.
  - A kész lépéseket átugorja.
  - A félbemaradt lépést onnan folytatja, ahol abbamaradt: a kész fordítási kötegek és a kész szegmenshangok megmaradnak.
- **Állapot:** az `uv run mp4translator status` a lépéseket, a szegmensek számát és a beszélők hangtípusát mutatja.
- Egy munkamappán egyszerre csak egy futás dolgozhat (`work/<videó>/run.pid`).

## iCloud

Ha a projekt iCloudon szinkronizált mappában van (pl. `Documents`), a `.venv`, a `work` és az `output` legyen a `.venv.nosync`, `work.nosync` és `output.nosync` mappákra mutató link. Az iCloud a `.nosync` végű mappákat nem szinkronizálja. Enélkül kiürítheti a fájlokat, és a Python nem találja a telepített csomagot.

## Tesztek

```bash
uv run --group dev pytest -q
```

## További dokumentumok

- [DESIGN.md](DESIGN.md): felépítés és ütemterv
- [OPTIMIZATION.md](OPTIMIZATION.md): sebességmérések és gyorsítások
- [STATUS.md](STATUS.md): jelenlegi állapot és következő lépések
