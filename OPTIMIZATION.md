# Gyorsítási terv (2026-10-27)

## Mérések (Barbie-videó, 9 perc 19 mp, 162 szegmens, MacBook Air M5 16 GB)

| Lépés | Idő | Megjegyzés |
|---|---|---|
| separate (Demucs, MPS) | 81 mp | rendben |
| transcribe (mlx-whisper) | 28 mp | rendben |
| voices | 18 mp (a gyorsítótárral) | rendben |
| translate (Gemma 3 12B, Ollama) | kb. 5–6 perc | elfogadható |
| **synthesize (XTTS-v2)** | **121 szegmens 50 perc alatt** | **ez a szűk keresztmetszet** |

A hanggenerálás szegmensenként: medián 12 mp, átlag 25 mp, a leglassabb 10% 76 mp felett, a leglassabb 147 mp. A probléma a hosszú „farok”, nem az átlagos sebesség.

**Oka:**
- A gép a futás közben 6,8–7 GB-ot swapelt, a szabad memória 11% volt.
- Az Ollama `llama-server` folyamata 12 GB-ot foglal. Ebben benne vannak a modell súlyai, a 8192 tokenes kontextus gyorsítótára és a Gemma 3 képfeldolgozó része.
- Ha egy mondatot rövidíteni kell, a hanggenerálás közben betöltődik a Gemma. Az XTTS mellett így elfogy a memória, és a gép lemezre lapoz.

## Eredmény az 1–3. pont után (2026-10-27, teljes futás az elejétől)

Teljes idő: **20 perc 18 mp** (a régi futás becsült 70–80 perce helyett). Hiba nélkül lefutott, a hőállapot végig „meleg” volt, „súlyos” egyszer sem.

| Lépés | Régi | Új |
|---|---|---|
| separate | 81 mp | 68 mp |
| transcribe | 28 mp | 26 mp |
| voices | 18 mp (gyorsítótárral) | 84 mp (új vektorok + tempómérés) |
| translate | 508 mp | 394 mp |
| **synthesize** | 121 szegmens kb. 50 perc alatt | **166 szegmens 610 mp alatt** |
| align + mix + mux | – | 25 mp |

- Hanggenerálás szegmensenként: medián 2,3 mp (régen 12), átlag 3,1 mp (régen 25), p90 6,3 mp (régen 76), maximum 13,8 mp (régen 147).
- A swap nem nőtt: 1,5 GB maradt (régen 6,8–7,7 GB). A pipeline-folyamat memóriacsúcsa 11,3 GB volt.
- Rövidítés: 20 szegmens egy kötegben, aztán még 2 egy második körben. Az eredmény: 92 ok, 56 borrow, 18 speedup. Overflow nem volt.
- **Új szűk pont:** 91 szegmensnél kellett újrapróbálkozni. Oka: a hossz-ellenőrzés (`plausible_duration`) a beállított 6,0 szótag/mp-cel számol, a klónozott hangok mért tempója viszont 4,0–4,6. Ha az ellenőrzés a beszélő mért tempóját használná, a próbálkozások nagy része elmaradna. **Javítva (2026-10-27):** az ellenőrzés már a beszélő mért tempójával számol (`XTTSEngine.set_rate`). Mérés a synthesize lépéstől újrafuttatva:
  - synthesize 610 mp → **395 mp** (−35%); a teljes újrafuttatás (synthesize → mux) 7 perc 5 mp alatt ment le;
  - szegmensenként: medián 2,3 → 1,4 mp, átlag 3,1 → 1,9 mp, p90 6,3 → 3,6 mp;
  - TTS-hívások 364 → 244, újrapróbált szegmensek 91 → 44.
  - Fenntartás: ez a futás már az első futásban rövidített szövegekből indult, ezért 15 rövidítés kellett 20 helyett. A javulás nagyobb része így is a kevesebb újrapróbálásból jön.

## Kódátnézés és gyorsítás, 3 perces részleten (2026-10-27)

Mérés a Barbie-videó első 3 percén (`work/clips/barbie3.mp4`), teljes futás az elejétől, hűvös gépről indítva.

| Lépés | Előtte | Utána | Megjegyzés |
|---|---|---|---|
| separate | 23 mp | 22 mp | |
| transcribe | 9 mp | 9 mp | utána a Whisper kikerül a memóriából |
| sounds | 4 mp | 4 mp | a 16 kHz-es sávot olvassa, librosa-átmintavételezés nélkül |
| voices | 47 mp | 31 mp | |
| translate | 127 mp | 96 mp | tömörebb JSON-válasz |
| synthesize | 154 mp | 168 mp | 9 rövidítés 5 helyett (ld. lent) |
| align | 6 mp | 4 mp | párhuzamos ffmpeg (4 szál) |
| **összesen** | **382 mp** | **344 mp** | **−10%** |

- **A két futás bemenete nem volt azonos.** A Demucs véletlen időeltolással dolgozik, így a beszédsáv futásonként kicsit más. A második futásban a Whisper írásjelek nélkül írta át a szöveget, és 50 helyett 43 hosszabb szegmens lett. Emiatt a lépésenkénti számok zajosak. Javítva: a separate lépés rögzített véletlenmagot használ, két egymás utáni szétválasztás bitre azonos.
- **Fordítás, azonos bemeneten mérve:** a válasz `{"s12": "…"}` alakú, nem `[{"id": 12, "text": "…"}]`. Ugyanarra a 20 szegmensre 464 helyett kb. 390 token (−16%), és a séma minden szegmensre kikényszeríti a választ. A fordítás ideje a kimenet hosszával arányos: a Gemma 3 12B hűvös gépen kb. 15, melegen 10 token/mp. Ez a fanless MacBook Air hőkorlátozása.
- **Mérési buktató:** egy betöltve hagyott Ollama-modell (7 GB) az első próbálkozásnál swapolást okozott, és minden lépés 1,5–2,5-szer lassabb lett. Mérés előtt: `ollama ps` legyen üres, a hőállapot normál.

## Javaslatok, várható hatás szerint

### 1. Az LLM és az XTTS soha ne legyen egyszerre a memóriában (legnagyobb hatás)
A hanggenerálás kétmenetes legyen:
1. Minden szegmens legenerálódik, a túl hosszúak megjelölve.
2. Az XTTS kiürül a memóriából, és **egyetlen kötegelt LLM-hívás** rövidíti az összes megjelölt mondatot.
3. Az LLM kiürül, és csak a rövidített szegmensek generálódnak újra.

Várható hatás: eltűnik a swapolásból adódó farok, az átlag a medián közelébe kerül (kb. 25 → 12–14 mp/szegmens). Ez a mostani hanggenerálási idő **kb. fele**.

### 2. Kevesebb kétszeres generálás
- **Kis gyorsítás utólag:** 1,10× alatti gyorsításnál ne generáljon újra, hanem nyújtsa a már kész hangot (`atempo`), ez ennyinél alig hallható. Ma minden gyorsított szegmens kétszer generálódik.
- **Tempó beszélőnként:** a `voices` lépés mérje meg minden klónozott hang beszédtempóját (szótag/mp) egy próbamondaton. A fordítás így pontosabb szótagkeretet kap, és ritkábban kell rövidíteni vagy gyorsítani.
- **Mérés:** a manifest szegmensenként rögzítse a generálási időt és a próbálkozások számát (`xtts_retries`), hogy lássuk, a hallucináció-ellenőrzés mennyit visz el.

### 3. Az Ollama memóriaigényének csökkentése
- `num_ctx` 8192 → 4096. A legnagyobb prompt, az összefoglaló, kb. 2000 token egy 10 perces videónál, így ez elég. Kevesebb kontextus-gyorsítótár kell, és kisebb a swap.
- A fordítás végén azonnal kiürül a memóriából. Ez ma is így van, de a rövidítés újratölti, ezt az 1. pont megoldja.
- A rövidítéshez elég lehet a `gemma3:4b` is (kb. 3–4 GB).

### 4. XTTS: MPS vagy CPU
Ellenőrizetlen források szerint az XTTS MPS-en lassú lehet vagy akadozhat Macen. Ezt egy 10 szegmenses mérés eldönti, mindkét eszközön. A mérést üres gépen kell futtatni, a mostani futás alatt nem.

### 5. Később: modellcsere
A kutatás szerint MLX-es XTTS-port nincs. Magyart tudó, hangklónozós jelöltek, amelyek **még ellenőrzésre és mérésre várnak**:
- **F5-TTS magyar finomhangolás:** CC-BY-NC licenc;
- **MOSS-TTS-Nano:** kicsi, CPU-ra készült, a licence nem egyértelmű;
- **Higgs Audio v3:** az mlx-audio listázza, nem kereskedelmi licenc.

A Chatterbox, a Qwen3-TTS és a Kokoro nem tud magyarul. Mivel a TTS cserélhető interfész mögött van, egy új motor egyetlen osztály, de előbb a magyar minőséget kell meghallgatni.

### 6. Kisebb tételek
- **Demucs:** a `demucs-mlx` port állítólag gyorsabb, de a szerző csak CPU-hoz mérte. A mostani 81 mp-ből legfeljebb 1 perc nyerhető.
- **Fordítás MLX-szel (mlx-lm):** a beszámolók szerint 10–30%-kal gyorsabb, azaz kb. 1 perc.
- **Más alkalmazások:** a futás alatt a Teams, az Edge és az Outlook együtt kb. 2 GB-ot foglal. Ezek bezárása csökkenti a swapolást.

## Becsült összhatás
Egy 10 perces videó ma kb. 70–80 perc (a hanggenerálás kb. 65 perc). Az 1–3. ponttal kb. **30–40 perc**, a 4. ponttal esetleg kevesebb. Ezek becslések, implementálás után mérni kell.
