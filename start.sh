#!/bin/bash
# Elindítja az Ollamát (ha nem fut) és a webes felületet; Ctrl+C mindkettőt leállítja.
#
#   ./start.sh               indítás (alapból a 7860-as porton, a böngésző magától megnyílik)
#   ./start.sh --port 7861   más porton
#   ./start.sh --no-open     a böngésző megnyitása nélkül
#   ./start.sh stop          a felület és az Ollama leállítása (egy másik ablakból)
#
# Az Ollamát csak akkor állítja le, ha ez a szkript indította, és nem fut éppen szinkron (annak kell a fordításhoz).

set -u
cd "$(dirname "$0")" || exit 1

OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
PORT=7860
OPEN="--open"
LOG_DIR="work"
STARTED_OLLAMA=""
UI_PID=""

say() { printf '%s\n' "$*"; }
ollama_up() { curl -sf -m 2 "$OLLAMA_HOST_URL/api/version" >/dev/null 2>&1; }
ui_up() { curl -sf -m 2 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; }
dub_running() { pgrep -f "mp4translator(\.cli)? dub" >/dev/null 2>&1; }

stop_ollama_serve() {
    if dub_running; then
        say "Fut egy szinkron, ezért az Ollama tovább fut (a fordításhoz kell)."
        return
    fi
    if pgrep -f "ollama serve" >/dev/null 2>&1; then
        pkill -f "ollama serve"
        say "Ollama leállítva."
    fi
}

if [ "${1:-}" = "stop" ]; then
    if pkill -f "mp4translator ui" 2>/dev/null; then say "Webes felület leállítva."; else say "A webes felület nem futott."; fi
    stop_ollama_serve
    if dub_running; then say "A futó szinkron leállítása: uv run mp4translator stop"; fi
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?a --port után szám kell}"; shift 2 ;;
        --no-open) OPEN="--no-open"; shift ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) say "Ismeretlen kapcsoló: $1 (lásd: ./start.sh --help)"; exit 2 ;;
    esac
done

# --- előfeltételek ---
missing=""
for tool in uv ffmpeg ollama; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    say "Hiányzik:$missing"
    say "Telepítés: brew install$missing"
    exit 1
fi

if ui_up; then
    say "A webes felület már fut: http://127.0.0.1:$PORT"
    [ "$OPEN" = "--open" ] && open "http://127.0.0.1:$PORT"
    exit 0
fi

say "Csomagok ellenőrzése…"
uv sync --quiet || { say "A csomagok telepítése nem sikerült (uv sync)."; exit 1; }

# --- Ollama ---
mkdir -p "$LOG_DIR"
if ollama_up; then
    say "Az Ollama már fut."
else
    say "Ollama indítása…"
    # saját folyamatcsoportban (set -m): a terminál Ctrl+C-je ne állítsa le közvetlenül, erről a szkript dönt
    set -m
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup ollama serve >>"$LOG_DIR/ollama.log" 2>&1 &
    STARTED_OLLAMA=$!
    set +m
    for _ in $(seq 1 30); do
        ollama_up && break
        sleep 1
    done
    if ollama_up; then
        say "Az Ollama elindult."
    else
        say "Az Ollama nem indult el (napló: $LOG_DIR/ollama.log). A felület enélkül is megnyílik, ott is elindítható."
    fi
fi

if ollama_up; then
    model=$(uv run --quiet python -c "from pathlib import Path; from mp4translator.config import Config; \
print(Config.load(Path('config.yaml')).llm.split(':', 1)[-1])" 2>/dev/null)
    if [ -n "$model" ] && ! curl -sf -m 3 "$OLLAMA_HOST_URL/api/tags" | grep -q "\"$model\""; then
        say "A fordítómodell ($model) még nincs letöltve. Letöltés: ollama pull $model (vagy a felület Beállítások fülén)."
    fi
fi

# --- leállítás Ctrl+C-re vagy a terminál bezárásakor ---
cleanup() {
    trap - INT TERM HUP EXIT
    say ""
    if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
        kill "$UI_PID" 2>/dev/null
        wait "$UI_PID" 2>/dev/null
        say "Webes felület leállítva."
    fi
    if [ -n "$STARTED_OLLAMA" ]; then
        stop_ollama_serve
    fi
    if dub_running; then
        say "Egy szinkron a háttérben tovább fut; a felület újraindítás után megtalálja."
    fi
    exit 0
}
trap cleanup INT TERM HUP EXIT

# --- webes felület ---
say "Webes felület indítása: http://127.0.0.1:$PORT  (leállítás: Ctrl+C)"
uv run mp4translator ui --port "$PORT" "$OPEN" &
UI_PID=$!
wait "$UI_PID"
