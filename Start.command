#!/bin/bash
# Dupla kattintással indítható (Finder): megnyit egy terminált, és elindítja az Ollamát és a webes felületet.
# A terminálablak bezárása vagy Ctrl+C leállítja őket.
exec "$(dirname "$0")/start.sh" "$@"
