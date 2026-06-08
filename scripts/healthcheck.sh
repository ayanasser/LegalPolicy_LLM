#!/usr/bin/env bash
# Quick health check for the unified-UI stack. Run any time:
#     bash scripts/healthcheck.sh
# Shows each service's /health, the UI port, the Gradio share link, Ollama, and
# the live processes — so you can see at a glance what's up and what's down.

c_up="\033[32m"; c_dn="\033[31m"; c_0="\033[0m"
ok()   { printf "  ${c_up}● UP${c_0}   %-22s %s\n" "$1" "$2"; }
down() { printf "  ${c_dn}● DOWN${c_0} %-22s %s\n" "$1" "$2"; }

check() {  # label  url
  local out; out=$(curl -s -m 5 "$2" 2>/dev/null)
  if [ -n "$out" ]; then ok "$1" "$out"; else down "$1" "(no response at $2)"; fi
}

echo "── Ollama ─────────────────────────────────────────────"
if curl -s -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  ok "ollama (:11434)" ""
  ollama ps 2>/dev/null | sed 's/^/    /'
else
  down "ollama (:11434)" "start it: nohup ollama serve &"
fi

echo "── RAG / answer services ──────────────────────────────"
check "Bilingual  (:8100)" http://localhost:8100/health
check "Neo4j Graph(:8000)" http://localhost:8000/health
check "Combined   (:8200)" http://localhost:8200/health

echo "── Unified UI (:${LP_UI_PORT:-7870}) ──────────────────────────────"
code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://localhost:${LP_UI_PORT:-7870}" 2>/dev/null)
[ "$code" = "200" ] && ok "gradio server" "http 200" || down "gradio server" "http ${code:-000}"
link=$(grep -ohE "https://[a-z0-9]+\.gradio\.live" /tmp/lp_ui.log 2>/dev/null | tail -1)
[ -n "$link" ] && echo "    public link: $link" || echo "    public link: (not in /tmp/lp_ui.log yet)"

echo "── Processes ──────────────────────────────────────────"
ps aux | grep -E "uvicorn apps|unified_ui.app|ollama serve" | grep -v grep \
  | awk '{printf "    pid %-7s %s %s %s\n", $2, $13, $14, $15}' || echo "    (none)"
