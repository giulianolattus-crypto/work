#!/bin/bash

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON_SCRIPT="${SCRIPT_DIR}/SAM_computation.py"
LOGFILE="${SCRIPT_DIR}/SAM_output.log"

# Your ntfy topic
NTFY_TOPIC="lim_notify_keodpjxjxk"

# Optional: server URL (default is ntfy.sh)
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"


# -----------------------------
# Run script
# -----------------------------

echo "Starting SAM computation..."
echo "Log file: ${LOGFILE}"


nice -n 10 python3 -u "${PYTHON_SCRIPT}" > "${LOGFILE}" 2>&1

EXIT_CODE=$?


# -----------------------------
# Notification
# -----------------------------

if [ ${EXIT_CODE} -eq 0 ]; then

    curl \
      -d "SAM computation finished successfully." \
      -H "Title: SAM finished" \
      -H "Priority: default" \
      "${NTFY_URL}"

else

    curl \
      -d "SAM computation failed (exit code ${EXIT_CODE}). Check log: ${LOGFILE}!" \
      -H "Title: SAM FAILED" \
      -H "Priority: high" \
      "${NTFY_URL}"

fi


exit ${EXIT_CODE}