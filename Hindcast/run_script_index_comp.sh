#!/bin/bash

# -----------------------------
# Configuration
# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Your ntfy topic
NTFY_TOPIC="lim_notify_keodpjxjxk"

# Optional: server URL
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"


# -----------------------------
# Check input
# -----------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

PYTHON_SCRIPT="$1"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found:"
    echo "${PYTHON_SCRIPT}"
    exit 1
fi


# -----------------------------
# Define logfile
# -----------------------------

SCRIPT_NAME=$(basename "${PYTHON_SCRIPT}" .py)
LOGFILE="${SCRIPT_DIR}/${SCRIPT_NAME}_output.log"


# -----------------------------
# Run script
# -----------------------------

echo "Starting ${SCRIPT_NAME}..."
echo "Log file: ${LOGFILE}"


nice -n 10 python3 -u "${PYTHON_SCRIPT}" > "${LOGFILE}" 2>&1

EXIT_CODE=$?


# -----------------------------
# Notification
# -----------------------------

if [ ${EXIT_CODE} -eq 0 ]; then

    curl \
      -d "${SCRIPT_NAME} finished successfully." \
      -H "Title: ${SCRIPT_NAME} finished" \
      -H "Priority: default" \
      "${NTFY_URL}"

else

    curl \
      -d "${SCRIPT_NAME} failed (exit code ${EXIT_CODE}). Check log: ${LOGFILE}" \
      -H "Title: ${SCRIPT_NAME} FAILED" \
      -H "Priority: high" \
      "${NTFY_URL}"

fi


exit ${EXIT_CODE}