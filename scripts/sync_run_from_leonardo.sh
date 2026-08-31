#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 3 ]; then echo "usage: $0 USER LOGIN_HOST REMOTE_RUN_DIR [LOCAL_DIR]"; exit 2; fi
USER_NAME=$1; HOST=$2; REMOTE=$3; LOCAL=${4:-runs/remote_$(basename "$REMOTE")}
mkdir -p "$LOCAL"
# Use the CINECA datamover hostname supplied for your account/event when available.
rsync -av --partial --append-verify "$USER_NAME@$HOST:$REMOTE/" "$LOCAL/"
