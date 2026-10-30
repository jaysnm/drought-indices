#!/bin/sh
# Runs `drought-indices update` now and then every UPDATE_INTERVAL_HOURS hours.
# A failed update is logged and retried at the next interval rather than stopping the loop.
set -u
interval_hours="${UPDATE_INTERVAL_HOURS:-24}"

while true; do
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') starting update"
    if drought-indices update "$@"; then
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') update finished"
    else
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') update failed with status $?" >&2
    fi
    echo "next update in ${interval_hours} h"
    sleep "$((interval_hours * 3600))"
done
