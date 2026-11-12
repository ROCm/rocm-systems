#!/bin/sh
if [ -n "$ROCJITSU_DAEMON_PID" ]; then
    printf 'RJ_CLI_DAEMON_PID=%s\n' "$ROCJITSU_DAEMON_PID"
else
    printf 'RJ_CLI_DAEMON_PID=missing\n'
fi

if [ "${RJ_CHECK_DAEMON_PROCESS_GROUP:-}" = 1 ]; then
    if [ -z "$ROCJITSU_DAEMON_PID" ]; then
        printf 'RJ_CLI_DAEMON_PROCESS_GROUP=missing-daemon\n'
        exit 11
    fi
    daemon_pgid="$(ps -o pgid= -p "$ROCJITSU_DAEMON_PID" | tr -d ' ')"
    rocgdb_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
    printf 'RJ_CLI_DAEMON_PGID=%s\n' "$daemon_pgid"
    printf 'RJ_CLI_ROCGDB_PGID=%s\n' "$rocgdb_pgid"
    if [ -z "$daemon_pgid" ] || [ -z "$rocgdb_pgid" ]; then
        printf 'RJ_CLI_DAEMON_PROCESS_GROUP=missing-pgid\n'
        exit 12
    fi
    if [ "$daemon_pgid" = "$rocgdb_pgid" ]; then
        printf 'RJ_CLI_DAEMON_PROCESS_GROUP=foreground\n'
        exit 13
    fi
    printf 'RJ_CLI_DAEMON_PROCESS_GROUP=detached\n'
fi

if [ "${RJ_CHECK_DAEMON_SIGINT:-}" = 1 ]; then
    trap ':' INT
    kill -INT 0
    sleep 0.2
    if [ -S "$ROCJITSU_INVOCATION_DIR/daemon.sock" ]; then
        printf 'RJ_CLI_DAEMON_SIGINT=survived\n'
    else
        printf 'RJ_CLI_DAEMON_SIGINT=stopped\n'
        exit 14
    fi
fi

first=1
for arg in "$@"; do
    if [ "$first" = 0 ]; then
        printf '|'
    fi
    first=0
    printf '%s' "$arg"
done
printf '\n'
