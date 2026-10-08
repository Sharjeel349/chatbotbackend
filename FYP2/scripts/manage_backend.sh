#!/usr/bin/env bash
PLIST="$HOME/Library/LaunchAgents/com.sharjeel.fypbackend.plist"
LOG_DIR="/Users/sharjeelahmed/PycharmProjects/FYP/logs"

case "$1" in
    start)
        launchctl load "$PLIST"
        echo "Backend service started."
        ;;
    stop)
        launchctl unload "$PLIST"
        echo "Backend service stopped."
        ;;
    restart)
        launchctl unload "$PLIST" 2>/dev/null || true
        sleep 1
        launchctl load "$PLIST"
        echo "Backend service restarted."
        ;;
    status)
        launchctl list | grep fypbackend || echo "Backend service is NOT running."
        curl -s http://127.0.0.1:8000/ || echo ""
        ;;
    logs)
        tail -f "$LOG_DIR/backend.stderr.log" "$LOG_DIR/backend.stdout.log"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
