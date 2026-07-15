#!/bin/sh
set -e

case "$1" in
    api)
        shift
        exec uvicorn docling_rag.api.app:app --host 0.0.0.0 --port 8000 "$@"
        ;;
    test)
        shift
        exec pytest "$@"
        ;;
    *)
        exec docling-rag "$@"
        ;;
esac
