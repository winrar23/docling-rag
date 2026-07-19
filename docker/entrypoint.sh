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
    worker)
        shift
        exec python -m docling_rag.worker "$@"
        ;;
    embed)
        shift
        exec uvicorn --factory docling_rag.api.embed_app:create_app --host 0.0.0.0 --port 8100
        ;;
    *)
        exec docling-rag "$@"
        ;;
esac
