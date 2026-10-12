"""
HydroTrace-AI Maritime Command Platform - Production Runner Entrypoint.
Starts the high-performance, cyber-hardened FastAPI server with WebSocket
streaming and Deck.gl / MapLibre GL geospatial command interface.

Usage:
    python run_platform.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
import uvicorn

from server.config import settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="HydroTrace-AI Maritime Command Platform"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.HOST,
        help="Bind socket host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.PORT,
        help="Bind socket port (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"================================================================================", file=sys.stderr)
    print(f"  ⚓ {settings.APP_NAME} (Version: {settings.APP_VERSION})", file=sys.stderr)
    print(f"  🛡️ DevSecOps: Zero-Trust Auth, RBAC Guards, Strict CSP, SHA-256 Audit Logs", file=sys.stderr)
    print(f"  🌐 Listening on: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  📚 Interactive API Docs: http://{args.host}:{args.port}/api/docs", file=sys.stderr)
    print(f"================================================================================", file=sys.stderr)

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
