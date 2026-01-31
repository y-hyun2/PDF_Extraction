"""간단한 Chroma 컬렉션 관리 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
import chromadb

BASE_DIR = Path("vector_db").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chroma 컬렉션 관리")
    parser.add_argument("--list", action="store_true", help="컬렉션 목록 출력")
    parser.add_argument("--remove", type=str, default=None, help="삭제할 컬렉션 이름")
    parser.add_argument("--confirm", action="store_true", help="실제 삭제 실행")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(BASE_DIR))
    if args.list:
        print("현재 컬렉션:")
        for col in client.list_collections():
            print(" -", col.name)

    if args.remove:
        if not args.confirm:
            print("--confirm 옵션을 주어야 삭제가 진행됩니다.")
            return
        try:
            client.delete_collection(args.remove)
            print(f"컬렉션 '{args.remove}' 삭제 완료")
        except Exception as exc:
            print(f"삭제 실패: {exc}")


if __name__ == "__main__":
    main()
