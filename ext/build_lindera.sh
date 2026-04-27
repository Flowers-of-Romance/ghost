#!/bin/bash
# Build lindera-sqlite FTS5 tokenizer extension for ghost.
#
# Reproduces /Users/jm/ghost/ext/lindera_fts5.dylib from upstream + local patch.
# Two upstream bugs are patched:
#   1. bind_fts5_pointer() binds the slot's value (null) instead of its address
#      due to method auto-deref on `target.cast::<c_void>()`.
#   2. ensure_fts5_api_version() requires iVersion == 2; SQLite 3.48+ uses 3.
#
# Requirements: rust toolchain (`brew install rust`).

set -euo pipefail

EXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${EXT_DIR}/.build/lindera-sqlite"

mkdir -p "${EXT_DIR}/.build"

if [ ! -d "${WORK_DIR}" ]; then
  git clone --depth 1 https://github.com/lindera/lindera-sqlite.git "${WORK_DIR}"
fi

cd "${WORK_DIR}"
git checkout -- src/extension.rs 2>/dev/null || true
git apply "${EXT_DIR}/lindera-sqlite.patch"

cargo build --release --features=embed-ipadic

cp target/release/liblindera_sqlite.dylib "${EXT_DIR}/lindera_fts5.dylib"
echo "built: ${EXT_DIR}/lindera_fts5.dylib"
