#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "macOS engine setup requires an Apple Silicon Mac." >&2
  exit 1
fi

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$workspace/engines/manifest.macos.json"
cache="$workspace/.cache/engines/macos"
destination="$workspace/engines/macos/arm64"

read_manifest() {
  node -e 'const m=require(process.argv[1]); let v=m; for (const key of process.argv[2].split(".")) v=v[key]; process.stdout.write(String(v));' "$manifest" "$1"
}

archive_name="$(read_manifest distribution.archiveName)"
layout_root="$(read_manifest distribution.archiveLayoutRoot)"
build_fallback="$(read_manifest distribution.buildFallback)"
archive="$cache/$archive_name"
checksum="$archive.sha256"
source_url="$(read_manifest distribution.sourceUrl)"
checksum_url="$(read_manifest distribution.archiveSha256File)"

install_from_archive() {
  expected="$(awk 'NF { print tolower($1); exit }' "$checksum")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid release checksum file: $checksum" >&2; exit 1; }
  actual="$(shasum -a 256 "$archive" | awk '{ print tolower($1) }')"
  [[ "$actual" == "$expected" ]] || { echo "macOS engine archive SHA-256 mismatch." >&2; exit 1; }

  if tar -tJf "$archive" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "Unsafe path found in macOS engine archive." >&2
    exit 1
  fi

  temporary="$(mktemp -d)"
  trap 'rm -rf -- "$temporary"' EXIT
  tar -xJf "$archive" -C "$temporary"
  runtime_root="$temporary/$layout_root"
  [[ -d "$runtime_root" ]] || {
    echo "Unexpected macOS engine archive layout: expected $layout_root/ at the archive root." >&2
    exit 1
  }

  staged="$temporary/runtime"
  mv "$runtime_root" "$staged"
  # The destination contains a tracked placeholder README so a fresh clone has the
  # resource directory before engines are installed. Preserve it across the
  # all-at-once runtime replacement; otherwise setup leaves the worktree dirty by
  # deleting a tracked file.
  if [[ -f "$destination/README.md" ]]; then
    cp "$destination/README.md" "$staged/README.md"
  fi
  rm -rf -- "$destination"
  mv "$staged" "$destination"
}

mkdir -p "$cache" "$(dirname "$destination")"
if [[ "${OOOSPLAT_ENGINE_BUILD_VERIFY:-}" != "1" ]]; then
  [[ "$(read_manifest distribution.archiveSha256)" =~ ^[a-fA-F0-9]{64}$ && "$(read_manifest distribution.integritySha256)" =~ ^[a-fA-F0-9]{64}$ ]] || {
    echo 'macOS engine archive is not yet hash-locked; build, review and lock the first release.' >&2; exit 1;
  }
fi
if [[ -n "${OOOSPLAT_MACOS_ENGINE_ARCHIVE:-}" ]]; then
  source_archive="$OOOSPLAT_MACOS_ENGINE_ARCHIVE"
  [[ -f "$source_archive" ]] || { echo "Missing local engine archive: $source_archive" >&2; exit 1; }
  cp "$source_archive" "$archive"
  if [[ -f "$source_archive.sha256" ]]; then
    cp "$source_archive.sha256" "$checksum"
  else
    (cd "$cache" && shasum -a 256 "$(basename "$archive")" > "$(basename "$checksum")")
  fi
  install_from_archive
  "$workspace/scripts/verify-engines-macos.sh"
  exit 0
fi

expected="$(read_manifest distribution.archiveSha256)"
if [[ "${OOOSPLAT_ENGINE_BUILD_VERIFY:-}" == "1" && -n "${OOOSPLAT_MACOS_ENGINE_ARCHIVE:-}" ]]; then
  expected="$(awk 'NF { print tolower($1); exit }' "$checksum")"
fi
expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
[[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid release checksum file." >&2; exit 1; }
actual="$(shasum -a 256 "$archive" | awk '{ print tolower($1) }')"
[[ "$actual" == "$expected" ]] || { echo "macOS engine archive SHA-256 mismatch." >&2; exit 1; }

rm -f -- "$archive" "$checksum"

echo "Could not download the pinned macOS engine assets:" >&2
echo "  $source_url" >&2

# A caller that restores the archive by another route (for example
# .github/workflows/macos.yml fetching the release asset with the GitHub CLI)
# must be able to stop here rather than have this script start a source build,
# which takes hours and would blow the job's timeout.
if [[ -n "${OOOSPLAT_MACOS_ENGINE_NO_BUILD:-}" ]]; then
  echo "OOOSPLAT_MACOS_ENGINE_NO_BUILD is set, so the local rebuild was not started." >&2
  echo "Install the runtime with OOOSPLAT_MACOS_ENGINE_ARCHIVE=<archive> instead." >&2
  exit 1
fi

temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT
tar -xJf "$archive" -C "$temporary"
runtime_root="$temporary/ooosplat-engines-macos-arm64"
[[ -d "$runtime_root" ]] || { echo "Unexpected macOS engine archive layout." >&2; exit 1; }
integrity_pin="$(read_manifest distribution.integritySha256)"
if [[ "${OOOSPLAT_ENGINE_BUILD_VERIFY:-}" == "1" && -n "${OOOSPLAT_MACOS_ENGINE_ARCHIVE:-}" ]]; then
  integrity_pin="$(shasum -a 256 "$runtime_root/SHA256SUMS" | awk '{print $1}')"
fi
node "$workspace/scripts/colmap-runtime.mjs" "$runtime_root" macos "$integrity_pin" integrity

staged="$temporary/runtime"
mv "$runtime_root" "$staged"
# The destination contains a tracked placeholder README so a fresh clone has the
# resource directory before engines are installed. Preserve it across the
# all-at-once runtime replacement; otherwise setup leaves the worktree dirty by
# deleting a tracked file.
if [[ -f "$destination/README.md" ]]; then
  cp "$destination/README.md" "$staged/README.md"
fi
# The mixed runtime does not own standalone COLMAP. Preserve the previous
# directory until the independent, atomic COLMAP installer has a verified
# replacement ready; a download failure must not remove a working local copy.
if [[ -d "$destination/colmap" ]]; then
  cp -R "$destination/colmap" "$staged/colmap"
fi
rm -rf -- "$destination"
mv "$staged" "$destination"

# COLMAP is an independently versioned runtime and is never merged into the
# FFmpeg/Brush directory, avoiding dylib and metadata collisions.
node "$workspace/scripts/setup-colmap-runtime.mjs" macos

"$workspace/scripts/verify-engines-macos.sh"
