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

# The pinned release asset is the fast path. It only exists once
# .github/workflows/macos-engines.yml has published the tag named in
# distribution.tag, so a failed download must not be fatal: rebuild the runtime
# from the pinned engine sources instead of leaving the checkout without engines
# and failing the build. The rebuild is reproducible because every input is
# pinned by SHA-256 in engines/manifest.macos.json.
if curl --fail --location --retry 3 "$source_url" --output "$archive" &&
  curl --fail --location --retry 3 "$checksum_url" --output "$checksum"; then
  install_from_archive
  "$workspace/scripts/verify-engines-macos.sh"
  exit 0
fi

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

echo "Falling back to a local build from the pinned engine sources ($build_fallback)." >&2
echo "This compiles FFmpeg and COLMAP and can take a long time." >&2
"$workspace/$build_fallback"
