#!/usr/bin/env bash
# Build minimal ONNX Runtime AAR for DocQuadNet-256 (NO XNNPACK, NO NNAPI)
set -euo pipefail

# ---------- config ----------
ORT_VERSION="${ORT_VERSION:-}"

# Repo root (script-relative)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Path to your operator config (repo-relative by default)
CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/model/docquadnet256_trained_opset17.required_operators.config}"

# Output folder where the resulting .aar will be copied
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/app/libs}"

# Android
ANDROID_API="${ANDROID_API:-29}"
ANDROID_TARGET_SDK="${ANDROID_TARGET_SDK:-34}"
ABIS_STR="${ABIS:-arm64-v8a armeabi-v7a x86 x86_64}"
read -r -a ABIS <<<"$ABIS_STR"

# Work dirs
WORK_DIR="${WORK_DIR:-/tmp/onnxruntime-minimal-build}"
ORT_DIR_ORIG="${ORT_DIR_ORIG:-$REPO_DIR/external/onnxruntime}"
SRC_DIR="${SRC_DIR:-/tmp/onnxruntime-src}"
BUILD_DIR="$WORK_DIR/build_android"

# ---------- helpers ----------
die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# ---------- checks ----------
need_cmd cmake
need_cmd python3

if [ ! -f "$CONFIG_FILE" ]; then
  die "operator config not found: $CONFIG_FILE"
fi

CONFIG_FILE_TMP="${CONFIG_FILE_TMP:-/tmp/ort_required_operators.config}"
cp -f "$CONFIG_FILE" "$CONFIG_FILE_TMP"
CONFIG_FILE="$CONFIG_FILE_TMP"

if [ ! -e "$ORT_DIR_ORIG/.git" ]; then
  die "onnxruntime submodule not found: $ORT_DIR_ORIG (run: git submodule update --init --recursive)"
fi

if [ -z "$ORT_VERSION" ]; then
  if command -v git >/dev/null 2>&1; then
    ORT_VERSION="$(git -C "$ORT_DIR_ORIG" describe --tags --abbrev=0 2>/dev/null || true)"
    if [ -z "$ORT_VERSION" ]; then
      ORT_VERSION="$(git -C "$ORT_DIR_ORIG" rev-parse --short HEAD 2>/dev/null || true)"
    fi
  fi
  ORT_VERSION="${ORT_VERSION:-unknown}"
fi

: "${ANDROID_SDK_ROOT:=${ANDROID_HOME:-}}"
[ -n "${ANDROID_SDK_ROOT:-}" ] || die "set ANDROID_SDK_ROOT (or ANDROID_HOME)"
[ -d "$ANDROID_SDK_ROOT" ] || die "ANDROID_SDK_ROOT not a directory: $ANDROID_SDK_ROOT"

: "${ANDROID_NDK_HOME:=${ANDROID_NDK:-}}"
[ -n "${ANDROID_NDK_HOME:-}" ] || die "set ANDROID_NDK_HOME (or ANDROID_NDK)"
[ -d "$ANDROID_NDK_HOME" ] || die "ANDROID_NDK_HOME not a directory: $ANDROID_NDK_HOME"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

echo "=== Minimal ORT AAR build (no XNNPACK/NNAPI) ==="
echo "ORT_VERSION      : $ORT_VERSION"
echo "CONFIG_FILE      : $CONFIG_FILE"
echo "ANDROID_SDK_ROOT : $ANDROID_SDK_ROOT"
echo "ANDROID_NDK_HOME : $ANDROID_NDK_HOME"
echo "ANDROID_API      : $ANDROID_API"
echo "ABIS             : ${ABIS[*]}"
echo "WORK_DIR         : $WORK_DIR"
echo "OUTPUT_DIR       : $OUTPUT_DIR"
echo ""

# ---------- create build_settings.json from default ----------
# Fresh copy of ONNX Runtime sources into a path that's stable across machines
rm -rf "$SRC_DIR"
mkdir -p "$SRC_DIR"
cp -a "$ORT_DIR_ORIG/." "$SRC_DIR"

cd "$SRC_DIR"

BASE_JSON="tools/ci_build/github/android/default_full_aar_build_settings.json"
[ -f "$BASE_JSON" ] || die "expected default settings json missing: $BASE_JSON"

SETTINGS_JSON="$WORK_DIR/build_settings.json"
cp -f "$BASE_JSON" "$SETTINGS_JSON"

python3 - <<PY
import json
p=r"$SETTINGS_JSON"
d=json.load(open(p,"r",encoding="utf-8"))

# Patch ABIs + SDKs
d["build_abis"] = ${ABIS[@]+"["}$(printf '"%s",' "${ABIS[@]}" | sed 's/,$//')${ABIS[@]+"]"}
d["android_min_sdk_version"] = int("$ANDROID_API")
d["android_target_sdk_version"] = int("$ANDROID_TARGET_SDK")

# Patch build params:
params = d.get("build_params", [])
# Drop stuff we don't want for a minimal Android build
drop_exact = {
  "--use_vcpkg",
  "--use_vcpkg_ms_internal_asset_cache",
  "--use_webgpu",
  "--enable_lto",
}
params = [p for p in params if p not in drop_exact]

# 1) remove xnnpack/nnapi if present
params = [x for x in params if not x.startswith("--use_xnnpack") and not x.startswith("--use_nnapi")]

# 2) ensure minimal build + ops config + no tests
# remove existing minimal/include_ops/skip_tests duplicates
params = [x for x in params if not x.startswith("--minimal_build=")]
params = [x for x in params if not x.startswith("--include_ops_by_config=")]
params = [x for x in params if x != "--skip_tests"]
params = [x for x in params if x != "--skip_submodule_sync"]

# add our desired flags
params += [
  "--skip_tests",
  "--skip_submodule_sync",
  "--minimal_build=basic",
  "--include_ops_by_config=" + r"$CONFIG_FILE",
  "--disable_exceptions",
  "--disable_rtti",
  "--disable_ml_ops",
]

# 3) hard-disable ORT tests at CMake level (skip_tests alone is not enough for AAR builds)
params = [x for x in params if not x.startswith("--cmake_extra_defines=")]
params += [
  "--cmake_extra_defines=onnxruntime_BUILD_UNIT_TESTS=OFF",
  "--cmake_extra_defines=onnxruntime_BUILD_SHARED_LIB_TESTS=OFF",
  "--cmake_extra_defines=onnxruntime_BUILD_MLAS_TESTS=OFF",
]

d["build_params"] = params

json.dump(d, open(p,"w",encoding="utf-8"), indent=2)
print("Wrote", p)
PY

echo ""
echo "Using build settings:"
sed -n '1,160p' "$SETTINGS_JSON"
echo ""

# ---------- build AAR ----------
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ---- size/repro flags (apply to all native compilation/link steps) ----
echo "Building AAR ..."
python3 tools/ci_build/github/android/build_aar_package.py \
  --android_sdk_path "$ANDROID_SDK_ROOT" \
  --android_ndk_path "$ANDROID_NDK_HOME" \
  --build_dir "$BUILD_DIR" \
  --config Release \
  "$SETTINGS_JSON"

echo ""
echo "Searching for final AAR ..."

# remove previous copies (optional)
rm -f "$OUTPUT_DIR"/onnxruntime-android-*.aar "$OUTPUT_DIR"/onnxruntime-android.aar

# pick the canonical release aar (prefer versioned)
AAR="$(find "$BUILD_DIR" -type f -name "onnxruntime-android-*.aar" 2>/dev/null | head -n 1 || true)"
if [ -z "$AAR" ]; then
  AAR="$(find "$BUILD_DIR" -type f -name "onnxruntime-android.aar" 2>/dev/null | head -n 1 || true)"
fi
[ -n "$AAR" ] || die "no onnxruntime-android*.aar produced under: $BUILD_DIR"

cp -f "$AAR" "$OUTPUT_DIR/onnxruntime-android-$ORT_VERSION.aar"
echo "Copied: $OUTPUT_DIR/onnxruntime-android-$ORT_VERSION.aar  (size: $(du -h "$AAR" | awk '{print $1}'))"

echo ""
echo "DONE."

