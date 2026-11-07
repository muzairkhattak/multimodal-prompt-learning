#!/usr/bin/env bash
# Helpers shared by the download scripts. Source this file, do not run it.
set -euo pipefail

require_data_dir() {
    if [ -z "${DATA:-}" ]; then
        echo "ERROR: expected DATA to name the datasets folder, found it unset or empty." >&2
        exit 1
    fi
    mkdir -p "${DATA}"
}

require_commands() {
    local command_name
    for command_name in "$@"; do
        if ! command -v "${command_name}" > /dev/null; then
            echo "ERROR: expected '${command_name}' on PATH, did not find it." >&2
            exit 1
        fi
    done
}

# Download to a .part file first, so an existing destination is always complete
# and an interrupted download resumes on the next run.
download() {
    local url="$1"
    local destination="$2"
    if [ -f "${destination}" ]; then
        echo "Already downloaded: ${destination}"
        return
    fi
    echo "Downloading ${url}"
    curl --fail --location --retry 5 --continue-at - --output "${destination}.part" "${url}"
    mv "${destination}.part" "${destination}"
}

check_sha256() {
    local file_path="$1"
    local expected="$2"
    local found
    found="$(sha256sum "${file_path}" | cut -d' ' -f1)"
    if [ "${found}" != "${expected}" ]; then
        echo "ERROR: ${file_path}: expected sha256 ${expected}, found ${found}." >&2
        exit 1
    fi
}

check_image_count() {
    local folder="$1"
    local name_pattern="$2"
    local expected="$3"
    local found
    found="$(find "${folder}" -type f -name "${name_pattern}" | wc -l | tr -d ' ')"
    if [ "${found}" -ne "${expected}" ]; then
        echo "ERROR: ${folder}: expected ${expected} images, found ${found}." >&2
        exit 1
    fi
    echo "${folder}: ${found} images"
}
