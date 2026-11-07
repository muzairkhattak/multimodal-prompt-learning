#!/usr/bin/env bash
# Download DomainNet from the official page, http://ai.bu.edu/M3SDA/, using the
# set it marks "cleaned version, recommended". Needs about 25 GB free while running.
#
#   DATA=/path/to/datasets bash tools/data/download_domainnet.sh
#
# Result:
#   $DATA/domainnet/<style>/<class>/<image>
#   $DATA/domainnet/splits/<style>_train.txt and <style>_test.txt
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BASE_URL="https://csr.bu.edu/ftp/visda/2019/multi-source"
STYLES="clipart infograph painting quickdraw real sketch"

zip_url() {
    local style="$1"
    # The official page serves these two from groundtruth/. The zips with the
    # same names one folder up have another layout: no listed path is in them.
    if [ "${style}" = "clipart" ] || [ "${style}" = "painting" ]; then
        echo "${BASE_URL}/groundtruth/${style}.zip"
    else
        echo "${BASE_URL}/${style}.zip"
    fi
}

download_lists() {
    local style split
    mkdir -p "${SPLIT_DIR}"
    for style in ${STYLES}; do
        for split in train test; do
            download "${BASE_URL}/domainnet/txt/${style}_${split}.txt" \
                "${SPLIT_DIR}/${style}_${split}.txt"
        done
    done
}

download_images() {
    local style="$1"
    local zip_path="${DOMAINNET_DIR}/${style}.zip"
    # The zip is deleted only after a full extraction, so a folder without a zip is complete.
    if [ -d "${DOMAINNET_DIR}/${style}" ] && [ ! -f "${zip_path}" ]; then
        echo "Already extracted: ${style}"
        return
    fi
    download "$(zip_url "${style}")" "${zip_path}"
    unzip -oq "${zip_path}" -d "${DOMAINNET_DIR}"
    rm "${zip_path}"
}

sorted_files_on_disk() {
    local style="$1"
    (cd "${DOMAINNET_DIR}" && find "${style}" -type f | LC_ALL=C sort)
}

# Count the images listed in the txt files that are not on disk.
count_missing() {
    local style="$1"
    cat "${SPLIT_DIR}/${style}_train.txt" "${SPLIT_DIR}/${style}_test.txt" \
        | cut -d' ' -f1 | LC_ALL=C sort \
        | LC_ALL=C comm -23 - <(sorted_files_on_disk "${style}") \
        | wc -l
}

print_counts() {
    local style on_disk train test missing
    local total_missing=0
    # on_disk is larger than train + test: the zips also hold the images that
    # the cleaned lists removed.
    printf '%-10s %8s %8s %8s %8s\n' style on_disk train test missing
    for style in ${STYLES}; do
        on_disk="$(sorted_files_on_disk "${style}" | wc -l)"
        train="$(wc -l < "${SPLIT_DIR}/${style}_train.txt")"
        test="$(wc -l < "${SPLIT_DIR}/${style}_test.txt")"
        missing="$(count_missing "${style}")"
        printf '%-10s %8d %8d %8d %8d\n' "${style}" "${on_disk}" "${train}" "${test}" "${missing}"
        total_missing=$((total_missing + missing))
    done
    if [ "${total_missing}" -ne 0 ]; then
        echo "ERROR: expected every listed image on disk, found ${total_missing} missing." >&2
        exit 1
    fi
}

main() {
    require_data_dir
    require_commands curl unzip
    DOMAINNET_DIR="${DATA}/domainnet"
    SPLIT_DIR="${DOMAINNET_DIR}/splits"
    mkdir -p "${DOMAINNET_DIR}"

    download_lists
    local style
    for style in ${STYLES}; do
        download_images "${style}"
    done
    print_counts
}

main
