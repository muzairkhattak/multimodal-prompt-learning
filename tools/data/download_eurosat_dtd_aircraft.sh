#!/usr/bin/env bash
# Download EuroSAT, DTD and FGVC Aircraft into the layout of docs/DATASETS.md.
#
#   DATA=/path/to/datasets bash tools/data/download_eurosat_dtd_aircraft.sh
#
# Result: $DATA/eurosat, $DATA/dtd and $DATA/fgvc_aircraft
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# docs/DATASETS.md points to madm.dfki.de, which timed out when this was written.
# This is the authors' Zenodo record, and it holds all 27,000 images that
# split_zhou_EuroSAT.json names.
EUROSAT_URL="https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
EUROSAT_SPLIT_URL="https://drive.google.com/uc?export=download&id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"
EUROSAT_SPLIT_SHA256="541503a77d0b26b735d85ec7a3e672804fe0aa466c2e5f07da772cc6c0af8ad2"
EUROSAT_IMAGES=27000

DTD_URL="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_SPLIT_URL="https://drive.google.com/uc?export=download&id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"
DTD_SPLIT_SHA256="f26ebeccd2b58e68d70f07a0dc39fe49bf7b69024bac34396b954dfd87969c38"
DTD_IMAGES=5640

AIRCRAFT_URL="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
AIRCRAFT_IMAGES=10000

# The dataset classes silently make a new random split when the split file is
# absent, so its checksum is verified here.
download_split() {
    local url="$1"
    local destination="$2"
    local expected_sha256="$3"
    download "${url}" "${destination}"
    check_sha256 "${destination}" "${expected_sha256}"
}

download_eurosat() {
    local dataset_dir="${DATA}/eurosat"
    local zip_path="${dataset_dir}/EuroSAT_RGB.zip"
    mkdir -p "${dataset_dir}"
    if [ ! -d "${dataset_dir}/2750" ]; then
        download "${EUROSAT_URL}" "${zip_path}"
        unzip -oq "${zip_path}" -d "${dataset_dir}"
        # datasets/eurosat.py reads the images from a folder named 2750.
        mv "${dataset_dir}/EuroSAT_RGB" "${dataset_dir}/2750"
        rm "${zip_path}"
    fi
    download_split "${EUROSAT_SPLIT_URL}" "${dataset_dir}/split_zhou_EuroSAT.json" \
        "${EUROSAT_SPLIT_SHA256}"
    check_image_count "${dataset_dir}/2750" "*.jpg" "${EUROSAT_IMAGES}"
}

download_dtd() {
    local dataset_dir="${DATA}/dtd"
    local archive="${DATA}/dtd-r1.0.1.tar.gz"
    if [ ! -d "${dataset_dir}/images" ] || [ -f "${archive}" ]; then
        download "${DTD_URL}" "${archive}"
        tar -xzf "${archive}" -C "${DATA}"
        rm "${archive}"
    fi
    download_split "${DTD_SPLIT_URL}" "${dataset_dir}/split_zhou_DescribableTextures.json" \
        "${DTD_SPLIT_SHA256}"
    check_image_count "${dataset_dir}/images" "*.jpg" "${DTD_IMAGES}"
}

download_aircraft() {
    local dataset_dir="${DATA}/fgvc_aircraft"
    local archive="${DATA}/fgvc-aircraft-2013b.tar.gz"
    local extracted="${DATA}/fgvc-aircraft-2013b"
    if [ ! -d "${dataset_dir}" ]; then
        download "${AIRCRAFT_URL}" "${archive}"
        # DATASETS.md keeps only data/, renamed to fgvc_aircraft/.
        tar -xzf "${archive}" -C "${DATA}" fgvc-aircraft-2013b/data
        mv "${extracted}/data" "${dataset_dir}"
        rm -r "${extracted}"
        rm "${archive}"
    fi
    check_image_count "${dataset_dir}/images" "*.jpg" "${AIRCRAFT_IMAGES}"
}

main() {
    require_data_dir
    require_commands curl unzip tar sha256sum
    download_eurosat
    download_dtd
    download_aircraft
}

main
