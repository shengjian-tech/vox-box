#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  git submodule update --init --recursive
  rm -rf "${ROOT_DIR}/speech_box/third_party/CosyVoice/third_party/Matcha-TTS/data"
  
  pip install poetry==1.8.3 pre-commit==4.0.1
  poetry install  
  pre-commit install
}

#
# main
#

speech_box::log::info "+++ DEPENDENCIES +++"
download_deps
speech_box::log::info "--- DEPENDENCIES ---"
