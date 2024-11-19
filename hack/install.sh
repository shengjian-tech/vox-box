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
  pip install poetry==1.8.3 pre-commit==4.0.1
  pip install pynini==2.1.5 WeTextProcessing==1.0.3
  poetry install
  pre-commit install
}

#
# main
#

speech_box::log::info "+++ DEPENDENCIES +++"
download_deps
speech_box::log::info "--- DEPENDENCIES ---"
