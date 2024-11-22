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
  
  ignore_thirdparty_invalid_file
  
  pip install poetry==1.8.3 pre-commit==4.0.1
  poetry install  
  pre-commit install
}

#
# main
#

vox_box::log::info "+++ DEPENDENCIES +++"
download_deps
vox_box::log::info "--- DEPENDENCIES ---"
