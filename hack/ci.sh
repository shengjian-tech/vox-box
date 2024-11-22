#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function ci() {
  make install "$@"
  make lint "$@"
  # make test "$@"
  make build "$@"
}

#
# main
#

vox_box::log::info "+++ CI +++"
ci "$@"
vox_box::log::info "--- CI ---"
