#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function test() {
  poetry run pytest
}

#
# main
#

vox_box::log::info "+++ TEST +++"
test
vox_box::log::info "--- TEST ---"
