#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  if vox_box::util::is_darwin; then
    build_platform "macosx_11_0_universal2"
  elif vox_box::util::is_linux; then
    # This is a temporary workaround to make the wheel files different.
    echo >> "${ROOT_DIR}/README.md"
    build_platform "manylinux2014_x86_64"
    echo >> "${ROOT_DIR}/README.md"
    build_platform "manylinux2014_aarch64"
    # Remove the extra newline.
    # shellcheck disable=SC2016
    vox_box::util::sed '${/^$/d;}' "${ROOT_DIR}/README.md"
  fi
}


function build_platform() {
  platform="${1:-}"
  if [ -z "$platform" ]; then
    vox_box::log::fatal "undefined platform to build"
  fi
  
  ignore_thirdparty_invalid_file

  poetry build

  dist_dir="$ROOT_DIR/dist"
  whl_files=$(find "$dist_dir" -name "*.whl")
  if [ -z "$whl_files" ]; then
      vox_box::log::fatal "no wheel files found in $dist_dir"
  fi

  for whl_file in $whl_files; do
      if [[ "$whl_file" == *-any* ]]; then
          original_name=$(basename "$whl_file")
          new_name="${original_name/any/$platform}"
          mv -f "$whl_file" "$dist_dir/$new_name"
          vox_box::log::info "renamed $original_name to $new_name"
      fi
  done
}

function prepare_dependencies() {
  bash "${ROOT_DIR}/hack/install.sh"
}

function set_version() {  
  local version_file="${ROOT_DIR}/vox_box/__init__.py"
  local git_commit="${GIT_COMMIT:-HEAD}"
  local git_commit_short="${git_commit:0:7}"

  vox_box::log::info "setting version to $GIT_VERSION"
  vox_box::log::info "setting git commit to $git_commit_short"

  # Replace the __version__ variable in the __init__.py file
  vox_box::util::sed "s/__version__ = .*/__version__ = '${GIT_VERSION}'/" "${version_file}"
  vox_box::util::sed "s/__git_commit__ = .*/__git_commit__ = '${git_commit_short}'/" "${version_file}"

  # Update the poetry version
  poetry version "${GIT_VERSION}"
}

function restore_version_file() {
  local version_file="${ROOT_DIR}/vox_box/__init__.py"
  git checkout -- "${version_file}"
}

#
# main
#

vox_box::log::info "+++ BUILD +++"
prepare_dependencies
set_version
build
restore_version_file
vox_box::log::info "--- BUILD ---"
