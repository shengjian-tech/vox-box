$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Test {
    poetry run pytest
    if ($LASTEXITCODE -ne 0) {
        VoxBox.Log.Fatal "failed to run poetry run pytest."
    }
}

#
# main
#

VoxBox.Log.Info "+++ TEST +++"
try {
    Test
} catch {
    VoxBox.Log.Fatal "failed to test: $($_.Exception.Message)"
}
VoxBox.Log.Info "--- TEST ---"
