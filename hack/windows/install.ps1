# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Install-Dependency {
    git submodule update --init --recursive
    Remove-Item -Recurse -Force "$ROOT_DIR/vox_box/third_party/CosyVoice/third_party/Matcha-TTS/data"

    pip install poetry==1.8.3 pre-commit==4.0.1
    if ($LASTEXITCODE -ne 0) {
        SpeechBox.Log.Fatal "failed to install poetry."
    }

    poetry install
    if ($LASTEXITCODE -ne 0) {
        SpeechBox.Log.Fatal "failed run poetry install."
    }

    poetry run pre-commit install
    if ($LASTEXITCODE -ne 0) {
        SpeechBox.Log.Fatal "failed run pre-commint install."
    }
}

#
# main
#

SpeechBox.Log.Info "+++ DEPENDENCIES +++"
try {
    Install-Dependency
}
catch {
    SpeechBox.Log.Fatal "failed to download dependencies: $($_.Exception.Message)"
}
SpeechBox.Log.Info "-- DEPENDENCIES ---"
