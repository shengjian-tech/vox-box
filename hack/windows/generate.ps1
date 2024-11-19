$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Generate {
    poetry run gen
    if ($LASTEXITCODE -ne 0) {
        SpeechBox.Log.Fatal "failed to run poetry run gen."
    }
}

#
# main
#

SpeechBox.Log.Info "+++ GENERATE +++"
try {
    Generate
} catch {
    SpeechBox.Log.Fatal "failed to generate: $($_.Exception.Message)"
}
SpeechBox.Log.Info "--- GENERATE ---"
