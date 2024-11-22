$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

function Ignore-Thirdparty-InvalidFile {
    $dataFileDir = Join-Path -Path $ROOT_DIR -ChildPath "vox_box/third_party/CosyVoice/third_party/Matcha-TTS"
    if (-Not (Test-Path -Path $dataFileDir)) {
        Write-Host "Directory $dataFileDir does not exist. Skipping."
        return
    }

    Push-Location -Path $dataFileDir
    try {
        if (Test-Path -Path "data") {
            git update-index --assume-unchanged "data"
            Remove-Item -Recurse -Force "data"
        }
        else {
            Write-Host "File 'data' does not exist. Skipping."
        }
    }
    finally {
        Pop-Location
    }
}
