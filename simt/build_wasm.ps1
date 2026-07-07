# Build the mini-GPU engine to WebAssembly for the browser playground ($0, Emscripten).
# Produces web/simt.js + web/simt.wasm from the REAL simt_core sources.
# Usage:  powershell -File build_wasm.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:EM_CONFIG = "C:\emsdk\.emscripten"
$emcc = "C:\emsdk\upstream\emscripten\emcc.exe"

New-Item -ItemType Directory -Force -Path (Join-Path $here "web") | Out-Null

& $emcc -O2 -std=c++17 -fexceptions "-I$here/include" `
    "$here/src/isa.cpp" "$here/src/assembler.cpp" "$here/src/core.cpp" "$here/src/wasm_api.cpp" `
    -s MODULARIZE=1 -s "EXPORT_NAME=SimtModule" `
    -s "EXPORTED_FUNCTIONS=['_run_kernel','_malloc','_free']" `
    -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']" `
    -s INITIAL_MEMORY=67108864 `
    -o "$here/web/simt.js"

Write-Output "built web/simt.js + web/simt.wasm"
