# MoE Cache Engine Windows Compilation Script

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Compiling MoE Cache Engine..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Ensure output binary and build directories exist
$OutputDir = "bin"
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Compiler = ""
$CompileCmd = ""

# 1. Check for local w64devkit g++
$LocalGpp = "$Home\w64devkit\bin\g++.exe"
if (Test-Path $LocalGpp) {
    $Compiler = "Local g++ (w64devkit)"
    $env:PATH = "$Home\w64devkit\bin;" + $env:PATH
    $CompileCmd = "& `"$LocalGpp`" -O3 -std=c++17 src/tensor.cpp src/cache.cpp src/ops.cpp src/model.cpp src/models/deepseek_v4.cpp src/models/glm_5_2.cpp src/models/kimi_k3.cpp src/tokenizer.cpp src/main.cpp -o bin/moe_cache_test.exe"
}
# 2. Check for system g++ (MinGW)
elseif (Get-Command "g++" -ErrorAction SilentlyContinue) {
    $Compiler = "g++"
    $CompileCmd = "g++ -O3 -std=c++17 src/tensor.cpp src/cache.cpp src/ops.cpp src/model.cpp src/models/deepseek_v4.cpp src/models/glm_5_2.cpp src/models/kimi_k3.cpp src/tokenizer.cpp src/main.cpp -o bin/moe_cache_test.exe"
}
# 3. Check for MSVC cl
elseif (Get-Command "cl" -ErrorAction SilentlyContinue) {
    $Compiler = "cl"
    $CompileCmd = "cl /EHsc /O2 /std:c++17 src/tensor.cpp src/cache.cpp src/ops.cpp src/model.cpp src/models/deepseek_v4.cpp src/models/glm_5_2.cpp src/models/kimi_k3.cpp src/tokenizer.cpp src/main.cpp /Febin/moe_cache_test.exe"
}

if ($Compiler -eq "") {
    Write-Error "No C++ compiler found. Please install GCC (MinGW) or MSVC C++ Build Tools."
    Exit 1
}

Write-Host "Detected Compiler: $Compiler" -ForegroundColor Green
Write-Host "Running Command: $CompileCmd" -ForegroundColor Yellow

# Execute compilation
Invoke-Expression $CompileCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host " Compilation Successful!" -ForegroundColor Green
    Write-Host " Binary generated at: bin/moe_cache_test.exe" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
} else {
    Write-Error "Compilation Failed. Review the compiler output above."
}
