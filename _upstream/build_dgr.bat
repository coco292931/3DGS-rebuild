@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 1>nul
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "TORCH_EXTENSIONS_DIR=C:\Users\tonyp\Downloads\3dgs\_upstream\_ext"
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
cd /d C:\Users\tonyp\Downloads\3dgs\_upstream\diff-gaussian-rasterization-main
echo --- nvcc ---
"%CUDA_HOME%\bin\nvcc.exe" --version | findstr release
echo --- building ---
python setup.py build_ext --inplace
echo BUILD_EXIT=%errorlevel%
