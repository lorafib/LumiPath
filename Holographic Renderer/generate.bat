@echo off

if "%~1"=="-r" del build_uwp\CMakeCache.txt
if "%~1"=="-d" rd /s /q build_uwp
mkdir build_uwp
cd build_uwp
cmake -G "Visual Studio 15 2017" -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=10.0.17134.0 ..
cd ..

if "%~1"=="-r" del build_desktop\CMakeCache.txt
if "%~1"=="-d" rd /s /q build_desktop
mkdir build_desktop
cd build_desktop
cmake -G "Visual Studio 15 2017 Win64" ..
cd ..
