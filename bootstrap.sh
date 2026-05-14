#!/bin/bash

if [ ! -d "build" ] ; then mkdir build ; fi
cd build
if [ ! -d "vcpkg" ] ; then git clone https://github.com/microsoft/vcpkg.git ; fi
cd vcpkg
if [ ! -f "vcpkg" ] ; then ./bootstrap-vcpkg.sh ; fi

VCPKG_DEFAULT_TRIPLET=x64-linux
./vcpkg install --overlay-ports=../../vcpkg-overlay-ports doctest eigen3
cd ..

cmake -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DBUILD_TESTS=ON ..
cmake --build . --clean-first --target all --config Release --parallel 8
ctest --output-on-failure
cd ..
