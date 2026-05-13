Package: eigen3:x64-linux@3.3.7

**Host Environment**

- Host: x64-linux
- Compiler: GNU 13.3.0
- CMake Version: 4.3.2
-    vcpkg-tool version: 2026-04-08-e0612b42ce44e55a0e630f2ee9d3c533a63d8bc1
    vcpkg-scripts version: ba27a182ef 2026-05-13 (13 minutes ago)

**To Reproduce**

`vcpkg install `

**Failure logs**

```
Downloading https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz -> eigenteam-eigen-git-mirror-3.3.7.tar.gz
Successfully downloaded eigenteam-eigen-git-mirror-3.3.7.tar.gz
-- Extracting source /home/runner/work/mcmc/mcmc/build/vcpkg/downloads/eigenteam-eigen-git-mirror-3.3.7.tar.gz
-- Using source at /home/runner/work/mcmc/mcmc/build/vcpkg/buildtrees/eigen3/src/3.3.7-c0bebf7460.clean
-- Configuring x64-linux
CMake Warning at /home/runner/work/mcmc/mcmc/vcpkg_installed/x64-linux/share/vcpkg-cmake/vcpkg_cmake_configure.cmake:344 (message):
  The following variables are not used in CMakeLists.txt:

      EIGEN_BUILD_BLAS
      EIGEN_BUILD_CMAKE_PACKAGE
      EIGEN_BUILD_DEMOS
      EIGEN_BUILD_DOC
      EIGEN_BUILD_LAPACK
      EIGEN_BUILD_SPBENCH

  Please recheck them and remove the unnecessary options from the
  `vcpkg_cmake_configure` call.

  If these options should still be passed for whatever reason, please use the
  `MAYBE_UNUSED_VARIABLES` argument.
Call Stack (most recent call first):
  /home/runner/work/mcmc/mcmc/vcpkg-overlay-ports/eigen3/portfile.cmake:11 (vcpkg_cmake_configure)
  scripts/ports.cmake:206 (include)


-- Building x64-linux-dbg
-- Building x64-linux-rel
-- Fixing pkgconfig file: /home/runner/work/mcmc/mcmc/build/vcpkg/packages/eigen3_x64-linux/lib/pkgconfig/eigen3.pc
-- Fixing pkgconfig file: /home/runner/work/mcmc/mcmc/build/vcpkg/packages/eigen3_x64-linux/debug/lib/pkgconfig/eigen3.pc
CMake Error at scripts/cmake/vcpkg_install_copyright.cmake:27 (message):


  vcpkg_install_copyright was passed a non-existing path:
  /home/runner/work/mcmc/mcmc/build/vcpkg/buildtrees/eigen3/src/3.3.7-c0bebf7460.clean/COPYING.APACHE

Call Stack (most recent call first):
  /home/runner/work/mcmc/mcmc/vcpkg-overlay-ports/eigen3/portfile.cmake:37 (vcpkg_install_copyright)
  scripts/ports.cmake:206 (include)



```

**Additional context**

<details><summary>vcpkg.json</summary>

```
{
  "name": "mcmc",
  "version-string": "1.0.0",
  "dependencies": [
    "doctest",
    "eigen3"
  ]
}

```
</details>
