vcpkg_buildpath_length_warning(37)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO eigenteam/eigen-git-mirror
    REF 3.3.7
    SHA512 270ab9b5c22e09aa0e70d1a26995523c5c21fb0f09da45c137c11ab4c7700fe2bdb2b343c1e063bea4be5ae61d2313ff29ebbcad519dc355a568792b4a6e9e48
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
        -DEIGEN_BUILD_BTL=OFF
        -DEIGEN_BUILD_PKGCONFIG=ON
    OPTIONS_RELEASE
        "-DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/share/${PORT}"
        "-DPKGCONFIG_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/lib/pkgconfig"
    OPTIONS_DEBUG
        "-DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/debug/share/${PORT}"
        "-DPKGCONFIG_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(
    FILE_LIST
        "${SOURCE_PATH}/COPYING.README"
        "${SOURCE_PATH}/COPYING.BSD"
        "${SOURCE_PATH}/COPYING.MINPACK"
        "${SOURCE_PATH}/COPYING.MPL2"
)
