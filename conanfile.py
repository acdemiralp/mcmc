from   conans       import ConanFile, CMake, tools
from   conans.tools import download, unzip
import os

class Project(ConanFile):
    name            = "mcmc"
    version         = "1.0.1"					
    description     = "Conan package for mcmc."												 
    url             = "https://devhub.vr.rwth-aachen.de/VR-Group/mcmc"
    license         = "MIT"
    settings        = "arch", "build_type", "compiler", "os"
    generators      = "cmake"

    def source(self):
        zip_name = "%s.zip" % self.version
        download ("%s/-/archive/%s/%s" % (self.url, self.version, zip_name), zip_name, verify=False)
        unzip    (zip_name)
        os.unlink(zip_name)

    def package(self):
        include_folder = "%s-%s/include" % (self.name, self.version)
        self.copy("*.h"  , dst="include", src=include_folder)
        self.copy("*.hpp", dst="include", src=include_folder)
        self.copy("*.inl", dst="include", src=include_folder)
