from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

def get_extra_compile_args():
    if os.name == "nt":
        return {"msvc": ["/std:c++20", "/O2", "/DNDEBUG"]}
    else:
        return {"cxx": ["-std=c++20", "-O3", "-DNDEBUG"]}

setup(
    name="transf_in_cpp",
    ext_modules=[
        CppExtension(
            "transf_in_cpp",
            ["transf-in-cpp.cpp"],
            extra_compile_args=get_extra_compile_args(),
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
