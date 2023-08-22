import numpy as np
from setuptools import Extension, find_packages, setup, find_namespace_packages
from Cython.Build import cythonize

req_list = []

with open("requirements.inference.txt", encoding="utf-8") as file:
    for line in map(str.strip, file):
        if line:
            req_list.append(line)

PACKAGE_NAME = "yolo_models"

exts = Extension(
    f"{PACKAGE_NAME}.nms.nms_native",
    sources=[f"{PACKAGE_NAME}/nms/*.pyx"],
    include_dirs=[np.get_include()],
)

setup(install_requires=req_list,
      packages=find_packages(include=[f"{PACKAGE_NAME}*"]),
      ext_modules=cythonize(exts, compiler_directives={"language_level": "3"},
                            )
      )
