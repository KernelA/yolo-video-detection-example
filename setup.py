import os

from setuptools import Extension, find_packages, setup
import numpy as np
from Cython.Build import cythonize

PACKAGE_NAME = "yolo_models"


def get_version():
    version = None

    path_to_file = os.path.join(PACKAGE_NAME, "__init__.py")

    with open(path_to_file, "r", encoding="utf-8") as f:
        for line in map(str.strip, f):
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().replace("\"", "")
                break
        else:
            raise RuntimeError(f"Cannot find __version__ variables in the '{path_to_file}'")

    return version


def get_req(filepath: str):
    req_list = []

    with open(filepath, encoding="utf-8") as file:
        for line in map(str.strip, file):
            if line:
                if line.startswith("--extra-index-url"):
                    continue
                req_list.append(line)

    return req_list


exts = Extension(
    f"{PACKAGE_NAME}.nms.nms_native",
    sources=[f"{PACKAGE_NAME}/nms/*.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

setup(install_requires=get_req("requirements.inference.txt"),
      version=get_version(),
      packages=find_packages(include=[f"{PACKAGE_NAME}*"]),
      package_data={"": [f"{PACKAGE_NAME}/log_set/*.yaml"]},
      extras_require={
          "torch": get_req("requirements.torch.gpu.txt")},
      ext_modules=cythonize(exts,
                            compiler_directives={"language_level": "3"},
                            )
      )
