from setuptools import Extension, find_packages, setup
import numpy as np
from Cython.Build import cythonize


def get_req(filepath: str):
    req_list = []

    with open(filepath, encoding="utf-8") as file:
        for line in map(str.strip, file):
            if line:
                if line.startswith("--extra-index-url"):
                    continue
                req_list.append(line)

    return req_list


PACKAGE_NAME = "yolo_models"

exts = Extension(
    f"{PACKAGE_NAME}.nms.nms_native",
    sources=[f"{PACKAGE_NAME}/nms/*.pyx"],
    include_dirs=[np.get_include()],
)

setup(install_requires=get_req("requirements.inference.txt"),
      packages=find_packages(include=[f"{PACKAGE_NAME}*"]),
      extras_require={
          "torch": get_req("requirements.torch.gpu.txt")
},
    ext_modules=cythonize(exts,
                          compiler_directives={"language_level": "3"},
                          )
)
