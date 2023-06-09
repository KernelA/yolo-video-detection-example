from setuptools import setup, find_packages

req_list = []

with open("requirements.txt", encoding="utf-8") as file:
    for line in map(str.strip, file):
        if line:
            req_list.append(line)

setup(install_requires=req_list,
      packages=find_packages(["video_detection*"])
      )
