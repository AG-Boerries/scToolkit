from setuptools import setup
from setuptools import find_packages

import re
import subprocess
# 'scToolkit' :["databases/**/*"]},
setup(
    packages=find_packages('src', exclude=['docs', 'tests', 'examples']),
    package_dir={'': 'src'},
    package_data={
            "scToolkit": [
            "databases/**/*",      # subfolders
        ]},
    include_package_data=True)

