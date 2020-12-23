from setuptools import setup, find_packages
import os

pkg_name = "pixace"
pkg_url = "https://github.com/vishnubob/pixace"
pkg_path = os.path.split(__file__)[0]
author = "Giles Hall"
author_email = "giles@polymerae.org"

def get_requirements():
    with open('requirements.txt') as fh:
        required = [req for req in fh if not (req.startswith('#') or not req.strip())]
    return required

def get_package_version():
    with open('VERSION') as fh:
        version = fh.read()
    return version.strip()

def run_setup():
    pkg_version = get_package_version()

    config = {
        "name": pkg_name,
        "version": pkg_version,
        "packages": find_packages(),
        "install_requires": get_requirements(),
        "entry_points": {
            "console_scripts": [
                "pixace = pixace.main:cli",
            ],
        },
        "author": author,
        "author_email": author_email,
        "url": pkg_url,
    }
    setup(**config)

if __name__ == "__main__":
    run_setup()
