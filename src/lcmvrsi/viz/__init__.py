# Visualization helpers. matplotlib lives in the optional `viz` extra, so submodules import
# it lazily (inside functions) and nothing here imports it at package-import time -- keeping
# the core package usable, and CI green, without the viz dependencies installed.
