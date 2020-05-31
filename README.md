<p align="center" style="text-align:center">
<img src="https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/hubble_emitting.jpg" alt="drawing" width="300"/>
</p>

# Ananke
Generalized C++ collocation optimization toolkit with Python bindings.

_Built with Python 3.8 in the Anaconda Python distribution._

Anaconda 64-bit Python Distribution is the package-manager used to download required libraries and headers, and is required to run AnankeC.

## Python Helpers
| Module | Description |
| - | - |
| `ananke.frames` | Frame conversion and transformation tools. |
| `ananke.orbit` | Keplerian orbital calculation tools. |
| `ananke.opt` | Optimization toolkit. |
| `ananke.planets` | Planetary information. |
| `ananke.util` | Generic math capability and utilities. |

## C++ Modules List
| Module | Description |
| - | - |
| `AnankeC` | Baseline module for trajectory leg/optimization confi structs |

# Setup

## High-Level requirements:
* Visual Studio 2019
* SNOPT (Windows 64-bit DLL)
* Anaconda 64-bit Python Distribution with requested packages listed below

## Required Anaconda Packages
The following should be installed with `conda install <toolkit>` in the Anaconda Prompt:
* `scipy`
* `numpy`
* `pygmo_plugins_nonfree`
* `pagmo`
* `pagmo-devel`
* `cspice`
* `pybind11`
* `eigen`

## Required Environment Variables
| Variable | Description |
| - | - |
| `CONDA_DIR` | Directory of the installed Anaconda environment with all of the requested packages. For example, if a new environment `ananke_env` was created, then this variable would be `<Anaconda-install-dir>/envs/ananke_env`. |
| `SNOPT_DLL` | Location of the SNOPT 64-bit Windows DLL on the computer. Please contact `jason.m.everett@nasa.got` for SNOPT issues. |




