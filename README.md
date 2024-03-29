# Dykes2DModel
[![Build Status](https://img.shields.io/badge/build-notready-red.svg)](https://travis-ci.org/gitpoint/git-point)[![Current Version](https://img.shields.io/badge/version-0.0.1-green.svg)](https://github.com/IgorAntun/node-chat)[![GitHub Issues](https://img.shields.io/badge/issues-7-red.svg)](https://github.com/Sorokin-MA/Dykes2DModel/issues) [![All Contributors](https://img.shields.io/badge/all_contributors-1-blue.svg)](./CONTRIBUTORS.md)[![Coveralls](https://img.shields.io/coveralls/github/gitpoint/git-point.svg)](https://github.com/Sorokin-MA/Dykes2DModel)
## Description

A project aimed at numerical modeling of the formation of dikes and silos. The code is based on the [code](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JB023008) of Ivan Utkin and takes into account the effect of plasticity.

## Getting Started

### Dependencies

CUDA.jl
HDF5.jl
Random.jl

### Executing program
```
julia
using Pkg; Pkg.activate("")
include("run.jl")
main()
```

## References
Melnik, O. E., Utkin, I. S., & Bindeman, I. N. (2021). Magma chamber formation by dike accretion and crustal melting: 2D thermo-compositional model with emphasis on eruptions and implication for zircon records. Journal of Geophysical Research: Solid Earth, 126, e2021JB023008. 
https://doi.org/10.1029/2021JB023008

