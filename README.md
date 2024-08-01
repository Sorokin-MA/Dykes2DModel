<div align="center">

# Dykes2DModel

[![GitHub tag](https://img.shields.io/github/v/release/Sorokin-MA/Dykes2DModel)](https://github.com/Sorokin-MA/Dykes2DModel/releases/latest) ![last commit](https://img.shields.io/github/last-commit/Sorokin-MA/Dykes2DModel) [![License](https://img.shields.io/github/license/Sorokin-MA/Dykes2DModel)](https://github.com/Sorokin-MA/Dykes2DModel/blob/main/LICENSE) 



</div>

A project aimed at numerical modeling of the formation of dikes and silos. The code is based on the [code](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JB023008) of Ivan Utkin and takes into account the effect of plasticity.

## Quick start
```
julia
using Pkg; Pkg.activate("");Pkg.instantiate();
include("run.jl")
dikes_gui()
```

## References
Melnik, O. E., Utkin, I. S., & Bindeman, I. N. (2021). Magma chamber formation by dike accretion and crustal melting: 2D thermo-compositional model with emphasis on eruptions and implication for zircon records. Journal of Geophysical Research: Solid Earth, 126, e2021JB023008. 
https://doi.org/10.1029/2021JB023008

