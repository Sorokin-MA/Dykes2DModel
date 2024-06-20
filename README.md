# Dykes2DModel

> **Warning**
>
> The code is in the pre-alpha phase, bugs and errors should be expected.

## Description

A project aimed at numerical modeling of the formation of dikes and silos. The code is based on the [code](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JB023008) of Ivan Utkin and takes into account the effect of plasticity.

## Getting Started

### Executing program
```
julia
using Pkg; Pkg.activate("");Pkg.instantiate();
include("run.jl")
main()
```

## References
Melnik, O. E., Utkin, I. S., & Bindeman, I. N. (2021). Magma chamber formation by dike accretion and crustal melting: 2D thermo-compositional model with emphasis on eruptions and implication for zircon records. Journal of Geophysical Research: Solid Earth, 126, e2021JB023008. 
https://doi.org/10.1029/2021JB023008

