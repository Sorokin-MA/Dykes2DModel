using Pkg

Pkg.activate("DEV_D2DM")
using Revise
Pkg.develop(path=".")
include("examples/d2dm_gui.jl")
