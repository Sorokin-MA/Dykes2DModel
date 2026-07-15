using Pkg

Pkg.activate("DEV_D2DM")
using Revise
Pkg.develop(path=".")
include("examples/D2DM_GUI_Main.jl")
include("examples/graphs.jl")
