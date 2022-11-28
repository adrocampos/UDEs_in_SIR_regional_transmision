## Environment and packages
cd(@__DIR__)
using Pkg;
Pkg.activate(".");
#;

Pkg.add("DiffEqFlux")
Pkg.add("DifferentialEquations")
Pkg.add("Optimization")
Pkg.add("OptimizationOptimJL")
Pkg.add("Random")
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("Lux")
Pkg.add("DataFrames")
Pkg.add("DataDrivenDiffEq")
Pkg.add("ModelingToolkit")
Pkg.add("LinearAlgebra")
Pkg.add("DiffEqSensitivity")
Pkg.add("Zygote")
Pkg.add("Optim")
Pkg.add("Flux")
Pkg.add("IJulia")

Pkg.instantiate()
Pkg.status()
Pkg.up()