using Pkg; Pkg.activate("/p/home/jusers/rojascampos1/juwels/MyProjects/covid19/env")

## from System
using OrdinaryDiffEq
using Optim
using CSV
using DataFrames

## from Environment
using DiffEqFlux
using Flux
using Random
using LinearAlgebra
using Dates
using ModelingToolkit
using DataDrivenDiffEq
using SciMLBase
using Plots
using JLD
using Interpolations
using Distributions
using MPI

## for plotting
gr()
ENV["GKSwstype"]="nul"

## MPI initialization
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
println("I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
MPI.Barrier(comm)


pairs = Tuple{Int, Int}[]

for r in 1:10 ## regions
    for i in 1:20 ## inits
        push!(pairs, (r,i))
    end
    for i in 51:70 ## inits
        push!(pairs, (r,i))
    end
end

target_region = pairs[rank+1][1]
init = pairs[rank+1][2]
println("rank = ", string(rank), " target_region = ", string(target_region), " init = ", string(init))


n_regions = 10

mobility = ["border", "inv_dist", "neighbor"][2]

## Loading data
data_dir = "/p/scratch/deepacf/deeprain/rojascampos1/data/synth_data/"
file_name = "SIR_" * string(n_regions) * "_regions_" * mobility * "_" * string(init) * ".csv"
println("init = ", init, " Input file = ", data_dir * file_name)

csv_reader = CSV.File(data_dir * file_name, types=Float64) 
df = DataFrame(csv_reader)
X = Matrix(df)[:,2:end] ## Filtering out t
mask = (1:10:5001) ## Defines resolution of input data
X = X[mask, :]'

## Selecting training and testing sets
tspan_train = (1, 201)
tspan_test  = (1, 501)

####################################################################
############################## Set up ##############################
####################################################################

println("###### Target region = ", string(target_region), " ######")

folder_name = "SIR_" * string(n_regions) * "_regions_" * mobility * "_" * string(tspan_train[2]) *  "/"
save_dir = "/p/project/deepacf/deeprain/rojascampos1/covid19/" * folder_name * "init_" * lpad(init,2,"0")  * "/region_" * lpad(target_region,2,"0") * "/"
println("save_dir = ", save_dir)
mkpath(save_dir)

universe = range(1, size(X)[1], step=1)
index_target = (target_region - 1) * 3 + 1
targets = [index_target, index_target+1, index_target+2]
adjacents = setdiff(universe , targets)

x_target = X[targets,:]
x_adjacent = X[adjacents,:]

x_test = x_target[:,1:tspan_test[2]]
x_train = x_target[:,1:tspan_train[2]]

u0 = x_train[:, 1]
println("init = ", init, " u0 = ", u0)

plt = plot(x_train', labels=["Train S" "Train I" "Train R"], lw=3, palette=:Dark2_3)
plt = plot!(x_test', labels=["Real S" "Real I" "Real R"], lw=1, palette=:Dark2_3)
plt = plot!(legend=:right)
savefig(plt, save_dir * "0_x_train.pdf")
savefig(plot(x_adjacent'), save_dir * "0_x_adjacent.pdf")

## Linear interpolation of the SIR model of adjacent region
interpolation_adjacent = interpolate(Array(x_adjacent), BSpline(Linear()))


####################################################################
############################ ODE Model #############################
####################################################################

println("###### ODE Model ######")

p = rand(Float64, 2)

function ODE(du, u, p, t)
    
    β = p[1]
    γ = p[2]
    
    S, I, R = u
    N = sum(u)

    du[1] = (-β * I / N) * S
    du[2] =  (β * I / N) * S - (γ * I)
    du[3] =  (γ * I)      
end

prob_ODE = ODEProblem(ODE, u0, tspan_train, p) 

    function predict(p)
    Array(concrete_solve(prob_ODE, Vern7(), u0=u0, p=p,
        tspan=tspan_train, saveat=1, abstol=1e-6, reltol=1e-6,
        sensealg=ForwardDiffSensitivity()
    ))
end

function loss(p)
    X̂ = predict(p)
    sum(abs2, x_train .- X̂) / size(x_train, 2)
end

losses = Float64[]
callback(θ, l) = begin
    l = loss(θ)
    push!(losses, l)
    if length(losses)%100==0
        println("init = ", string(init), "region = ",string(target_region), " Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
    
end

initial_loss = loss(p)
println("init = ", init, " Loss before training = ", initial_loss)


try
    res_ODE = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters=1_000)
    save(save_dir * "1_ODE_params.jld", "ODE_params", Array(res_ODE))

    pred_ODE = Array(concrete_solve(prob_ODE, Vern7(), u0=u0, p=res_ODE,
                tspan=tspan_test, saveat=1, abstol=1e-6, reltol=1e-6,
                sensealg=ForwardDiffSensitivity()
            ))

    ## Saving results
    plt = plot(pred_ODE', labels=["ODE pred S" "ODE pred I" "ODE pred R"], lw=2, ls=:dot, palette=:Dark2_3)
    plt = plot!(x_test', labels=["S" "I" "R"], lw=1, palette=:Dark2_3, linealpha=.5)
    plt = plot!(x_train', labels=["Train S" " Train I" "Train R"], lw=3, palette = :Dark2_3)
    plt = plot!(legend=:right)
    savefig(plt, save_dir * "1_ODE_pred.pdf")
    save(save_dir * "1_ODE_pred.jld", "ODE_pred", Array(pred_ODE))
    CSV.write(save_dir * "1_ODE_pred.csv", Tables.table(pred_ODE))

    savefig(plot(losses, labels=["ODE Loss"]), save_dir * "1_ODE_loss.pdf")
    CSV.write(save_dir * "1_ANN_loss.csv", Tables.table(losses))
    save(save_dir * "1_ODE_loss.jld", "ODE_loss", losses)
    println("init = ", init, " target region = ", target_region, " training completed with loss = ", string(losses[end]))

catch e
    println("init = ", init, " target region = ", target_region, " INTERRUPTED.", e)

end 

