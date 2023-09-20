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

# pairs = load("/p/home/jusers/rojascampos1/juwels/MyProjects/covid19/SIR_discovery/UODE_interrupted.jld", "UODE_interrupted")

target_region = pairs[rank+1][1]
init = pairs[rank+1][2]
id = "rank = " * string(rank) * " target_region = " * string(target_region) * " init = " * string(init)

## parameters
n_regions = 10
mobility = ["border", "inv_dist", "neighbor"][2]

## Loading data
data_dir = "/p/scratch/deepacf/deeprain/rojascampos1/data/synth_data/"
file_name = "SIR_" * string(n_regions) * "_regions_" * mobility * "_" * string(init) * ".csv"
println("Input file = ", data_dir * file_name)

csv_reader = CSV.File(data_dir * file_name, types=Float64) 
df = DataFrame(csv_reader)
X = Matrix(df)[:,2:end] ## Filtering out t
mask = (1:10:5001) ## Defines resolution of input data
X = X[mask, :]'

## Selecting training and testing sets
tspan_train = (1, 201)
tspan_test  = (1, 501)


println("###### Target region = ", string(target_region),  "######")

####################################################################
############################## Set up ##############################
####################################################################

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
println(id, " u0 = ", u0)

plt = plot(x_train', labels=["Train S" "Train I" "Train R"], lw=3, palette=:Dark2_3)
plt = plot!(x_test', labels=["Real S" "Real I" "Real R"], lw=1, palette=:Dark2_3)
plt = plot!(legend=:right)
savefig(plt, save_dir * "0_x_train.pdf")
savefig(plot(x_adjacent'), save_dir * "0_x_adjacent.pdf")

## Linear interpolation of the SIR model of adjacent region
interpolation_adjacent = interpolate(Array(x_adjacent), BSpline(Linear()))


#####################################################################
############################ UODE Model #############################
#####################################################################

println("###### UODE Model ######")

ann = FastChain(FastDense((n_regions-1)*3,16,tanh; initW=Flux.glorot_normal), FastDense(16,1,softplus,initW=Flux.kaiming_normal))
p = [rand(Float64, 2); Float64.(initial_params(ann))]

function dudt!(du, u, p, t)
    
    β = p[1]
    γ = p[2]

    S, I, R = u
    N = sum(u)

    adjacent_SIR = interpolation_adjacent[:,t]        
    adjacent_SIR = adjacent_SIR./sum(adjacent_SIR)
    z = ann(adjacent_SIR, p[3:end])

    du[1] = (-β * I / N) * S - z[1]
    du[2] =  (β * I / N) * S + z[1] - (γ * I)
    du[3] =  (γ * I)      
end

prob_UODE = ODEProblem(dudt!, u0, tspan_train, p) 

function predict(θ)
    Array(concrete_solve(prob_UODE, Vern7(), u0=u0, p=θ,
        tspan=tspan_train, saveat=1, abstol=1e-6, reltol=1e-6,
        sensealg=ForwardDiffSensitivity()
    ))
end

function loss(θ)
    X̂ = predict(θ)
    sum(abs2, x_train .- X̂) / size(x_train, 2)
end

losses = Float64[]
callback(θ, l) = begin
    l = loss(θ)
    push!(losses, l)
    if length(losses)%500==0
        println(id, " Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

initial_loss = loss(p)
println(id, " Loss before training = ", initial_loss)


try
    res_UODE_1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=20_000) 
    save(save_dir * "3_UODE_params_1.jld", "UODE_params", Array(res_UODE_1))

    res_UODE_2 = DiffEqFlux.sciml_train(loss, res_UODE_1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters=2_000) ##2_000 
    save(save_dir * "3_UODE_params_2.jld", "UODE_params", Array(res_UODE_2))

    pred_UODE = Array(concrete_solve(prob_UODE, Vern7(), u0=u0, p=res_UODE_2,
                    tspan=tspan_test, saveat=1, abstol=1e-6, reltol=1e-6,
                    sensealg=ForwardDiffSensitivity()
                ))

    ## Saving results
    plt = plot(pred_UODE', labels=["UODE pred S" "UODE pred I" "UODE pred R"], lw=2, ls=:dot, palette=:Dark2_3)
    plt = plot!(x_test', labels=["S" "I" "R"], lw=1, palette=:Dark2_3, linealpha=.5)
    plt = plot!(x_train', labels=["Train S" " Train I" "Train R"], lw=3, palette = :Dark2_3)
    plt = plot!(legend=:right)

    savefig(plt, save_dir * "3_UODE_pred.pdf")
    save(save_dir * "3_UODE_pred.jld", "UODE_pred", Array(pred_UODE))
    savefig(plot(losses, labels=["UODE Loss"]), save_dir * "3_UODE_loss.pdf")
    save(save_dir * "3_UODE_loss.jld", "UODE_loss", losses)
    println(id, " training completed with loss = ", string(losses[end]))

catch e
    println(id, " INTERRUPTED.")
    println(e)

end 

