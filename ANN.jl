using Pkg
using OrdinaryDiffEq
using Optim
using CSV
using DataFrames
using DifferentialEquations
using Random
using LinearAlgebra
using Dates
using DataDrivenDiffEq
using SciMLBase
using Plots
using JLD
using Interpolations
using Distributions
using ModelingToolkit
using Lux
using Optimization
using OptimizationFlux
using OptimizationOptimisers
using DiffEqFlux
using Flux
using ComponentArrays
using MPI
using Tables


## MPI initialization
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
println("I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
MPI.Barrier(comm)


## for plotting
gr()
ENV["GKSwstype"]="nul"

## parameters

for init = 1:1

    target_region = rank + 1
    n_regions = 10

    mobility = ["border", "inv_dist", "neighbor"][2]

    ## Loading data
    data_dir = "C:\\Users\\Adrian Rojas Campos\\Documents\\covid19\\synth_data/"
    file_name = "SIR_" * string(n_regions) * "_regions_" * mobility * "_" * string(init) * ".csv"
    println("init = ", init, " Input file = ", data_dir * file_name)

    csv_reader = CSV.File(data_dir * file_name, types=Float64) 
    df = DataFrame(csv_reader)
    X = Matrix(df)[:,2:end] ## Filtering out t
    mask = (1:10:5001) ## Defines resolution of input data
    X = X[mask, :]'

    ## Selecting training and testing sets
    tspan_train = (1, 251)
    tspan_test  = (1, 501)


    ####################################################################
    ############################## Set up ##############################
    ####################################################################

    println("\n ###### Target region = ", string(target_region), " ###### \n")

    folder_name = "SIR_" * string(n_regions) * "_regions_" * mobility * "/"
    save_dir = folder_name * "init_" * lpad(init,2,"0")  * "/region_" * lpad(target_region,2,"0") * "/"
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


    #####################################################################
    ############################# ANN Model #############################
    #####################################################################

    println("init = ", init, " ###### ANN Model ######")

    ann = FastChain(FastDense(n_regions*3,16,tanh), FastDense(16,3))

    p = Float64.(initial_params(ann))

    function dudt!(du, u, p, t)
        
        adjacent_SIR = interpolation_adjacent[:,t]
        z = ann([u./sum(u); adjacent_SIR./sum(adjacent_SIR)], p)

        du[1] = z[1]
        du[2] = z[2]
        du[3] = z[3]
    end

    prob_ANN = ODEProblem(dudt!, u0, tspan_train, p) 

    function predict(p)
        Array(concrete_solve(prob_ANN, Vern7(), u0=u0, p=p,
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


    res_ANN_1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.1), cb=callback, maxiters=500)
    save(save_dir * "2_ANN_params_1.jld", "ANN_params", Array(res_ANN_1))

    res_ANN_2 = DiffEqFlux.sciml_train(loss, res_ANN_1.minimizer, ADAM(0.01), cb=callback, maxiters=500)
    save(save_dir * "2_ANN_params_2.jld", "ANN_params", Array(res_ANN_2))


    pred_ANN = Array(concrete_solve(prob_ANN, Vern7(), u0=u0, p=res_ANN,
                    tspan=tspan_test, saveat=1, abstol=1e-6, reltol=1e-6,
                    sensealg=ForwardDiffSensitivity()
                ))

    ## Saving results
    plt = plot(pred_ANN', labels=["ANN pred S" "ANN pred I" "ANN pred R"], lw=2, ls=:dot, palette=:Dark2_3)
    plt = plot!(x_test', labels=["S" "I" "R"], lw=1, palette=:Dark2_3, linealpha=.5)
    plt = plot!(x_train', labels=["Train S" " Train I" "Train R"], lw=3, palette = :Dark2_3)
    plt = plot!(legend=:right)

    savefig(plt, save_dir * "2_ANN_pred.pdf")
    save(save_dir * "2_ANN_pred.jld", "ANN_pred", Array(pred_ANN))
    CSV.write(save_dir * "2_ANN_pred.csv", Tables.table(pred_ANN))
    savefig(plot(losses, labels=["ANN Loss"]), save_dir * "2_ANN_loss.pdf")
    save(save_dir * "2_ANN_loss.jld", "ANN_loss", losses)
    CSV.write(save_dir * "2_ANN_loss.csv", Tables.table(losses))
    save(save_dir * "2_ANN_params.jld", "ANN_params", Array(res_ANN))
    CSV.write(save_dir * "2_ANN_params.csv", Tables.table(res_ANN))

    println("init = ", init, " ANN model done!")

    end