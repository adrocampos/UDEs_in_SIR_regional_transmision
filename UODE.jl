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

for target_region = 1:1

    init = rank + 1
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
    ############################ UODE Model #############################
    #####################################################################

    println("\n ###### UODE Model ###### \n")

    ann = DiffEqFlux.FastChain(FastDense((n_regions-1)*3,16,tanh), FastDense(16,1,softplus))
    p_init = [rand(Float64, 2); Float64.(initial_params(ann))]


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

    prob_UODE = ODEProblem(dudt!, u0, tspan_train, p_init) 


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
        l = loss(θ)[1]
        push!(losses, l)
        if length(losses)%10==0
            println("init = ", string(init), " region = ",string(target_region), " Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        false
        
    end

    initial_loss = loss(p_init)
    println("init = ", init, " Loss before training = ", initial_loss)


    try

        res_UODE_1 = DiffEqFlux.sciml_train(loss, p_init, ADAM(0.01), callback=callback, maxiters=300)
        save(save_dir * "3_UODE_params_1.jld", "UODE_params", Array(res_UODE_1))

        res_UODE_2 = DiffEqFlux.sciml_train(loss, res_UODE_1.minimizer, BFGS(initial_stepnorm=0.01f0), callback=callback, maxiters=300)
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
        CSV.write(save_dir * "3_UODE_pred.csv", Tables.table(pred_UODE))

        savefig(plot(losses, labels=["UODE Loss"]), save_dir * "3_UODE_loss.pdf")
        save(save_dir * "3_UODE_loss.jld", "UODE_loss", losses)
        CSV.write(save_dir * "3_UODE_loss.csv", Tables.table(losses))
        
        println("init = ", init, " target region = ", target_region, " training completed with loss = ", string(losses[end]))

    catch e

        println("\n init = ", init, "target region = ", target_region, " INTERRUPTED. \n")
        println(e)

    end 


end