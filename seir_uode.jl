# ## Environment and packages
# cd(@__DIR__)
using Pkg;
# Pkg.activate(".");

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux

using Random
using CSV
using Lux
using DataFrames



Pkg.status()



data_dir = "/Users/adrocampos/covid19/synth_data/"
regions = ["2", "3", "5", "10", "15", "20", "30"][1]
mobility_type = ["inv_dist", "border", "neighbor"][2]
initially_recovered = false

# positions = CSV.File(data_dir * "positions_" * regions * "_regions.csv")
# positions = DataFrame(positions)

file = "1"

csv_reader = CSV.File(data_dir * "SIR_" * regions * "_regions_" * mobility_type * "_" * file * ".csv")
df = DataFrame(csv_reader)


index = range(1, stop=5001, step=50)
df = df[index, :]

X = Matrix(df[:, [:S1, :I1, :R1]])'
t = df.t

tspan = (t[begin], t[end])
tsteps = range(tspan[1], tspan[2], length=size(t)[1])

u0 = X[:, 1]


ann = FastChain(
    FastDense(3, 64, tanh), FastDense(64, 64, tanh), FastDense(64, 1)
)

p = initial_params(ann)


# rng = Random.default_rng()
# ann = Lux.Chain(Lux.Dense(3, 50, tanh),
#     Lux.Dense(50, 50, tanh),
#     Lux.Dense(50, 1))

# p, st = Lux.setup(rng, ann)

# p = [rand(Float32, 2); p]

## p here and p outside? Overwrite?

p_ = [0.1, 0.1, 2000]

function dudt!(du, u, p, t)
    
    S, I, R = u
    β, γ, N  = p_

    z  = ann(u, p)
    dS = -β * S * I/N - z[1]  # susceptible
    dI =  β * S * I/N #- γ*I - z[1] # infected
    dR =  γ * I

    print("z =", z, "u =", u)

    du[1] = dS
    du[2] = dI
    du[3] = dR

end




# # Define the problem
prob_UODE = ODEProblem(dudt!, u0, tspan, p) ##prob_neuralode
# # prob_UODE = NeuralODE(dudt!, tspan, Tsit5(),saveat=tsteps) ##prob_neuralode




## Function to train the network
# Define a predictor
function predict(θ, X=X[:, 1], T=t)
    Array(solve(prob_UODE, Vern7(), u0=X, p=θ,
        tspan=(T[1], T[end]), saveat=T,
        abstol=1e-6, reltol=1e-6,
        sensealg=ForwardDiffSensitivity()
    ))
end


## .- instead of -?
function loss(θ)
    X̂ = predict(θ)
    sum(abs2, X .- X̂) #/ size(X, 2) + convert(eltype(θ), 1e-3) * sum(abs2, θ[3:end]) ./ length(θ[3:end])
end

# Container to track the losses
losses = Float32[]

# Callback to show the loss during training
callback(θ, args...) = begin
    l = loss(θ) # Equivalent L2 loss
    println("l = ", l)
    push!(losses, l)
    # if length(losses) % 5 == 0
    #     println("Current loss after $(length(losses)) iterations: $(losses[end])")
    # end
    false
end

# # use Optimization.jl to solve the problem

# # pinit = Lux.ComponentArray(p)
# # adtype = Optimization.AutoZygote()
# # optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
# # optprob = Optimization.OptimizationProblem(optf, pinit)

# # result_neuralode = Optimization.solve(optprob,
# #     ADAM(0.01),
# #     # callback=callback,
# #     maxiters=1)

res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.001),  maxiters=1)



## Training -> First shooting / batching to get a rough estimate

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
# res1 = DiffEqFlux.sciml_train(shooting_loss, p, ADAM(0.1f0), cb=callback, maxiters=100)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")
# # Train with BFGS to achieve partial fit of the data
# res2 = DiffEqFlux.sciml_train(shooting_loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters=500)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")
# # Full L2-Loss for full prediction
# res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters=100)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")



# function predict(θ, X=X[:, 1], T=t)#     Array(solve(prob_UODE, Vern7(), u0=X, p=θ,
#         tspan=(T[1], T[end]), saveat=T,
#         abstol=1e-6, reltol=1e-6,
#         sensealg=ForwardDiffSensitivity()
#     ))
# end

# function loss(θ)
#     X̂ = predict(θ)
#     sum(abs2, X - X̂) / size(X, 2) + convert(eltype(θ), 1e-3) * sum(abs2, θ[3:end]) ./ length(θ[3:end])
# end






# # Container to track the losses
# losses = Float32[]

# # Callback to show the loss during training
# callback(θ, args...) = begin
#     l = loss(θ) # Equivalent L2 loss
#     push!(losses, l)
#     if length(losses) % 5 == 0
#         println("Current loss after $(length(losses)) iterations: $(losses[end])")
#     end
#     false
# end









# res_fit = DiffEqFlux.sciml_train(loss, p, BFGS(initial_stepnorm=0.1f0), cb=callback, maxiters=2)
# ## Training -> First shooting / batching to get a rough estimate

# # First train with ADAM for better convergence -> move the parameters into a
# # favourable starting positing for BFGS


# # res1 = DiffEqFlux.sciml_train(shooting_loss, p, Adam(0.1f0), cb=callback, maxiters=1)
# # println("Training loss after $(length(losses)) iterations: $(losses[end])")
# # Train with BFGS to achieve partial fit of the data
# # res2 = DiffEqFlux.sciml_train(shooting_loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 500)
# # println("Training loss after $(length(losses)) iterations: $(losses[end])")
# # # Full L2-Loss for full prediction
# # res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000)
# # println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# # use Optimization.jl to solve the problem
# # adtype = Optimization.AutoZygote()

# optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
# optprob = Optimization.OptimizationProblem(optf, p)

# result_neuralode = Optimization.solve(optprob, ADAM(0.005), tspan=tspan, saveat=t, maxiters=10)
