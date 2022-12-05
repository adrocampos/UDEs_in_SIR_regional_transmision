# ## Environment and packages
cd(@__DIR__)
using Pkg;
Pkg.activate(".");

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux
using Random
using CSV
using Lux
using DataFrames
using JLD
using Plots
gr()

data_dir = "/Users/adrocampos/covid19/synth_data/"
regions = ["2", "3", "5", "10", "15", "20", "30"][1]
mobility_type = ["inv_dist", "border", "neighbor"][2]
initially_recovered = false


file = "1"

csv_reader = CSV.File(data_dir * "SIR_" * regions * "_regions_" * mobility_type * "_" * file * ".csv")
df = DataFrame(csv_reader)


index = range(1, stop=5001, step=200)
df = df[index, :]

X = Matrix(df[:, [:S1, :I1, :R1]])
X = transpose(X)
println("size(X) =", size(X))



tspan = (1, size(X)[2])

t = range(tspan[1], tspan[2], step=1)

u0 = X[:, 1]

# ann = FastChain(
#     FastDense(3, 32, tanh), FastDense(32, 32, tanh), FastDense(32, 1)
# )

# p = Float64.(initial_params(ann))


rng = Random.default_rng()
ann = Lux.Chain(Lux.Dense(3, 32, NNlib.tanh), 
                Lux.Dense(32, 32, NNlib.tanh), 
                Lux.Dense(32, 1))

# Initialize the model weights and state
p, st = Lux.setup(rng, ann)


function dudt!(du, u, p, t)

    N = sum(u)
    S, I, R = u
    β = 0.1
    γ = 1 / 10

    z = ann(u, p)

    du[1] = dS = -β * S * I / N - z[1]
    du[2] = dI = β * S * I / N + z[1] - γ * I
    du[3] = dR = γ * I

end




## Define the problem
## Prediction with initial random weights
prob_UODE = ODEProblem(dudt!, u0, tspan, p) ##prob_neuralode
s = concrete_solve(prob_UODE, Tsit5(), u0, p, saveat=t)

## Function to train the network
# Define a predictor
function predict(θ)
    Array(concrete_solve(prob_UODE, Vern7(), u0=u0, p=θ,
        tspan=tspan, saveat=1, abstol=1e-6, reltol=1e-6,
        sensealg=ForwardDiffSensitivity()
    ))
end
a = predict(p)

pS = plot(t, [X[1, :], a[1, :]], label=["data S" "prediction S"])
pI = plot(t, [X[2, :], a[2, :]], label=["data I" "prediction I"])
pR = plot(t, [X[3, :], a[3, :]], label=["data R" "prediction R"])

display(plot(pS, pI, pR, layout=(3, 1), size=(800, 800)))

## .- instead of -?
function loss(θ)
    X̂ = predict(θ)
    sum(abs2, X .- X̂) / size(X, 2) + convert(eltype(θ), 1e-3) * sum(abs2, θ[3:end]) ./ length(θ[3:end])
end
println("initial loss = ", loss(p))

losses = Float32[]
callback(θ, args...) = begin
    l = loss(θ) # Equivalent L2 loss
    push!(losses, l)
    if length(losses) % 5 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end


pinit = Lux.ComponentArray(p)
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

## Gives the parameters back!
res1_uode = Optimization.solve(optprob, Adam(0.01, (0.8, 0.8)), callback=callback, maxiters=10)