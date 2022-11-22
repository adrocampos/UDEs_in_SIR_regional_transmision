## Environment and packages
cd(@__DIR__)
using Pkg;
Pkg.activate(".");
# Pkg.instantiate();


using DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, CSV, Lux, DataFrames
using DataDrivenDiffEq, ModelingToolkit, LinearAlgebra, DiffEqSensitivity, Zygote, Optim, CSV, Lux, Pkg, Flux
gr()
Pkg.status()


data_dir = "/Users/adrocampos/covid19/synth_data/"
regions = ["2", "3", "5", "10", "15", "20", "30"][1]
mobility_type = ["inv_dist", "border", "neighbor"][2]
initially_recovered = false

positions = CSV.File(data_dir * "positions_" * regions * "_regions.csv")
positions = DataFrame(positions)

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


ann = FastChain(FastDense(3, 50, tanh),
    FastDense(50, 50, tanh),
    FastDense(50, 1))


# Get the initial parameters, first two is linear birth / decay of prey and predator

# ps, st = Lux.setup(rng, ann)

## Firs the parameters for Beta, gama und N, then the weigths. 

# Get the initial parameters, first two is linear birth / decay of prey and predator
p = [rand(Float32, 2); 2000; initial_params(ann)]

function dudt_(du, u, p, t)

    S, I, R = u
    β, γ, N = p[1:3]

    z = ann(u, p[3:end])
    dS = -β * S * I / N - z[1]  # susceptible
    dI = β * S * I / N - γ * I - z[1] # infected
    dR = γ * I

    du[1] = dS
    du[2] = dI
    du[3] = dR

end



# Define the problem
prob_UODE = ODEProblem(dudt_, u0, tspan, p) ##prob_neuralode



# Define parameters for Multiple Shooting
# group_size = 5
# continuity_term = 200.0f0

# function loss(data, pred)
#     return sum(abs2, data - pred)
# end

# function shooting_loss(p)
#     return multiple_shoot(p, X, t, prob_UODE, loss, Vern7(),
#         group_size; continuity_term)
# end

function loss(θ)
    X̂ = predict(θ)
    sum(abs2, X - X̂) / size(X, 2) + convert(eltype(θ), 1e-3) * sum(abs2, θ[3:end]) ./ length(θ[3:end])
end







## Training -> First shooting / batching to get a rough estimate

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS


# res1 = DiffEqFlux.sciml_train(shooting_loss, p, Adam(0.1f0), cb=callback, maxiters=1)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS to achieve partial fit of the data
# res2 = DiffEqFlux.sciml_train(shooting_loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 500)
# println("Training loss after $(length(losses)) iterations: $(losses[end])")
# # Full L2-Loss for full prediction
# res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_neuralode = Optimization.solve(prob_UODE,
    Tsit5(),
    callback=callback,
    maxiters=300)


