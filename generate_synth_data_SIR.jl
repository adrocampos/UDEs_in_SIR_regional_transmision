using DifferentialEquations
using StatsBase, Random
using ComponentArrays
using GeometryBasics, VoronoiCells
using DataFrames, CSV
using Plots, ColorSchemes

projpath = normpath(@__DIR__, "../")
# also imports DifferentialEquations
include(joinpath(projpath, "src/multigroup_SIR_no_callbacks.jl"))

Random.seed!(100)

n = 10

"""`region_setup(n::Int)` takes the number of regions `n` and returns:

- `N`: a Vector of random population sizes between 1.000 and 10.000
- `ρ`: a Vector of corresponding population densities based on `N` and `A`
- `ρ_tot`: the population density over the whole geography
- `pos`: a Vector of the positions of each region inside a `Rectangle(Point2(0,0), Point2(4,4))`
- `A`: a Vector of areas of each region calculated using the voronoi cells
- `tess`: the `Tessellation` containg all the voronoi cells"""
function region_setup(n::Int)
    pos = 4. .* rand(Point2, n)
    tess = voronoicells(pos, Rectangle(Point2(0,0), Point2(4,4)))
    N = round.(Int,10 .^ (3 .+ rand(n)))
    A = voronoiarea(tess)
    ρ = N ./ A
    ρ_tot = sum(N) / sum(N ./ ρ)

    return N, ρ, ρ_tot, pos, A, tess
end

N, ρ, ρ_tot, pos, A, tess = region_setup(n)

# mobilities from i to j
mobilities_dist = [inv_dist_mobility(pos[[i,j]]..., ρ[j], ρ_tot, x -> min(x,1)) for i in eachindex(N), j in eachindex(N)] .* .05
mobilities_border = [border_length_mobility(tess, i,j,ρ[j]) for i in eachindex(N), j in eachindex(N)] .* 0.04
mobilities_neighbor = .!(iszero.(mobilities_border)) .* 0.04


# infection rates:
β = fill(0.1, n)

# recovery rates:
γ = 1/10


#--------------------------------------------------------------------#
#                             Simulation                             #
#--------------------------------------------------------------------#

# Initial conditions
function generate_initial_conditions(N, I0tot=100, R0tot=0)
    I0 = zeros(length(N))
    R0 = zeros(length(N))

    for _ = 1:I0tot
        i = sample(weights(N))
        I0[i] += 1
    end

    for _ = 1:R0tot
        i = sample(weights(N))
        R0[i] += 1
    end

    u0 = vcat([[N[i] - I0[i] - R0[i], I0[i], R0[i]] for i in eachindex(N)]...)

    return u0
end

u0 = generate_initial_conditions(N, 10)

function simulate_SIR(u0, N, β, γ, mobilities)
    tspan = (0.,500.)
    param = ComponentVector(; β, γ, N, mobilities)

    compartment_names = vcat([Symbol.(["S", "I", "R"] .* "$i") for i in eachindex(N)]...)

    func = ODEFunction(SIR!, syms=compartment_names)
    prob = ODEProblem(func, u0, tspan, param)
    sol = solve(prob, saveat=.1)

    return sol
end


sol = simulate_SIR(u0, N, β, γ, mobilities_dist)
sol2 = simulate_SIR(u0, N, β, γ, mobilities_border)
sol3 = simulate_SIR(u0, N, β, γ, mobilities_neighbor)

p = plot(sol, idxs=(0,[3 * (0:n-1);].+2), xlim=(0,100))
plot!(p, sol2, idxs=(0,[3 * (0:n-1);].+2), ls = :dot, c = [j for i=1:1, j=1:n])
plot!(p, sol3, idxs=(0,[3 * (0:n-1);].+2), ls = :dash, c = 6 .+ [j for i=1:1, j=1:n])
display(p)


#--------------------------------------------------------------------#
#                             save data                              #
#--------------------------------------------------------------------#

datadir = normpath(@__DIR__, "../data/synth_data")

function save_sol(sol, mobility::String, idx)
    df = DataFrame(sol)
    rename!(df, :timestamp => :t)
    n_reg = (ncol(df) - 1) ÷ 3

    filename = "SIR_$(n_reg)_regions_$(mobility)_$(idx).csv"
    filepath = joinpath(datadir, filename)
    CSV.write(filepath, df)
end

function save_mobilities(mobilities, mobility_str::String)
    n_reg = size(mobilities,2)

    idx_df = DataFrame("from\\to" => 1:n_reg)
    mobility_df = DataFrame(mobilities, string.(1:n_reg))

    filename = "mobility_$(n_reg)_regions_$(mobility_str).csv"
    filepath = joinpath(datadir,filename)
    CSV.write(filepath, hcat(idx_df, mobility_df))
end

function save_border_lenghts(tess::Tessellation)
    n_reg = length(tess.Cells)
    borders = [border_length(tess,i,j) for i=1:n_reg, j=1:n_reg]

    idx_df = DataFrame("from\\to" => 1:n_reg)
    borders_df = DataFrame(borders, string.(1:n_reg))

    filename = "border_lengths_$(n_reg)_regions.csv"
    filepath = joinpath(datadir,filename)
    CSV.write(filepath, hcat(idx_df, borders_df))
end

function save_distances(pos)
    n_reg = length(pos)
    dists = [sum((x1 .- x2).^2) for x1 in pos, x2 in pos]

    idx_df = DataFrame("from\\to" => 1:n_reg)
    dists_dt = DataFrame(dists, string.(1:n_reg))

    filename = "distances_$(n_reg)_regions.csv"
    filepath = joinpath(datadir,filename)
    CSV.write(filepath, hcat(idx_df, dists_dt))
end


function save_regions(N,ρ,pos)
    n_reg = length(N)
    df = DataFrame(id=1:n_reg, N=N, density=ρ, x=first.(pos), y=last.(pos))

    filename = "positions_$(n_reg)_regions.csv"
    filepath = joinpath(datadir,filename)
    CSV.write(filepath, df)
end


#--------------------------------------------------------------------#
#                             final run                              #
#--------------------------------------------------------------------#

function run_all_simulations(ns = [2, 3, 5, 10, 15, 20, 30]; β0= 0.1, γ = 1/10, tspan = (0.,500.))
    for n in ns
        β = fill(β0, n)
        N, ρ, ρ_tot, pos, A, tess = region_setup(n)
        plt = plot(tess)
        display(plt)

        println("saving $n regions")
        save_regions(N,ρ, pos)
        println("saving distances for $n regions")
        save_distances(pos)
        println("saveing border lengths for $n regions")
        save_border_lenghts(tess)

        # mobilities from i to j
        mobilities_dist = [inv_dist_mobility(pos[[i,j]]..., ρ[j], ρ_tot, x -> min(x,1)) for i in eachindex(N), j in eachindex(N)] .* .05
        mobilities_border = [border_length_mobility(tess, i,j,ρ[j]) for i in eachindex(N), j in eachindex(N)] .* 0.04
        mobilities_neighbor = .!(iszero.(mobilities_border)) .* 0.04

        println("saving inverse distance mobility for $n regions")
        save_mobilities(mobilities_dist, "inv_dist")
        println("saving border length mobility for $n regions")
        save_mobilities(mobilities_border, "border")
        println("saving neighbor mobility for $n regions")
        save_mobilities(mobilities_neighbor, "neighbor")

        for i = 1:100
            u0 = if i <= 50
                generate_initial_conditions(N,10)
            else
                generate_initial_conditions(N,10, sum(N) / 4)
            end

            sol_dist = simulate_SIR(u0, N, β, γ, mobilities_dist)
            sol_border = simulate_SIR(u0, N, β, γ, mobilities_border)
            sol_neighbor = simulate_SIR(u0, N, β, γ, mobilities_neighbor)

            save_sol(sol_dist, "inv_dist", i)
            save_sol(sol_border, "border", i)
            save_sol(sol_neighbor, "neighbor", i)
        end

        println("All simulations done and saved for $n regions.")
    end

    nothing
end

run_all_simulations()
