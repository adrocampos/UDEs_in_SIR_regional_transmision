using VoronoiCells
using LinearAlgebra:norm
using DifferentialEquations

smooth_min_1(x, n=1) = 1 - 1 / (x + 1)^n

"""calculate mobility from region 1 to region 2 based on the inverse distance

`scale` could be either the mean density or half the max"""
function inv_dist_mobility(x1, x2, ρ2, scale=1000, cut_off=x->smooth_min_1(x,3))
    dist = sum((x1 .- x2).^2)
    weight = 1 + 0.5 * ρ2 / (scale + ρ2)

    return cut_off(weight / dist)
end


function border_length(tess::Tessellation, i::Int, j::Int)
    c1 = tess.Cells[i]
    c2 = tess.Cells[j]

    border = intersect(c1, c2)
    
    return norm(diff(border))
end


"""calculate mobility from voronoi patch i to voronoi patch j based on the
border length between them

"""
function border_length_mobility(tess::Tessellation, i::Int, j::Int, ρj::Float64, scale=2000)
    bl = border_length(tess, i, j)

    return ρj * bl / scale
end

λ_self(I_in, β, N) = β * I_in / N


"""
    λ_cross_border(I_in, mobility12, mobility21, β2, N2)

Returns the contribution to λ due to infections from another patch.

# Parameters
* `E_in` I in the other patch
* `mobility12` The mobility to the other patch
* `mobility21` The mobility from the other patch
* `β2` the transmission rate in the other patch 
* `N2` The population size of the other patch 
    
"""
function λ_cross_border(I_in, mobility12, mobility21, β2, N2)
    ## abstract mobility from 1 to 2 mobility12 and vice versa mobility21

    # compartment-wise abstract mobilities
    β = β2 .* (mobility21 + mobility12) # combined beta

    return λ_self(I_in, β, N2)
end

"calculate β Matrix"
function calc_β(β_fac, ρ, n=1)
    w = 2000
    β = β_fac * (w^n + ρ^n) / (( 2 * w )^n + ρ^n)

    return β
end

function SIR!(du, u, param, t)
    (;β, γ, N, mobilities) = param
        
    for i in eachindex(N)
        m = (i - 1) * 3
        S, I, R = u[m + 1:m + 3]
        
        λ = 0.
        for j in eachindex(N)
            if i == j
                λ += λ_self(I, β[i], N[i])
            else
                n = (j - 1) * 3
                I_in = u[n + 3]
                λ += λ_cross_border(I_in, mobilities[i,j], mobilities[j,i], β[j], N[j])
            end
        end

        du[m + 1] = dS = -λ * S
        du[m + 2] = dI = λ * S - γ * I
        du[m + 3] = dR = γ * I
    end

    nothing
end
