using Polyhedra
using JuMP
using CDDLib
using LinearAlgebra
using Clarabel
using JLD2
using CSV
using DataFrames

# =============================================================================
# Helper functions
# =============================================================================

"""
    robust_mean(Ξ, P, r)

Compute the Wasserstein robust hub location for support Ξ with weights P
and Wasserstein radius r. Returns the optimal location μ and optimal value.

The conic LP dual of  min_μ sup_{W(Q,P₀)≤r} E_Q[‖ξ-μ‖²]  is solved via
a RotatedSecondOrderCone reformulation.
"""
function robust_mean(Ξ, P, r; optimizer = Clarabel.Optimizer)
    N, d = size(Ξ)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, μ[1:d])       # hub location
    @variable(model, λ >= 0)       # dual Wasserstein multiplier
    @variable(model, s[1:N])       # epigraph variables

    @objective(model, Min, λ * r + sum(P[i] * s[i] for i in 1:N))
    @constraint(model, [i=1:N, j=1:N],
        vcat(s[i] + λ * norm(Ξ[i,:] - Ξ[j,:]), 0.5, μ .- Ξ[j,:])
        in RotatedSecondOrderCone())

    optimize!(model)
    @assert is_solved_and_feasible(model)

    return value.(μ), objective_value(model)
end

"""
    robust_objective(μ, Ξ, P, r)

Evaluate the robust worst-case cost  sup_{W(Q,P₀)≤r} E_Q[‖ξ-μ‖²]
for a fixed hub location μ.
"""
function robust_objective(μ, Ξ, P, r; optimizer = Clarabel.Optimizer)
    N, d = size(Ξ)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, λ >= 0)       # dual Wasserstein multiplier
    @variable(model, s[1:N])       # epigraph variables

    @objective(model, Min, λ * r + sum(P[i] * s[i] for i in 1:N))
    @constraint(model, [i=1:N, j=1:N],
        vcat(s[i] + λ * norm(Ξ[i,:] - Ξ[j,:]), 0.5, μ .- Ξ[j,:])
        in RotatedSecondOrderCone())

    optimize!(model)
    @assert is_solved_and_feasible(model)

    return objective_value(model)
end

"""
    one_center(p)

Compute the Chebyshev center (circumcenter) and circumradius of polytope p.
"""
function one_center(p; optimizer = Clarabel.Optimizer)
    model = Model(optimizer)
    set_silent(model)

    @variable(model, c[1:fulldim(p)])
    @variable(model, R >= 0)
    for v in points(vrep(p))
        @constraint(model, [R; c .- v] in SecondOrderCone())
    end
    @objective(model, Min, R)

    optimize!(model)
    @assert is_solved_and_feasible(model)

    return value.(c), value(R)
end

"""
    regret_mean(Ξ, P, r)

Compute the regret-optimal hub location: the Chebyshev center of the set M
of attainable means under the Wasserstein ball of radius r.

The set M is obtained by projecting the transport polytope onto the mean
variable μ using exact vertex enumeration via CDD.
"""
function regret_mean(Ξ, P, r; optimizer = Clarabel.Optimizer)
    N, d = size(Ξ)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, μ[1:d])              # mean of the worst-case distribution
    @variable(model, Q[1:N] >= 0)         # marginal weights
    @variable(model, T[1:N, 1:N] >= 0)   # transport plan

    # μ is the mean of Q; Q and P are coupled by transport plan T within radius r
    @constraint(model, μ .== sum(Ξ[i, :] * Q[i] for i in 1:N))
    @constraint(model, sum(Q) == 1)
    @constraint(model, sum(T) == 1)
    @constraint(model, sum(T, dims=1)[:] == Q)
    @constraint(model, sum(T, dims=2)[:] == P)
    @constraint(model,
        sum(T[i, j] * norm(Ξ[i,:] - Ξ[j,:]) for i in 1:N, j in 1:N) <= r)

    # Project the feasible set onto the mean variable μ
    poly_full = polyhedron(model, CDDLib.Library())
    M_r = Polyhedra.project(poly_full, 1:d, ProjectGenerators())
    removevredundancy!(M_r)

    μ_reg, R_reg = one_center(M_r)
    return μ_reg, R_reg, M_r
end

"""
    sat_mean(Ξ, P, f0; R)

Compute the satisficing hub location of Long (2023): minimise the slope α
such that the worst-case cost stays below f0 + α·γ for all γ ∈ [0, R].
The Wasserstein ball constraint is discretised on a grid of radii.
"""
function sat_mean(Ξ, P, f0; R=1, optimizer = Clarabel.Optimizer)
    N, d = size(Ξ)
    rs   = range(0, stop=R, length=200)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, μ[1:d])                    # hub location
    @variable(model, α >= 0)                    # satisficing slope
    @variable(model, λ[1:length(rs)] >= 0)      # dual Wasserstein multipliers
    @variable(model, s[1:N, 1:length(rs)])      # epigraph variables

    @objective(model, Min, α)
    @constraint(model, [k=eachindex(rs)],
        λ[k] * rs[k] + sum(P[i] * s[i,k] for i in 1:N) <= f0 + α * rs[k])
    @constraint(model, [i=1:N, j=1:N, k=eachindex(rs)],
        vcat(s[i,k] + λ[k] * norm(Ξ[i,:] - Ξ[j,:]), 0.5, μ .- Ξ[j,:])
        in RotatedSecondOrderCone())

    optimize!(model)
    # @assert is_solved_and_feasible(model)

    return value.(μ), objective_value(model)
end

"""
    glob_robust_mean(Ξ, P; R, q)

Compute the GARO hub location: minimise α such that the adversarial regret
satisfies A(μ, γ) ≤ α·ϕ(γ) for all γ ∈ [0, R].
The rate function φ(γ) = 1 gives an absolute guarantee.
The Wasserstein ball constraint is discretised on a grid of radii.
"""
function glob_robust_mean(Ξ, P; R=1, ϕ=γ->1, optimizer = Clarabel.Optimizer)
    N, d = size(Ξ)
    rs   = range(0, stop=R, length=200)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, μ[1:d])                    # hub location
    @variable(model, α >= 0)                    # adversarial regret bound
    @variable(model, λ[1:length(rs)] >= 0)      # dual Wasserstein multipliers
    @variable(model, s[1:N, 1:length(rs)])      # epigraph variables

    @objective(model, Min, α)
    @constraint(model, [k=eachindex(rs)],
        λ[k] * rs[k] + sum(P[i] * s[i,k] for i in 1:N)
        <= α * ϕ(rs[k]) + robust_mean(Ξ, P, rs[k])[2])
    @constraint(model, [i=1:N, j=1:N, k=eachindex(rs)],
        vcat(s[i,k] + λ[k] * norm(Ξ[i,:] - Ξ[j,:]), 0.5, μ .- Ξ[j,:])
        in RotatedSecondOrderCone())

    optimize!(model)
    @assert is_solved_and_feasible(model)

    return value.(μ), objective_value(model)
end

function ordered_vertices(poly)
    verts = [collect(v) for v in points(poly)]
    c = sum(verts) / length(verts)
    return sort(verts, by = v -> atan(v[2]-c[2], v[1]-c[1]))
end

# =============================================================================
# Main script: load instance, solve, write CSV output
# =============================================================================

# --- Problem instance ---------------------------------------------------------

@load "problem_instance.jld2" Ξ P r Pstar

CSV.write("csv/instance.csv",
    DataFrame(xi_1=Ξ[:,1], xi_2=Ξ[:,2], P=P, Pstar=Pstar, zero=zeros(length(P))))

# --- Geometry of Ξ -----------------------------------------------------------

# Convex hull of Ξ and its circumscribed ball
Ξh  = polyhedron(vrep(Ξ), CDDLib.Library())
removevredundancy!(Ξh)
V_Ξ = ordered_vertices(Ξh)
CSV.write("csv/ch-Xi.csv",
    DataFrame(xi_1=[v[1] for v in V_Ξ], xi_2=[v[2] for v in V_Ξ]))

μ_circ, R_circ = one_center(Ξh)
CSV.write("csv/min-ball-Xi.csv", DataFrame(xi_1=μ_circ[1], xi_2=μ_circ[2], R=R_circ))
CSV.write("csv/circle1.csv",
    DataFrame(xi_1 = μ_circ[1] .+ R_circ*cos.(0:0.01:2π),
              xi_2 = μ_circ[2] .+ R_circ*sin.(0:0.01:2π),
              P    = zeros(length(0:0.01:2π))))

# --- Hub locations ------------------------------------------------------------

# Nominal hub: mean of P₀
μ_nom = sum(Ξ[i,:] * P[i] for i in eachindex(P))
CSV.write("csv/mean.csv", DataFrame(xi_1=μ_nom[1], xi_2=μ_nom[2]))

# Regret-optimal hub and the set M of attainable means
μ_reg, R_reg, M_r = regret_mean(Ξ, P, r)
V_M = ordered_vertices(M_r)
CSV.write("csv/M_r.csv",
    DataFrame(xi_1=[v[1] for v in V_M], xi_2=[v[2] for v in V_M]))
CSV.write("csv/min-ball-M_r.csv", DataFrame(xi_1=μ_reg[1], xi_2=μ_reg[2], R=R_reg))
CSV.write("csv/circle2.csv",
    DataFrame(xi_1 = μ_reg[1] .+ R_reg*cos.(0:0.01:2π),
              xi_2 = μ_reg[2] .+ R_reg*sin.(0:0.01:2π),
              P    = zeros(length(0:0.01:2π))))

# Robust hub
μ_rob, _ = robust_mean(Ξ, P, r)
CSV.write("csv/robust_mean.csv", DataFrame(xi_1=μ_rob[1], xi_2=μ_rob[2]))

# Satisficing hub (satisfaction level 4% above nominal cost)
f0    = 1.04 * sum(norm(Ξ[i,:] - μ_nom)^2 * P[i] for i in eachindex(P))
μ_sat, α_sat = sat_mean(Ξ, P, f0, R=2*R_circ)
CSV.write("csv/sat_mean.csv", DataFrame(xi_1=μ_sat[1], xi_2=μ_sat[2]))

# GARO hub (absolute guarantee: φ(γ) = 1, i.e. q = 0)
ϕ=γ->1
μ_garo, α_garo = glob_robust_mean(Ξ, P, ϕ=ϕ, R=2*R_circ)
CSV.write("csv/glob_robust_mean.csv", DataFrame(xi_1=μ_garo[1], xi_2=μ_garo[2]))

# Diagnostic: adversarial regret at the nominal radius r
println("Adversarial regret at r:")
println("  nominal hub  : ", robust_objective(μ_nom,  Ξ, P, r) - robust_objective(μ_rob, Ξ, P, r))
println("  robust hub   : ", 0.0)
println("  regret hub   : ", robust_objective(μ_reg,  Ξ, P, r) - robust_objective(μ_rob, Ξ, P, r))

# --- Adversarial regret curves -----------------------------------------------

# Paths of robust and regret hub locations as γ varies
γs = range(0, stop=2*R_circ, length=200)

robust_path = [robust_mean(Ξ, P, γ)[1] for γ in γs]
CSV.write("csv/robust_hubs.csv",
    DataFrame(gamma=γs,
              xi_1=[μ[1] for μ in robust_path],
              xi_2=[μ[2] for μ in robust_path]))

regret_path = [regret_mean(Ξ, P, γ)[1] for γ in γs]
CSV.write("csv/regret_hubs.csv",
    DataFrame(gamma=γs,
              xi_1=[μ[1] for μ in regret_path],
              xi_2=[μ[2] for μ in regret_path]))

# Adversarial regret A(μ, γ) = robust_objective(μ, γ) - oracle_cost(γ)
oracle_cost   = [robust_mean(Ξ, P, γ)[2]             for γ in γs]
regret_nom    = [robust_objective(μ_nom,  Ξ, P, γ)   for γ in γs] .- oracle_cost
regret_rob    = [robust_objective(μ_rob,  Ξ, P, γ)   for γ in γs] .- oracle_cost
regret_sat    = [robust_objective(μ_sat,  Ξ, P, γ)   for γ in γs] .- oracle_cost
regret_garo   = [robust_objective(μ_garo, Ξ, P, γ)   for γ in γs] .- oracle_cost
regret_reg    = [robust_objective(μ_reg,  Ξ, P, γ)   for γ in γs] .- oracle_cost
guarantee_garo = [α_garo * ϕ(γ)                for γ in γs]
guarantee_sat  = [f0 - robust_mean(Ξ, P, γ)[2] + α_sat * γ for γ in γs]

CSV.write("csv/costs.csv",
    DataFrame(gamma         = γs,
              nom           = regret_nom,
              rob           = regret_rob,
              sat           = regret_sat,
              grob          = regret_garo,
              reg           = regret_reg,
              oracle        = oracle_cost,
              guarantee_garo = guarantee_garo,
              guarantee_sat  = guarantee_sat))
