"""
    T3Dist(ν, μ, σ)

The *non-standardized Students T distribution* with `ν` degrees of freedom, location `μ`, and scale `σ` 
has probability density function
"""
struct T3Dist{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    μ::T
    σ::T
    T3Dist{T}(ν::T, μ::T, σ::T) where {T<:Real} = new{T}(ν, μ, σ)
end

function T3Dist(ν::T, μ::T, σ::T; check_args::Bool = true) where {T<:Real}
    @check_args T3Dist (ν, ν > zero(ν)) (σ, σ > zero(σ))
    return T3Dist{T}(ν, μ, σ)
end

T3Dist(ν::Real, μ::Real, σ::Real; check_args::Bool = true) =
    T3Dist(promote(ν, μ, σ)...; check_args = check_args)
T3Dist(ν::Integer, μ::Integer, σ::Integer; check_args::Bool = true) =
    T3Dist(float(ν), float(μ), float(σ); check_args = check_args)
T3Dist(ν::Real, μ::Real, σ::Real = 1.0) = T3Dist(ν, μ, σ)
T3Dist(ν::Real, μ::Real = 0.0, σ::Real = 1.0) = TDist(ν)

@distr_support T3Dist -Inf Inf

#### Conversions

#### Parameters

params(d::T3Dist) = (d.ν, d.μ, d.σ)
dof(d::T3Dist) = d.ν
location(d::T3Dist) = d.μ
scale(d::T3Dist) = d.σ

@inline partype(d::T3Dist{T}) where {T<:Real} = T

#### Statistics

mean(d::T3Dist{T}) where {T<:Real} = d.ν > 1 ? d.μ : T(NaN)
mode(d::T3Dist) = d.μ
var(d::T3Dist{T}) where {T<:Real} = d.ν > 2 ? d.σ^2 * d.ν / (d.ν - 2) : T(NaN)

#### Evaluation

function logpdf(d::T3Dist, x::Real)
    ν, μ, σ = params(d)
    νp12 = (ν + 1) / 2
    loggamma(νp12) - (logπ + log(ν) + 2log(σ)) / 2 - loggamma(ν / 2) -
    νp12 * log1p(((x - μ) / σ)^2 / ν)
end
