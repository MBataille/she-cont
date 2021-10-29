using FFTW
FFTW.set_provider!("mkl")

struct Params
    N::Int
    L::Float    
    ϵ::Float
    ν::Float
end

struct Flags
    useFFT::Bool
end

struct State
    p::Params
    u::Matrix
end


function deriv_mat(A::Matrix{Float64}, L::Float64; axis::Char='x', order::Int64=1)
    if axis == 'y'
        dim = 1;
    elseif axis == 'x'
        dim = 2;
    else
        println("Axis not understood");
    factor = (2π * im / L)^order;
    N = size(A)[1];
    k = fftfreq(N) * N;

    ifft( factor * k.* fft(A, (dim, )) )
end

# Definitions
∂ₓ(S::State; axis='x') = deriv_mat(S.u, S.p.L, axis=axis, order=1);
∂ₓₓ(S::State; axis='x') = deriv_mat(S.u, S.p.L, axis=axis, order=2);
∂ₓₓₓₓ(S::State; axis='x') = deriv_mat(S.u, S.p.L, axis=axis, order=2);

∇²(S::State) = ∂ₓₓ(S) + ∂ₓₓ(S, axis='y');
∇⁴(S::State) = ∂ₓₓₓₓ(S) + ∂ₓₓₓₓ(S, axis='y');

# Right hand side of Swith-Hohenberg equation
function RHS_SHE(S::State)
    ϵ = S.p.ϵ;
    ν = S.p.ν;
    u = S.u;
    @. ϵ * u - u ^ 3 - ν * ∇²(S) - ∇⁴(S)
end

using DelimitedFiles

u = readdlm("data/epsilon1.16.values", ' ', Float64)

## Params are the following
N = 128
Δx = 0.7
L = N *Δx
ν = 1
ϵ = 1.16

p = Params(N, L, ν, ϵ)
