module SubSIt

using LinearAlgebra
using SparseArrays
# using MKLSparse
import SparseArrays: findnz
using ThreadedSparseCSR

findnz(A::T) where {T<:LinearAlgebra.Symmetric{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}}  = findnz(A.data)

"""
    ssit(K, M; nev=6, v0=fill(zero(T), 0, 0), tol=1.0e-3, maxiter=300, verbose=false)

Subspace  Iteration method for the generalized eigenvalue problem.

Block inverse power (subspace iteration) method for k smallest eigenvalues of
the generalized eigenvalue problem `K*v = lambda*M*v`.

# Arguments
* `K` =  square symmetric stiffness matrix (if necessary mass-shifted),
* `M` =  square symmetric mass matrix,

Keyword arguments
* `v0` =  initial guess of the eigenvectors (for instance random),
* `nev` = the number of eigenvalues sought
* `tol` = relative tolerance on the eigenvalue, expressed in terms of norms 
      of the change of the eigenvalue estimates from iteration to iteration.
* `maxiter` =  maximum number of allowed iterations
* `verbose` = verbose? (default is false)

# Output
* `labm` = computed eigenvalues,
* `v` = computed eigenvectors,
* `nconv` = number of converged eigenvalues
* `niter` = number of iterations taken
* `lamberr` = eigenvalue errors, defined as normalized  differences  of
    successive  estimates of the eigenvalues (or not normalized if the 
    eigenvalues converge to zero).
"""
function ssit1(K, M; nev = 6, ncv=max(20, 2*nev+1), v0 = fill(eltype(K), 0, 0), tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, check=0) 
    @assert which == :SM
    @assert nev >= 1
    if size(v0) == (0, 0)
        v0 = fill(zero(eltype(K)), size(K,1), ncv)
        for j in axes(M, 1)
            v0[j, 1] = M[j, j]
        end
        dMK = diag(M) ./ diag(K)
        ix = sortperm(dMK)
        k = 1
        for j in 2:ncv-1
            v0[ix[k], j] = 1.0
        end
        v0[:, end] = rand(size(K,1))
    end
    v = deepcopy(v0)
    @assert nev <= size(v0, 2)
    nvecs = size(v, 2)  # How many eigenvalues are iterated?
    plamb = zeros(nvecs)  # previous eigenvalue
    lamb = zeros(nvecs)
    lamberr = zeros(nvecs)
    converged = falses(nev)  # not yet
    niter = 0
    nconv = 0
    factor = cholesky(K)
    Kv = zeros(size(K, 1), nvecs)
    Mv = zeros(size(M, 1), nvecs)

    @show ThreadedSparseCSR.get_num_threads()
    # ThreadedSparseCSR.set_num_threads(4)

    K = ThreadedSparseCSR.sparsecsr(findnz(K)..., size(K)...)
    
    for i = 1:maxiter
        mul!(Mv, M, v)
        v .= factor \ Mv
        mul!(Mv, M, v)
        __mass_normalize!(v, Mv)
        for k in axes(Kv, 2)
            ThreadedSparseCSR.bmul!(view(Kv, :, k), K, view(v, :, k))
        end
        mul!(Mv, M, v)
        for k in 1:nvecs
            lamb[k] = dot(v[:, k], Kv[:, k]) / dot(v[:, k], Mv[:, k])
        end
        ix = sortperm(real.(lamb))
        lamb .= lamb[ix]
        decomp = eigen(transpose(v)*Kv, transpose(v)*Mv)
        ix = sortperm(real.(decomp.values))
        evalues = real.(decomp.values[ix])
        evectors = decomp.vectors[:, ix]
        mul!(Mv, v, real.(evectors))
        v, Mv = Mv, v
        plamb .= evalues
        for j in 1:nvecs
            if abs(lamb[j]) <= tol  # zero eigenvalues
                lamberr[j] = abs(lamb[j]) 
            else # non zero eigenvalues
                lamberr[j] = abs(lamb[j] - plamb[j]) / abs(plamb[j])
            end
        end
        converged .= (lamberr .<= tol)[1:nev]
        nconv = length(findall(converged))
        verbose && println("nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        niter = niter + 1
    end
    __mass_normalize_M!(v, M)
    return lamb[1:nev], v[:, 1:nev], nconv, niter, lamberr
end

function __mass_normalize!(v, Mv)
    for k in axes(v, 2)
        v[:, k] .*= @views 1.0 / sqrt(dot(v[:, k], Mv[:, k]))
    end
    v
end

function __mass_normalize_M!(v, M)
    for k in axes(v, 2)
        v[:, k] ./= @views sqrt(v[:, k]' * M * v[:, k])
    end
    v
end

function ssit(K, M; nev = 6, ncv=max(20, 2*nev+1), v0 = fill(eltype(K), 0, 0), tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, check=0) 
    @assert which == :SM
    @assert nev >= 1
    ncv = max(ncv, size(v0, 2))
    @assert nev < ncv
    nvecs = ncv
    if size(v0) == (0, 0)
        v0 = fill(zero(eltype(K)), size(K,1), nvecs)
        for j in axes(M, 1)
            v0[j, 1] = M[j, j]
        end
        dMK = diag(M) ./ diag(K)
        ix = sortperm(dMK)
        k = 1
        for j in 2:nvecs-1
            v0[ix[k], j] = 1.0
        end
        v0[:, end] = rand(size(K,1))
    end
    X = v0
    Y = deepcopy(X)
    Kr = fill(zero(eltype(K)), nvecs, nvecs)
    Mr = fill(zero(eltype(K)), nvecs, nvecs)
    plamb = fill(zero(eltype(K)), nvecs) .+ Inf
    lamb = fill(zero(eltype(K)), nvecs)
    lamberr = fill(zero(eltype(K)), nvecs)
    converged = falses(nev)  # not yet
    niter = 0
    nconv = 0

    factor = cholesky(K)
    mul!(Y, M, X)
    for i = 1:maxiter
        X .= factor \ Y
        mul!(Kr, transpose(X), Y)
        mul!(Y, M, X)
        mul!(Mr, transpose(X), Y)
        decomp = eigen(Kr, Mr)
        ix = sortperm(real.(decomp.values))
        evalues = real.(decomp.values[ix])
        evectors = decomp.vectors[:, ix]
        mul!(X, Y, real.(evectors))
        __mass_normalize_M!(X, M)
        X, Y = Y, X
        lamb .= evalues
        for j in 1:nvecs
            lamberr[j] = abs(lamb[j] - plamb[j]) / abs(plamb[j])
        end
        converged .= (lamberr .<= tol)[1:nev]
        nconv = length(findall(converged))
        verbose && println("nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        lamb, plamb = plamb, lamb
        niter = niter + 1
    end
    return lamb[1:nev], Y[:, 1:nev], nconv, niter, lamberr
end

end