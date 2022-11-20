module SubSIt

using LinearAlgebra
using SparseArrays
# using MKLSparse
import SparseArrays: findnz
# using ThreadedSparseCSR

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
function ssit(K, M; nev = 6, ncv=0, X = fill(eltype(K), 0, 0), tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, check=0) 
    @assert which == :SM
    @assert nev >= 1

    factor = cholesky(K)

    _nev = max(Int(round(nev/4)+1), 20)
    while true
        ncv = min(_nev + 50, Int(round(_nev * 2.0)))
        ncv = min(_nev + 50, Int(round(_nev * 2.0)))
        if size(X) == (0, 0)
            X = fill(zero(eltype(K)), size(K,1), ncv)
            for j in axes(M, 1)
                X[j, 1] = M[j, j]
            end
            dMK = diag(M) ./ diag(K)
            ix = sortperm(dMK)
            k = 1
            for j in 2:ncv-1
                X[ix[k], j] = 1.0
            end
            X[:, end] = rand(size(K,1))
        else
            X = hcat(X, rand(eltype(K), size(X, 1), ncv - size(X, 2)))
        end
        _maxiter = ifelse(_nev == nev, maxiter, 4)
        lamb, X, nconv, niter, lamberr = _ssit(factor, M, _nev, X, tol, _maxiter, verbose) 
        if _nev == nev
            return lamb[1:nev], X[:, 1:nev], nconv, niter, lamberr
        end
        _nev = _nev * 2
        _nev = min(nev, _nev)
    end
    return nothing
end

function _ssit(K, M, nev, X, tol, maxiter, verbose) 
    nvecs = size(X, 2)
    @show nev, nvecs
    Y = deepcopy(X)
    Kr = fill(zero(eltype(K)), nvecs, nvecs)
    Mr = fill(zero(eltype(K)), nvecs, nvecs)
    evalues = fill(zero(eltype(K)), nvecs)
    evectors = fill(zero(eltype(K)), nvecs, nvecs)
    plamb = fill(zero(eltype(K)), nvecs) .+ Inf
    lamb = fill(zero(eltype(K)), nvecs)
    lamberr = fill(zero(eltype(K)), nvecs)
    converged = falses(nvecs)  # not yet
    niter = 0
    nconv = 0

    mul!(Y, M, X)
    for i = 1:maxiter
        qrd = qr!(Y)
        Y .= Matrix(qrd.Q)
        X .= K \ Y
        mul!(Kr, Transpose(X), Y)
        mul!(Y, M, X)
        mul!(Mr, Transpose(X), Y)
        decomp = eigen(Kr, Mr)
        ix = sortperm(real.(decomp.values))
        evalues .= real.(@view decomp.values[ix])
        evectors .= real.(@view decomp.vectors[:, ix])
        mul!(X, Y, evectors)
        X, Y = Y, X
        lamb .= evalues
        for j in 1:nvecs
            lamberr[j] = abs(lamb[j] - plamb[j]) / abs(plamb[j])
        end
        converged .= (lamberr .<= tol)
        nconv = length(findall(converged))
        verbose && println("maximum(lamberr)=$(maximum(lamberr)), nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        lamb, plamb = plamb, lamb
        niter = niter + 1
    end
    return lamb, Y, nconv, niter, lamberr
end

end

# function ssit(K, M; nev = 6, ncv=max(20, 2*nev+1), v0 = fill(eltype(K), 0, 0), tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, check=0) 
#     @assert which == :SM
#     @assert nev >= 1
#     ncv = max(ncv, size(v0, 2))
#     @assert nev < ncv
#     nvecs = ncv
#     if size(v0) == (0, 0)
#         v0 = fill(zero(eltype(K)), size(K,1), nvecs)
#         for j in axes(M, 1)
#             v0[j, 1] = M[j, j]
#         end
#         dMK = diag(M) ./ diag(K)
#         ix = sortperm(dMK)
#         k = 1
#         for j in 2:nvecs-1
#             v0[ix[k], j] = 1.0
#         end
#         v0[:, end] = rand(size(K,1))
#     end
#     factor = cholesky(K)
#     return _ssit(factor, M, nev, v0, tol, maxiter, verbose) 
# end