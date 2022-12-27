module SubSIt

using LinearAlgebra

function __coldot(A, j, i)
    m = size(A, 1)
    r = zero(eltype(A))
    @simd for k in 1:m
        r += A[k, i] * A[k, j]
    end
    return r; 
end

function __colnorm(A, j)
    return sqrt(__coldot(A, j, j)); 
end

function __colsubt!(A, i, j, r)
    m = size(A, 1)
    @simd for k in 1:m
        A[k, i] -= r * A[k, j]
    end
end

function __normalizecol!(A, j)
    m = size(A, 1)
    r = 1.0 / __colnorm(A, j)
    @simd for k in 1:m
        A[k, j] *= r
    end
end

function __mgsthr!(A)
    m, n = size(A)
    __normalizecol!(A, 1)
    for j in 2:n 
        Base.Threads.@threads for i in j:n
            __colsubt!(A, i, j-1, __coldot(A, j-1, i))
        end
        __normalizecol!(A, j)
    end
    return A
end

function __mgsl3!(A)
    m, n = size(A)
    _one = one(eltype(A))
    r = fill(zero(eltype(A)), 1, n)
    __normalizecol!(A, 1)
    for j in 2:n 
        vr = view(r, 1:1, j:n)
        vAc = view(A, :, j-1:j-1)
        vAb = view(A, :, j:n)
        mul!(vr, transpose(vAc), vAb)
        mul!(vAb, vAc, vr, -_one, _one)
        __normalizecol!(A, j)
    end
    return A
end

# Copyright 2022, Michael Stewart
function __mgs3thr!(A; block_size = 32)
    m, n = size(A)
    _one = one(eltype(A))
    _zero = zero(eltype(A))
    num_blocks, rem_block = divrem(n, block_size)
    work = zeros(eltype(A), block_size, n)
    @views for b = 0:(num_blocks - 1)
        c0 = block_size * b + 1
        c1 = block_size * (b + 1)
        A1 = A[:, c0:c1]
        A2 = A[:, (c1 + 1):n]
        work2 = work[:, (c1 + 1):n]
        __mgsthr!(A1)
        mul!(work2, A1', A2, _one, _zero)
        mul!(A2, A1, work2, -_one, _one)
    end
    if rem_block != 0
        __mgsthr!(@views A[:, (end - rem_block + 1):end])
    end
    return A
end

# Copyright 2022, Michael Stewart
function __mgs3!(A; block_size = 32)
    m, n = size(A)
    _one = one(eltype(A))
    _zero = zero(eltype(A))
    num_blocks, rem_block = divrem(n, block_size)
    work = zeros(eltype(A), block_size, n)
    @views for b = 0:(num_blocks - 1)
        c0 = block_size * b + 1
        c1 = block_size * (b + 1)
        A1 = A[:, c0:c1]
        A2 = A[:, (c1 + 1):n]
        work2 = work[:, (c1 + 1):n]
        __mgsl3!(A1)
        mul!(work2, A1', A2, _one, _zero)
        mul!(A2, A1, work2, -_one, _one)
    end
    if rem_block != 0
        __mgsl3!(@views A[:, (end - rem_block + 1):end])
    end
    return A
end

# Copyright 2022, Michael Stewart
function __computeQ!(A)
    m, n = size(A)
    k = min(m, n)
    tau = zeros(eltype(A), k)
    LAPACK.geqrf!(A, tau)
    LAPACK.orgqr!(A, tau, k)
    return A
end

# Copyright 2022, Michael Stewart
function __computeQT!(A; nb = 32)
    m, n = size(A)
    k = min(m, n)
    T = zeros(eltype(A), nb, k)
    tau = zeros(eltype(A), k)
    LAPACK.geqrt!(A, T)
    b = Int(ceil(k / nb))
    ib = k - (b - 1) * nb
    for l = 1:(b - 1)
        for j = 1:nb
            tau[(l - 1) * nb + j] = T[j, (l - 1) * nb + j]
        end
    end
    for j = 1:ib
        tau[(b - 1) * nb + j] = T[j, (b - 1) * nb + j]
    end
    LAPACK.orgqr!(A, tau, k)
    return A
end


"""
    ssit(K, M; nev = 6, ncv=0, tol = 2.0e-3, maxiter = 300, verbose=false, which=:SM, X = fill(eltype(K), 0, 0), check=0, ritzvec=true, sigma=0.0) 

Subspace Iteration method for the generalized eigenvalue problem.

Implementation of the subspace iteration method for `k` smallest eigenvalues of
the generalized eigenvalue problem `K*X = M*X*Lambda`.

# Arguments

See also the documentation for Arpack `eigs`.

* `K` =  square symmetric stiffness matrix (if necessary mass-shifted to avoid
  singularity),
* `M` =  square symmetric mass matrix,

# Keyword arguments
* `nev` = the number of eigenvalues sought,
* `ncv` = ignored, number of iteration vectors used in the computation; 
* `tol` = relative tolerance on the eigenvalue, expressed in terms of norms 
      of the change of the eigenvalue estimates from iteration to iteration.
* `maxiter` =  maximum number of allowed iterations (default 300),
* `verbose` = verbose? (default is false),
* `which` =    type of eigenvalues, only `:SM` (eigenvalues of smallest
  magnitude) is accommodated, all other types raise an error,
* `X` =  initial guess of the eigenvectors (for instance random), of dimension `size(M, 1)`x`nev`,
* `check` = ignored
* `ritzvec`= ignored, this function always returns the Ritz vectors,
* `sigma` = ignored,
* `explicittransform` = :none (ignored).

# Output
* `labm` = computed eigenvalues,
* `v` = computed eigenvectors,
* `nconv` = number of converged eigenvalues
* `niter` = number of iterations taken
* `lamberr` = eigenvalue errors, defined as normalized  differences  of
    successive  estimates of the eigenvalues.
"""
function ssit(K, M; nev = 6, ncv=0, tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, X = fill(eltype(K), 0, 0), check=0, ritzvec=true, sigma=0.0, explicittransform=:none) 
    which != :SM && error("Wrong type of eigenvalue requested; only :SM accepted")
    nev < 1 && error("Wrong number of eigenvalues: needs to be > 1") 
    if ncv <= 0 && size(X, 2) > 0
        ncv = size(X, 2)
        ncv < nev+1 && error("Insufficient number of iteration vectors: must be >= nev+1")
    end

    # Tactics to build up the iteration space by starting from a smaller
    # number eigenvalues
    _iteration_tactics(_nev) = begin
        if ncv == 0 # Assume we can freely control the number of iteration vectors
            # Increase the number of eigenvalues
            _nev = min(nev, Int(round(_nev * 2)))
            # ... and adjust the number of iteration vectors
            _ncv = min(_nev + 100, Int(round(_nev * 2)))
        else # otherwise we are constrained by the number of requested iteration vectors
            _nev = nev
            _ncv = ncv
        end
        _nev, _ncv
    end

    # Initial number of eigenvalues requested
    _nev = ifelse(ncv == 0, max(Int(round(nev/4)+1), 20), nev)

    # If the number of iteration vectors was not specified, but we have the
    # iteration vectors, we will go with that
    _ncv = _iteration_tactics(_nev)[2]

    if size(X) == (0, 0)
        X = rand(eltype(M), (size(M, 1), _ncv))  
    end

    factor = cholesky(K)
    
    iter = 0
    while iter < maxiter
        @show _nev, _ncv
        _maxiter = ifelse(_nev == nev, maxiter, 4)
        lamb, X, nconv, niter, lamberr = ss_iterate(factor, M, _nev, X, tol, iter, _maxiter, verbose) 
        if _nev == nev
            return lamb[1:nev], X[:, 1:nev], nconv, niter, lamberr
        end
        _nev, _ncv = _iteration_tactics(_nev)
        if _ncv - size(X, 2) > 0
            X = hcat(X, rand(eltype(K), size(X, 1), _ncv - size(X, 2)))
        end
    end
    return nothing
end

"""
    ss_iterate(K, M, nev, X, tol, maxiter, verbose) 

Iterate eigenvector subspace of the generalized eigenvalue problem
`K*X=M*X*Lambda`.

* `K` = _factorization_ of the (mass-shifted) stiffness matrix,
* `M` =  square symmetric mass matrix,
* `nev` = the number of eigenvalues sought,
* `X` =  initial guess of the eigenvectors, of dimension `size(M, 1)`x`ncv`,
    where `ncv > nev` = number of iteration vectors,
* `tol` = relative tolerance on the eigenvalue, expressed in terms of norms 
      of the change of the eigenvalue estimates from iteration to iteration,
* `maxiter` =  maximum number of allowed iterations,
* `verbose` = verbose? (default is false).
"""
function ss_iterate(K, M, nev, X, tol, iter, maxiter, verbose) 
    nvecs = size(X, 2)
    verbose && println("Number of requested eigenvalues: $nev, number of iteration vectors: $nvecs")
    Y = deepcopy(X)
    Kr = fill(zero(eltype(K)), nvecs, nvecs)
    Mr = fill(zero(eltype(K)), nvecs, nvecs)
    Pr = fill(zero(eltype(K)), nvecs, nvecs)
    plamb = fill(zero(eltype(K)), nvecs) .+ Inf
    lamb = fill(zero(eltype(K)), nvecs)
    lamberr = fill(zero(eltype(K)), nvecs)
    converged = falses(nvecs)  # not yet
    nconv = 0
    __fastapproxqr! = (Threads.nthreads() > 1 ? __mgs3thr! : __mgs3!)
    # __fastapproxqr! = __computeQT!

    niter = 0
    Z = X #  Z ← Xₖ
    for i in iter:maxiter
        mul!(Y, M, Z) # Y ← M Xₖ
        Z .= K \ Y # Z ← ̅Xₖ₊₁ 
        mul!(Kr, Transpose(Z), Y) # ← ̅Xₖ₊₁ᵀ K ̅Xₖ₊₁ = ̅Xₖ₊₁ᵀ M Xₖ
        mul!(Y, M, Z) # Y ← M X̅ₖ₊₁
        mul!(Mr, Transpose(Z), Y) # ← ̅Xₖ₊₁ᵀ M ̅Xₖ₊₁
        decomp = eigen(Kr, Mr)
        ix = sortperm(real.(decomp.values))
        lamb .= real.(@view decomp.values[ix])
        Pr .= real.(@view decomp.vectors[:, ix])
        mul!(Y, Z, Pr) #  Xₖ₊₁ ← ̅Xₖ₊₁ Pᵣ
        __fastapproxqr!(Y) # orthonormalize the vectors in the subspace
        Z, Y = Y, Z # Z ←  Xₖ₊₁, Y is scratch space
        @. lamberr = abs(lamb - plamb) / abs(plamb)
        converged .= (lamberr .<= tol)
        nconv = length(findall(converged))
        verbose && println("Iteration $i: nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        lamb, plamb = plamb, lamb
        niter = niter + 1
    end
    return lamb, Z, nconv, niter, lamberr
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
#     return ss_iterate(factor, M, nev, v0, tol, maxiter, verbose) 
# end