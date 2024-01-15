module SubSIt

using LinearAlgebra

# function __coldot(A, j, i)
#     m = size(A, 1)
#     r = zero(eltype(A))
#     @simd for k in 1:m
#         r += A[k, i] * A[k, j]
#     end
#     return r;
# end

# function __colnorm(A, j)
#     return sqrt(__coldot(A, j, j));
# end

# function __colsubt!(A, i, j, r)
#     m = size(A, 1)
#     @simd for k in 1:m
#         A[k, i] -= r * A[k, j]
#     end
# end

# function __normalizecol!(A, j)
#     m = size(A, 1)
#     r = 1.0 / __colnorm(A, j)
#     @simd for k in 1:m
#         A[k, j] *= r
#     end
# end

# function __mgsthr!(A)
#     m, n = size(A)
#     __normalizecol!(A, 1)
#     for j in 2:n
#         Base.Threads.@threads for i in j:n
#             __colsubt!(A, i, j-1, __coldot(A, j-1, i))
#         end
#         __normalizecol!(A, j)
#     end
#     return A
# end

# function __mgsl3!(A)
#     m, n = size(A)
#     _one = one(eltype(A))
#     r = fill(zero(eltype(A)), 1, n)
#     __normalizecol!(A, 1)
#     for j in 2:n
#         vr = view(r, 1:1, j:n)
#         vAc = view(A, :, j-1:j-1)
#         vAb = view(A, :, j:n)
#         mul!(vr, transpose(vAc), vAb)
#         mul!(vAb, vAc, vr, -_one, _one)
#         __normalizecol!(A, j)
#     end
#     return A
# end

# # Copyright 2022, Michael Stewart
# function __mgs3thr!(A; block_size = 32)
#     m, n = size(A)
#     _one = one(eltype(A))
#     _zero = zero(eltype(A))
#     num_blocks, rem_block = divrem(n, block_size)
#     work = zeros(eltype(A), block_size, n)
#     @views for b = 0:(num_blocks - 1)
#         c0 = block_size * b + 1
#         c1 = block_size * (b + 1)
#         A1 = A[:, c0:c1]
#         A2 = A[:, (c1 + 1):n]
#         work2 = work[:, (c1 + 1):n]
#         __mgsthr!(A1)
#         mul!(work2, A1', A2, _one, _zero)
#         mul!(A2, A1, work2, -_one, _one)
#     end
#     if rem_block != 0
#         __mgsthr!(@views A[:, (end - rem_block + 1):end])
#     end
#     return A
# end

# # Copyright 2022, Michael Stewart
# function __mgs3!(A; block_size = 32)
#     m, n = size(A)
#     _one = one(eltype(A))
#     _zero = zero(eltype(A))
#     num_blocks, rem_block = divrem(n, block_size)
#     work = zeros(eltype(A), block_size, n)
#     @views for b = 0:(num_blocks - 1)
#         c0 = block_size * b + 1
#         c1 = block_size * (b + 1)
#         A1 = A[:, c0:c1]
#         A2 = A[:, (c1 + 1):n]
#         work2 = work[:, (c1 + 1):n]
#         __mgsl3!(A1)
#         mul!(work2, A1', A2, _one, _zero)
#         mul!(A2, A1, work2, -_one, _one)
#     end
#     if rem_block != 0
#         __mgsl3!(@views A[:, (end - rem_block + 1):end])
#     end
#     return A
# end

# # Copyright 2022, Michael Stewart
# function __computeQ!(A)
#     m, n = size(A)
#     k = min(m, n)
#     tau = zeros(eltype(A), k)
#     LAPACK.geqrf!(A, tau)
#     LAPACK.orgqr!(A, tau, k)
#     return A
# end

# # Copyright 2022, Michael Stewart
# function __computeQT!(A; nb = 32)
#     m, n = size(A)
#     k = min(m, n)
#     T = zeros(eltype(A), nb, k)
#     tau = zeros(eltype(A), k)
#     LAPACK.geqrt!(A, T)
#     b = Int(ceil(k / nb))
#     ib = k - (b - 1) * nb
#     for l = 1:(b - 1)
#         for j = 1:nb
#             tau[(l - 1) * nb + j] = T[j, (l - 1) * nb + j]
#         end
#     end
#     for j = 1:ib
#         tau[(b - 1) * nb + j] = T[j, (b - 1) * nb + j]
#     end
#     LAPACK.orgqr!(A, tau, k)
#     return A
# end

function __noop!(A)
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
function ssit(K, M; nev = 6, ncv=0, tol = 1.0e-4, maxiter = 300, verbose=false, which=:SM, X = fill(zero(eltype(K)), 0, 0), check=0, ritzvec=true, sigma=0.0, explicittransform=:none)
    which != :SM && error("Wrong type of eigenvalue requested; only :SM accepted")
    nev < 1 && error("Wrong number of eigenvalues: needs to be > 1") 
    if ncv <= 0 && size(X, 2) > 0
        ncv = size(X, 2)
        ncv < nev+1 && error("Insufficient number of iteration vectors: must be >= nev+1")
    end
    if tol  == 0.0
        tol = sqrt(eps(one(eltype(X))))
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
    _nev = ifelse(ncv == 0, max(Int(round(nev/4)+1), min(nev, 20)), nev)

    # If the number of iteration vectors was not specified, but we have the
    # iteration vectors, we will go with that
    _ncv = _iteration_tactics(_nev)[2]

    if size(X) == (0, 0)
        X = rand(eltype(M), (size(M, 1), _ncv))  
    end

    Kfactor = cholesky(K)
    
    iter = 0
    while iter < maxiter
        _maxiter = ifelse(_nev == nev, maxiter, 8)
        lamb, X, nconv, niter, lamberr = ss_iterate(Kfactor, M, _nev, X, tol, iter, _maxiter, verbose)
        if _nev == nev && nconv >= nev
            __mass_orthogonalize!(X, M)
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

* `Kfactor` = _factorization_ of the (mass-shifted) stiffness matrix,
* `M` =  square symmetric mass matrix,
* `nev` = the number of eigenvalues sought,
* `X` =  initial guess of the eigenvectors, of dimension `size(M, 1)`x`ncv`,
    where `ncv > nev` = number of iteration vectors,
* `tol` = relative tolerance on the eigenvalue, expressed in terms of norms 
      of the change of the eigenvalue estimates from iteration to iteration,
* `maxiter` =  maximum number of allowed iterations,
* `verbose` = verbose? (default is false).
"""
function ss_iterate(Kfactor, M, nev, X, tol, iter, maxiter, verbose)
    nvecs = size(X, 2)
    verbose && println("Number of requested eigenvalues: $nev, number of iteration vectors: $nvecs")
    Y = deepcopy(X)
    T = eltype(X)
    Kr = fill(zero(T), nvecs, nvecs)
    Mr = fill(zero(T), nvecs, nvecs)
    Pr = fill(zero(T), nvecs, nvecs)
    plamb = fill(zero(T), nvecs) .+ Inf
    lamb = fill(zero(T), nvecs)
    lamberr = fill(zero(T), nvecs)
    converged = falses(nvecs)  # not yet
    nconv = 0
    # __fastapproxqr! = (Threads.nthreads() > 1 ? __mgs3thr! : __mgs3!)
    # @info "No orthogonalization"
    __fastapproxqr! = __noop!

    niter = 0
    Z = X #  Z ← Xₖ
    for i in iter:maxiter
        mul!(Y, M, Z) # Y ← M Xₖ
        Z .= Kfactor \ Y # Z ← ̅Xₖ₊₁
        # begin
        #     Threads.@threads for i in axes(Y, 2)
        #         @views Z[:, i] .= Kfactor \ Y[:, i] # Z ← ̅Xₖ₊₁
        #     end
        # end
        mul!(Kr, Transpose(Z), Y) # ← ̅Xₖ₊₁ᵀ K ̅Xₖ₊₁ = ̅Xₖ₊₁ᵀ M Xₖ
        mul!(Y, M, Z) # Y ← M X̅ₖ₊₁
        mul!(Mr, Transpose(Z), Y) # ← ̅Xₖ₊₁ᵀ M ̅Xₖ₊₁
        decomp = eigen(Symmetric(Kr), Symmetric(Mr))
        ix = sortperm(real.(decomp.values))
        lamb .= real.(@view decomp.values[ix])
        Pr .= real.(@view decomp.vectors[:, ix])
        mul!(Y, Z, Pr) #  Xₖ₊₁ ← ̅Xₖ₊₁ Pᵣ
        @. lamberr = abs(lamb - plamb) / abs(plamb)
        converged .= (lamberr .<= tol)
        nconv = length(findall(converged))
        verbose && println("Iteration $i: nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            return lamb, Y, nconv, niter, lamberr
        end
        __fastapproxqr!(Y) # orthonormalize the vectors in the subspace
        Z, Y = Y, Z # Z ←  Xₖ₊₁, Y is scratch space
        lamb, plamb = plamb, lamb
        niter = niter + 1
    end
    # If we fell through, the variables have gone through a swap: revert it
    return plamb, Z, nconv, niter, lamberr
end

function __mass_orthogonalize!(v, M)
    for i in axes(v, 2)
        v[:, i] /= sqrt(dot(v[:, i], M * v[:, i]))
    end
    return v
end

"""
    check_M_orthogonality(v, M)

Check the mass-orthogonality of the eigenvectors.

# Returns

- `max_vMv_diag_error`, `max_vMv_offdiag_error`: absolute deviations of the
  diagonal entries of the reduced mass matrix from unity, absolute deviations
  of the off-diagonal entries of the reduced mass matrix from zero.
"""
function check_M_orthogonality(v, M)
    max_vMv_diag_error = 0.0
    max_vMv_offdiag_error = 0.0
    Mred = v' * M * v
    for i in 1:size(Mred, 1), j in i:size(Mred, 2)
        p = (Mred[i, j]+Mred[j, i]) / 2
        if i == j
            max_vMv_diag_error = max(max_vMv_diag_error, abs(p - 1))
        else
            max_vMv_offdiag_error = max(max_vMv_offdiag_error, abs(p))
        end
    end
    return max_vMv_diag_error, max_vMv_offdiag_error
end

"""
    check_K_orthogonality(d, v, K)

Check the stiffness-orthogonality of the eigenvectors.

# Returns

- `max_vKv_diag_error`, `max_vKv_offdiag_error`: absolute deviations of the
  diagonal entries of the reduced stiffness matrix from the eigenvalue squared,
  absolute deviations of the off-diagonal entries of the reduced stiffness
  matrix from zero.
"""
function check_K_orthogonality(d, v, K)
    max_vKv_diag_error = 0.0
    max_vKv_offdiag_error = 0.0
    Kred = v' * K * v
    for i in 1:length(d), j in i:length(d)
        p = (Kred[i, j]+Kred[j, i]) / 2
        if i == j
            max_vKv_diag_error = max(max_vKv_diag_error, abs(p - d[i]))
        else
            max_vKv_offdiag_error = max(max_vKv_offdiag_error, abs(p))
        end
    end
    return max_vKv_diag_error, max_vKv_offdiag_error
end

end # module
