module SubSIt

using LinearAlgebra
using SparseArrays
# using MKLSparse
import SparseArrays: findnz
# using ThreadedSparseCSR

findnz(A::T) where {T<:LinearAlgebra.Symmetric{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}}  = findnz(A.data)

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
* `sigma` = ignored.

# Output
* `labm` = computed eigenvalues,
* `v` = computed eigenvectors,
* `nconv` = number of converged eigenvalues
* `niter` = number of iterations taken
* `lamberr` = eigenvalue errors, defined as normalized  differences  of
    successive  estimates of the eigenvalues.
"""
function ssit(K, M; nev = 6, ncv=0, tol = 2.0e-3, maxiter = 300, verbose=false, which=:SM, X = fill(eltype(K), 0, 0), check=0, ritzvec=true, sigma=0.0) 
    which != :SM && error("Wrong type of eigenvalue requested; only :SM accepted")
    nev < 1 && error("Wrong number of eigenvalues: needs to be > 1") 

    ncv_tactics(_nev) = min(_nev + 50, Int(round(_nev * 2.0)))
    
    _nev = max(Int(round(nev/4)+1), 20)
    ncv = ncv_tactics(_nev)
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
        ncv = size(X, 2)
        ncv < nev+1 && error("Insufficient number of iteration vectors")
    end

    factor = cholesky(K)

    
    iter = 0
    while iter < maxiter
        _maxiter = ifelse(_nev == nev, maxiter, 4)
        lamb, X, nconv, niter, lamberr = ss_iterate(factor, M, _nev, X, tol, iter, _maxiter, verbose) 
        if _nev == nev
            return lamb[1:nev], X[:, 1:nev], nconv, niter, lamberr
        end
        _nev = _nev * 2
        _nev = min(nev, _nev)
        ncv = ncv_tactics(_nev)
        X = hcat(X, rand(eltype(K), size(X, 1), ncv - size(X, 2)))
    end
    return nothing
end

"""
    ss_iterate(K, M, nev, X, tol, maxiter, verbose) 

Iterate subspace.

Iterate eigenvector subspace of the generalized eigenvalue problem
`K*X=M*X*Lambda`.

`K` = factorization of the stiffness matrix,
"""
function ss_iterate(K, M, nev, X, tol, iter, maxiter, verbose) 
    nvecs = size(X, 2)
    verbose && println("Number of requested eigenvalues: $nev, number of iteration vectors: $nvecs")
    Y = deepcopy(X)
    Kr = fill(zero(eltype(K)), nvecs, nvecs)
    Mr = fill(zero(eltype(K)), nvecs, nvecs)
    evalues = fill(zero(eltype(K)), nvecs)
    evectors = fill(zero(eltype(K)), nvecs, nvecs)
    plamb = fill(zero(eltype(K)), nvecs) .+ Inf
    lamb = fill(zero(eltype(K)), nvecs)
    lamberr = fill(zero(eltype(K)), nvecs)
    converged = falses(nvecs)  # not yet
    nconv = 0

    niter = 0
    mul!(Y, M, X)
    for i in iter:maxiter
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
        verbose && println("Iteration $i: nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        lamb, plamb = plamb, lamb
        niter = niter + 1
    end
    return lamb, Y, nconv, niter, lamberr
end

function gsqr!(A)
    @show n = size(A, 2)
    R = fill(zero(eltype(A)), n, n); # initialize R
    
    for col in 1:n # for the first, second, third, ..., m-th column of Q
        R[col,col] = norm(A[:,col]);
        if (abs(R[col,col]) <= 0)# < eps(one(eltype(A))))
            @show R
            error("Matrix is singular");
        end
        A[:,col] ./= R[col,col]; # normalize column
        for ncol in col+1:n
            R[col,ncol] = dot(A[:,col], A[:,ncol]);
            A[:,ncol] -= R[col,ncol] .* A[:,col]; # subtract projection
        end
    end
    return A
end

function gramschmidt!(U)
n,k = size(U);
U = zeros(n,k);
U[:,1] = U[:,1]/norm(U[:,1]);
for i = 2:k
    for j=1:i-1
        U[:,i]=U[:,i]-(U[:,j]'*U[:,i]) * U[:,j];
    end
    U[:,i] = U[:,i]/norm(U[:,i]);
end
end

# function [Q,R] = gsqr (A)
#     % Check that the input matrix is square
#     [m,n] = size(A);
#     if (m < n)
#         disp('Error: matrix A has not sufficient rank: more columns than rows');
#         Q = zeros(m,n); % return something initialized
#         R = eye(m,n); % return something initialized
#         return
#     end
    
#     R = zeros(n,n); % initialize R
    
#     for col = 1:n % for the first, second, third, ..., m-th column of Q
#         R(col,col) = norm(A(:,col));
#         if (abs(R(col,col)) < eps)
#             error(['Matrix is singular']);
#         end
#         A(:,col) = A(:,col)/R(col,col); % normalize column
#         for ncol = col+1:n
#             R(col,ncol) = A(:,col)'*A(:,ncol);
#             A(:,ncol) = A(:,ncol) -  R(col,ncol) * A(:,col); % subtract projection
#         end
#     end
#     Q=A; % output Q
# end

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