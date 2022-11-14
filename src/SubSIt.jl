module SubSIt

using LinearAlgebra

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
function ssit(K, M; nev = 6, ncv=max(20, 2*nev+1), v0 = fill(eltype(K), 0, 0), tol = 1.0e-3, maxiter = 300, verbose=false, which=:SM, check=0) 
    @assert which == :SM
    @assert nev >= 1
    if size(v0) == (0, 0)
        v0 = [i==j ? one(eltype(K)) : zero(eltype(K)) for i=1:size(K,1), j=1:ncv]
    end
    v = deepcopy(v0)
    @assert nev <= size(v0, 2)
    nvecs = size(v, 2)  # How many eigenvalues are iterated?
    plamb = zeros(nvecs)  # previous eigenvalue
    lamb = zeros(nvecs)
    lamberr = zeros(nvecs)
    converged = falses(nvecs)  # not yet
    niter = 0
    nconv = 0
    factor = cholesky(K)
    Kv = zeros(size(K, 1), nvecs)
    Mv = zeros(size(M, 1), nvecs)
    
    for i = 1:maxiter
        mul!(Mv, M, v)
        v .= factor \ Mv
        _mass_normalize!(v, M)
        mul!(Kv, K, v)
        mul!(Mv, M, v)
        decomp = eigen(transpose(v)*Kv, transpose(v)*Mv)
        ix = sortperm(real.(decomp.values))
        evalues = decomp.values[ix]
        evectors = decomp.vectors[:, ix]
        # v .= v * real.(evectors)
        mul!(Mv, v, real.(evectors))
        v, Mv = Mv, v
        lamb .= real.(evalues)
        for j = 1:nvecs
            if abs(lamb[j]) <= tol # zero eigenvalues
                lamberr[j] = abs(lamb[j])
            else # non zero eigenvalues
                lamberr[j] = abs(lamb[j] - plamb[j])/abs(lamb[j])
            end
            converged[j] = lamberr[j] <= tol
        end
        nconv = length(findall(converged[1:nev]))
        verbose && println("nconv = $(nconv)")
        if nconv >= nev # converged on all requested eigenvalues
            break
        end
        plamb, lamb = lamb, plamb # swap the eigenvalue arrays
        niter = niter + 1
    end
    _mass_normalize!(v, M)
    return lamb[1:nev], v[:, 1:nev], nconv, niter, lamberr
end

function _mass_normalize!(v, M)
    for k in axes(v, 2)
        v[:, k] ./= @views sqrt(v[:, k]' * M * v[:, k])
    end
    v
end

end