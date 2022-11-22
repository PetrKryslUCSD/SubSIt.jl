module unit_cube_tet_examples

using Test
using FinEtools
using FinEtools.MeshExportModule
using FinEtoolsDeforLinear
using SubSIt: ssit
using LinearAlgebra
using Arpack
# using KrylovKit
using DataDrop

E = 1*phun("PA");
nu = 0.499;
rho = 1*phun("KG/M^3");
a = 1*phun("M"); b = a; h =  a;
n1 = 16;# How many element edges per side?
na =  n1; nb =  n1; nh  = n1;
                  # how many eigenvalues
OmegaShift = (0.01*2*pi)^2;

function unit_cube_esnice_ssit(N, neigvs = 20, tol=0.001)
    na,nb,nh = N, N, N
    # println("""
    # Vibration modes of unit cube  of almost incompressible material.

    # Reference: Puso MA, Solberg J (2006) A stabilized nodally integrated
    # tetrahedral. International Journal for Numerical Methods in
    # Engineering 67: 841-867.
    # """)

    MR = DeforModelRed3D
    fens,fes  = T4block(a,b,h, na,nb,nh)

    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz,1),3)) # displacement field

    numberdofs!(u)

    material = MatDeforElastIso(MR, rho, E, nu, 0.0)

    femm = FEMMDeforLinearESNICET4(MR, IntegDomain(fes, NodalSimplexRule(3)), material)
    associategeometry!(femm,  geom)
    K  = stiffness(femm, geom, u)
    M = mass(femm, geom, u)

    @info "size(M) = $(size(M))"
    # DataDrop.store_matrix("unit_cube_$N.h5", "/K", K)
    # DataDrop.retrieve_matrix("unit_cube_$N.h5", "/K")
    # DataDrop.store_matrix("unit_cube_$N.h5", "/M", M)


    @info "N=$(N), neigvs=$(neigvs), eigs"
    @time d,v,nconv = eigs(K+OmegaShift*M, M; nev=neigvs, which=:SM, explicittransform=:none)
    d = d .- OmegaShift;
    fs = real(sqrt.(complex(d)))/(2*pi)
    # println("Eigenvalues: $fs [Hz]")
    reffs = fs

    # @info "N=$(N), neigvs=$(neigvs), eigs"
    # @time d, v, info = geneigsolve((K+OmegaShift*M, M), neigvs, :SR; krylovdim = 2*neigvs, issymmetric = true, verbosity = 1)
    # @show info
    # d = d .- OmegaShift;
    # fs = real(sqrt.(complex(d)))/(2*pi)
    # # println("Eigenvalues: $fs [Hz]")
    # @test norm(fs - reffs) / norm(reffs) < tol

    @info "N=$(N), neigvs=$(neigvs), ssit"
    @time d,v,nconv = ssit(K+OmegaShift*M, M; nev=neigvs, verbose=true)
    d = d .- OmegaShift;
    fs = real(sqrt.(complex(d)))/(2*pi)
    # println("Eigenvalues: $fs [Hz]")
    @test norm(fs - reffs) / norm(reffs) < tol

    # mode = 17
    # scattersysvec!(u, v[:,mode])
    # File =  "unit_cube_esnice.vtk"
    # vtkexportmesh(File, fens, fes; vectors=[("mode$mode", u.values)])
    # @async run(`"paraview.exe" $File`)
    true
end # unit_cube_esnice

for N in (8, 16, 32)
    unit_cube_esnice_ssit(N, 20)
end


for N in (32, )
    unit_cube_esnice_ssit(N, 100)
    # unit_cube_esnice_ssit(N, 500)
end
end # module 
nothing