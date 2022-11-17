# module mcub001
# using Test
# using LinearAlgebra
# using Arpack
# using SubSIt: ssit
# using DataDrop
# neigvs = 20                   # how many eigenvalues
# OmegaShift = (0.01*2*pi)^2;

# function test_ssit(f, reffs, tol = 0.001)
#     K = DataDrop.retrieve_matrix(f, "/K")
#     M = DataDrop.retrieve_matrix(f, "/M")
#     @info "Matrix size = $(size(K))"
#     d,v,nconv = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, which=:SM, check=1)
#     d = d .- OmegaShift;
#     fs = real(sqrt.(complex(d)))/(2*pi)
#     @test norm(fs - reffs) < tol
#     true
# end
# for d in (
#     ("unit_cube_8.h5", [0.00000e+00, 0.00000e+00, 3.16833e-08, 1.09714e-07, 1.12730e-07, 1.87727e-07, 2.51509e-01, 2.56099e-01, 3.37083e-01, 3.39137e-01, 3.42637e-01, 3.48691e-01, 3.49059e-01, 3.50360e-01, 3.77199e-01, 3.96925e-01, 3.97760e-01, 4.27904e-01, 4.29469e-01, 4.30729e-01]),
#     ("unit_cube_16.h5", [0.00000e+00, 0.00000e+00, 1.39971e-07, 1.98557e-07, 2.46956e-07, 3.04491e-07, 2.59393e-01, 2.60642e-01, 3.51863e-01, 3.52430e-01, 3.53351e-01, 3.57428e-01, 3.57552e-01, 3.57918e-01, 3.99576e-01, 4.05492e-01, 4.05636e-01, 4.52988e-01, 4.53485e-01, 4.54343e-01])
#     )
#     test_ssit(d...)
# end

# end

module unit_cube_tet_examples

using Test
using FinEtools
using FinEtools.MeshExportModule
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule: ssit
using LinearAlgebra
using Arpack
using DataDrop

E = 1*phun("PA");
nu = 0.499;
rho = 1*phun("KG/M^3");
a = 1*phun("M"); b = a; h =  a;
n1 = 16;# How many element edges per side?
na =  n1; nb =  n1; nh  = n1;
neigvs = 20                   # how many eigenvalues
OmegaShift = (0.01*2*pi)^2;

function unit_cube_esnice_ssit(N, reffs, tol=0.001)
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
    @time d,v,nconv = eigs(K+OmegaShift*M, M; nev=neigvs)
    @time d,v,nconv = ssit(K+OmegaShift*M, M; nev=neigvs)
    d = d .- OmegaShift;
    fs = real(sqrt.(complex(d)))/(2*pi)
    println("Eigenvalues: $fs [Hz]")
    @test norm(fs - reffs) < tol

    # mode = 17
    # scattersysvec!(u, v[:,mode])
    # File =  "unit_cube_esnice.vtk"
    # vtkexportmesh(File, fens, fes; vectors=[("mode$mode", u.values)])
    # @async run(`"paraview.exe" $File`)
    true
end # unit_cube_esnice

for d in (
    (8, [0.00000e+00, 0.00000e+00, 3.16833e-08, 1.09714e-07, 1.12730e-07, 1.87727e-07, 2.51509e-01, 2.56099e-01, 3.37083e-01, 3.39137e-01, 3.42637e-01, 3.48691e-01, 3.49059e-01, 3.50360e-01, 3.77199e-01, 3.96925e-01, 3.97760e-01, 4.27904e-01, 4.29469e-01, 4.30729e-01]),
    (16, [0.00000e+00, 0.00000e+00, 1.39971e-07, 1.98557e-07, 2.46956e-07, 3.04491e-07, 2.59393e-01, 2.60642e-01, 3.51863e-01, 3.52430e-01, 3.53351e-01, 3.57428e-01, 3.57552e-01, 3.57918e-01, 3.99576e-01, 4.05492e-01, 4.05636e-01, 4.52988e-01, 4.53485e-01, 4.54343e-01])
    )
    unit_cube_esnice_ssit(d...)
end
end # module 
nothing