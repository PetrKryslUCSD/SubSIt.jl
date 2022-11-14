module mbas001
using Test
using LinearAlgebra
using Arpack
using SubSIt: ssit
using DataDrop
neigvs = 20                   # how many eigenvalues
OmegaShift = (0.01*2*pi)^2;

function test_ssit(f, reffs, tol = 0.001)
    K = DataDrop.retrieve_matrix(f, "/K")
    M = DataDrop.retrieve_matrix(f, "/M")
    d,v,nconv = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, which=:SM, check=1)
    d = d .- OmegaShift;
    fs = real(sqrt.(complex(d)))/(2*pi)
    @test norm(fs - reffs) < tol
    true
end
for d in (
    ("unit_cube_8.h5", [0.00000e+00, 0.00000e+00, 3.16833e-08, 1.09714e-07, 1.12730e-07, 1.87727e-07, 2.51509e-01, 2.56099e-01, 3.37083e-01, 3.39137e-01, 3.42637e-01, 3.48691e-01, 3.49059e-01, 3.50360e-01, 3.77199e-01, 3.96925e-01, 3.97760e-01, 4.27904e-01, 4.29469e-01, 4.30729e-01]),
    ("unit_cube_16.h5", [0.00000e+00, 0.00000e+00, 1.39971e-07, 1.98557e-07, 2.46956e-07, 3.04491e-07, 2.59393e-01, 2.60642e-01, 3.51863e-01, 3.52430e-01, 3.53351e-01, 3.57428e-01, 3.57552e-01, 3.57918e-01, 3.99576e-01, 4.05492e-01, 4.05636e-01, 4.52988e-01, 4.53485e-01, 4.54343e-01])
    )
    test_ssit(d...)
end

end
