module trunc_cyl_shell_examples
using Test
using FinEtools
using FinEtoolsDeforLinear
using LinearAlgebra
using Arpack
using SubSIt: ssit

function trunc_cyl_shell()
    println("""
    Vibration modes of truncated cylindrical shell.
    """)

    # t0 = time()

    E = 205000*phun("MPa");# Young's modulus
    nu = 0.3;# Poisson ratio
    rho = 7850*phun("KG*M^-3");# mass density
    OmegaShift = (2*pi*100) ^ 2; # to resolve rigid body modes
    h = 0.05*phun("M");
    l = 10*h;
    Rmed = h/0.2;
    psi   = 0;    # Cylinder
    nh = 5; nl  = 30; nc = 80;
    tolerance = h/nh/100;
    neigvs = 20;

    MR = DeforModelRed3D
    fens,fes  = H8block(h,l,2*pi,nh,nl,nc)
    # Shape into a cylinder
    R = zeros(3, 3)
    for i = 1:count(fens)
        x, y, z = fens.xyz[i,:];
        rotmat3!(R, [0, z, 0])
        Q = [cos(psi*pi/180) sin(psi*pi/180) 0;
        -sin(psi*pi/180) cos(psi*pi/180) 0;
        0 0 1]
        fens.xyz[i,:] = reshape([x+Rmed-h/2, y-l/2, 0], 1, 3)*Q*R;
    end
    candidates = selectnode(fens, plane = [0.0 0.0 1.0 0.0], thickness = h/1000)
    fens,fes = mergenodes(fens, fes,  tolerance, candidates);

    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz,1),3)) # displacement field

    numberdofs!(u)

    material=MatDeforElastIso(MR, rho, E, nu, 0.0)

    femm = FEMMDeforLinearMSH8(MR, IntegDomain(fes, GaussRule(3,2)), material)
    femm = associategeometry!(femm, geom)
    K =stiffness(femm, geom, u)
    femm = FEMMDeforLinear(MR, IntegDomain(fes, GaussRule(3,3)), material)
    M =mass(femm, geom, u)
    
    tim = @elapsed begin
        evals, evecs, convinfo = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "SubSIt: Time $tim [sec]"
    reffs = fs

    tim = @elapsed begin
        evals, evecs, convinfo = eigs(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, which=:SM, explicittransform=:none)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "Arpack: Time $tim [sec]"
    @test norm(reffs - fs) / norm(reffs) < 1.0e-3


    tim = @elapsed begin
        evals, evecs, convinfo = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, ncv=2*neigvs)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "SubSIt: Time $tim [sec]"

    true

end # trunc_cyl_shell

trunc_cyl_shell()

end # module 
nothing