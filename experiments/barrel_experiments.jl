
# """
# Vibration analysis of free-floating thin plate

# Free-vibration problem is solved for a homogeneous free-floating
# (unsupported) square plate. This is the NAFEMS Benchmark, Test No. FV12.

# The plate is discretized with shell elements. Because no displacements are
# prevented, the structure has six rigid body modes (six zero vibration
# frequencies).

# The nonzero benchmark frequencies are (in hertz): 

# 1.622, 2.360, 2.922, 4.190, 4.190,  7.356, 7.356, 7.668.
# """
module barrel_w_stiffeners_examples

using Test
using LinearAlgebra
using Arpack
using ArnoldiMethodConvenienceWrapper
using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsFlexStructures.FESetShellT3Module: FESetShellT3
using FinEtoolsFlexStructures.FEMMShellT3FFModule
using FinEtoolsFlexStructures.RotUtilModule: initial_Rfield, update_rotation_field!
using DataDrop
using DataDrop: with_extension
using SubSIt: ssit
using FinEtools.MeshExportModule.VTKWrite: vtkwrite

function _execute_model(formul, input, visualize = true)
    E = 200e3*phun("MPa")
    nu = 0.3;
    rho= 7850*phun("KG/M^3");
    thickness = 2.0*phun("mm");
    OmegaShift = (100*2*pi)^2
    
    # Report
    @info "Mesh: $input"

    if !isfile(joinpath(dirname(@__FILE__()), input))
        success(run(`unzip -qq -d $(dirname(@__FILE__())) $(joinpath(dirname(@__FILE__()), "barrel_w_stiffeners-s3-mesh.zip"))`; wait = false))
    end
    output = FinEtools.MeshImportModule.import_H5MESH(joinpath(dirname(@__FILE__()), input))
    fens, fes  = output["fens"], output["fesets"][1]
    fens.xyz .*= phun("mm");

    # output = import_ABAQUS(joinpath(dirname(@__FILE__()), input))
    # fens = output["fens"]
    # fes = output["fesets"][1]

    # connected = findunconnnodes(fens, fes);
    # fens, new_numbering = compactnodes(fens, connected);
    # fes = renumberconn!(fes, new_numbering);

    # fens, fes = mergenodes(fens, fes, thickness/10)
    
    # FinEtools.MeshExportModule.H5MESH.write_H5MESH(with_extension(input, "h5mesh"), fens, fes)
    
    @info "Number of nodes and elements: $(count(fens)), $(count(fes))"

    mater = MatDeforElastIso(DeforModelRed3D, rho, E, nu, 0.0)
    
    sfes = FESetShellT3()
    accepttodelegate(fes, sfes)
    femm = formul.make(IntegDomain(fes, TriRule(1), thickness), mater)
    associate = formul.associategeometry!
    stiffness = formul.stiffness
    mass = formul.mass

    # Construct the requisite fields, geometry and displacement
    # Initialize configuration variables
    geom0 = NodalField(fens.xyz)
    u0 = NodalField(zeros(size(fens.xyz,1), 3))
    Rfield0 = initial_Rfield(fens)
    dchi = NodalField(zeros(size(fens.xyz,1), 6))

    # Apply EBC's
    # l1 = collect(1:count(fens))
    # for i in [6]
    #     setebc!(dchi, l1, true, i)
    # end
    applyebc!(dchi)
    numberdofs!(dchi);

    # Assemble the system matrix
    tim = @elapsed begin
        associategeometry!(femm, geom0)
    end
    @info "Associate Geometry time $tim [sec]"
    tim = @elapsed begin
        K = stiffness(femm, geom0, u0, Rfield0, dchi);
    end
    @info "Stiffness assembly time $tim [sec]"
    tim = @elapsed begin
        M = mass(femm, geom0, dchi);
    end
    @info "Mass assembly time $tim [sec]"

    # DataDrop.store_matrix("barrel_w_stiffeners.h5", "/K", K)
    # DataDrop.store_matrix("barrel_w_stiffeners.h5", "/M", M)

    # Solve
    neigvs = 40

    tim = @elapsed begin
        evals, evecs, convinfo = Arpack.eigs(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, which=:SM, explicittransform=:none)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "Arpack: Time $tim [sec]"
    reffs = fs
    

    tim = @elapsed begin
        evals, evecs, convinfo = ArnoldiMethodConvenienceWrapper.eigs(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, which=:SM, explicittransform=:none)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "ArnoldiMethod: Time $tim [sec]"
    reffs = fs
    
    tim = @elapsed begin
        evals, evecs, convinfo = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, verbose=true)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "SubSIt: Time $tim [sec]"
    @test norm(reffs - fs, Inf) / norm(reffs, Inf) < 2.0e-3


    tim = @elapsed begin
        evals, evecs, convinfo = ssit(Symmetric(K+OmegaShift*M), Symmetric(M); nev=neigvs, X = rand(size(K, 1), neigvs*2), verbose=true)
    end
    @show convinfo
    evals[:] = evals .- OmegaShift;
    fs = real(sqrt.(complex(evals)))/(2*pi)
    @info "Frequencies: $(round.(fs[7:15], digits=4))"
    @info "SubSIt w/ initial guess: Time $tim [sec]"
    @test norm(reffs - fs, Inf) / norm(reffs, Inf) < 2.0e-3

    return true
end


function test_convergence()
    input = joinpath("../test", "barrel_w_stiffeners-s3.h5mesh")
    formul = FEMMShellT3FFModule
    @info "Barrel With Stiffeners, free vibration"
    @time _execute_model(formul, input, true)
    return true
end

test_convergence()

end # module
nothing
