using DataDrop
using LinearAlgebra
using SubSIt: ssit, check_M_orthogonality, check_K_orthogonality

function __load_pencil(b)
    K = DataDrop.retrieve_matrix(b, "/K")
    M = DataDrop.retrieve_matrix(b, "/M")
    K, M
end

function __load_frequencies(b)
    DataDrop.retrieve_matrix(b, "/frequencies")
end

orthogonality_tol = 1.0e-9
frequency_tol = 1.0e-6
residual_tol = 1.0e-6

b = "unit_cube_tet-16"
if !isfile(joinpath(dirname(@__FILE__()), b * ".h5"))
    success(run(`unzip -qq -d $(dirname(@__FILE__())) $(joinpath(dirname(@__FILE__()), "unit_cube.zip"))`; wait = false))
end

let

    omega_shift = 2 * pi * 0.1


    for b  in  ["unit_cube_modes-h20-n1=3", "unit_cube_modes-h8-n1=3", "unit_cube_tet-16"]
        @info "Input $b"

        K, M = __load_pencil(b)
        @show fs = __load_frequencies(b)

        for neigvs in [3, 6, 9, 12, 17, 20]
            @info "Number of eigenvalues $(neigvs)"

            @time d, v, nconv = ssit(K+omega_shift^2*M, M; nev=neigvs, tol = 0.0, verbose=true)
            d .-= omega_shift^2

            @test length(d) == size(v, 2)
            @test length(d) == neigvs

            @show fs1 = sqrt.(abs.(d)) ./ (2*pi)

            @test nconv >= neigvs

            @test norm(fs1 - fs[1:length(fs1)]) / norm(fs) <= frequency_tol

            r = check_K_orthogonality(d, v, K)
            @test norm(r, Inf) <= max(10000 * eps(1.0), orthogonality_tol * maximum(d))
            r = check_M_orthogonality(v, M)
            @test norm(r, Inf) <= orthogonality_tol
        end
    end


    nothing
end
