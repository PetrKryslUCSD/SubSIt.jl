using SubSIt
using Test
using ZipFile


function unzip(file,exdir="")
    fileFullPath = isabspath(file) ?  file : joinpath(pwd(),file)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    zarchive = ZipFile.Reader(fileFullPath)
    for f in zarchive.files
        fullFilePath = joinpath(outPath,f.name)
        if (endswith(f.name,"/") || endswith(f.name,"\\"))
            mkdir(fullFilePath)
        else
            write(fullFilePath, read(f))
        end
    end
    close(zarchive)
end

@testset "Incompressible cube given by matrices" begin
    include("test_cube_mat.jl")
end

@testset "Truncated cylindrical solid shell" begin
    include("test_trunc_cyl_shell.jl")
end

@testset "Incompressible cube" begin
    include("test_cube.jl")
end

@testset "Barrel shell" begin
    unzip("barrel_w_stiffeners-s3-mesh.zip")
    include("test_barrel.jl")
end
