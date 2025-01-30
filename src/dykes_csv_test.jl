using CSV, DataFrames

function d2d_create_csv()
    #capri
    EruptionVolumesVec = Vector{Float64}([265, 16, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.201, 0.06, 0.05, 0.02, 0.07, 0.930, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029])
    EruptionTimesVec = Vector{Float64}([39.8, 29.3, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 10.6, 9.6, 9.3, 5.1, 4.7, 4.9, 4.5, 4.3, 4.2, 4.1, 3.9, 0.5])

    A = DataFrame(EruptionVolumes=EruptionVolumesVec, EruptionTimes=EruptionTimesVec)
    filename = "test.csv"
    filepath = joinpath(@__DIR__, filename)
    open(filepath, "w") do io
        CSV.write(io, A)
    end
end

function d2d_read_csv()
    filename = "test.csv"
    filepath = joinpath(@__DIR__, filename)
    #println(Vector{Float64}(collect(A[1:end, :EruptionVolumes])))
    return CSV.read(filepath, DataFrame)
end
