

function dykes_graph()
    fid = h5open(data_folder * "grid.120000.h5", "r")
    T = read(fid, "T")
    close(fid)
end
