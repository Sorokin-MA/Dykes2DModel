"""
Print graphs for d2dm output
"""

using CSV, DataFrames
using Dykes2DModel
using PlotlyJS
using Plots

#include("dykes_init.jl")
#include("dykes_funcs.jl")


function dykes_graph()

    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")
    #Lx = read(fid, "Lx")
    #Ly = read(fid, "Ly")
    #close(fid)

    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")

    # dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
    # ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script
    #
    # io = open(data_folder * "pa.bin", "r")
    # read!(io, dpa)
    # read!(io, ipa)
    #
    # ipar = 1
    # Lx, ipar = read_par(dpa, ipar)
    # Ly, ipar = read_par(dpa, ipar)
    # lam_r_rhoCp, ipar = read_par(dpa, ipar)
    # lam_m_rhoCp, ipar = read_par(dpa, ipar)
    # L_Cp, ipar = read_par(dpa, ipar)
    # T_top, ipar = read_par(dpa, ipar)
    # T_bot, ipar = read_par(dpa, ipar)
    # T_magma, ipar = read_par(dpa, ipar)
    # tsh, ipar = read_par(dpa, ipar)
    # gamma, ipar = read_par(dpa, ipar)
    # Ly_eruption, ipar = read_par(dpa, ipar)
    # nu, ipar = read_par(dpa, ipar)
    # G, ipar = read_par(dpa, ipar)
    # dt, ipar = read_par(dpa, ipar)
    # dx, ipar = read_par(dpa, ipar)
    # dy, ipar = read_par(dpa, ipar)
    # eiter, ipar = read_par(dpa, ipar)
    # pic_amount, ipar = read_par(dpa, ipar)
    # tfin, ipar = read_par(dpa, ipar)
    #
    # ipar = 1
    #
    # pmlt, ipar = read_par(ipa, ipar)
    # nx, ipar = read_par(ipa, ipar)
    # ny, ipar = read_par(ipa, ipar)
    # nl, ipar = read_par(ipa, ipar)
    # nt, ipar = read_par(ipa, ipar)
    # niter, ipar = read_par(ipa, ipar)
    # nout, ipar = read_par(ipa, ipar)
    # nsub, ipar = read_par(ipa, ipar)
    # nerupt, ipar = read_par(ipa, ipar)
    # npartcl, ipar = read_par(ipa, ipar)
    # nmarker, ipar = read_par(ipa, ipar)
    # nSample, ipar = read_par(ipa, ipar)
    #
    # close(io)
    #
    #Lx = 4000
    #Lx = 5000
    # xs = 0:dx:Lx
    # ys = 0:dy:Ly

    #    fid = h5open(data_folder * "julia_grid.40001.h5", "r")
    # fid = h5open(data_folder * "julia_grid.60.h5", "r")
    # T = read(fid, "T")
    # C = read(fid, "C")
    # close(fid)
    #
    # l = @layout [[grid(1, 2)]
    #     a]

    #=
    	campi_calc = Int32[10000, 20000, 25000, 26000]
    	println(typeof(campi_calc))
    	file = open(data_folder*"eruptions.bin", "w")
    	write(file, campi_calc)
    	close(file)
    =#

    # fz = filesize(data_folder * "eruptions.bin")
    # fz_int = Int32(floor(fz / sizeof(Int32)))
    #
    # println(fz / sizeof(Int32))
    # if (fz_int <= 1)
    #     println("No eruptions!!!")
    #     campi_calc_fake = Int32[10000, 20000, 25000, 26000]
    #     println(typeof(campi_calc_fake))
    #     file = open(data_folder * "eruptions.bin", "w")
    #     write(file, campi_calc_fake)
    #     close(file)
    #
    # end
    # campi_calc = Array{Int32,1}(undef, fz_int)#array of int values from matlab script
    # read!(data_folder * "eruptions.bin", campi_calc)
    #
    # campi_calc = @view campi_calc[2:end]
    #
    # println(campi_calc)
    # tyear = 365 * 24 * 3600#seconds in year
    # tfin = (tfin / tyear) / 1.e3
    # campi_calc = -(1 .- campi_calc ./ nt) .* (tfin)
    # println(campi_calc)
    # display(campi_calc)
    # campi_real = -vcat(39.8, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 11.5, 11, 10.6, 9.6, 9.3, 5.1, 4.9, 4.5, 4.3, 4.2, 4.2, 4.2, 4.1, 3.9, 0.5)


    A_x = range(699.2355, 1186.2385, 51);



    p_rock = PlotlyJS.scatter(x=A_x, y=d2dm_mf_rock(A_x), mode="lines", name="mf, rock", marker_color="rgba(255, 0, 0, 1)")
    # p_rock = plot(A_x, d2dm_mf_rock(A_x), markersize=7,
    #     markershape=:circle, color=:red, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="mf, rock")

    p_magma = PlotlyJS.scatter(x=A_x, y=d2dm_mf_magma.(A_x), mode="lines", name="mf, magma", marker_color="rgba(0, 0, 255, 1)")
    # p_magma = plot(A_x, d2dm_mf_magma.(A_x), markersize=7,
    #     markershape=:circle, color=:red, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="mf, magma")
    #
    #

# Example DataFrame
df = DataFrame(mf_rock_x = A_x, mf_rock_y = d2dm_mf_rock(A_x), mf_magma_x = A_x, mf_magma_y = d2dm_mf_magma.(A_x))

# Save the DataFrame to a CSV file
# CSV.write("d2dm_crystalisation.csv", df)
    data = [p_rock, p_magma]

    layout = Layout(title="Crystalisation graph",
        xaxis=attr(title="T, (°C)"),
        yaxis=attr(title="mf"))


    p = PlotlyJS.plot(data, layout)


    # p = Plots.scatter!(campi_calc, zeros(length(campi_calc)), markersize=4,
    #     markershape=:circle, color=:blue, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campi", markeralpha=0.5)

    # T = reshape(T, (length(xs), length(ys)))
    # C = reshape(C, (length(xs), length(ys)))
    # p1 = Plots.plot(Plots.heatmap(ys, xs, transpose(T)), Plots.heatmap(ys, xs, transpose(C)), p, layout=l)

    display(p)

    savefig(p, "array_plot.png")
    # return p
end

function d2dm_markers_graph()
    fid_after = h5open(data_folder * "julia_grid.22001.h5", "r")
    #fid_after = h5open(data_folder * "julia_grid.6382.after_eruption.h5", "r")
    fid_before= h5open(data_folder * "markers.h5", "r")

    mx_before = read(fid_before, "/0/mx")
    my_before = read(fid_before, "/0/my")
    mT_before = read(fid_before, "/0/mT")

    mx_before = mx_before[1:10:end]
    my_before = my_before[1:10:end]

    mx = read(fid_after, "mx")
    my = read(fid_after, "my")

    mx = mx[1:10:end]
    my = my[1:10:end]

    mx = mx[my.>5000]
    my = my[my.>5000]


    erupt_x = read(fid_after, "erupt_x")
    erupt_y = read(fid_after, "erupt_y")

    erupt_x = erupt_x[erupt_x.>0]
    erupt_y = erupt_y[erupt_y.>0]

	println(erupt_x)
	println(erupt_y)

	println("Minimum value: ", minimum(my))
	println("Maximum value: ", maximum(my))

    #close(fid)

    #println(mx)

    A_x = range(699.2355, 1186.2385, 51);

	println(typeof(mx))

	#mx = range(0, 10, length=100)
	#my = sin.(mx)
    #plotik = Plots.scatter(x=mx, y=my, xlims=(-10000, 20000), ylims=(-10000, 20000))
    #Plots.plot(x=mx, y=my, xlims=(-10000, 20000), ylims=(-10000, 20000))
    # Plots.scatter!(plotik, x=A_x, y=d2dm_mf_magma.(A_x), mode="lines", name="mf, magma", marker_color="rgba(0, 0, 255, 1)")


	colors = 1:length(erupt_x)  # Color by order of points

    l = @layout [[grid(1, 3)]]

	x = range(0, 10, length=100)
	y = sin.(x)
	Plots.plot(mx_before, my_before, xlims=(0, 20000), ylims=(0, 20000), xlabel='x', ylabel='y', title="Downlift on Campi Flegrei", label="Init")
	Plots.scatter!(erupt_x, erupt_y, xlims=(0, 20000), ylims=(0, 20000), label="Eruption Centers")
for i in 1:length(erupt_x)
    Plots.annotate!(erupt_x[i], erupt_y[i], text(string(i), 3, :black))  # Adding labels with index numbers
end
	p2 = Plots.scatter!(mx, my, xlims=(0, 20000), ylims=(0, 20000), label="Final")
	# p3 = Plots.scatter(mT_before)


	p_final = Plots.plot(p2)

	#Plots.savefig(p_final, "markers.png")
    #return p_final

    # p = Plots.plot(data, layout)


    # p = Plots.scatter!(campi_calc, zeros(length(campi_calc)), markersize=4,
    #     markershape=:circle, color=:blue, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campi", markeralpha=0.5)

    # T = reshape(T, (length(xs), length(ys)))
    # C = reshape(C, (length(xs), length(ys)))
    # p1 = Plots.plot(Plots.heatmap(ys, xs, transpose(T)), Plots.heatmap(ys, xs, transpose(C)), p, layout=l)

    # display(p)

    # savefig(p, "array_plot.png")
    # return p
end


function d2dm_mf_graph()
    #fid_after = h5open(data_folder * "d2d_snapshot_23000_2025_07_12_18_09_17.hdf5", "r")
    fid_after = h5open("E:\\Melnik\\campri\\Dykes2DModel\\d2d_snapshot_23000_2025_07_12_18_09_17.hdf5", "r")

    cumulutive_time = read(fid_after, "cumulutive_time")
    nt = read(fid_after, "nt")
    calc_years = 230000
    cum_mf_01 = read(fid_after, "cum_mf_01")
    cum_mf_05 = read(fid_after, "cum_mf_05")
    cum_mf_10 = read(fid_after, "cum_mf_10")
    cum_mf_25 = read(fid_after, "cum_mf_25")
    cum_mf_50 = read(fid_after, "cum_mf_50")
    cum_mf_75 = read(fid_after, "cum_mf_75")
    cum_mf_85 = read(fid_after, "cum_mf_85")

	cumulutive_time = -(nt .- cumulutive_time) ./ nt .* calc_years / 1000


	p2 = Plots.plot(cumulutive_time, cum_mf_01, xlabel="time (ka)", ylabel="volume, km^3", title="Accomulated melt fraction under CF", label="1%")
#=
	Plots.plot!(cumulutive_time, cum_mf_05, label="5%")
=#

	Plots.plot!(cumulutive_time, cum_mf_10, label="10%")

#=
	Plots.plot!(cumulutive_time, cum_mf_25, label="25%")
	Plots.plot!(cumulutive_time, cum_mf_50, label="50%")
	Plots.plot!(cumulutive_time, cum_mf_75, label="75%")

=#
	Plots.plot!(cumulutive_time, cum_mf_85, label="85%")

	p_final = Plots.plot(p2)

	#Plots.savefig(p_final, "markers.png")
    #return p_final

    # p = Plots.plot(data, layout)


    # p = Plots.scatter!(campi_calc, zeros(length(campi_calc)), markersize=4,
    #     markershape=:circle, color=:blue, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campi", markeralpha=0.5)

    # T = reshape(T, (length(xs), length(ys)))
    # C = reshape(C, (length(xs), length(ys)))
    # p1 = Plots.plot(Plots.heatmap(ys, xs, transpose(T)), Plots.heatmap(ys, xs, transpose(C)), p, layout=l)

    # display(p)

    # savefig(p, "array_plot.png")
    # return p

end

function d2dm_Q_graph()
    fid_after = h5open(data_folder * "julia_grid.22001.true.h5", "r")
    fid_plus_10 = h5open(data_folder * "julia_grid.110001.plus10.h5", "r")
    fid_plus_20 = h5open(data_folder * "julia_grid.55001.plus20.h5", "r")
    fid_minus_10 = h5open(data_folder * "julia_grid.22001.minus10.h5", "r")
    fid_minus_20 = h5open(data_folder * "julia_grid.22001.minus20.h5", "r")


    cumulutive_time_true = read(fid_after, "cumulutive_time")
    cumulutive_time_plus10 = read(fid_plus_10, "cumulutive_time")
    cumulutive_time_plus20 = read(fid_plus_20, "cumulutive_time")
    cumulutive_time_minus10 = read(fid_minus_10, "cumulutive_time")
    cumulutive_time_minus20 = read(fid_minus_20, "cumulutive_time")
    nt = read(fid_after, "nt")
    nt_plus10 = read(fid_plus_10, "nt")
    nt_plus20 = read(fid_plus_20, "nt")

    calc_years = 220000
    cumulutive_calc_true = read(fid_after, "cumulutive_calc")
    cumulutive_calc_real= read(fid_after, "cumulutive_real")
    cumulutive_calc_plus10= read(fid_plus_10, "cumulutive_calc")
    cumulutive_calc_plus20= read(fid_plus_20, "cumulutive_calc")
    cumulutive_calc_minus10= read(fid_minus_10, "cumulutive_calc")
    cumulutive_calc_minus20= read(fid_minus_20, "cumulutive_calc")



	cumulutive_time_true = -(nt .- cumulutive_time_true) ./ nt .* calc_years / 1000
	cumulutive_time_plus10 = -(nt_plus10 .- cumulutive_time_plus10) ./ nt_plus10 .* calc_years / 1000
	cumulutive_time_plus20 = -(nt_plus20 .- cumulutive_time_plus20) ./ nt_plus20 .* calc_years / 1000
	cumulutive_time_minus10 = -(nt .- cumulutive_time_minus10) ./ nt .* calc_years / 1000
	cumulutive_time_minus20 = -(nt .- cumulutive_time_minus20) ./ nt .* calc_years / 1000


	p2 = Plots.plot(cumulutive_time_true, cumulutive_calc_true, lw=4, xlabel="time (ka)", ylabel="Cum. erupt. volume, km^3", title="Compartion of different Q", label="4.4e-3km^3/y")
	Plots.plot!(cumulutive_time_true, cumulutive_calc_real, label="real", lw=4)
	Plots.plot!(cumulutive_time_plus10, cumulutive_calc_plus10, label="+10%")
	Plots.plot!(cumulutive_time_plus20, cumulutive_calc_plus20, label="+20%")
	Plots.plot!(cumulutive_time_minus10, cumulutive_calc_minus10, label="-10%")
	Plots.plot!(cumulutive_time_minus20, cumulutive_calc_minus20, label="-20%")

	p_final = Plots.plot(p2)

	#Plots.savefig(p_final, "markers.png")
    #return p_final

    # p = Plots.plot(data, layout)


    # p = Plots.scatter!(campi_calc, zeros(length(campi_calc)), markersize=4,
    #     markershape=:circle, color=:blue, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campi", markeralpha=0.5)

    # T = reshape(T, (length(xs), length(ys)))
    # C = reshape(C, (length(xs), length(ys)))
    # p1 = Plots.plot(Plots.heatmap(ys, xs, transpose(T)), Plots.heatmap(ys, xs, transpose(C)), p, layout=l)

    # display(p)

    # savefig(p, "array_plot.png")
    # return p

end

function d2dm_resol_graph()
    fid_after = h5open(data_folder * "julia_grid.22001.true.h5", "r")
    fid_after_hr = h5open(data_folder * "julia_grid.22001.high_resolution.h5", "r")


    cumulutive_time_true = read(fid_after, "cumulutive_time")
    cumulutive_time_hr = read(fid_after_hr, "cumulutive_time")
    nt = read(fid_after, "nt")
    nt_hr = read(fid_after_hr, "nt")

    calc_years = 220000
    cumulutive_calc_true = read(fid_after, "cumulutive_calc")
    cumulutive_calc_real= read(fid_after, "cumulutive_real")
    cumulutive_calc_hr= read(fid_after_hr, "cumulutive_calc")


	cumulutive_time_true = -(nt .- cumulutive_time_true) ./ nt .* calc_years / 1000
	cumulutive_time_hr = -(nt_hr .- cumulutive_time_hr) ./ nt_hr .* calc_years / 1000

	#campi_calc = -(nt .- gp.eruptionSteps) ./ vp.nt .* init_vp.calc_years / 1000
	#campi_real = -vcat(init_vp.critVolTime)

	#show num cumulutive volume
	#campi_cumulut_graph = PlotlyJS.scatter(x=campi_cumulut, y=gp.cumulutive_calc, mode="lines", color=1, name="cumulative volume calc", marker_color="rgba(0, 0, 255, 1)")
	#show real cumulutive volume
	#campi_next_eruption = PlotlyJS.scatter(x=campi_cumulut, y=gp.cumulutive_real, mode="lines", color=2, name="cumulutive volume real", marker_color="rgba(255, 0, 0, 1)")
	#points of calc eruptions
	#campi_cal_graph = PlotlyJS.scatter(x=campi_calc, y=zeros(length(campi_calc)), mode="markers", name="eruptions, calc", showlegend=true, marker_size=14, marker_color="rgba(0, 0, 255, 1)")
	#points of real eruptions
	#campi_real_graph = PlotlyJS.scatter(x=campi_real, y=zeros(length(campi_real)), marker_color="rgba(255, 0, 0, 1)", mode="markers", color=2, name="eruptions, real", showlegend=true, marker_size=10)


	p2 = Plots.plot(cumulutive_time_true, cumulutive_calc_true, xlabel="time (ka)", ylabel="Cum. erupt. volume, km^3", title="Compartion of different resolutions", label="10m")
	Plots.plot!(cumulutive_time_hr, cumulutive_calc_hr, label="5m")
	Plots.plot!(cumulutive_time_true, cumulutive_calc_real, label="real")

	p_final = Plots.plot(p2)

	#Plots.savefig(p_final, "markers.png")
    #return p_final

    # p = Plots.plot(data, layout)


    # p = Plots.scatter!(campi_calc, zeros(length(campi_calc)), markersize=4,
    #     markershape=:circle, color=:blue, legend=true,
    #     framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campi", markeralpha=0.5)

    # T = reshape(T, (length(xs), length(ys)))
    # C = reshape(C, (length(xs), length(ys)))
    # p1 = Plots.plot(Plots.heatmap(ys, xs, transpose(T)), Plots.heatmap(ys, xs, transpose(C)), p, layout=l)

    # display(p)

    # savefig(p, "array_plot.png")
    # return p

end





