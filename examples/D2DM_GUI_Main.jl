"""
GUI for d2dm
"""
module D2DM_GUI_Main
    using Base64, Printf, Dates, HDF5, Random, Interpolations, Adapt, CUDA
    using NativeFileDialog, Dash, PlotlyJS, DataFrames, CSV
    using Dykes2DModel
    
    # Import our modules
    include("gui/D2DM_GUI_Constants.jl")
    include("gui/D2DM_GUI_State.jl")
    include("gui/D2DM_GUI_Interpolations.jl")
    include("gui/D2DM_GUI_Utils.jl")
    
    using .D2DM_GUI_Constants
    using .D2DM_GUI_State
    using .D2DM_GUI_Interpolations
    using .D2DM_GUI_Utils
    
    # Initialize interpolation
    init_interpolation!()
    
    # Export main function
    export dykes_gui
    
    # Main function
    function dykes_gui()
		# Init dash
		app = dash()
		
		# Init main structs
		gp = GridParams()
		vp = VarParams()
		init_vp = InitVarParams()


		# Initialize and get the interpolation object
		itp = get_cpu_interpolation()

		# Use GPU version (for CUDA operations)
		cuitp_gpu = get_interpolation()

		data_folder = DATA_FOLDER

		# Get constants from module
		eruption_volumes = ERUPTION_VOLUMES
		eruption_times = ERUPTION_TIMES
		
		init_vp.critVol = copy(eruption_volumes)
		init_vp.critVolTime = copy(eruption_times)

		mf_boundaries = 0;
		
		num_columns = 5
		
		# App layout (truncated for brevity - keep same as original)
		app.layout = html_div(style=Dict("background_color" => "blue")) do
			html_h1(
				"Dykes2DModel",
				style=Dict("color" => "#000000", "textAlign" => "center"),
			),
			dcc_upload(id="load-config-upload", html_button("Load config", id="load-config-but")),
			html_div() do
				html_button("Save config", id="save-config-but"),
				dcc_download(id="save-config-download")
			end,
			html_div() do
				html_h2(
					"1. Set parameters.",
					style=Dict("color" => "#000000", "textAlign" => "left"),
				),
				#html_button("Load config", id="load-config-but"),
				dcc_tabs(id="tabs-example-graph", value="tab-1-example-graph", children=[
					dcc_tab(label="Physics", value="tab-1-example-graph"),
					dcc_tab(label="Numerics", value="tab-2-example-graph"),
					dcc_tab(label="Eruptions", value="tab-3-example-graph")
				]
				),
				html_div(id="tabs-content-example-graph")
			end,
			html_br(),
			html_div(id="my-output"),
			html_h2(
				"2. Generate data.",
				style=Dict("color" => "#000000", "textAlign" => "left"),
			),
			html_button("Generate", id="generate-but"),
			html_button("Show dykes", id="show-dykes-but"),
			html_div(id="dykes-graph"),
			html_h2(
				"3. Start\\Stop.",
				style=Dict("color" => "#000000", "textAlign" => "left"),
			),
			#buttons
			html_button("Start", id="start-but", disabled=false),
			html_button("Stop", id="stop-but", disabled=false),
			html_button("Refresh", id="refresh-but", disabled=false),
			html_div(
				children=[
					dcc_input(id="time_left_label", value="Time left", debounce=true)
				],
			),
			html_div(
				children=[
					dcc_input(id="percent_done_label", value="Done", debounce=true)
				],
			),
			#html_progress(id = "progress_bar", value = string(vp.it), max = vp.nt, style=Dict("width" => "100%")),
			html_div(id="eruptions-timeline", className="row", style=Dict("columnCount" => 1)) do
				#dcc_graph(id="T_graph",figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))
				#dcc_graph(id="eruptions-timeline-graph")
			end,
			html_div(id="cumul-timeline", className="row", style=Dict("columnCount" => 1)) do
			end,
			html_div(id="next-cumul-timeline", className="row", style=Dict("columnCount" => 1)) do
			end,
			html_h2(
				"Log",
				style=Dict("color" => "#000000", "textAlign" => "left"),
			),
			html_div([
				dcc_textarea(
					id="log_buffer",
					value="Dykes2D GUI successfully uploaded!\n",
					style=Dict("width" => "100%", "overflow" => "scroll", "resize" => "none"),
					readOnly=true, rows=15
				),
				dcc_interval(id="interval-component",
					interval=1 * 5000, # in milliseconds
					n_intervals=1)
			]),
			html_h2(
				"4. Snapshots.",
				style=Dict("color" => "#000000", "textAlign" => "left"),
			),
			html_div() do
				html_button(
					"🔄 Load Snapshot",
					id="load-snapshot-but",
					n_clicks=0,
					style=Dict(
						"backgroundColor" => "#28a745",
						"color" => "white",
						"padding" => "10px 20px",
						"border" => "none",
						"borderRadius" => "5px",
						"cursor" => "pointer",
						"margin" => "5px"
					)
				),
				html_div(id="load-snapshot-div")
			end,
			html_div(id="snapshot-file-display", style=Dict("margin" => "10px 0")),
			html_div(id="snapshot-status-text", style=Dict("margin" => "10px 0")),
			html_div(id="snapshot-load-result"), 
			html_div() do
				html_div(
					children=[
						simple_imput("mf_boundaries", mf_boundaries)
					],
				)
			end,
			html_button("Show current T", id="show-cur-T-but", disabled=false),
			html_button("Show current C", id="show-cur-C-but", disabled=false),
			html_button("Show current mf", id="show-cur-mf-but", disabled=false),
			html_button("Show current dmf", id="show-cur-dmf-but", disabled=false),
			html_button("Show current chambers", id="show-cur-chambers-but", disabled=false),
			html_div(style=Dict("columnCount" => 2)) do
				html_div(id="T-graph"),
				html_div(id="mf-graph"),
				html_div(id="C-graph"),
				html_div(id="dmf-graph"),
				html_div(id="chambers-graph")
			end	
		end
		
		# Callbacks
		begin
			# Callback to load snapshot
			callback!(app, 
				[Output("load-snapshot-div", "children")], 
				[Input("load-snapshot-but", "n_clicks")],
				prevent_initial_call=true
			) do n_clicks
				# This opens a native OS dialog window to pick a file
				chosen_path = pick_file() 
				
				if isempty(chosen_path)
					return ["Path is empty"]
				end

				#filename = @sprintf("d2d_snapshot.hdf5")
			
				filename = chosen_path
			
				fid = h5open(filename, "r")
				
				for n in fieldnames(typeof(vp))
					setfield!(vp, n, read(fid, string(n)))
					println(string(n))
					println(getfield(vp, n))
				end
				
				for n in fieldnames(typeof(gp))
					if (getfield(gp, n) isa CuArray)
						d2d_cu_type = eltype(getfield(gp, n))
						nn::CuArray{d2d_cu_type,1} = CuArray{d2d_cu_type,1}(undef, size(getfield(gp, n))[1])
						nn = read(fid, string(n))
						setfield!(gp, n, nn)
						println("GPU")
						#println(getfield(gp,n))
					else
						setfield!(gp, n, read(fid, string(n)))
						println("CPU")
						#println(getfield(gp,n))
					end
				end
				

				println("snapshot loaded from " * filename)

				close(fid)
				
				return ["Path - $chosen_path"]
			end

			#Tabs content callback
			callback!(app, [Output("tabs-content-example-graph", "children")],
				[Input("tabs-example-graph", "value")]) do tab
				if tab == "tab-1-example-graph"
					return [html_div(className="row") do
						html_div(className="info", style=Dict("columnCount" => num_columns)) do
						end,
						html_div(className="row", style=Dict("columnCount" => num_columns)) do
							simple_imput_desc("Lx", init_vp.Lx, DESCR_LX),
							simple_imput_desc("Ly", init_vp.Ly, DESCR_LY),
							simple_imput_desc("Lz", init_vp.Lz, DESCR_LZ),
							simple_imput("calc_years", init_vp.calc_years),
							simple_imput("dyke_to_sill", init_vp.dyke_to_sill),
							simple_imput("narrow_fact", init_vp.narrow_fact),
							simple_imput("Lam_r", init_vp.Lam_r),
							simple_imput("Lam_m", init_vp.Lam_m),
							simple_imput("rho", init_vp.rho),
							simple_imput("Cp", init_vp.Cp),
							simple_imput("L_heat", init_vp.L_heat),
							simple_imput("T_top", init_vp.T_top),
							simple_imput("dTdy", init_vp.dTdy),
							simple_imput("T_magma", init_vp.T_magma),
							simple_imput("T_ch", init_vp.T_ch),
							simple_imput("Qv", init_vp.Qv),
							simple_imput("dyke_x_W", init_vp.dyke_x_W),
							#simple_imput("dyke_y_rng_bot", init_vp.dyke_y_rng_bot),
							#simple_imput("dyke_y_rng_top", init_vp.dyke_y_rng_top),
							simple_imput("Ly_eruption", init_vp.Ly_eruption),
							simple_imput("dT", init_vp.dT),
							simple_imput("E", init_vp.E),
							simple_imput("nu", init_vp.nu),
							simple_imput("tsh", init_vp.tsh),
							simple_imput("gamma", init_vp.gamma),
							simple_imput("dyke_nu", init_vp.dyke_nu),
							simple_imput("dyke_dev", init_vp.dyke_dev),
							simple_imput("dyke_type", init_vp.dyke_type)
						end
					end]
				end
				if tab == "tab-2-example-graph"
					return [html_div(className="row") do
						simple_imput("seed", init_vp.seed),
						html_div(className="row", style=Dict("columnCount" => num_columns)) do
							simple_imput("nx", init_vp.nx),
							simple_imput("ny", init_vp.ny),
							simple_imput("dt", init_vp.dt),
							simple_imput("steph", init_vp.steph),
							simple_imput("nl", init_vp.nl),
							simple_imput("nmy", init_vp.nmy),
							simple_imput("pmlt", init_vp.pmlt),
							simple_imput("eiter", init_vp.eiter),
							simple_imput("CFL", init_vp.CFL),
							simple_imput("pic_amount", init_vp.pic_amount),
							simple_imput("nout", init_vp.nout)
						end
					end]
				end
				if tab == "tab-3-example-graph"
					return [html_div(className="row") do
						#callback for eruption csv file

						dcc_upload(
							id="upload-data",
							children=html_div([html_a("Select Files")
							]),
							style=Dict(
								"width" => "100%",
								"height" => "60px",
								"lineHeight" => "60px",
								"borderWidth" => "1px",
								"borderStyle" => "dashed",
								"borderRadius" => "5px",
								"textAlign" => "center",
								"margin" => "10px"
							),
							# Allow multiple files to be uploaded
							multiple=true
						),
						html_div(id="output-data-upload")
						# dash_datatable(
						# 	id="table",
						# 	columns=[Dict("name" =>i, "id" => i) for i in names(df)],
						# 	data = Dict.(pairs.(eachrow(df)))
						# )
					end]
				end
			end

			#Callback for generate button
			callback!(app, [Output("generate-but", "n_clicks")], [Input("generate-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("generate button clicked")
				println(init_vp.critVol)
				dykes_rand_param(init_vp)
				return [n_clicks + 1]
			end

			#Callback for showing dykes which was generated
			callback!(app, [Output("dykes-graph", "children")], [Input("show-dykes-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
				ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

				io = open(data_folder * "pa.bin", "r")
				read!(io, dpa)
				read!(io, ipa)

				ipar = 1
				Lx, ipar = read_par(dpa, ipar)
				Ly, ipar = read_par(dpa, ipar)
				lam_r_rhoCp, ipar = read_par(dpa, ipar)
				lam_m_rhoCp, ipar = read_par(dpa, ipar)
				L_Cp, ipar = read_par(dpa, ipar)
				T_top, ipar = read_par(dpa, ipar)
				T_bot, ipar = read_par(dpa, ipar)
				T_magma, ipar = read_par(dpa, ipar)
				tsh, ipar = read_par(dpa, ipar)
				gamma, ipar = read_par(dpa, ipar)
				Ly_eruption, ipar = read_par(dpa, ipar)
				nu, ipar = read_par(dpa, ipar)
				G, ipar = read_par(dpa, ipar)
				dt, ipar = read_par(dpa, ipar)
				dx, ipar = read_par(dpa, ipar)
				dy, ipar = read_par(dpa, ipar)
				eiter, ipar = read_par(dpa, ipar)
				pic_amount, ipar = read_par(dpa, ipar)
				tfin, ipar = read_par(dpa, ipar)

				ipar = 1

				pmlt, ipar = read_par(ipa, ipar)
				nx, ipar = read_par(ipa, ipar)
				ny, ipar = read_par(ipa, ipar)
				nl, ipar = read_par(ipa, ipar)
				nt, ipar = read_par(ipa, ipar)
				niter, ipar = read_par(ipa, ipar)
				nout, ipar = read_par(ipa, ipar)
				nsub, ipar = read_par(ipa, ipar)
				nerupt, ipar = read_par(ipa, ipar)
				npartcl, ipar = read_par(ipa, ipar)
				nmarker, ipar = read_par(ipa, ipar)
				nSample, ipar = read_par(ipa, ipar)

				critVol = Array{Float64,1}(undef, nSample) #???#Critical volume when eruption appears, predefined variable
				read!(io, critVol)

				#array 0 0 1 0 0 ... like, where 1 -instrusion
				ndykes = Array{Int32,1}(undef, nt)#number of dykes intruded on n-th time step
				read!(io, ndykes)

				ndykes_all = 0

				#count all dykes
				for istep in 1:nt
					ndykes_all = ndykes_all + ndykes[istep]
				end

				#array which describes amount of particles in new dyke
				particle_edges = Array{Int32,1}(undef, ndykes_all + 1)
				read!(io, particle_edges)

				marker_edges = Array{Int32,1}(undef, ndykes_all + 1)
				read!(io, marker_edges)

				close(io)

				cap_frac = 3  #value to spcify how much particles we allow to inject in runtime
				npartcl0 = npartcl #initial amount of particles
				max_npartcl = convert(Int64, npartcl * cap_frac) + particle_edges[ndykes_all+1] #???#count max particles
				println(npartcl)
				println(particle_edges[ndykes_all+1])
				println("max_npartcl")
				println(max_npartcl)
				nmarker0 = nmarker


				#	max_nmarker = nmarker + marker_edges[ndykes_all+1]



				np_dykes = particle_edges[ndykes_all+1]#number of particles in each dyke during intrusion

				fid = h5open(data_folder * "particles.h5", "r")

				h_px = Array{Float64,1}(undef, max_npartcl)
				h_py = Array{Float64,1}(undef, max_npartcl)

				h_px = read(fid, "px")
				h_py = read(fid, "py")


				h_px_dykes = Array{Float64,1}(undef, np_dykes)
				h_py_dykes = Array{Float64,1}(undef, np_dykes)

				h_px_dykes = read(fid, "px_dykes")
				h_py_dykes = read(fid, "py_dykes")

				#PlotlyJS.scatter([1,2,3],[4,5,6])

				close(fid)

				#	PlotlyJS.plot([
				#	test_fig = PlotlyJS.scatter(x=h_px_dykes, y=h_py_dykes, mode="markers", name="markers")

				d2d_limit = np_dykes
				d2d_limit_gap = 100
				#ret_plot = Plots.scatter(markersize = 0.1, h_px_dykes[1:d2d_limit_gap:d2d_limit],h_py_dykes[1:d2d_limit_gap:d2d_limit], xlimit = [1, 20000], ylimit = [1, 20000])

				new_plot = Plot([
						PlotlyJS.scatter(x=h_px_dykes[1:d2d_limit_gap:d2d_limit], y=h_py_dykes[1:d2d_limit_gap:d2d_limit], mode="markers", line_width=0.001)
					], Layout(title="Dash Data Visualization", xaxis_range=[1, Lx], yaxis_range=[1, Ly]))

				#	layout = Layout(xaxis_range=[1, 20000], yaxis_range=[1, 20000])

				return [html_div(id="dykes_figures_dykes", className="row", style=Dict("columnCount" => 2)) do
					#dcc_graph(id="T_graph",figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))

					dcc_graph(id="dd_graph", figure=new_plot)

					#dcc_graph(id="dykes_graph", figure = Plot(ret_plot))
				end]
			end

			#Callback to show current T distribution
			callback!(app, [Output("T-graph", "children")], [Input("show-cur-T-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("show-cur-T-but clicked")
				if (vp.dx == 0.0)
					return nothing
				end
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly

				h_T = Array{Float64,1}(undef, vp.nx * vp.ny)
				copyto!(h_T, gp.T)

				title_string = "T, (time, " * string(-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000) * " ka)"
				layout_inner = Layout(title=title_string,
				#yaxis=attr(
				#range=[0, 20],  # Reverse the range: max to min
				#	title="Depth (km)",
				#	tickvals=0:5000:20000,
				#	ticktext=["0", "5k", "10k", "15k", "20k"])
				)

				p = Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(h_T, (vp.nx, vp.ny))))), layout_inner)

				return [
					html_div(id="dykes_figures_T", className="row") do
						dcc_graph(id="T_graph", figure=p)
					end
				]
			end

			#Callback to show current C distribution
			callback!(app, [Output("C-graph", "children")], [Input("show-cur-C-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("show-cur-C-but clicked")
				if (vp.dx == 0.0)
					return nothing
				end
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly

				h_C = Array{Float64,1}(undef, vp.nx * vp.ny)
				copyto!(h_C, gp.C)

				title_string = "C, (time, " * string(-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000) * " ka)"
				layout_inner = Layout(title=title_string)

				p = Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(h_C, (vp.nx, vp.ny))))), layout_inner)
				return [
					html_div(id="dykes_figures_C", className="row") do
						dcc_graph(id="C_graph", figure=p)
					end
				]
			end

			#Callback to show current mf distribution
			callback!(app, [Output("mf-graph", "children")], [Input("show-cur-mf-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("show-cur-mf-but clicked")
				if (vp.dx == 0.0)
					return nothing
				end
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly

				h_T = Array{Float64,1}(undef, vp.nx * vp.ny)
				copyto!(h_T, gp.T)

				h_C = Array{Float64,1}(undef, vp.nx * vp.ny)
				copyto!(h_C, gp.C)

				mf = d2dm_mf_magma.(h_T) .* h_C + itp.(h_T) .* (1.0 .- h_C)
				mf[mf.>mf_boundaries] .= 1
				mf[mf.<=mf_boundaries] .= 0

				dxl = vp.dx * vp.nl
				dyl = vp.dy * vp.nl

				non_zero_count = count(x -> x != 0, mf)

				#i guess 1.e4 here is z
				#TODO: research this formula
				real_vol_2 = (non_zero_count * (vp.dx * vp.dy) / 1.e9) * 1.e4 * (1)

				println("accomulated material " * string(real_vol_2) * " km^3")

				high_melt_mask = mf .> mf_boundaries
				high_melt_cells = count(high_melt_mask)

				# Calculate areas and volumes
				cell_area_m2 = vp.dx * vp.dy
				dz_m = 1e4  # z-extent of dykes in meters

				# Area of high melt regions (in km²)
				high_melt_area_km2 = high_melt_cells * cell_area_m2 / 1e6  # m² to km²

				# Volume of high melt regions (in km³)
				high_melt_volume_km3 = high_melt_cells * cell_area_m2 * dz_m / 1e9  # m³ to km³

				# ALTERNATIVE: Weighted volume (accounts for partial melt fractions)
				# This is more accurate if you want total melt volume, not just volume of high-melt regions
				weighted_melt_volume_km3 = sum(mf[high_melt_mask]) * cell_area_m2 * dz_m / 1e9

				# Percentage of domain that is high-melt
				total_domain_area_km2 = vp.nx * vp.ny * cell_area_m2 / 1e6
				high_melt_percentage = (high_melt_area_km2 / total_domain_area_km2) * 100

				# Output results
				println("=== Melt Fraction Analysis ===")
				println("Threshold: mf > ", mf_boundaries)
				println("High melt cells: ", high_melt_cells, " / ", length(mf))
				println("High melt area: ", high_melt_area_km2, " km²")
				println("High melt volume: ", high_melt_volume_km3, " km³")
				println("Weighted melt volume: ", weighted_melt_volume_km3, " km³ (accounts for partial melt)")
				println("High melt percentage: ", high_melt_percentage, "% of domain")

				title_string = "Melt fraction (mf), (time, " * string(-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000) * " ka)"
				layout_inner = Layout(title=title_string)

				p = Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(mf, (vp.nx, vp.ny))))), layout_inner)

				return [
					html_div(id="dykes_figures_mf", className="row") do
						dcc_graph(id="mf_graph", figure=p)
					end
				]
			end

			#Callback to show current dmf distribution
			callback!(app, [Output("dmf-graph", "children")], [Input("show-cur-dmf-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("show-cur-dmf-but clicked")
				if (vp.dx == 0.0)
					return nothing
				end
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly

				dmf = CuArray{Float64,1}(undef, vp.nx * vp.ny)
				h_dmf = Array{Float64,1}(undef, vp.nx * vp.ny)

				# h_T = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
				# copyto!(h_T, gp.T)

				# h_C = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
				# copyto!(h_C, gp.C)
				CUDA.allowscalar(true)
				dmf = d2dm_dmf_magma.(gp.T) .* gp.C + only.(Interpolations.gradient.(Ref(cuitp), gp.T)) .* (1.0 .- gp.C)
				CUDA.allowscalar(false)
				copyto!(h_dmf, gp.T_old)

				title_string = "Vs (m/s), (time, " * string(-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000) * " ka)"
				layout_inner = Layout(title=title_string)

				p = Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(h_dmf, (vp.nx, vp.ny)))), reversescale=true), layout_inner)

				return [
					html_div(id="dykes_figures_dmf", className="row") do
						dcc_graph(id="dmf_graph", figure=p)
					end
				]
			end

			#Callback to show current chambers
			callback!(app, [Output("chambers-graph", "children")], [Input("show-cur-chambers-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("show-cur-chambers-but clicked")
				if (vp.dx == 0.0)
					return nothing
				end

				xs = 0:vp.dx*vp.nl:vp.Lx
				ys = 0:vp.dy*vp.nl:vp.Ly
				xss = collect(xs)
				yss = collect(ys)

				title_string = "Unique labeled chambers, (time, " * string(-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000) * " ka)"
				layout_inner = Layout(title=title_string)
				#data = DataFrame(x=hcat(xss for j = 1:length(xss)),y=vcat(yss for j = 1:length(yss)),color=collect(eachrow(reshape(gp.L_host, (vp.nxl, vp.nyl)))))

				println(unique(gp.L_host))

				a = 0
				for x in unique(gp.L_host)
					replace!(gp.L_host, x => a)
					a = a + 1
				end

				#data = DataFrame(x=(hcat(xss) for j = 1:((size(yss))[1])),y=(vcat(yss) for j = 1:((size(xss))[1])), color=gp.L_host)
				#data = DataFrame(x=(hcat(xss) for j = 1:size(xss)),y=(vcat(yss) for j = 1:size(yss)),color=collect(eachrow(reshape(gp.L_host, (vp.nxl, vp.nyl)))))

				#foreach(println, names(data))
				#println(data)

				p = Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(gp.L_host, (vp.nxl, vp.nyl))))), layout_inner)

				return [
					html_div(id="dykes_figures_chambers", className="row") do
						dcc_graph(id="chambers_graph", figure=p)
					end
				]
			end

			#Callback for start button
			callback!(app, [Output("stop-but", "disabled")], [Input("start-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("start_button clicked")

				global FLAGS.d2dm_started[] = true
				global FLAGS.d2dm_stopped[] = false
				global FLAGS.d2dm_break[] = false

				if (FLAGS.init[])
					gp = GridParams()
					vp = VarParams()
				end

				main_test_gui(gp, vp, init_vp)
				return [false]
			end

			#Callback for refresh botton
			callback!(app, [Output("refresh-but", "disabled")], [Input("refresh-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				global FLAGS.init[] = true
				return [false]
			end

			#Callback for stop button
			callback!(app, [Output("stop-but", "n_clicks")], [Input("stop-but", "n_clicks")], prevent_initial_call=true) do n_clicks
				println("stop button clicked")

				global FLAGS.d2dm_started[] = false
				global FLAGS.d2dm_stopped[] = true 
				global FLAGS.d2dm_break[] = true

				return [n_clicks + 1]
			end

			#Callback for figures panel
			callback!(app, [Output("tabs-content-figure-graph", "children")], [Input("tabs-figure-graph", "value")], prevent_initial_call=true) do tab
				if tab == "tab-1-figure-graph"
					if (vp.dx == 0.0)
						return nothing
					end
					xs = 0:vp.dx:vp.Lx
					ys = 0:vp.dy:vp.Ly

					h_T = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
					copyto!(h_T, gp.T)
					return [html_div(id="dykes_figures_T", className="row", style=Dict("columnCount" => 3)) do
						#dcc_graph(id="T_graph",figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))
						dcc_graph(id="T_graph", figure=Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(h_T, (vp.nx, vp.ny)))), title="T")))
					end]
				end
				if tab == "tab-2-figure-graph"
					if (vp.dx == 0.0)
						return nothing
					end
					xs = 0:vp.dx:vp.Lx
					ys = 0:vp.dy:vp.Ly
					h_C = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
					copyto!(h_C, gp.C)
					return [html_div(id="dykes_figures_C", className="row", style=Dict("columnCount" => 3)) do
						#NOTE: dash bug, see https://github.com/plotly/Dash.jl/issues/60

						#p = Plot(PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(h_C)), title="C"))
						#data = Plots.plotly_series(p)
						#data[1][:z] = [c for c in eachcol(data[1][:z])] # <-- As a temporary
						#layout = Plots.plotly_layout(p)
						#dcc_graph(id="C_graph", figure = (;data, layout), title="C")

						dcc_graph(id="C_graph", figure=Plot(PlotlyJS.heatmap(x=xs, y=ys, z=collect(eachrow(reshape(h_C, (vp.nx, vp.ny)))), title="C")))
					end]
				end
			end

			#Callback to update eruptions timeline
			callback!(app, [Output("eruptions-timeline", "children")], [Input("interval-component", "n_intervals")]) do n_intervals

				campi_calc = -(vp.nt .- gp.eruptionSteps) ./ vp.nt .* init_vp.calc_years / 1000
				campi_real = -vcat(init_vp.critVolTime)
				campi_now = [-(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000]
				#campi_cumulut = -(vp.nt .- gp.cumulutive_time)./vp.nt.*init_vp.calc_years/1000;

				campi_cal_graph = PlotlyJS.scatter(x=campi_calc, y=zeros(length(campi_calc)), marker_color="rgba(0, 0, 255, 1)", mode="markers", color=1, name="campi_calc", showlegend=true, marker_size=14)
				campi_real_graph = PlotlyJS.scatter(x=campi_real, y=zeros(length(campi_real)), marker_color="rgba(255, 0, 0, 1)", mode="markers", color=2, name="campi_real", showlegend=true, marker_size=10)
				campi_now_graph = PlotlyJS.scatter(x=campi_now, y=zeros(length(campi_now)), marker_color="rgba(0, 255, 0, 1)", mode="markers", color=3, name="now", showlegend=true, marker_size=10)
				#campi_cumulut_graph = PlotlyJS.scatter(x=campi_cumulut, y=gp.cumulutive_vol, mode="lines", name="cumulative volume")
				#		#campi_next_eruption = PlotlyJS.scatter(x=campi_cumulut, y=(gp.critVol[vp.iSample]/ 1.e9) * 1.e4 * vp.gamma, mode="lines", name="next eruption")
				data = [campi_cal_graph, campi_real_graph, campi_now_graph]

				layout = Layout(title="Eruptions graph",
					xaxis=attr(title="time, (ka)", showgrid=false),
					yaxis=attr(showgrid=false))

				p = PlotlyJS.plot(data, layout)

				return [html_div(className="row") do
					dcc_graph(id="eruptions-timeline-graph", figure=p)
				end]
			end

			#Callback to update cumul timeline
			callback!(app, [Output("cumul-timeline", "children")], [Input("interval-component", "n_intervals")]) do n_intervals

				#campi_cumulut = [-(vp.nt .- gp.cumulutive_time)./vp.nt.*init_vp.calc_years/1000];
				campi_cumulut = -(vp.nt .- gp.cumulutive_time) ./ vp.nt .* init_vp.calc_years / 1000
				campi_calc = -(vp.nt .- gp.eruptionSteps) ./ vp.nt .* init_vp.calc_years / 1000
				campi_real = -vcat(init_vp.critVolTime)

				#show num cumulutive volume
				campi_cumulut_graph = PlotlyJS.scatter(x=campi_cumulut, y=gp.cumulutive_calc, mode="lines", color=1, name="cumulative volume calc", marker_color="rgba(0, 0, 255, 1)")
				#show real cumulutive volume
				campi_next_eruption = PlotlyJS.scatter(x=campi_cumulut, y=gp.cumulutive_real, mode="lines", color=2, name="cumulutive volume real", marker_color="rgba(255, 0, 0, 1)")
				#points of calc eruptions
				campi_cal_graph = PlotlyJS.scatter(x=campi_calc, y=zeros(length(campi_calc)), mode="markers", name="eruptions, calc", showlegend=true, marker_size=14, marker_color="rgba(0, 0, 255, 1)")
				#points of real eruptions
				campi_real_graph = PlotlyJS.scatter(x=campi_real, y=zeros(length(campi_real)), marker_color="rgba(255, 0, 0, 1)", mode="markers", color=2, name="eruptions, real", showlegend=true, marker_size=10)

				data = [campi_next_eruption, campi_cumulut_graph, campi_cal_graph, campi_real_graph]

				layout = Layout(title="Cumulutive volume graph",
					xaxis=attr(title="time, (ka)"),
					yaxis=attr(title="volume, (km^3)", showgrid=false))

				p = PlotlyJS.plot(data, layout)

				return [html_div(className="row") do
					dcc_graph(id="eruptions-timeline-graph", figure=p)
				end]
			end

			#Callback to update next-cumul timeline
			callback!(app, [Output("next-cumul-timeline", "children")], [Input("interval-component", "n_intervals")]) do n_intervals

				#campi_cumulut = [-(vp.nt .- gp.cumulutive_time)./vp.nt.*init_vp.calc_years/1000];
				campi_calc = -(vp.nt .- gp.eruptionSteps) ./ vp.nt .* init_vp.calc_years / 1000
				campi_cumulut = -(vp.nt .- gp.cumulutive_time) ./ vp.nt .* init_vp.calc_years / 1000

				campi_cumulut_graph = PlotlyJS.scatter(x=campi_cumulut, y=gp.next_cumulutive_vol, mode="lines", name="current max volume", marker_color="rgba(0, 0, 255, 1)")
				campi_next_eruption = PlotlyJS.scatter(x=campi_cumulut, y=gp.next_cumulutive_erupt, mode="lines", name="next eruption volume", marker_color="rgba(255, 0, 0, 1)")
				campi_cal_graph = PlotlyJS.scatter(x=campi_calc, y=zeros(length(campi_calc)), mode="markers", name="calculated eruptions", marker_color="rgba(0, 0, 255, 1)", showlegend=true, marker_size=14)
				#		campi_next_eruption = PlotlyJS.scatter(x=campi_cumulut, y=(gp.critVol[vp.iSample]/ 1.e9) * 1.e4 * vp.gamma, mode="lines", name="next eruption")

				#		campi_cumulut_graph = PlotlyJS.scatter(x=gp.cumulutive_time, y=gp.cumulutive_vol, mode="lines", name="current volume")
				#		campi_next_eruption = PlotlyJS.scatter(x=gp.cumulutive_time, y=(gp.critVol[vp.iSample]/ 1.e9) * 1.e4 * vp.gamma, mode="lines", name="next eruption")
				data = [campi_next_eruption, campi_cal_graph, campi_cumulut_graph]

				layout = Layout(title="Next volume graph",
					xaxis=attr(title="time, (ka)"),
					yaxis=attr(title="volume, (km^3)", showgrid=false))

				p = PlotlyJS.plot(data, layout)

				return [html_div(className="row") do
					dcc_graph(id="eruptions-timeline-graph", figure=p)
				end]
			end

			#Callback to update time left 
			callback!(app, [Output("time_left_label", "value"), Output("percent_done_label", "value")], [Input("interval-component", "n_intervals")]) do n_intervals
				return ["Time left:\t" * string(D2DM_GUI_State.TIMING.str_time_left[]), "Done:\t" * @sprintf("%03s", round(((Float64(vp.it) - 1) / Float64(vp.nt - 1)) * 100, digits=2)) * "%"]
			end

			#Callback to update log buffer 
			callback!(app, [Output("log_buffer", "value")], [Input("interval-component", "n_intervals")]) do n_intervals
				return ["No buffer"]
			end

			#Callback to update variables in tabs
			#Pnysics
			begin
				callback!(app, [Output("Lx", "value")], [Input("Lx", "value")]) do input_value
					init_vp.Lx = input_value
					return [init_vp.Lx]
				end

				callback!(app, [Output("Ly", "value")], [Input("Ly", "value")]) do input_value
					init_vp.Ly = input_value
					return [init_vp.Ly]
				end

				callback!(app, [Output("Lz", "value")], [Input("Lz", "value")]) do input_value
					init_vp.Lz = input_value
					return [init_vp.Lz]
				end

				callback!(app, [Output("calc_years", "value")], [Input("calc_years", "value")]) do input_value
					init_vp.calc_years = input_value
					return [init_vp.calc_years]
				end

				callback!(app, [Output("narrow_fact", "value")], [Input("narrow_fact", "value")]) do input_value
					init_vp.narrow_fact = input_value
					return [init_vp.narrow_fact]
				end

				callback!(app, [Output("dyke_to_sill", "value")], [Input("dyke_to_sill", "value")]) do input_value
					init_vp.dyke_to_sill = input_value
					return [init_vp.dyke_to_sill]
				end

				callback!(app, [Output("Lam_r", "value")], [Input("Lam_r", "value")]) do input_value
					init_vp.Lam_r = input_value
					return [init_vp.Lam_r]
				end

				callback!(app, [Output("Lam_m", "value")], [Input("Lam_m", "value")]) do input_value
					init_vp.Lam_m = input_value
					return [init_vp.Lam_m]
				end

				callback!(app, [Output("rho", "value")], [Input("rho", "value")]) do input_value
					init_vp.rho = input_value
					return [init_vp.rho]
				end

				callback!(app, [Output("Cp", "value")], [Input("Cp", "value")]) do input_value
					init_vp.Cp = input_value
					return [init_vp.Cp]
				end

				callback!(app, [Output("L_heat", "value")], [Input("L_heat", "value")]) do input_value
					init_vp.L_heat = input_value
					return [init_vp.L_heat]
				end

				callback!(app, [Output("T_top", "value")], [Input("T_top", "value")]) do input_value
					init_vp.T_top = input_value
					return [init_vp.T_top]
				end

				callback!(app, [Output("dTdy", "value")], [Input("dTdy", "value")]) do input_value
					init_vp.dTdy = input_value
					return [init_vp.dTdy]
				end

				callback!(app, [Output("T_magma", "value")], [Input("T_magma", "value")]) do input_value
					init_vp.T_magma = input_value
					return [init_vp.T_magma]
				end

				callback!(app, [Output("T_ch", "value")], [Input("T_ch", "value")]) do input_value
					init_vp.T_ch = input_value
					return [init_vp.T_ch]
				end

				callback!(app, [Output("Qv", "value")], [Input("Qv", "value")]) do input_value
					init_vp.Qv = input_value
					return [init_vp.Qv]
				end

				callback!(app, [Output("dyke_x_W", "value")], [Input("dyke_x_W", "value")]) do input_value
					init_vp.dyke_x_W = input_value
					return [init_vp.dyke_x_W]
				end

				callback!(app, [Output("dyke_y_rng_bot", "value")], [Input("dyke_y_rng_bot", "value")]) do input_value
					init_vp.dyke_y_rng_bot = input_value
					return [init_vp.dyke_y_rng_bot]
				end

				callback!(app, [Output("dyke_y_rng_top", "value")], [Input("dyke_y_rng_top", "value")]) do input_value
					init_vp.dyke_y_rng_top = input_value
					return [init_vp.dyke_y_rng_top]
				end

				callback!(app, [Output("Ly_eruption", "value")], [Input("Ly_eruption", "value")]) do input_value
					init_vp.Ly_eruption = input_value
					return [init_vp.Ly_eruption]
				end

				callback!(app, [Output("dT", "value")], [Input("dT", "value")]) do input_value
					init_vp.dT = input_value
					return [init_vp.dT]
				end

				callback!(app, [Output("E", "value")], [Input("E", "value")]) do input_value
					init_vp.E = input_value
					return [init_vp.E]
				end

				callback!(app, [Output("nu", "value")], [Input("nu", "value")]) do input_value
					init_vp.nu = input_value
					return [init_vp.nu]
				end

				callback!(app, [Output("tsh", "value")], [Input("tsh", "value")]) do input_value
					init_vp.tsh = input_value
					return [init_vp.tsh]
				end

				callback!(app, [Output("gamma", "value")], [Input("gamma", "value")]) do input_value
					init_vp.gamma = input_value
					return [init_vp.gamma]
				end

				callback!(app, [Output("dyke_nu", "value")], [Input("dyke_nu", "value")]) do input_value
					init_vp.dyke_nu = input_value
					return [init_vp.dyke_nu]
				end

				callback!(app, [Output("dyke_dev", "value")], [Input("dyke_dev", "value")]) do input_value
					init_vp.dyke_dev = input_value
					return [init_vp.dyke_dev]
				end

				callback!(app, [Output("dyke_type", "value")], [Input("dyke_type", "value")]) do input_value
					init_vp.dyke_type = input_value
					return [init_vp.dyke_type]
				end

			end

			#Numerics
			begin
				callback!(app, [Output("nx", "value")], [Input("nx", "value")]) do input_value
					init_vp.nx = input_value
					return [init_vp.nx]
				end

				callback!(app, [Output("ny", "value")], [Input("ny", "value")]) do input_value
					init_vp.ny = input_value
					return [init_vp.ny]
				end

				callback!(app, [Output("dt", "value")], [Input("dt", "value")]) do input_value
					init_vp.dt = input_value
					return [init_vp.dt]
				end

				callback!(app, [Output("steph", "value")], [Input("steph", "value")]) do input_value
					init_vp.steph = input_value
					return [init_vp.steph]
				end

				callback!(app, [Output("nl", "value")], [Input("nl", "value")]) do input_value
					init_vp.nl = input_value
					return [init_vp.nl]
				end

				callback!(app, [Output("nmy", "value")], [Input("nmy", "value")]) do input_value
					init_vp.nmy = input_value
					return [init_vp.nmy]
				end

				callback!(app, [Output("pmlt", "value")], [Input("pmlt", "value")]) do input_value
					init_vp.pmlt = input_value
					return [init_vp.pmlt]
				end

				callback!(app, [Output("eiter", "value")], [Input("eiter", "value")]) do input_value
					init_vp.eiter = input_value
					return [init_vp.eiter]
				end

				callback!(app, [Output("CFL", "value")], [Input("CFL", "value")]) do input_value
					init_vp.CFL = input_value
					return [init_vp.CFL]
				end

				callback!(app, [Output("pic_amount", "value")], [Input("pic_amount", "value")]) do input_value
					init_vp.pic_amount = input_value
					return [init_vp.pic_amount]
				end

				callback!(app, [Output("nout", "value")], [Input("nout", "value")]) do input_value
					init_vp.nout = input_value
					return [init_vp.nout]
				end
			end

			#Callback to update mf field for graph
			callback!(app, [Output("mf_boundaries", "value")], [Input("mf_boundaries", "value")]) do input_value
					mf_boundaries = input_value
					return [mf_boundaries]
			end

			#Callback to load our own eruptions history
			callback!(app,
				[Output("output-data-upload", "children")],
				[Input("upload-data", "contents")],
				[State("upload-data", "filename"), State("upload-data", "last_modified")],
			) do contents, filename, last_modified
				if !(contents isa Nothing)
					children = [
						parse_contents_csv(c..., init_vp) for c in
						zip(contents, filename, last_modified)]
					return [children]
				end
			end

			#Callback to load config
			callback!(app,
			[Output("load-config-upload", "contents")],
			[Input("load-config-upload", "contents")],
			[State("load-config-upload", "filename")], prevent_initial_call=true
			) do contents, filename
				fid = h5open(filename, "r")

				# Store default values before loading
				default_init_vp = deepcopy(init_vp)

				for n in fieldnames(typeof(init_vp))
					field_name = string(n)
					
					# Check if the field exists in the HDF5 file
					if haskey(fid, field_name)
						println("Loading field: ", n)
						setfield!(init_vp, n, read(fid, field_name))
						println("New value: ", getfield(init_vp, n))
					else
						println("Field not found in config, keeping default: ", n)
						# Keep existing value (already set from defaults)
					end
				end

				# Reset these to their default values regardless
				init_vp.critVol = ERUPTION_VOLUMES
				init_vp.critVolTime = ERUPTION_TIMES

				println("config loaded from " * filename)
				close(fid)

				return [contents]
			end

			#Callback to save config
			callback!(app,
				[Output("save-config-but", "n_clicks")],
				[Input("save-config-but", "n_clicks")], prevent_initial_call=true
			) do n_clicks
				filename_donwload = @sprintf("d2d_config_%s.hdf5", Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))

				#file_as_bytes = h5open("AnyName_InMemory", "w"; driver=Drivers.Core(; backing_store=false)) do fid
				fid = h5open(filename_donwload, "w")

				for n in fieldnames(typeof(init_vp))
					println(getfield(init_vp, n))
					write(fid, string(n), getfield(init_vp, n))
				end

				#return Dict("content" => Vector{UInt8}(fid), "filename" => filename_donwload) # get a byte vector to send, e.g., using HTTP, MQTT or similar.
				println("config saved to " * filename_donwload)
				close(fid)

				return [n_clicks]
			end

			#Callback to save snapshot
			callback!(app,
				[Output("save-snapshot-but", "n_clicks")],
				[Input("save-snapshot-but", "n_clicks")], prevent_initial_call=true
			) do n_clicks
				filename_donwload = @sprintf("d2d_snapshot_%d_%s.hdf5", vp.it, Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))
				fid = h5open(filename_donwload, "w")

				for n in fieldnames(typeof(vp))
					println(getfield(vp, n))
					write(fid, string(n), getfield(vp, n))
				end

				for n in fieldnames(typeof(gp))
					if (getfield(gp, n) isa CuArray)
						d2d_cu_type = eltype(getfield(gp, n))

						nn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp, n))[1])
						copyto!(nn, getfield(gp, n))

						write(fid, string(n), nn)
						println("sucess!!")
					else
						write(fid, string(n), getfield(gp, n))
					end
				end

				println("snapshot saved to " * filename_donwload)

				close(fid)

				return [n_clicks]
			end
		end

		#run_server(app)
		run_server(app, "0.0.0.0", 8050)    
	end

	function main_test_gui(gp::GridParams, vp::VarParams, init_vp::InitVarParams)
		#TODO: fix this 
		data_folder = DATA_FOLDER
		bar1 = BAR1
		bar2 = BAR2
		eruption_times = ERUPTION_TIMES
		
		cuitp = get_interpolation()

		# Reset timing
		reset_flags!()
		
		local_buff = ""

		Random.seed!(1234)

		checker = Array{Float64}(undef, 1)
		
		# Initialize if needed
		if FLAGS.init[]
			@printf("%s reading params\n", bar1)
			d2dm_read_params(gp, vp, data_folder)
			
			@printf("%s initialization\n", bar1)
			d2dm_init(gp, vp, FLAGS.d2dm_markers[])
			FLAGS.init[] = false
		end
		
		filename = Array{Char,1}(undef, 1024)
		eruption_counter = 1
		real_eruption_counter = 1
		
		total_time = @elapsed begin
			# Main loop
			for vp.it in vp.it:vp.nt
				@printf("%s it = %d\n", bar1, vp.it)
				
				# Update timing
				update_runtime!(TIMING.time_of_loop[], vp.it, vp.nt)
				
				time_of_loop_local = @elapsed begin
					vp.is_eruption = false
					eruption_counter -= 1
					is_intrusion = gp.ndykes[vp.it] > 0
					nerupt = 1
					
					# Check eruption criteria
					if vp.it % nerupt == 0
						mf_rock_arr = CuArray{Float64,1}(undef, vp.nx * vp.ny)
						mf_rock_arr = cuitp.(gp.T)
						
						dxl = vp.dx * vp.nl
						dyl = vp.dy * vp.nl
						
						tsh_tmp = vp.tsh
						
						# Calculate melt fractions

						#=
						vp.tsh = 0.01
						maxVol, maxIdx, sumVol = d2dm_check_melt_fracton(gp, vp, mf_rock_arr)
						sumVol = (sumVol * (dxl * dyl) / 1.e9) * 1.e4 * (1 - vp.gamma)
						append!(gp.cum_mf_01, sumVol)
						
						vp.tsh = 0.50
						maxVol, maxIdx, sumVol = d2dm_check_melt_fracton(gp, vp, mf_rock_arr)
						sumVol = (sumVol * (dxl * dyl) / 1.e9) * 1.e4 * (1 - vp.gamma)
						append!(gp.cum_mf_50, sumVol)
						=#
						
						vp.tsh = tsh_tmp
						maxVol, maxIdx, sumVol = d2dm_check_melt_fracton(gp, vp, mf_rock_arr)
						
						if maxVol == -1
							return 0
						end
						
						real_vol_2 = (maxVol * (dxl * dyl) / 1.e9) * 1.e4 * (1 - vp.gamma)
						append!(gp.next_cumulutive_erupt, (gp.critVol[vp.iSample] / 1.e9) * 1.e4 * (1 - vp.gamma))
						append!(gp.next_cumulutive_vol, real_vol_2)
						
						# Real eruptions
						if vp.iSample_real <= length(eruption_times)
							if -eruption_times[vp.iSample_real] <= -(vp.nt - vp.it) / vp.nt * init_vp.calc_years / 1000
								vp.sum_erupted_real += gp.critVol[vp.iSample_real]
								vp.iSample_real += 1
							end
						end
						append!(gp.cumulutive_real, (vp.sum_erupted_real / 1.e9) * 1.e4 * (1 - vp.gamma))
						append!(gp.cumulutive_calc, vp.sum_erupted_calc)
						append!(gp.cumulutive_time, vp.it)
						
						if maxVol * dxl * dyl >= gp.critVol[vp.iSample] && eruption_counter <= 0
							@printf("%s erupting %07d cells\n", bar2, maxVol)
							vp.sum_erupted_calc += real_vol_2
							
							filename = data_folder * "julia_grid." * string(vp.it) * ".before_eruption" * ".h5"
							d2dm_make_snapshot(vp, gp, filename, FLAGS.make_snapshot[])
							d2dm_eruption_advection(gp, vp, maxVol, maxIdx, vp.it, FLAGS.d2dm_markers[])
							eruption_counter = 10
						end
					end
					
					# Process intrusions
					if is_intrusion
						@printf("%s inserting %02d dykes\n", bar2, gp.ndykes[vp.it])
						d2dm_inserting_dykes(gp, vp, vp.it, FLAGS.d2dm_markers[])
					end
					
					# p2g interpolation
					if vp.is_eruption || is_intrusion
						@printf("%s p2g interpolation\n", bar2)
						d2dm_p2g_interpolation(gp, vp)
						
						@printf("%s particle injection\n", bar2)
						d2dm_particles_injection(gp, vp)
					end
					
					# Solve heat equation
					@time begin
						@printf("%s solving heat diffusion\n", bar2)
						
						blockSize = (28, 32)
						gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), 
								   Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))
						
						copyto!(gp.T_old, gp.T)
						for isub = 0:vp.nsub-1
							dmf_rock_arr = CuArray{Float64,1}(undef, vp.nx * vp.ny)
							dmf_rock_arr = only.(Interpolations.gradient.(Ref(cuitp), gp.T))
							
							d2dm_update_T_NG!(gp.T, gp.T_old,
								vp.T_top, vp.T_bot,
								gp.C, vp.lam_r_rhoCp, vp.lam_m_rhoCp, vp.L_Cp,
								vp.dx, vp.dy, vp.dt,
								vp.nx, vp.ny, dmf_rock_arr)
							synchronize()
						end
						
						# Check for explosion
						T_for_check = @view gp.T[1]
						copyto!(checker, T_for_check)
						if isnan(checker[1])
							println("EXPLOSION!!!")
							return -1
						end
					end
					
					# g2p interpolation
					@time begin
						@printf("%s g2p interpolation\n", bar2)
						blockSize1D = 896
						gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D
						@cuda blocks=gridSize1D threads=blockSize1D d2dm_g2p!(
							gp.T, gp.T_old, gp.px, gp.py, gp.pT, 
							vp.dx, vp.dy, vp.pic_amount, 
							vp.nx, vp.ny, vp.npartcl
						)
						
						if FLAGS.d2dm_markers[]
							gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D
							@cuda blocks=gridSize1D threads=blockSize1D d2dm_g2p!(
								gp.T, gp.T_old, gp.mx, gp.my, gp.mT, 
								vp.dx, vp.dy, vp.pic_amount, 
								vp.nx, vp.ny, vp.nmarker
							)
						end
						synchronize()
					end
					
					# Write results
					if vp.it % vp.nout == 0 || vp.is_eruption
						@time begin
							@printf("%s writing results to disk\n", bar2)
							if vp.is_eruption
								filename = data_folder * "julia_grid." * string(vp.it) * ".after_eruption" * ".h5"
							else
								filename = data_folder * "julia_grid." * string(vp.it) * ".h5"
							end
							d2dm_make_snapshot(vp, gp, filename, FLAGS.make_snapshot[])
						end
					end
					
					# Check break flag
					if FLAGS.d2dm_break[]
						return 0
					end
				end
				
				# Update timing
				TIMING.time_of_loop[] = time_of_loop_local
			end
		end
		
		# Final write
		@printf("%s writing results to disk\n", bar2)
		filename = data_folder * "julia_grid." * string(vp.nt + 1) * ".h5"
		d2dm_make_snapshot(vp, gp, filename, true)
		
		@printf("\nTotal time: %s\n", total_time)
		
		# Save eruption data
		fid = open(data_folder * "eruptions.bin", "w")
		write(fid, vp.iSample)
		write(fid, gp.eruptionSteps)
		close(fid)
		
		FLAGS.init[] = true
		return 0
	end
end

# Optional: Convenience wrapper
function main()
    D2DM_GUI_Main.dykes_gui()
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
