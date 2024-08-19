using Dash
using Dash
using DashHtmlComponents
using DashCoreComponents

using Plots
plotly()

function simple_imput(name::String, init_val)
    html_div() do
        html_label(name * ": "),
        html_div(
            children=[
                dcc_input(id=name, value=init_val, type="number", debounce=true)
            ],
        )
    end
end


function simple_imput_desc(name::String, init_val, description)
    html_div() do
        #html_label(name * ": "),
        html_label(children="$name :", title=description),
        html_div(
            children=[
                dcc_input(id=name, value=init_val, type="number", debounce=true)
            ],
        )
    end
end


function dikes_gui()
	#init dash
	app = dash()

	num_columns = 6

	#init main structs
	gp = GridParams()			#array params
	vp = VarParams()			#scalar params
	init_vp = InitVarParams()	#params for generate random

	# campri_calc = Int32[10000, 20000, 25000, 26000]
	# tyear = 365 * 24 * 3600		#seconds in year
	# tfin = (tfin/tyear)/1.e3
	# campri_calc = -(1 .- campri_calc./nt) .* (tfin)

	#cat(159.8, 124.9, 18, 5, 1);

	#append!(gp.eruptionSteps, 1000)
	#append!(gp.eruptionSteps, 20000)

	#append!(gp.eruptionSteps, 200)
	#append!(gp.eruptionSteps, 10000)
	#vp.nt = 20000
	#
		#p = PlotlyJS.figure()

			#main layout
	app.layout = html_div(style=Dict("background_color" => "blue")) do
		html_h1(
			"Dykes2DModel",
			style=Dict("color" => "#000000", "textAlign" => "center"),
		),
		html_div() do
			html_h2(
				"1. Set parameters.",
				style=Dict("color" => "#000000", "textAlign" => "left"),
			),
			html_button("Load config", id="load-config-but"),
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

		html_h2(
			"3. Start\\Stop.",
			style=Dict("color" => "#000000", "textAlign" => "left"),
		),
		#buttons
		html_button("Start", id="start-but", disabled = false),
		html_button("Stop", id="stop-but", disabled = false),
		html_div(
			children=[
				dcc_input(id="time_left_label", value="Time left",  debounce=true)
			],
		),
		html_div(
			children=[
				dcc_input(id="percent_done_label", value="Done",  debounce=true)
			],
		),
		html_progress(id = "progress_bar", value = string(vp.it), max = vp.nt, style=Dict("width" => "100%")),
		html_div(id="eruptions-timeline", className="row",style=Dict("columnCount" => 1) ) do
        			#dcc_graph(id="T_graph",figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))
			#dcc_graph(id="eruptions-timeline-graph")
		end,
		html_h2(
			"Log",
			style=Dict("color" => "#000000", "textAlign" => "left"),
		),
		html_div([
			dcc_textarea(
				id = "log_buffer",
				value="Dykes2D GUI successfully uploaded!\n",
				style=Dict("width" => "100%","overflow" => "scroll", "resize" => "none"),
				readOnly=true, rows=15
			),
			dcc_interval(id="interval-component",
				interval=1*1000, # in milliseconds
				n_intervals=1)
		]),
		html_h2(
			"4. Snapshots.",
			style=Dict("color" => "#000000", "textAlign" => "left"),
		),
		html_button("Make snapshot", id="make-snap-but", disabled = false),
		html_button("Load snapshot", id="load-snap-but", disabled = false),
		html_div() do
			dcc_tabs(id="tabs-figure-graph", value="tab-1-figure-graph", children=[
					dcc_tab(label="T", value="tab-1-figure-graph"),
					dcc_tab(label="C", value="tab-2-figure-graph"),
					dcc_tab(label="P", value="tab-3-figure-graph")
				]
			),
			html_div(id="tabs-content-figure-graph")
		end
	end

	#table for input params
	callback!(app, Output("tabs-content-example-graph", "children"),
		Input("tabs-example-graph", "value")) do tab
			if tab == "tab-1-example-graph"
			    return html_div(className="row") do
			        html_div(className="info", style=Dict("columnCount" => num_columns)) do
			        end,
			        html_div(className="row", style=Dict("columnCount" => num_columns)) do
			            simple_imput_desc("Lx", init_vp.Lx, descr_Lx),
			            simple_imput_desc("Ly", init_vp.Ly, descr_Ly),
			            simple_imput_desc("Lz", init_vp.Lz, descr_Lz),
			            simple_imput("calc_years", init_vp.calc_years),
			            simple_imput("dike_to_sill", init_vp.dike_to_sill),
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
			            simple_imput("Ly_eruption", init_vp.Ly_eruption),
			            simple_imput("dT", init_vp.dT),
			            simple_imput("E", init_vp.E),
			            simple_imput("nu", init_vp.nu),
			            simple_imput("tsh", init_vp.tsh),
			            simple_imput("gamma", init_vp.gamma)
			        end
			    end

			end
			if tab == "tab-2-example-graph"
			    return html_div(className="row") do
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
				end
			end
			if tab == "tab-3-example-graph"
				return html_div(className="row") do
					#callback for eruption csv file
					callback!(app,
						Output("output-data-upload", "children"),
						Input("upload-data", "contents"),
						State("upload-data", "filename"),
						State("upload-data", "last_modified"),
						) do contents, filename, last_modified
							if !(contents isa Nothing)
							children = [
							parse_contents_csv(c..., init_vp) for c in
								zip(contents, filename, last_modified)]
								return children
							end
						end
					dcc_upload(
						id="upload-data",
						children=html_div([

						html_a("Select Files")
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
				end
			end
	end

	#callback for generate button
	callback!(app, Output("generate-but", "n_clicks"), Input("generate-but", "n_clicks"), prevent_initial_call=true) do n_clicks
		println("generate button clicked")
		dikes_rand_param(init_vp)
		return n_clicks + 1
	end

	#callback for start button
	callback!(app, Output("stop-but", "disabled"), Input("start-but", "n_clicks"), prevent_initial_call=true) do n_clicks
		println("start_button clicked")

		global D2DM_STARTED = true;
		global D2DM_STOPED = false;
		global flag_break = false;

		main_test_gui(gp, vp, G_FLAG_INIT)
		return false
	end

	#callback for stop button
	callback!(app, Output("stop-but", "n_clicks"), Input("stop-but", "n_clicks"), prevent_initial_call=true) do n_clicks
		println("stop button clicked")
	
		global D2DM_STARTED = false;
		global D2DM_STOPED = true;
		global flag_break = true;
	
		return n_clicks + 1
	end
	
	# callback!(app, Output("start-but", "disabled"), Input("stop-but", "disabled"),  prevent_initial_call=true) do n_clicks
	# 	return true
	# end

	#callback for figures panel
	callback!(app, Output("tabs-content-figure-graph", "children"), Input("tabs-figure-graph", "value"), prevent_initial_call=true) do tab
		if tab == "tab-1-figure-graph"
			if(vp.dx == 0.0)
				return nothing
			end
			xs = 0:vp.dx:vp.Lx
			ys = 0:vp.dy:vp.Ly

			h_T = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
			copyto!(h_T, gp.T)
			return html_div(id="dikes_figures_T", className="row",style=Dict("columnCount" => 3) ) do
				#dcc_graph(id="T_graph",figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))
				dcc_graph(id="T_graph", figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(reshape(h_T, (vp.nx, vp.ny)))), title="T")))
			end
		end
		if tab == "tab-2-figure-graph"
			if(vp.dx == 0.0)
				return nothing
			end
			xs = 0:vp.dx:vp.Lx
			ys = 0:vp.dy:vp.Ly
			h_C = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
			copyto!(h_C, gp.C)
			return html_div(id="dikes_figures_C", className="row",style=Dict("columnCount" => 3)) do
				#NOTE: dash bug, see https://github.com/plotly/Dash.jl/issues/60
				
				#p = Plot(PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(h_C)), title="C"))
				#data = Plots.plotly_series(p)
				#data[1][:z] = [c for c in eachcol(data[1][:z])] # <-- As a temporary
				#layout = Plots.plotly_layout(p)
				#dcc_graph(id="C_graph", figure = (;data, layout), title="C")

				dcc_graph(id="C_graph", figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(reshape(h_C, (vp.nx, vp.ny)))), title="C")))
			end
		end
	end

	#update progress bar
	callback!(app, Output("progress_bar", "value"),Output("progress_bar", "max"), Input("interval-component", "n_intervals")) do n_intervals
		return string(vp.it), vp.nt
	end

	#update progress bar
	callback!(app, Output("eruptions-timeline", "children"), Input("interval-component", "n_intervals")) do n_intervals
		
		return html_div(className="row") do

			campri_calc = -(vp.nt .- gp.eruptionSteps)./vp.nt.*init_vp.calc_years/1000;
			campri_real = -vcat(39.8, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 11.5, 11, 10.6, 9.6, 9.3, 5.1, 4.9, 4.5, 4.3, 4.2, 4.2, 4.2, 4.1, 3.9, 0.5);
			campri_now = [-(vp.nt - vp.it)/vp.nt*init_vp.calc_years/1000];

			campri_cal_graph  = PlotlyJS.scatter(x=campri_calc, y= zeros(length(campri_calc)), mode="markers", name="campri_calc", showlegend=true, marker_size=14)
			campri_real_graph = PlotlyJS.scatter(x=campri_real, y= zeros(length(campri_real)), mode="markers", name="campri_real", showlegend=true, marker_size=10)
			campri_now_graph = PlotlyJS.scatter(x=campri_now, y= zeros(length(campri_now)), mode="markers", name="now", showlegend=true, marker_size=10)
			data = [campri_cal_graph, campri_real_graph,campri_now_graph]

			layout = Layout(title="Eruptions graph",
							xaxis=attr(title="time, (ka)", showgrid=false, zeroline=false),
							yaxis=attr(showgrid=false, range=[0, 0]))

			p = PlotlyJS.plot(data, layout)

			dcc_graph(id="eruptions-timeline-graph", figure = p)
		end
	end

	#time left label callback
	callback!(app, Output("time_left_label", "value"),Output("percent_done_label", "value"), Input("interval-component", "n_intervals")) do n_intervals
		return "Time left: "*string(str_time_left), "Done: "*string(@sprintf("%03s", ((Float64(vp.it)-1)/Float64(vp.nt))*100))*"%"
	end

	#refresh buffer every n_intrervals seconds
	callback!(app, Output("log_buffer", "value"), Input("interval-component", "n_intervals")) do n_intervals
		#println("debug time_interval")
		return buf
	end


	#Output for Lx
	callback!(app, Output("Lx", "value"), Input("Lx", "value")) do input_value
		init_vp.Lx = input_value
		return init_vp.Lx
	end


	run_server(app, debug=true)
#    run_server(app)
end

function log_to_buffer(input_string)
	global buf= input_string*buf
end

#parse csv file
function parse_contents_csv(contents, filename, date, init_vp)
	content_type, content_string = split(contents, ',')
	decoded = base64decode(content_string)
	df = DataFrame()
	try
		if occursin("csv", filename)
			str = String(decoded)
			df =  CSV.read(IOBuffer(str), DataFrame)
			init_vp.critVol = collect(df[1:end, :EruptionVolumes])
			init_vp.critVolTime = collect(df[1:end, :EruptionTimes])
			println(init_vp.critVolTime)
		end
	catch e
		print(e)
		return html_div([
		"There was an error processing this file."
		])
	end

	return html_div([
		html_h5(filename),
		html_h6(Libc.strftime(date)),

		dash_datatable(
				data=[Dict(pairs(NamedTuple(eachrow(df)[j]))) for j in 1:nrow(df)],
				columns=[Dict("name" =>i, "id" => i) for i in names(df)]
		),

		# horizontal line
		html_hr()

		# For debugging, display the raw contents provided by the web browser
		# html_div("Raw Content"),
		# html_pre(string(contents[1:200], "..."), style=Dict(
		#     "whiteSpace" => "pre-wrap",
		#     "wordBreak" => "break-all"
		# ))
	])
end
