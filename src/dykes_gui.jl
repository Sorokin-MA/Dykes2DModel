using Dash

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

#=
    dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
    ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

    io = open(data_folder*"pa.bin", "r")
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

    close(io)

	#Lx = 4000
	#Lx = 5000
    xs = 0:dx:Lx
    ys = 0:dy:Ly

#    fid = h5open(data_folder * "julia_grid.40001.h5", "r")
    fid = h5open(data_folder * "julia_grid.40001.h5", "r")
    T = read(fid, "T")
    C = read(fid, "C")
    close(fid)

	T = reshape(T,(length(xs), length(ys)))
	C = reshape(C,(length(xs), length(ys)))
=#

	#init dash
	app = dash()

	num_columns = 6

	#init main structs
	gp = GridParams()
	vp = VarParams()
	init_vp = InitVarParams()

	#main layout
	app.layout = html_div(style=Dict("background_color" => "blue")) do
		html_h1(
			"Dykes2D",
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

#=
        html_div() do
            html_label(title="test label, ∂(2x + 3)/∂x,\nDimention [m]:, \nApproximate range: 1000 - 2000", children="test_label"),
            html_div(
                children=[
                    dcc_input(id="test_input", value=11, type="number", debounce=true),
                    dcc_tooltip(id="test_tooltip", direction="bottom", background_color="darkblue", border_color="blue")
                ],
            )
        end,
=#

        html_br(),
        html_div(id="my-output"),

        html_h2(
            "2. Generate data.",
            style=Dict("color" => "#000000", "textAlign" => "left"),
        ),

        html_button("Generate", id="generate-but"),

        html_h2(
            "3. Start calculations.",
            style=Dict("color" => "#000000", "textAlign" => "left"),
        ),

		#buttons
        html_button("Start", id="start-but"),
        html_button("Stop", id="stop-but"),
        html_button("Show data", id="show-but"),

		html_div() do
            dcc_tabs(id="tabs-figure-graph", value="tab-1-figure-graph", children=[
                dcc_tab(label="T", value="tab-1-figure-graph"),
                dcc_tab(label="C", value="tab-2-figure-graph"),
                dcc_tab(label="P", value="tab-3-figure-graph")
            ]
            ),
            html_div(id="tabs-content-figure-graph")
        end, 

        html_h2(
            "Log",
            style=Dict("color" => "#000000", "textAlign" => "left"),
        ),
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
		html_progress(id = "progress_bar", value = vp.it, max = vp.nt, style=Dict("width" => "100%")),
		html_div([
        dcc_textarea(
			id = "log_buffer",
            value="Dykes2D GUI successfully uploaded!\n",
            style=Dict("width" => "100%","overflow" => "scroll", "resize" => "none"),
            readOnly=true, rows=15
        ),
		dcc_interval(id="interval-component",
            interval=1*1000, # in milliseconds)
            n_intervals=1
		)])
    end

	#callback for choose figures panel
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
				dcc_graph(id="T_graph",
				figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T")))
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
				dcc_graph(id="C_graph", figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_C)), title="C")))
			end
		end
	end


	#callback!(app, Output("log_buffer", "value"), Input("log_buffer", "value"), Input("start-but", "n_clicks")) do log_buffer, n_clicks
	#	return log_buffer * "\nasdf\n"
	#end
	
	#=
	callback!(app, Output("stop-but", "n_clicks"), Input("stop-but", "n_clicks")) do n_intervals
		for i in 1:100
			buf = buf*"22"
			sleep(2)
		end
		return 1
	end
	=#	

	#callback for generate button
	callback!(app, Output("generate-but", "n_clicks"), Input("generate-but", "n_clicks"), prevent_initial_call=true) do n_clicks
		println("generate button clicked")
		dikes_rand_param(init_vp)
		return n_clicks + 1
    end

	#refresh buffer every n_intrervals seconds
	callback!(app, Output("progress_bar", "value"),Output("progress_bar", "max"), Input("interval-component", "n_intervals")) do n_intervals
		return vp.it, vp.nt
	end

	callback!(app, Output("time_left_label", "value"),Output("percent_done_label", "value"), Input("interval-component", "n_intervals")) do n_intervals
		return "Time left: "*string(str_time_left), "Done: "*string(((Float64(vp.it)-1)/Float64(vp.nt))*100)*"%"
	end

	callback!(app, Output("log_buffer", "value"), Input("interval-component", "n_intervals")) do n_intervals
		#println("debug time_interval")
		return buf
	end

	#callback for start button
	callback!(app, Output("start-but", "n_clicks"), Input("start-but", "n_clicks"),prevent_initial_call=true) do n_clicks
		println("start_button clicked")
		global flag_break = false;
		main_test_gui(gp, vp, G_FLAG_INIT)
		return n_clicks + 1
    end

	#callback for stop button
	callback!(app, Output("stop-but", "n_clicks"), Input("stop-but", "n_clicks"),prevent_initial_call=true) do n_clicks
		println("stop button clicked")
		global flag_break = true;
		return n_clicks + 1
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
			            simple_imput("ka_years", init_vp.ka_years),
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
				#df = CSV.read(download("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv"), DataFrame)
				#filename = "test.csv"
				#filepath = joinpath(@__DIR__, filename)
				#df = CSV.read(filepath, DataFrame)
					#
			    return html_div(className="row") do
					dcc_upload(
						id="upload-datan",
						children=html_div([
						"Drag and Drop or ",
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
					html_div(id="output-data-uploadn")
					# dash_datatable(
					# 	id="table",
					# 	columns=[Dict("name" =>i, "id" => i) for i in names(df)],
					# 	data = Dict.(pairs.(eachrow(df)))
					# )
				end
			end
        end

	#callback for eruption csv file
	callback!(
		app,
		Output("output-data-uploadn", "children"),
		Input("upload-datan", "contents"),
		State("upload-datan", "filename"),
		State("upload-datan", "last_modified"),
		) do contents, filename, last_modified
			if !(contents isa Nothing)
			children = [
			parse_contents(c..., init_vp) for c in
				zip(contents, filename, last_modified)]
			return children
		end
	end

    callback!(app, Output("Lx", "value"), Input("Lx", "value")) do input_value
        vp.Lx = input_value
		return vp.Lx
    end

    run_server(app)
end

function log_to_buffer(stringg)
	#Dates.format(now(), "[yyyy-mm-dd HH:MM:SS]|")*
	global buf= stringg*buf
end

function parse_contents(contents, filename, date, init_vp)
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

      html_hr()  # horizontal line

      # For debugging, display the raw contents provided by the web browser
      # html_div("Raw Content"),
      # html_pre(string(contents[1:200], "..."), style=Dict(
      #     "whiteSpace" => "pre-wrap",
      #     "wordBreak" => "break-all"
      # ))
  ])
end
