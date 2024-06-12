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
    my_string = 0
    app = dash()
    num_columns = 6

    gp = GridParams()
    vp = VarParams()

    hm = PlotlyJS.heatmap(
    z=[[1, missing, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
    x=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    y=["Morning", "Afternoon", "Evening"],
    hoverongaps=false
)

    app.layout = html_div() do
        html_h1(
            "Dykes2D",
            style=Dict("color" => "#000000", "textAlign" => "center"),
        ), 
		html_div() do
            html_h2(
                "1.Set parameters.",
                style=Dict("color" => "#000000", "textAlign" => "left"),
            ),
            html_button("Load config", id="load-config-but"),
            dcc_tabs(id="tabs-example-graph", value="tab-1-example-graph", children=[
                dcc_tab(label="Physics", value="tab-1-example-graph"),
                dcc_tab(label="Numerics", value="tab-2-example-graph")
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

        html_button("Start", id="start-but"),
        html_button("Stop", id="stop-but"),
        html_button("Show data", id="show-but"),

		#html_div(id="dikes_figures", className="row", style=Dict("columnCount" => 3)) do
        #        dcc_graph(
		#		figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(transpose(T))), title="T"))),
         #       dcc_graph(
		#		figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(transpose(C))), title="C")))
        #end,

		html_div(id="dikes_figures", className="row", style=Dict("columnCount" => 3)) do
        #       dcc_graph(
		#		figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(transpose(T))), title="T"))),
        #       dcc_graph(
		#		figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(transpose(C))), title="C")))
		end,
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

		html_div([
        dcc_textarea(
			id = "log_buffer",
            value="Dykes2D GUI successfully uploaded!",
            style=Dict("width" => "100%"),
            readOnly=true
        ),
		dcc_interval( id="interval-component",
            interval=1*1000, # in milliseconds)
            n_intervals=0
		)])
    end
#=
    callback!(app, Output("my-output", "children"), Input("Lx", "value")) do input_value
        vp.Lx = input_value
    end
=#
#=
    callback!(app, Output("test_tooltip", "show"), Input("test_label", "value")) do input_value
        my_string = input_value
        println(my_string)
    end


=#


    callback!(app, Output("tabs-content-figure-graph", "children"),
        Input("tabs-figure-graph", "value")) do tab
        if tab == "tab-1-figure-graph"
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly

				h_T = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
				copyto!(h_T, gp.T)
				return html_div(id="dikes_figures_T", className="row", style=Dict("columnCount" => 3)) do
				dcc_graph(id="T_graph",
				figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_T)), title="T"))),
				dcc_loading(id="ls-loading-1", children=[html_div(id="ls-loading-output-1")], type="default")
				end
			if tab == "tab-2-figure-graph"
				xs = 0:vp.dx:vp.Lx
				ys = 0:vp.dy:vp.Ly
				h_C = Array{Float64,1}(undef, vp.nx * vp.ny)#array of double values from matlab script
				copyto!(h_C, gp.C)
				return html_div(id="dikes_figures_C") do
				dcc_graph(
					figure = Plot(PlotlyJS.heatmap(x = xs, y =ys, z=collect(eachcol(h_C)), title="C")))

				end


			end
		end
	end


			#callback!(app, Output("log_buffer", "value"), Input("log_buffer", "value"), Input("start-but", "n_clicks")) do log_buffer, n_clicks
			#	return log_buffer * "\nasdf\n"
			#end
	#
	#=
	callback!(app, Output("stop-but", "n_clicks"), Input("stop-but", "n_clicks")) do n_intervals
		for i in 1:100
			buf = buf*"22"
			sleep(2)
		end
		return 1
	end
=#	
	callback!(app, Output("log_buffer", "value"), Input("interval-component", "n_intervals")) do n_intervals
		println("debug time_interval")
		return buf
	end

#	callback!(app, Output("ls-loading-output-1", "children"), Input("T_graph", "figure")) do n_intervals
#		return n_intervals
#	end

	callback!(app, Output("start-but", "n_clicks"), Input("start-but", "n_clicks"),prevent_initial_call=true) do n_clicks
		println("start_button clicked")
		global flag_break = false;
		main_test_gui(gp, vp, G_FLAG_INIT)
		return n_clicks + 1
    end

	callback!(app, Output("stop-but", "n_clicks"), Input("stop-but", "n_clicks"),prevent_initial_call=true) do n_clicks
		println("stop button clicked")
	global flag_break = true;
		return n_clicks + 1
    end

    callback!(app, Output("tabs-content-example-graph", "children"),
        Input("tabs-example-graph", "value")) do tab
        if tab == "tab-1-example-graph"
            return html_div(className="row") do
                html_div(className="info", style=Dict("columnCount" => num_columns)) do
                end,
                html_div(className="row", style=Dict("columnCount" => num_columns)) do
                    simple_imput("Lx", 10),
                    simple_imput("Ly", 10),
                    simple_imput("Lz", 10),
                    simple_imput("dike_to_sill", 10),
                    simple_imput("narrow_factor", 10),
                    simple_imput("Lam_r", 10),
                    simple_imput("Lam_m", 10),
                    simple_imput("rho", 10),
                    simple_imput("Cp", 10),
                    simple_imput("L_heat", 10),
                    simple_imput("T_top", 10),
                    simple_imput("dTdy", 10),
                    simple_imput("T_magma", 10),
                    simple_imput("T_ch", 10),
                    simple_imput("Qv", 10),
                    simple_imput("ka_years", 10),
                    simple_imput("Ly_eruption", 10),
                    simple_imput("dT", 10),
                    simple_imput("E", 10),
                    simple_imput("nu", 10),
                    simple_imput("tsh", 10),
                    simple_imput("gamma", 10)
                end
            end
        end

        if tab == "tab-2-example-graph"
            return html_div(className="row") do
            simple_imput("seed", 10),
            html_div(className="row", style=Dict("columnCount" => num_columns)) do
                simple_imput("nx", 10),
                simple_imput("ny", 10),
                simple_imput("dt", 10),
                simple_imput("steph", 10),
                simple_imput("nl", 10),
                simple_imput("nmy", 10),
                simple_imput("pmlt", 10),
                simple_imput("eiter", 10),
                simple_imput("CFL", 10),
                simple_imput("pic_amount", 10),
                simple_imput("nout", 10)
            end
        end
        end
    end

    run_server(app)
end
