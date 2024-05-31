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

    my_string = 0
    app = dash()
    num_columns =5 

    app.layout = html_div() do
        html_h1(
            "Dykes2D",
            style=Dict("color" => "#000000", "textAlign" => "center"),
        ),
        html_div(className="row") do
            html_h2(
                "Physics",
                style=Dict("color" => "#000000", "textAlign" => "left"),
            ),
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
        end,
        html_div(className="row") do
            html_h2(
                "Numerics",
                style=Dict("color" => "#000000", "textAlign" => "left"),
            ),
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
        end,
        html_div(className="row") do
            html_div(className="nine columns",
                dcc_graph(
                    id="graphic",
                    animate=true,
                )
            )
        end,
        html_br(),
        html_div(id="my-output")

    end

    callback!(app, Output("my-output", "children"), Input("Lx", "value")) do input_value
        #"Output: $(input_value)"
        my_string = input_value
        println(my_string)
    end

    run_server(app)

end
