#TODO: header doc
module D2DM_GUI_Utils
    export read_par, parse_contents_csv, simple_imput, simple_imput_desc, log_to_buffer!
    
    using DataFrames, CSV, Base64, Dates, Dash
    
    function read_par(par, ipar)
        par_name_2 = par[ipar]
        ipar_2 = ipar + 1
        return par_name_2, ipar_2
    end
    
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
            html_label(children="$name :", title=description),
            html_div(
                children=[
                    dcc_input(id=name, value=init_val, type="number", debounce=true)
                ],
            )
        end
    end
    
    function log_to_buffer!(buf_ref, input_string)
        buf_ref[] = input_string * buf_ref[]
        return nothing
    end
    
    function parse_contents_csv(contents, filename, date, init_vp)
        content_type, content_string = split(contents, ',')
        decoded = base64decode(content_string)
        df = DataFrame()
        try
            if occursin("csv", filename)
                str = String(decoded)
                df = CSV.read(IOBuffer(str), DataFrame)
                init_vp.critVol = collect(df[1:end, :EruptionVolumes])
                init_vp.critVolTime = collect(df[1:end, :EruptionTimes])
            end
        catch e
            print(e)
            return html_div(["There was an error processing this file."])
        end
        
        return html_div([
            html_h5(filename),
            html_h6(Libc.strftime(date)),
            dash_datatable(
                data=[Dict(pairs(NamedTuple(eachrow(df)[j]))) for j in 1:nrow(df)],
                columns=[Dict("name" => i, "id" => i) for i in names(df)]
            ),
            html_hr()
        ])
    end
end