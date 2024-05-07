#import Pkg; Pkg.add


include("dykes_init.jl")
include("dykes_structs.jl")
include("dykes_funcs.jl")

using Gtk
#using Plots
function on_button_clicked(w)
  println("The button has been clicked")
end

using Cairo
using Gtk
using Plots

const io = PipeBuffer()


histio(n) = show(io, MIME("image/png"),  histogram(randn(n)))

function plotincanvas(h = 900, w = 800)
    win = GtkWindow("Normal Histogram Widget", h, w) |> (vbox = GtkBox(:v) |> (slide = GtkScale(false, 1:500)))
    Gtk.G_.value(slide, 250.0)
    can = GtkCanvas()
    push!(vbox, can)
    set_gtk_property!(vbox, :expand, can, true)
    @guarded draw(can) do widget
        ctx = getgc(can)
        n = Int(Gtk.GAccessor.value(slide))
        histio(n)
        img = read_from_png(io)
        set_source_surface(ctx, img, 0, 0)
        paint(ctx)
    end
    id = signal_connect((w) -> draw(can), slide, "value-changed")
    showall(win)
    show(can)
end


function start_buttom_test(widget, event, log_widget, progress_widget_id)
    @async begin
			sleep(5)
				progress_widget_id.fraction[Float64] = Float64((17%20))/20;
				#log_widget.text[String] *=  "$i\n"
				Cint(false)

	end
end

flag_start::Bool = false;

function start_buttom(widget)
	flag_start = true
end


function log_to_gui(log_buffer, str)
		log_buffer.text[String] *= str 
end

function scroll_adj(widget, event, vadjustment)
	println("changed!! c")
	println(GAccessor.value(vadjustment))
    #set_gtk_property!(vadjustment, :value, 10000)
	#max_vl = get_gtk_property!(vadjustment,:upper, Int32)
	#println(max_vl)
	#=
	min_vl = get_gtk_property!(widget,:lower, Int32)
	println(min_vl)
	=#
end

function dykes_gui()

gtk_builder = GtkBuilder(filename="src/dykes_test.glade")
win = gtk_builder["main_id"]


log_widget = gtk_builder["log_id"]
	vadjustment = GAccessor.vadjustment( gtk_builder["scroller"])
log_buffer  = log_widget[:buffer, GtkTextBuffer]

log_buffer.text[String] *=  "my text \n kek"
start_button_windget = gtk_builder["start_button_id"]
progress_widget_id = gtk_builder["progress_id"]

id = addprocs(1)[1]

#id = signal_connect((widget, event) -> start_buttom_test(widget, event, log_buffer, progress_widget_id), start_button_windget, "button-press-event")
id = signal_connect(start_buttom, start_button_windget, "clicked")

#id_2 = signal_connect((widget, event) -> scroll_adj(widget, event, vadjustment),log_buffer, "changed") 
	
id_2 = signal_connect(log_buffer, "changed") do widget 
	println("changed!! ")
	#println(GAccessor.value(vadjustment))
		GAccessor.value(vadjustment,GAccessor.upper(vadjustment))
end
		#=
	max_vl = get_gtk_property!(vadjustment,:upper, Int32)
	println(max_vl)
	min_vl = get_gtk_property!(vadjustment,:lower, Int32)
	println(min_vl)
    set_gtk_property!(vadjustment, :value, max_vl)
		=#


showall(win)

		log_to_gui(log_buffer, "\nStart!\n")
	GAccessor.value(vadjustment, GAccessor.upper(vadjustment))

    @async begin
	#Gtk.GLib.g_idle_add(nothing) do
		
	#end
	#while(flag_start!=true)

	#end

    #Initialization of inner random
    Random.seed!(1234)

    #TODO:Настроить фильтр
    #TODO:Настроить девайс если не выбран

    #print_gpu_properties()
    gp = GridParams()
    vp = VarParams()

    #reading params from hdf5 files
    @printf("%s reading params			  ", bar1)
    read_params(gp, vp)

    #initialisation of T and Ph variables
    @printf("%s initialization			  ", bar1)
    init(gp, vp)


    filename = Array{Char,1}(undef, 1024)
    global iSample = Int32(1)
    eruptionSteps = Vector{Int32}()
    is_eruption = false


    #main loop
    for it in 1:vp.nt
        @printf("%s it = %d", bar1, it)
        is_eruption = false
        is_intrusion = (gp.ndikes[it] > 0)
        nerupt = 1


		#log_buffer.text[String] *=  "my text \n kek"
		log_to_gui(log_buffer, "kek\n")

			println(GAccessor.value(vadjustment))
		GAccessor.value(vadjustment, 1000)

        #checking eruption criteria and advect particles if eruption
        if (it % nerupt == 0)
            maxVol, maxIdx = check_melt_fracton(gp, vp)

            dxl = vp.dx * vp.nl
            dyl = vp.dy * vp.nl

            if (maxVol * dxl * dyl >= gp.critVol[iSample])
                @printf("%s erupting %07d cells   | ", bar2, maxVol)
                eruption_advection(gp, vp, maxVol, maxIdx, iSample, is_eruption)
            end

        end


        #processing intrusions of dikes
        if (is_intrusion)
            @printf("%s inserting %02d dikes	   | ", bar2, gp.ndikes[it])
            inserting_dykes(gp, vp, it)
        end


        #if eruption or injection happend, taking into account their effcto on grid with p2g
        if (is_eruption || is_intrusion)
            @printf("%s p2g interpolation		| ", bar2)
            p2g_interpolation(gp, vp)


            @printf("%s particle injection	   | ", bar2)
            particles_injection(gp, vp)
        end


        #solving heat equation
        #NOTE:difference like 2.e-1, mb make sense to fix it
        @time begin
            @printf("%s solving heat diffusion   | ", bar2)

            blockSize = (16, 32)
            gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

            copyto!(gp.T_old, gp.T)
            for isub = 0:vp.nsub-1
                @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] update_T!(gp.T, gp.T_old, vp.T_top, vp.T_bot, gp.C, vp.lam_r_rhoCp, vp.lam_m_rhoCp, vp.L_Cp, vp.dx, vp.dy, vp.dt, vp.nx, vp.ny)
                synchronize()
            end
        end


        #g2p interpolation
        @time begin
            @printf("%s g2p interpolation		| ", bar2)
            blockSize1D = 512
            gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D
            @cuda blocks = gridSize1D threads = blockSize1D g2p!(gp.T, gp.T_old, gp.px, gp.py, gp.pT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.npartcl)

            gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D
            pic_amount_tmp = vp.pic_amount
            pic_amount = 1.0
            @cuda blocks = gridSize1D threads = blockSize1D g2p!(gp.T, gp.T_old, gp.mx, gp.my, gp.mT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.nmarker)
            synchronize()
            vp.pic_amount = pic_amount_tmp
        end


        #mailbox output
        if (it % vp.nout == 0 || is_eruption)
            @time begin
                @printf("%s writing results to disk  | ", bar2)
                filename = "data/julia_grid." * string(it) * ".h5"

                small_mailbox_out(filename, gp.T, gp.pT, gp.C, gp.mT, gp.staging, is_eruption, gp.L, vp.nx, vp.ny, vp.nxl, vp.nyl, vp.max_npartcl, vp.max_nmarker, gp.px, gp.py, gp.mx, gp.my, gp.h_px_dikes, gp.pcnt, gp.mfl)
                #mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl, max_nmarker, px, py, mx ,my, h_px_dikes,pcnt, mfl);
            end
        end
    end

    @printf("\nTotal time: ")

    fid = open("data/eruptions.bin", "w")
    write(fid, iSample)
    write(fid, gd.eruptionSteps)
    close(fid)
    return 0

	end

end
