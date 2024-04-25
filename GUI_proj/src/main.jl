#import Pkg; Pkg.add
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

#plotincanvas()

function main()


b = GtkBuilder(filename="dykes_test.glade")
win = b["main_id"]
log_widget = b["log_id"]
progress_widget_id = b["progress_id"]
#log_buffer = GtkTextBuffer()
#log_widget = GtkTextView(log_buffer)
	
log_buffer  = log_widget[:buffer, GtkTextBuffer]
	
#G_.text(tb, "test", -1)
#println(set_gtk_property(log_buffer, :text, String, "Lol?"))
#log_buffer.text[String] = "my text \n kek"
println(get_gtk_property(log_buffer, :text, String))
#progress_widget_id = GtkProgressBar()
#setproperty!(progress_widget_id, :fraction, 1/20)
progress_widget_id.fraction[Float64] = 6.0/20;
#progress_widget_id[:fraction, 9/20]
#println(GtkTree(log_widget))
#log_buffer = GtkTextBufferLeaf();
#b["log_id"] = log_widget

image_widget_id = b["box_widget_id"]

Lx_widget_id = b["Lx"]
io = PipeBuffer()
histio(n) = show(io, MIME("image/png"),  histogram(randn(n)))
#histio(n) = show(io, MIME("image/png"),  FileIO.load("T_julia.png"))
can = GtkCanvas()
    push!(image_widget_id, can)
    set_gtk_property!(image_widget_id, :expand, can, true)
    @guarded draw(can) do widget
        ctx = getgc(can)
        #n = Int(Gtk.GAccessor.value(Lx_widget_id))
		n = parse(Int, (get_gtk_property(Lx_widget_id,:text, String)))
		histio(n)
        img = read_from_png(io)
        set_source_surface(ctx, img, 0, 0)
        paint(ctx)
    end
    #id = signal_connect((w) -> draw(can), Lx_widget_id, "value-changed")



showall(win)

end
