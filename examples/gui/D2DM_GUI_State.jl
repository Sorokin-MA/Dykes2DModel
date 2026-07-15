module D2DM_GUI_State
    export FLAGS, RUNTIME, TIMING, reset_flags!, update_runtime!
    
    using Dates
    
    # Mutable state using Ref for thread-safety and type-stability
    const FLAGS = (
        init = Ref(true),
        d2dm_break = Ref(false),
        d2dm_started = Ref(false),
        d2dm_stopped = Ref(true),
        d2dm_markers = Ref(true),
        make_snapshot = Ref(false),
    )
    
    # Runtime state
    const RUNTIME = (
        buf = Ref("\n\nWelcome to Dykes2DModel!\n 1.Set parameters and upload history of eruptions \n 2. Generate dykes \n 3. Start calculations\n"),
        mf_boundaries = Ref(0.1),
    )
    
    # Timing state
    const TIMING = (
        time_of_loop = Ref(0.0),
        str_time_spend = Ref(0.0),
        str_time_left = Ref(Time(0)),
    )
    
    function reset_flags!()
        FLAGS.init[] = true
        FLAGS.d2dm_break[] = false
        FLAGS.d2dm_started[] = false
        FLAGS.d2dm_stopped[] = true
        FLAGS.make_snapshot[] = false
        TIMING.time_of_loop[] = 0.0
        TIMING.str_time_spend[] = 0.0
        TIMING.str_time_left[] = Time(0)
    end
    
    function update_runtime!(time_loop::Float64, it::Int, nt::Int32)
        TIMING.time_of_loop[] = time_loop
        TIMING.str_time_spend[] += time_loop
        TIMING.str_time_left[] = Time(0) + Second(Int64(floor(time_loop * (nt - it))))
    end
end