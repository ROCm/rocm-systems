using Base: String
using JSON, DataFrames, Dates, CSV

function check_input_args()
    message = "Needs only one json file and at least one trace data file(s)."
    @assert length(ARGS) >= 2 && occursin(".json", join(ARGS)) message
    json_file, trace_files = nothing, String[]
    for arg in ARGS
        @assert Base.Filesystem.ispath(arg) "'$(arg)' not found!"
        if endswith(arg, ".json")
            json_file = arg
        else
            push!(trace_files, arg)
        end
    end
    json_file, trace_files
end

json_file, trace_files = check_input_args()

out = JSON.parsefile(json_file)
for trace in trace_files
    @info "processing $(trace) ..."
    f = tempname() * ".json"
    run(`trace_to_text json $trace $f`)
    todo = JSON.parsefile(f)
    out["traceEvents"] = vcat(
        out["traceEvents"],
        get(todo, "traceEvents", [])
    )
    out["systemTraceEvents"] = (*)(
        get(out, "systemTraceEvents", ""),
        get(todo, "systemTraceEvents", "")
    )
end

output_file = replace("$(string(now())).json", ":" => "-")
open(output_file, "w") do f
    write(f, JSON.json(out))
    @info "output file: $(output_file)"
end

