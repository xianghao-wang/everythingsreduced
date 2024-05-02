module Main {
    use IO;
    
    proc main(args: [] string) {
        if args.size < 2 {
            try! stderr.write("Usage: ", args[0], " <benchmark> <options>\n"
                , "\n"
                , "Valid benchmarks:\n"
                , "  dot, complex_sum, complex_sum_soa, complex_min, field_summary, describe"
                , "field_summary, describe"
                , "\n");
            
            exit(1);
        }

        const run = args[1];

        writeln("Unit of time: milliseconds\n");

        //////////////////////////////////////////////////////////////////////////////
        // Run Dot Product Benchmark
        //////////////////////////////////////////////////////////////////////////////
        if run == "dot" {
            // check_for_option(args.size);
            // const N = get_problem_size(args[2]);
            // bench_dot(N);
        }
    }

    proc check_for_option(argc: int) {
        if argc != 3 {
            try! stderr.writeln("Missing problem size");
            exit(1);
        }
    }

    proc get_problem_size(option: string) {
        return try! option: int;
    }
}