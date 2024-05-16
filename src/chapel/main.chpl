module Main {
    use IO;

    use Dot;
    use ComplexSum;
    use ComplexSumSOA;
    use FieldSummary;

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

        const name = args[1];
        const gpuLocale = here.gpus[0];

        writeln("Unit of time: milliseconds\n");

        select name {
            when "dot" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                bench_dot(N);
            }

            when "complex_sum" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                bench_complex_sum(N);
            }

            when "complex_sum_soa" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                bench_complex_sum_soa(N);
            }

            when "field_summary" {
                bench_field_summary();
            }

            otherwise {
                try! stderr.writeln("Invalid benchmark: ", name);
                exit(1);
            }
        }
    }

    /* Check #arguments for getting problem dimension */
    proc check_for_option(argc: int) {
        if argc != 3 {
            try! stderr.writeln("Missing problem size");
            exit(1);
        }
    }

    /* Get problem dimension */
    proc get_problem_size(option: string) {
        const N = try! option: int;
        try! stderr.writeln("Problem size: ", N);
        return N;
    }
}