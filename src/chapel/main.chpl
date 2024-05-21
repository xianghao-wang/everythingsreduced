module Main {
    use IO;

    use Shared;
    use Dot;
    use ComplexSum;
    // use ComplexSumSOA;
    // use ComplexMin;
    // use FieldSummary;

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
        const reduceLocale = if useGPU then here.gpus[0] else here;

        writeln("Unit of time: milliseconds\n");

        select name {
            when "dot" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                if useGPU {
                    on reduceLocale do bench_dot(N);
                } else {
                    bench_dot(N);
                }
            }

            when "complex_sum" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                if useGPU {
                    on reduceLocale do bench_complex_sum(N);
                } else {
                    bench_complex_sum(N);
                }
            }

            when "complex_sum_soa" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                // bench_complex_sum_soa(N);
            }

            when "complex_min" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);
                // bench_complex_min(N);
            }

            when "field_summary" {
                // bench_field_summary();
            }

            when "all" {
                check_for_option(args.size);
                const N = get_problem_size(args[2]);

                if useGPU {
                    on reduceLocale {
                        bench_dot(N);
                        bench_complex_sum(N);
                    }
                } else {
                    bench_dot(N);
                    bench_complex_sum(N);
                }
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