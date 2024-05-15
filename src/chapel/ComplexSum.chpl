module ComplexSum {
    use GPU;
    use CTypes;
    
    use Shared;

    class ComplexSum {
        var N: int;
        const vecDom = 0..#N;
        var A: [vecDom] Complex = noinit;

        proc init(N: int) {
            this.N = N;
        }

        proc setup() {
            const v = 2.0 * 1024 / N: real;
            @assertOnGpu forall i in vecDom {
                A[i].re = v;
                A[i].im = v;
            }
        }

        proc run(): Complex {
            var sum = new Complex(0.0, 0.0);

            return sum;
        }

        proc expect(): Complex {
            const v = 2.0 * 1024.0;
            return new Complex(v, v);
        }

        proc gigabytes(): real {
            return 1e-9 * 2 * 8 * N;
        }
    }

    proc bench_complex_sum(N: int) {
        var results: [0..#NITERS] Complex = noinit;
        var start, tConstruct, tSetup, tRun, tCheck, tTeardown: real;
        var bench_: unmanaged ComplexSum?;

        // Construct
        start = current_seconds();
        on gpuLocale do bench_ = new unmanaged ComplexSum(N);
        tConstruct = current_seconds() - start;

        var bench = try! bench_ : unmanaged ComplexSum;
        const amount = bench.gigabytes();

        // Setup
        start = current_seconds();
        on gpuLocale do bench.setup();
        tSetup = current_seconds() - start;

        // Run
        start = current_seconds();
        for i in results.domain {
            results[i] = bench.run();
        }
        tRun = current_seconds() - start;

        // Check
        start = current_seconds();
        for i in results.domain {
            const res = results[i];
            const exp = bench.expect();
            const diff = res.distance(exp);
            if diff > epsilon * 100.0 {
                print_difference("Complex Sum", "reuslt incorrect", i
                        , exp.str()
                        , res.str()
                        , diff: string);
                break;
            }
        }
        tCheck = current_seconds() - start;

        // Teardown
        start = current_seconds();
        delete bench;
        tTeardown = current_seconds() - start;

        print_timing("Complex Sum"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS * amount);
    }
}