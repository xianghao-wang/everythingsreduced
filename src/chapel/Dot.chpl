module Dot {
    use Time;

    use Shared;

    class Dot {
        var N: int;
        const vecDom = 0..#N;
        var A: [vecDom] real = noinit;
        var B: [vecDom] real = noinit;

        proc init(N: int) {
            this.N = N;
        }

        proc setup() {
            forall i in vecDom {
                A[i] = 1.0 * 1024.0 / N: real;
                B[i] = 2.0 * 1024.0 / N: real;
            }
        }

        proc run(): real {
            var sum: real = 0.0;
            forall (a, b) in zip(A, B) with (+ reduce sum) {
                sum += a * b;
            }
            return sum;
        }

        proc expect(): real {
            const a = 1.0 * 1024.0 / N: real;
            const b = 2.0 * 1024.0 / N: real;
            return a * b * N: real;
        }

        proc gigabytes(): real {
            return 1e-9 * 8.0 * 2.0 * N: real;
        }
    }

    proc bench_dot(N: int) {
        var results: [0..#NITERS] real = noinit;
        var start, tConstruct, tSetup, tRun, tCheck, tTeardown: real;
        
        // Construct
        start = current_seconds();
        const dot = new unmanaged Dot(N);
        tConstruct = current_seconds() - start;

        const amount = dot.gigabytes();

        // Setup
        start = current_seconds();
        dot.setup();
        tSetup = current_seconds() - start;

        // Run
        start = current_seconds();
        for i in results.domain {
            results[i] = dot.run();
        }
        tRun = current_seconds() - start;

        // Check
        start = current_seconds();
        for i in results.domain {
            const res = results[i];
            const exp = dot.expect();
            const diff = abs(exp - res);

            if diff > epsilon * 100.0 {
                print_difference("Dot", "reuslt incorrect", i
                        , exp: string
                        , res: string
                        , diff: string);
                break;
            }
        }
        tCheck = current_seconds() - start;

        // Teardown
        start = current_seconds();
        // delete dot;
        delete dot;
        tTeardown = current_seconds() - start;

        print_timing("Dot Product"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS: real * amount);
    }
}