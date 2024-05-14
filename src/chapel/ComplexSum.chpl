module ComplexSum {
    use GPU;
    use CTypes;
    
    use Shared;

    class ComplexSum {
        var N: int;
        const vecDom = 0..#N;
        var A: [vecDom] complex = noinit;

        proc init(N: int) {
            this.N = N;
        }

        proc setup() {
            const v = 2.0 * 1024 / N: real;
            const u = (v, v): complex;
            @assertOnGpu forall i in vecDom {
                A[i] = u;
            }
        }

        proc run(): complex {
            var sum = (0.0, 0.0): complex;
            const DOT_NUM_BLOCKS = (N + TBSIZE - 1) / TBSIZE;

            on gpuLocale {
                const zero = (0.0, 0.0): complex;
                var blockSum: [0..#DOT_NUM_BLOCKS] complex = noinit;
                const numThreads = TBSIZE * DOT_NUM_BLOCKS;

                @assertOnGpu @gpu.blockSize(TBSIZE) foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(complex, TBSIZE);
                    const localI = i % TBSIZE;
                    
                    tbSum[localI] = zero;
                    var j = i;
                    while j < N {
                        tbSum[localI] += A[j];
                        j += numThreads;
                    }

                    var offset = TBSIZE / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            tbSum[localI] += tbSum[localI+offset];
                        }
                        offset /= 2;
                    }

                    if localI == 0 {
                        const blockIdxX = i / TBSIZE;
                        blockSum[blockIdxX] = tbSum[localI];
                    }
                }

                sum = + reduce blockSum;
            }

            return sum;
        }

        proc expect(): complex {
            const v = 2.0 * 1024.0;
            return (v, v): complex;
        }

        proc gigabytes(): real {
            return 1e-9 * c_sizeof(complex) * N;
        }
    }

    proc bench_complex_sum(N: int) {
        var results: [0..#NITERS] complex = noinit;
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
            const diff = abs(exp - res);

            if diff > epsilon * 100.0 {
                print_difference("Complex Sum", "reuslt incorrect", i
                        , exp: string
                        , res: string
                        , diff: string);
                break;
            }
        }
        tCheck = current_seconds() - start;
    }
}