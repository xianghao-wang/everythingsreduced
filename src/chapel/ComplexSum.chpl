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

            if useGPU {
                const DOT_NUM_BLOCKS = min((N + TBSIZE - 1) / TBSIZE, 256);
                const numThreads = TBSIZE * DOT_NUM_BLOCKS;
                var blockSum: [0..#DOT_NUM_BLOCKS] Complex = noinit;
                
                @assertOnGpu @gpu.blockSize(TBSIZE) foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(real, TBSIZE * 2);
                    const localI = i % TBSIZE;

                    // reduce elements to each thread
                    tbSum[2 * localI] = 0.0;
                    tbSum[2 * localI + 1] = 0.0;
                    var j = i;
                    while j < N {
                        tbSum[2 * localI] += A[j].re;
                        tbSum[2 * localI + 1] += A[j].im;
                        j += numThreads;
                    } 

                    // reduce threads in a block
                    var offset = TBSIZE / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            tbSum[2 * localI] += tbSum[2 * (localI+offset)];
                            tbSum[2 * localI + 1] += tbSum[2 * (localI+offset) + 1];
                        }
                        offset /= 2;
                    }

                    if localI == 0 {
                        const blockIdxX = i / TBSIZE;
                        blockSum[blockIdxX].re = tbSum[0];
                        blockSum[blockIdxX].im = tbSum[1];
                    }
                }

                on hostLocale {
                    var blockSumHost: [0..#DOT_NUM_BLOCKS] Complex;
                    blockSumHost = blockSum;
                    sum = + reduce blockSumHost;
                }
            } else {
                
            }

            // return + reduce blockSumHost;
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

        // Construct
        start = current_seconds();
        var bench = new unmanaged ComplexSum(N);
        tConstruct = current_seconds() - start;

        const amount = bench.gigabytes();

        // Setup
        start = current_seconds();
        bench.setup();
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