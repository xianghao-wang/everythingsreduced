module ComplexMin {
    use GPU;

    use Shared;

    class ComplexMin {
        var N: int;
        const vecDom = 0..#N;
        var A: [vecDom] Complex = noinit;

        const maxComplex = new Complex(max(real), max(real));

        proc init(N: int) {
            this.N = N;
        }

        proc setup() {
             @assertOnGpu forall i in vecDom {
                const v = N / 2.0 - i;
                A[i].re = v;
                A[i].im = v;
             }
        }

        proc run(): Complex {
            const DOT_NUM_BLOCKS = (N + TBSIZE - 1) / TBSIZE;
            var blockSumHost: [0..#DOT_NUM_BLOCKS] Complex = noinit;

            on gpuLocale {
                
                const numThreads = TBSIZE * DOT_NUM_BLOCKS;
                var blockSum: [0..#DOT_NUM_BLOCKS] Complex = noinit;
                
                @assertOnGpu @gpu.blockSize(TBSIZE) foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(real, TBSIZE * 2);
                    const localI = i % TBSIZE;

                    // reduce elements to each thread
                    tbSum[2 * localI] = maxComplex.re;
                    tbSum[2 * localI + 1] = maxComplex.im;
                    var j = i;
                    while j < N {
                        const less = A[j].re * A[j].re + A[j].im * A[j].im 
                            < tbSum[2 * localI] * tbSum[2 * localI] + tbSum[2 * localI + 1] * tbSum[2 * localI + 1];
                        if (less) {
                            tbSum[2 * localI] = A[j].re;
                            tbSum[2 * localI + 1] = A[j].im;
                        }
                        j += numThreads;
                    } 

                    // reduce threads in a block
                    var offset = TBSIZE / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            const less = tbSum[2 * (localI+offset)] * tbSum[2 * (localI+offset)] + tbSum[2 * (localI+offset) + 1] * tbSum[2 * (localI+offset) + 1] 
                                < tbSum[2 * localI] * tbSum[2 * localI] + tbSum[2 * localI + 1] * tbSum[2 * localI + 1];
                            if (less) {
                                tbSum[2 * localI] = tbSum[2 * (localI+offset)];
                                tbSum[2 * localI + 1] = tbSum[2 * (localI+offset) + 1];
                            }
                        }
                        offset /= 2;
                    }

                    if localI == 0 {
                        const blockIdxX = i / TBSIZE;
                        blockSum[blockIdxX].re = tbSum[0];
                        blockSum[blockIdxX].im = tbSum[1];
                    }
                }

                blockSumHost = blockSum;
            }

            var result = maxComplex; 
            for c in blockSumHost {
                const less = c.re * c.re + c.im * c.im < result.re * result.re + result.im * result.im;
                result = if less then c else result;
            }

            return result;
        }

        proc gigabytes(): real {
            return 1e-9 * 2 * 8 * N;
        }

        proc expect(): Complex {
            if N % 2 == 1 {
                return new Complex(0.5, 0.5);
            } else {
                return new Complex(0.0, 0.0);
            }
        }
    }

    proc bench_complex_min(N: int) {
        var results: [0..#NITERS] Complex = noinit;
        var start, tConstruct, tSetup, tRun, tCheck, tTeardown: real;
        var bench_: unmanaged ComplexMin?;

        // Construct
        start = current_seconds();
        on gpuLocale do bench_ = new unmanaged ComplexMin(N);
        tConstruct = current_seconds() - start;

        var bench = try! bench_ : unmanaged ComplexMin;
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
                print_difference("Complex Min", "reuslt incorrect", i
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

        print_timing("Complex Min"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS * amount);
    }
}