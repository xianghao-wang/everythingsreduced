module ComplexSumSOA {
    use GPU;
    use Time;
    
    use Shared;

    class ComplexSumSOA {
        var N: int;
        const vecDom = 0..#N;
        var reals: [vecDom] real = noinit;
        var imgs: [vecDom] real = noinit;

        proc init(N: int) {
            this.N = N;
        }

        proc setup() {
            const v = 2.0 * 1024 / N: real;
            @assertOnGpu forall i in vecDom {
                reals[i] = v;
                imgs[i] = v;
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
                    tbSum[localI] = 0.0;
                    tbSum[localI + TBSIZE] = 0.0;
                    var j = i;
                    while j < N {
                        tbSum[localI] += reals[j];
                        tbSum[localI + TBSIZE] += imgs[j];
                        j += numThreads;
                    } 

                    // reduce threads in a block
                    var offset = TBSIZE / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            tbSum[localI] += tbSum[localI + offset];
                            tbSum[localI + TBSIZE] += tbSum[localI + offset + TBSIZE];
                        }
                        offset /= 2;
                    }

                    if localI == 0 {
                        const blockIdxX = i / TBSIZE;
                        blockSum[blockIdxX].re = tbSum[0];
                        blockSum[blockIdxX].im = tbSum[TBSIZE];
                    }
                }

                blockSumHost = blockSum;
            }

            return + reduce blockSumHost;
        }

        proc expect(): Complex {
            const v = 2.0 * 1024.0;
            return new Complex(v, v);
        }

        proc gigabytes(): real {
            return 1e-9 * 2 * 8 * N;
        }
    } 

    proc bench_complex_sum_soa(N: int) {
        var results: [0..#NITERS] Complex = noinit;
        var start, tConstruct, tSetup, tRun, tCheck, tTeardown: real;
        var bench_: unmanaged ComplexSumSOA?;

        // Construct
        start = current_seconds();
        on gpuLocale do bench_ = new unmanaged ComplexSumSOA(N);
        tConstruct = current_seconds() - start;

        var bench = try! bench_ : unmanaged ComplexSumSOA;
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
                print_difference("Complex Sum SoA", "reuslt incorrect", i
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

        print_timing("Complex Sum SoA"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS * amount);
    }
}