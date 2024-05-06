module Dot {
    use Time;
    use GPU;
    use GpuDiagnostics;

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
            // startVerboseGpu();

            var sum: real = 0.0;
            const DOT_NUM_BLOCKS = (N + TBSIZE - 1) / TBSIZE;
            
            on gpuLocale {
                var blockSum: [0..#DOT_NUM_BLOCKS] real = noinit;
                const numThreads = TBSIZE * DOT_NUM_BLOCKS;
                
                @assertOnGpu @gpu.blockSize(TBSIZE) foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(real, TBSIZE);
                    const localI = i % TBSIZE;

                    // reduce elements to each thread
                    tbSum[localI] = 0.0;
                    var j = i;
                    while j < N {
                        tbSum[localI] += A[j] * B[j];
                        j += numThreads;
                    }

                    // reduce threads in a block
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

            // stopVerboseGpu();

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
        var dot_: unmanaged Dot?;

        // Construct
        start = current_seconds();
        on gpuLocale do dot_ = new unmanaged Dot(N);
        tConstruct = current_seconds() - start;
        
        var dot = try! dot_ : unmanaged Dot;
        const amount = dot.gigabytes();

        // Setup
        start = current_seconds();
        on gpuLocale do dot.setup();
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
        delete dot;
        tTeardown = current_seconds() - start;

        print_timing("Dot Product"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS * amount);
    }
}