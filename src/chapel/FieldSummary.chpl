module FieldSummary {
    use CTypes;
    use GPU;
    
    use Shared;

    record FieldStatus {
        var vol: real;
        var mass: real;
        var ie: real;
        var ke: real;
        var press: real;

        operator +(a: FieldStatus, b: FieldStatus) {
            return new FieldStatus(a.vol + b.vol, a.mass + b.mass, a.ie + b.ie, a.ke + b.ke, a.press + b.press);
        }
    }

    class FieldSummary {
        const nx = 3840;
        const ny = 3840;
        
        const velDomain = {0..#((nx+1)*(ny+1))};
        const fieldDomain = {0..#(nx * ny)};

        var xvel: [velDomain] real;
        var yvel: [velDomain] real;
        var volume: [fieldDomain] real;
        var density: [fieldDomain] real;
        var energy: [fieldDomain] real;
        var pressure: [fieldDomain] real;

        proc setup() {
            const dx = 10.0 / nx: real;
            const dy = 10.0 / ny: real;
            forall k in 0..#ny {
                foreach j in 0..#nx {
                    volume[j + k * nx] = dx * dy;
                    density[j + k * nx] = 0.2;
                    energy[j + k * nx] = 1.0;
                    pressure[j + k * nx] = (1.4 - 1.0) * density[j + k * nx] * energy[j + k * nx];
                }
            }

            forall k in 0..#ny/5 {
                foreach j in 0..#nx/2 {
                    density[j + k * nx] = 1.0;
                    energy[j + k * nx] = 2.5;
                    pressure[j + k * nx] = (1.4 - 1.0) * density[j + k * nx] * energy[j + k * nx];
                }
            }

            forall k in 0..#ny+1 {
                foreach j in 0..#nx+1 {
                    xvel[j + k * (nx + 1)] = 0.0;
                    yvel[j + k * (nx + 1)] = 0.0;
                }
            }
        }

        proc run(): FieldStatus {
            const DOT_NUM_BLOCKS = (ny + TBSIZE - 1) / TBSIZE;
            var blockSumHost: [0..#DOT_NUM_BLOCKS] FieldStatus = noinit;
            
            on gpuLocale {
                const numThreads = TBSIZE * DOT_NUM_BLOCKS;
                var blockSum: [0..#DOT_NUM_BLOCKS] FieldStatus = noinit;

                @assertOnGpu @gpu.blockSize(TBSIZE) foreach i in 0..#numThreads {
                    var tbSum = createSharedArray(real, TBSIZE * 5);
                    const localI = i % TBSIZE;

                    tbSum[5 * localI] = 0.0;        // vol
                    tbSum[5 * localI + 1] = 0.0;    // mass
                    tbSum[5 * localI + 2] = 0.0;    // ie
                    tbSum[5 * localI + 3] = 0.0;    // ke
                    tbSum[5 * localI + 4] = 0.0;    // press

                    // Reduce to threads
                    var y = i;
                    while y < ny {
                        for x in 0..#nx {
                            var vsqrd = 0.0;
                            for yv in y..y+1 {
                                for xv in x..x+1 {
                                    vsqrd += 0.25 * (
                                        xvel[xv + yv * (nx + 1)] * xvel[xv + yv * (nx + 1)] +
                                        yvel[xv + yv * (nx + 1)] * yvel[xv + yv * (nx + 1)]
                                    );
                                }
                            }

                            const cell_volume = volume[x + y * nx];
                            const cell_mass = cell_volume * density[x + y * nx];
                            tbSum[5 * localI] += cell_volume;
                            tbSum[5 * localI + 1] += cell_mass;
                            tbSum[5 * localI + 2] += cell_mass * energy[x + y * nx];
                            tbSum[5 * localI + 3] += cell_mass * 0.5 * vsqrd;
                            tbSum[5 * localI + 4] += cell_volume * pressure[x + y * nx];
                        }

                        y += numThreads;
                    }

                    // reduce threads in a block
                    var offset = TBSIZE / 2;
                    while offset > 0 {
                        syncThreads();
                        if localI < offset {
                            tbSum[5 * localI] += tbSum[5 * (localI + offset)];
                            tbSum[5 * localI + 1] += tbSum[5 * (localI + offset) + 1];
                            tbSum[5 * localI + 2] += tbSum[5 * (localI + offset) + 2];
                            tbSum[5 * localI + 3] += tbSum[5 * (localI + offset) + 3];
                            tbSum[5 * localI + 4] += tbSum[5 * (localI + offset) + 4];
                        }
                        offset /= 2;
                    }

                    if localI == 0 {
                        const blockIdxX = i / TBSIZE;
                        blockSum[blockIdxX].vol = tbSum[0];
                        blockSum[blockIdxX].mass = tbSum[1];
                        blockSum[blockIdxX].ie = tbSum[2];
                        blockSum[blockIdxX].ke = tbSum[3];
                        blockSum[blockIdxX].press = tbSum[4];
                    }
                }

                blockSumHost = blockSum;
            }

            return + reduce blockSumHost;
        }

        proc expect(): FieldStatus {
            return new FieldStatus(
                0.1000E+03, 0.2800E+02, 0.4300E+02, 0.0000E+00,
                0.1720E+00 * 0.1000E+03
            );
        }

        proc gigabytes(): real { return 1.0E-9 * c_sizeof(real) * ((4.0 * nx * ny) + (2.0 * (nx + 1) * (ny + 1))); }
    }

    proc bench_field_summary() {
        var results: [0..#NITERS] FieldStatus = noinit;
        var start, tConstruct, tSetup, tRun, tCheck, tTeardown: real;
        var bench_: unmanaged FieldSummary?;

        // Construct
        start = current_seconds();
        on gpuLocale do bench_ = new unmanaged FieldSummary();
        tConstruct = current_seconds() - start;

        var bench = try! bench_ : unmanaged FieldSummary;
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
            const exp = bench.expect();
            const res = results[i];
            var wrong = false;

            if abs(exp.vol - res.vol) > 1e-8 {
                print_difference("Field Summary", "vol result incorrect", i 
                            , exp.vol: string
                            , res.vol: string
                            , abs(exp.vol - res.vol): string);
                break;
            }

            if abs(exp.mass - res.mass) > 1e-8 {
                print_difference("Field Summary", "mass result incorrect", i
                            , exp.mass: string
                            , res.mass: string
                            , abs(exp.mass - res.mass): string);
                break;
            }

            if abs(exp.ie - res.ie) > 1e-8 {
                print_difference("Field Summary", "ie result incorrect", i
                            , exp.ie: string
                            , res.ie: string
                            , abs(exp.ie - res.ie): string);
                break;
            }

            if abs(exp.ke - res.ke) > 1e-8 {
                print_difference("Field Summary", "ke result incorrect", i
                            , exp.ke: string
                            , res.ke: string
                            , abs(exp.ke - res.ke): string);
                break;
            }

            if abs(exp.press - res.press) > 1e-8 {
                print_difference("Field Summary", "press result incorrect", i
                            , exp.press: string
                            , res.press: string
                            , abs(exp.press - res.press): string);
                break;
            }
        }
        tCheck = current_seconds() - start;

        // Teardown
        start = current_seconds();
        delete bench;
        tTeardown = current_seconds() - start;

        print_timing("Field Summary"
            , tConstruct, tSetup, tRun, tCheck, tTeardown
            , NITERS * amount);
    }
}