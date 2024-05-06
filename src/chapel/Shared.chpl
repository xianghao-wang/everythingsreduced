module Shared {
    use IO;
    use Time;
    
    config param TBSIZE = 1024;
    config param NITERS = 100;

    param LINE = "------------------------------------------------------------"
                 + "--------------------";

    param epsilon: real = 1e-16;

    proc current_seconds(): real {
        return timeSinceEpoch().totalSeconds();
    }

    proc print_difference(const name: string, const detail: string, const resultId: int
            , const expectStr: string
            , const resultStr: string
            , const differenceStr) {
        try! stderr.write(name, ": ", detail, "\n"
                    , "Result: ", resultId, " (skipping rest)\n"
                    , "Expected: ", expectStr, "\n"
                    , "Result: ", resultStr, "\n"
                    , "Difference: ", differenceStr, "\n"
                    , "Eps: ", epsilon, "\n");
    }

    proc print_timing(const name: string, const constructor: real, const setup: real, const run: real, const check: real,
                const teardown: real, const gigabytes: real) {
        writeln("");
        writeln(" ", name);
        writeln("  Constructor: ", constructor * 1e3);
        writeln("  Setup:       ", setup * 1e3);
        writeln("  Run:         ", run * 1e3);
        writeln("  Verify:      ", check * 1e3);
        writeln("  Teardown:    ", teardown * 1e3);
        writeln("");
        writeln("  Sustained GB/s: ", gigabytes / run);
        writeln(LINE);
    }

    
}