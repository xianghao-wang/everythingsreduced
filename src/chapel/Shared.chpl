module Shared {
    config param useGPU: bool = false;
    config param TBSIZE: int = 1024;
    config param NITERS: int = 100;

    param epsilon: real = 1e-16;
    param LINE = "------------------------------------------------------------"
                 + "--------------------";

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