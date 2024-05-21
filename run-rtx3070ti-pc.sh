for exe in ./bin/*-reduced
do
   ${exe} dot 268435456
   ${exe} complex_sum 268435456
   ${exe} complex_sum_soa 268435456
   ${exe} complex_min 268435456
   ${exe} field_summary
done