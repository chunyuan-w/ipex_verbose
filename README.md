# ipex_verbose
## Reorder analysis

If you find that `reorder` takes a large amount of time and want to find which OPs lead to these `reorder`s, you could:

1. install pandas.

    ```
    pip install -r requirements.txt
    ```

1. set the `_DEBUG` flag in `DevOPs.cpp` by 
    ```
    #define _DEBUG
    ```

1. dump the verbose output into a file for example named `output.log` by enabling `MKLDNN_VERBOSE=1`.

    Example of the verbose output:
    ```
    AtenIpexCPUDev::matmul_common
    dnnl_verbose,exec,cpu,matmul,gemm:jit,undef,src_bf16::blocked:abc:f0 wei_bf16::blocked:bac:f0 dst_bf16::blocked:abc:f0,,,b1408m52n64k52,0.417969
    AtenIpexCPUDefault::transpose
    AtenIpexCPUDev::dil_transpose
    AtenIpexCPUDefault::empty
    AtenIpexCPUDefault::copy_
    dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_bf16::blocked:abc:f0 dst_f32::blocked:abc:f0,,,1408x52x64,3.99487
    AtenIpexCPUDefault::view
    AtenIpexCPUDev::dil_view
    AtenIpexCPUDev::dil_linear
    dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abc:f0 dst_bf16::blocked:abc:f0,,,52x88x1024,0.13916
    AtenIpexCPUDev::dil_size
    AtenIpexCPUDev::dil_reshape
    dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_bf16::blocked:ab:f0,,,1024x1024,0.0690918
    dnnl_verbose,exec,cpu,inner_product,gemm:blas,forward_training,src_bf16::blocked:ab:f0 wei_bf16::blocked:ab:f0 bia_f32::blocked:a:f0 dst_bf16::blocked:ab:f0,,,mb4576ic1024oc1024,1.59692
    AtenIpexCPUDev::dil_size
    AtenIpexCPUDev::dil_view
    AtenIpexCPUDefault::view
    ```

1. use `reorder.py` to find the OPs which lead to the reorder and get a summary of the time spent per OP and per shape.

    ```
    python reorder.py -f path:/to/the/verbose.log -g name -t 10 -e src_f32 -l 20 -p AtenIpexCPUDefault AtenIpexCPUDev -o directory:to/save/output/
    ```

    * `-f` -path to the verbose file
    * `-g` -column names to groupby to calculate the total time
    * `-t` -only show the top `k` result within each group
    * `-e` -format starts with the given strings will be excluded
    * `-l` -only print the first `l` lines in the table
    * `-o` -directory to save the output file
    * `-p` -op to include

    ```
    **************************************************
    excluded format that starts with: src_f32
    only show the top 10 result in each group
    **************************************************
    ***** Only print first 20 lines in table *****
                                                        total_time
    name                                                            
    AtenIpexCPUDefault::copy_                           11639.369703
    AtenIpexCPUDefault::mul                              2043.746809
    AtenIpexCPUDefault::native_layer_norm                1684.674033
    AtenIpexCPUDefault::expand                            650.644742
    AtenIpexCPUDefault::mul_                              361.822256
    AtenIpexCPUDefault::sum                               339.461665
    AtenIpexCPUDefault::add_                              280.614487
    AtenIpexCPUDefault::native_layer_norm_backward        123.052239
    AtenIpexCPUDefault::norAtenIpexCPUDefault::nati...      0.800782
    AtenIpexCPUDefault:AtenIpexCPUDefault::native_l...      0.513184
    AtenIpexCPUDefault:AtenIpexCPUDefault::add_             0.309082

    ***** Only print first 20 lines in table *****
                                                time    total_time
    name                      shape                               
    AtenIpexCPUDefault::copy_ 5016x42720  575.986200  11639.369703
                              5120x42720  562.807000  11639.369703
                              5040x42720  521.028200  11639.369703
                              4992x42720  294.528500  11639.369703
                              4896x42720  289.565400  11639.369703
                              4752x42720  281.221200  11639.369703
                              5096x42720  225.600800  11639.369703
                              5032x42720  222.723300  11639.369703
                              4928x42720  218.197400  11639.369703
                              5120x1024   171.200931  11639.369703
    AtenIpexCPUDefault::mul   5120x1024    49.610103   2043.746809
                              33x2432x64   48.380130   2043.746809
                              5040x1024    44.238524   2043.746809
                              21x3840x64   42.923827   2043.746809
                              37x2176x64   42.661621   2043.746809
                              46x1664x64   39.604003   2043.746809
                              40x2048x64   39.142090   2043.746809
                              4992x1024    38.011959   2043.746809
                              42x1920x64   36.992190   2043.746809
                              54x1408x64   35.770019   2043.746809
    ```
    The complete result will be saved at:
    `directory:to/save/output/op_level.csv` and `directory:to/save/output/op_shape_level.csv`
    if `-o` option is set.
