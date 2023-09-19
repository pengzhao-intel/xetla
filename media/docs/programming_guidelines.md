[README](/README.md#documentation) > **Programming Guidelines**

![ALT](/media/docs/workflow.png "Step by step GEMM decomposition")

# Basic Concepts

The central idea behind Intel® XeTLA revolves around the concept of `building blocks`, which used to create larger and more complex kernels. These building blocks consist of highly performant device code that harnesses advanced GPU instructions such as 2D block load/store and DPAS. Furthermore, it means the most intricacies of computation and data is offloaded into these essential building blocks. XeTLA empowers developers to concentrate exclusively on their algorithm design, encompassing task allocation, fusion, and memory hierarchy utilization. 

There are there groups of APIs for user, each serving with different purposes. 
- [kernel-level APIs](https://github.com/pengzhao-intel/xetla/tree/main/include/kernel) is designed for the easiest user experience by combining various `group-level APIs`. For instance, `gemm_universal` is specifically tailored for GEMM (General Matrix Multiply), where users only need to set the input shapes of A, B, C, a few basic parameters and post functions,  without delving into the intricacies of computation. Of course, developers have the option to customize their own GEMM implementations using the `group-level APIs`, potentially achieving better performance for their specific input shapes.
- [group-level APIs](https://github.com/pengzhao-intel/xetla/tree/main/include/group) serves as the primary component for building your own kernels. These group functions are mapped to `workgroup` and executed in the Dual Subslice (DSS) on the GPU. Therefore, it's crucial to understand how to divide the workload into smaller pieces and allocate it to the workgroups. One major performance concern is having too few workgroups to fully utilize all available DSS resources on the GPU.
- [subgroup](https://github.com/pengzhao-intel/xetla/tree/main/include/subgroup) represents the next lower level of group APIs. In most cases, creating high-performance kernels can be achieved using the `group-level APIs`. However, for developers who seek finer control over algorithm details, such as when to perform data prefetch or manage data reuse within a workgroup, the `subgroup-level APIs` offers the utmost flexibility.

| API level | Example                                  |
| :-------- | :----------------------------------------|
| kernel    | `gpu::xetla::kernel::gemm_universal`     |
| group     | `gpu::xetla::group::gemm`                |
| subgroup  | `gpu::xetla::subgroup::tile_prefetch`    |  


## Kernel-level API 
The `kernel-level API` operates at the GPU-wide scale, where both input and output rely on global memory. Local shared memory and synchronization are handled internally within workgroups, transparent to the developer. Consequently, developers remain unaware of these low-level details.

For instance, consider the `gemm_universal` function. When using this API, developers are required to make choices regarding dispatch policies, select the appropriate GEMM building block, and specify any post-processing operators. The API example is outlined below:

```c++
using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;
```
And then this GEMM can be executed inside `parallel_for`.
```c++
auto gpu_event = queue.submit([&](handler &cgh) {
    // GPU kernel
    cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
        xetla_exec_item<3> ei(item);
        // allocate slm and nbarrier resource
        slm_barrier_init<gemm_op_t>();
        gemm_op_t gemm_op;
        gemm_op(ei, gemm_arg);
    });
});
```
For a runnable code example, you can refer to the code in the [01_basic_gemm](/examples/01_basic_gemm), which also includes explanations of the idea behind the implementation.

## Group-level API 
The use of a `group-level API` in parallel computing provides several notable advantages. Firstly, it offers developers greater flexibility in constructing custom kernels tailored to their specific needs. This flexibility extends to workload distribution across GPU workgroups. In this context, the allocation of workgroups is based on the output matrix C, with each workgroup handling a distinct sub-matrix sized `wg_tile_m` * `wg_tile_n`. Within each workgroup, intricate computations related to the `K` dimension are encapsulated within the GEMM building block, sparing developers from delving into these details at the group level

![ALT](/media/docs/code_map.jpg "Code Example to show workload mapping")

Moreover, a key benefit of the group-level API is the empowerment it grants developers over accumulator variables (`matAcc` in below example). This control enables developers to implement more sophisticated and innovative operations, seamlessly fused with the GEMM computation. This level of customization proves invaluable when striving for optimized performance tailored to specific computational tasks, such as example in [02_basic_gemm](/examples/02_basic_gemm)

```c++
gemm_t::matAcc_t matAcc;
matAcc.init(0);

gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

// the results is in the matAcc rather than real output C
gemm_t::work_group_t g(ei.get_local_linear_id());
gemm(g, matAcc, gemm_args);

// any customized operation here based on matACC

// write the results from matACC to real output C
epilogue_t epilogue;
epilogue(g, matAcc, md_c);
```

### Subgroup-level API
The micro-kernel is a crucial component of GEMM, and correctly setting it is essential to its implementation. 
To help developers customize their micro-kernels, the `brgemm_select_t` class provides a simple interface as below.
In this template, the memory layout, computation engine and work-group/sub-gourp shape will be provided and the developer can
decide the location of input and output matrix which is either from global or shared local memory.

```c++
template <typename dtype_a,
          typename dtype_b,
          mem_layout mem_layout_a,
          mem_layout mem_layout_b,
          mem_space mem_space_a,
          mem_space mem_space_b,
          int alignment_a,
          int alignment_b,
          typename dtype_acc,
          typename tile_shape,
          int k_stride,
          mma_engine engine,
          gpu_arch arch>
class brgemm_selector_t {};
```

- `dtype_a` and `dtype_b` are the memory data type of matrix A and B
- `mem_layout_a` and `mem_layout_b` are the memory layout of matrix A and B, can be either `mem_layout::row_major` or `mem_layout::col_major`.
- `mem_space_a` and `mem_space_b` are the memory space of matrix A and B, can be either `mem_space::global` or `mem_layout::local`.
- `alignment_a` and `alignment_b` are the memory alignment of matrix A and B, in unit of element count.
- `dtype_acc` is the accumulate data type of mma compute.
- `tile_shape` is the problem size of each group and subgroup.
- `k_stride` is the size of how many elements will be compuated in the inner loop.
- `engine` is the computing engine: xmx, fpu..
- `arch` is the intel hardware architecture: Xe, Xe2...

### Define Epilogue

The fusion of post-operations, such as `bias add`, `relu`, `gelu`,  after GEMM computation can significantly reduce unnecessary memory transitions and greatly improve performance. In Intel® XeTLA, the `epilogue` is specifically designed to seamlessly integrate post-operations into the GEMM computation at the register level. Beside the fusion, the `epilogue` is also used to update the buffer `c` or data conversion and fusing with some post-processing ops, such as `bias add`, `relu`, `gelu`,.etc.

```c++
template <typename epilogue_policy,
          typename tile_shape,
          typename mem_desc_c>
class epilogue_t {};
```

- `epilogue_policy` tells the epilogue behavior, as well as the related configurations, such as `tile_op_t`, `update_method`, ...
  - `tile_op_t` is the post-processing ops that can be fused together with `brgemm`. When there are multiple post-processing ops, Intel® XeTLA provides `chained_tile_op_t<tile_op_0, tile_op_1, ...>` to fuse all the tile ops first, then feed into `epilogue_t`.
  - `update_method` is the method to update buffer `c`, can be either `result_overwrite` or `result_reduce_sum`.
- `tile_shape` is the problem size of each group and subgroup.
- `mem_desc_c` is the description of buffer `c`, which includes `memory data type`, `memory space` and `memory layout`...

In example [03_gemm_fusion](/examples/03_gemm_fusion), a chain of operations is effectively fused into the GEMM computation. 
First, using pre-defined post-operations `bias_add` and `relu`, and then pass it to `epilogue_policy::tile_op_t`.

```c++
using tile_op_t = chained_tile_op_t<
                  relu_op_t, // apply elementwise ReLU
                  bias_op_t // apply elementwise BiasAdd
                  >;
```

### Construct GEMM 

After configuration of BRGEMM and epilogue, it's simple to build entire GEMM with:
- assigning tasks to each group, setting working boundaries and starting position accordingly.
- ordering the execution of work-group within the kernel
- performing any synchronization in between that may be necessary
- performing any necessary group remapping logic to maximize data locality

As below interface, GEMM is constructd by `dispatch_policy`, `brgemm` and `epilogue`.

```c++
template <typename dispatch_policy,
          typename brgemm_t,
          typename epilogue_t>
class gemm_t {};

using gemm_op_t = gpu::xetla::kernel::gemm_t<
                  gpu::xetla::kernel::dispatch_policy_default<gpu_arch::Xe>, brgemm_t,
                  epilogue_t>;
```

- `dispatch_policy` is the kernel launch attribute, which includes the hardware architecture tag, group remapping information, and special parameters for task splitting, e.g., `l3_kslicing` can be used to split the group-level problem along the `K` dimension in order to get higher occupancy.
- `brgemm_t` is the brgemm operation as describe above.
- `epilogue_t` is the epilogue operation as describe above.

Finally, the actual data will be passed using gemm_op_t::arguments_t, and all of these configurations will be instantiated during the compilation stage for the actual kernel.

```c++
typename gemm_op_t::arguments_t arg(matrix_n, matrix_k,
                     matrix_m, A, matrix_k, B, matrix_n, C, matrix_n);
```
```c++ 
gemm_op_t gemm_op;
xetla_exec_item<3> ei(item);
gemm_op(ei, arg);
```






## The Key Things for Better Performance
Intel® XeTLA provides the basic building block of GEMM unit; however, it still needs to implement the kernel carefully for the better perforamnce in both algorithm and hardware level.
1. Number of work-group / sub-group
2. K slicing algorithm
3. Reuse register for post operations
4. Data sharing through shared local memory
5. Reduction

## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
