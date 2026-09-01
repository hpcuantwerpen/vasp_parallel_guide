## A Guide to Parallelization in VASP

### Parameters & Scaling Tests

This guide is intended to aid researchers in performing (and understanding) their scaling tests for a VSC Tier-1 application. Though, understanding parallelism in VASP is essential for all HPC users working with VASP. The guide is based on the VASP manual, my own experience running it on VSC clusters and discussions with the developers at the 2025 VASP workshop. The examples provided are run on the UAntwerpen Tier-2 Vaughan cluster, whose hardware is similar to Tier-1 Hortense and likewise uses the Slurm scheduler.
VASP makes use of distributed memory parallelization with MPI. As of VASP6 OpenMP (shared memory) is supported. More details are provided later.

#### Parameters
The parameters that control parallelization in VASP are [VASP wiki]: 
- IMAGES
- KPAR 
- LPANE 
- NCORE 
- NPAR 
- NSIM 
- NOMEGAPAR 
- NTAUPAR 
- LUSENCCL 

Out of these, only **KPAR** and **NCORE** are essential. The other parameters are described at the end of this document.

The first (or ‘outer’) level of parallelization that VASP offers is distribution of work over k-points. Different **k-points** can be **calculated in parallel**. KPAR defines the number of k-points that are treated in parallel. Thus, the total number of cores are divided in KPAR groups of `N` cores, with `N = #total_cores/KPAR`.

The next level of parallelization is distribution of the work per k-point over electronic bands. NPAR defines the number of **bands** that are **calculated in parallel**. Closely intertwined with this parameter is NCORE, which are the number of cores that work together on a single band (the plane wave coefficients are distributed over these NCORE cores). In other words, each group of N cores is now further divided into NPAR groups of NCORE cores. Only one of the parameters NCORE and NPAR need to set, because the value of the other is imposed according to:

```#total_cores = KPAR\*NPAR\*NCORE```

Note that there is no memory distribution over KPAR: increasing this parameter will increase the required total memory. When increasing NCORE, the memory requirement per core is lowered, because the non-local projector functions can be distributed over NCORE cores. On the other hand, with an increased NCORE, more communication will be needed as accessing memory of another MPI task requires communication.

*An example is shown in the figure: VASP is launched with 32 cores. These cores are split into 4 groups (KPAR=4) of 8 cores. Each of these groups will work on #kpoints/4 k-points, one at a time. Every k-point group is further divided into 2 groups (NPAR=2). Each of these groups will work on NBANDS/2 bands. We are left with groups of 4 cores (therefore, NCORE=4). Each of these 4 cores will handle a subset of the plane wave coefficients for the given band and k-point.*

// FIGURE
 
Optimal values for these parameters are system-dependent, you therefore need to perform scaling tests and investigate which values lead to efficient use of the available HPC infrastructure. General rules of thumb are outlined here with considerations regarding the topology of the compute nodes.

As KPAR divides the k-points of the calculation in groups, KPAR is optimally chosen as a divisor of the number of k-points. This ensures a **correct load balance**, as all KPAR groups work on an equal number of k-points.
 
 //FIGURE

*Visualization of bad (left, KPAR=3) and good (right, KPAR=4) load balancing with 4 k-points.*

Good KPAR values are not only linked to the number of k-points, but also to the number of cores/nodes. Communication is affected by how the groups are divided over the nodes.

-	`KPAR = N #nodes`: on each node there is an integer number of KPAR groups. Each KPAR-group is fully contained within a node, hence all communication between the NPAR-groups (and NCORE cores) remains within the node, which is faster.
-	`KPAR < #nodes` might be necessary if there are very little k-points or due to memory constraints. In this case, make sure that on each node there is an integer number of NPAR-groups, so that communication within every NPAR-group does not span over nodes. (In other words: NCORE is a divisor of the number of cores per node.)

**Maximizing KPAR is beneficial**. Use it to the maximal extent, even at the cost of using less cores per compute node. This is the most efficient parallelization option, but unfortunately in some cases (low #k-points) or algorithms (methods beyond regular DFT) it cannot be exploited.

Like KPAR, NPAR should be a divisor of the number of bands (NBANDS). However, VASP does this automatically by increasing NBANDS so that it is a multiple of NPAR (NBANDS = the smallest multiple of NPAR higher than the requested (or default) bands). Like KPAR, this ensures a good load balance by having the NPAR groups calculate the same number of bands. Note that it is always best to manually set NBANDS and then choose NPAR, because successive calculations (especially when starting from a WAVECAR) need the same NBANDS. For scaling tests, it is also important to keep NBANDS fixed, so you must choose a value that is compatible with the range of NPAR you will test.

It is **not advised to maximize NPAR**, because then NCORE=1 or you may end up with too little bands per NPAR-group to offset the communication overhead. Instead, use **scaling tests** to figure out the best variation of NPAR and NCORE, as they are system- and computer-dependent. 

> ##### How do I know how many bands and irreducible k-points my system contains?
>
> An easy way to figure out the number of k-points, the default number of bands (which you could also calculate from the number of valence electrons for each atom) and other ‘dimension’ parameters, is to perform a **dry-run** [VASP wiki]:
>
> ```
> $ vasp_std --dry-run
> ```
>
> VASP will then parse your input and let you know potential issues and incompatibilities.
>
> ```
> $ grep -A 12 'Dimension of arrays' OUTCAR
> \> Dimension of arrays:
>    k-points           NKPTS =     20   k-points in BZ     **NKDIM =     20**   number of bands    **NBANDS=    161**
>    number of dos      NEDOS =    301   number of ions     NIONS =     64
>    non local maximal  LDIM  =      5   non local SUM 2l+1 LMDIM =     13
>    total plane-waves  **NPLWV = 175616**
>    max r-space proj   IRMAX =      1   max aug-charges    IRDMAX=   8497
>    dimension x,y,z NGX =    56 NGY =   56 NGZ =   56
>    dimension x,y,z **NGXF=   112 NGYF=  112 NGZF=  112**
>    support grid    NGXF=   112 NGYF=  112 NGZF=  112
>    ions per type =              32  32
>    NGX,Y,Z   is equivalent  to a cutoff of   8.10,  8.10,  8.10 a.u.
>    NGXF,Y,Z  is equivalent  to a cutoff of  16.19, 16.19, 16.19 a.u.
> ```
> 
> If you are using an older VASP version (< 6.4), you can instead use ALGO=None in the INCAR. This ALGO does not update the orbitals and energies, but unlike the dry run it does generate output (DOSCAR, EIGENVAL, vaspout.h5, …).

To optimize the parallelization parameters, you should also keep the architecture of the compute nodes in mind. The bulk of the inter-process communication is due to the parallel FFT, which is between the NCORE cores. In the worst case, a group of NCORE cores is spread over multiple nodes, which deteriorates the performance massively. Furthermore, within a single node, multiple memory domains coexist as data access is generally not uniform in a modern computer: these domains are referred to as NUMA (Non-Uniform Memory Access) nodes. Access to a memory space handled by a NUMA node is faster than accessing the same data from another NUMA node, as the latter involves additional hops across interconnects.

Let’s look at the architecture of the UA tier-2 Vaughan CPUs (both zen2 and zen3 AMD EPYC 7452 & 7543). Each node has 2 sockets (grey) hosting an AMD EPYC processor with 32 cores each, forming the first level of the memory hierarchy. Each socket consists of 4 NUMA nodes (light pink), each with 8 cores (black) and two memory controllers. 

// FIGURE

The distances between the NUMA nodes are displayed as follows, where you can see that NUMA nodes 0-3 and 4-7 are on different sockets, because the distance is much larger.

```
$ numactl --hardware
> node distances:
node   0   1   2   3   4   5   6   7
  0:  10  12  12  12  32  32  32  32
  1:  12  10  12  12  32  32  32  32
  2:  12  12  10  12  32  32  32  32
  3:  12  12  12  10  32  32  32  32
  4:  32  32  32  32  10  12  12  12
  5:  32  32  32  32  12  10  12  12
  6:  32  32  32  32  12  12  10  12
  7:  32  32  32  32  12  12  12  10
```

Looking at the core configuration only, a good NCORE value would be 4 or 8 rather than 16, because in case of the latter, a group of NCORE cores spans over different NUMA domains. You should perform scaling tests to find the optimal values for these parameters, but understanding the hardware of the compute nodes helps you interpret the results of these tests.
 
#### caling tests

Instead of investigating all possible values for the parallelization parameters for all different core counts, which yields a lot of unnecessary calculations, a smarter approach is recommended. **Firstly**, you should **establish a minimal baseline**. Secondly, you should **assess the performance on 1 compute unit**, where you find the appropriate values for KPAR and NCORE. Then you scale up, while keeping the configuration per compute unit similar (i.e. keeping KPAR-per-core fixed and keeping NCORE (and NPAR) fixed).

The recommended approach (if you have **sufficient k-points**) would be:
1.	Start with 1 core with the default parameters `KPAR=NCORE=1`, `NPAR=#cores`. If the run fails (due to out-of-memory errors or the 3-day walltime), double the core count until the calculation succeeds. As soon as `#cores>NUMA_domain`, start increasing NCORE as well. This calculation forms your baseline.
2.	Investigate all possible KPAR and NCORE values using 1 compute unit. The size of this compute unit is up to you to decide: if you are planning to use multiple nodes for your production runs, go for a compute unit equal to a socket or a full node, if on the other hand, you plan to use less than a node for your runs, choose a NUMA domain or a socket as your compute unit[^1]. You want the compute unit to be larger than your baseline and to be able to test an appropriate range in parallelization parameters. 
3.	Vary the number of cores with the best values from step 2. You want NPAR and NCORE to stay fixed, thus KPAR-per-core should remain the same. Remember to scale down if possible (i.e. less than a full compute unit). 
4.	[optional] Step 2 is typically severely limited by the number of k-points and how well KPAR-per-unit divides the number of k-points. If needed, repeat step 2 with a lower KPAR-per-core value. This way you can use a different number of cores in your calculation that was not compatible with the previous KPAR-per-core value. 
(note, it is possible that by doing this, KPAR-per-unit<1. In this case, I would recommend checking the ideal values for NCORE and NPAR again, using KPAR=1, and the larger compute unit)

> **[^1] Footnote:** you are also free to choose a compute unit consisting of an odd number of cores to accommodate an odd/prime number of k-points. This will require proper task-placement and falls outside the scope of the current guide; contact user support if you’d need assistance with this.

ALTERNATIVE: if you have a large system with **very little or only 1 k-point**, you will have to rely on NPAR parallelization.
1.	Step 1 and 2 remains the same. You want to find a baseline and the ideal NCORE value.
3.	You will not be able to increase KPAR with the number of compute units. Use your largest possible KPAR value and use the ideal NCORE value from step 2 as the minimum value for NCORE. This ideal NCORE value from step 2 will generally be a good choice, but as you increase the number of cores, NPAR will do so as well. Eventually there is not sufficient work (NBANDS) for each NPAR group to outperform the communication overhead, hence if you balance the increase of NPAR and NCORE, you can squeeze out maximum efficiency. 

Other essential tips to **make sure your tests are consistent**:
-	Use the timings written in the OUTCAR for LOOP+ (use NELM=15 to limit the number of electronic steps and therefore the time needed). 
-	Always explicitly write NBANDS in the INCAR. Choose a value that is compatible with all NPAR values you’re testing. NBANDS needs to be the same for all your tests.
-	Always explicitly specify the partition you want to use: `--partition=<partition>`
-	Use `--switches=1` so that communication between nodes only passes 1 network switch and that this is the same for all your tests.
-	Use `--nodes=n` rather than `--exclusive` when not using all the cores in a node. The latter will give you more memory per core, possibly making comparison between the calculations unfair.

##### EXAMPLE 1

*An example is shown in the figures. An SCF calculation was performed on a GaAs supercell on the UAntwerpen Tier-2 Vaughan Zen3 partition. In this case, NBANDS = 256 is chosen because it can nicely be divided by different NPAR values. The number of irreducible k-points = 20 because of the symmetry of the system. This is a relatively small system with enough k-points, so the first approach is followed. Possible values for KPAR are a divisor of #k-points and of the number of cores: KPAR = {1,2,4}. Possible values of NCORE are a divisor of #cores/KPAR and NPAR(=#cores/KPAR/NCORE) is a divisor of NBANDS: NCORE = {1,2,4,8,16(,32(,64))}, depending on KPAR.*

*According to the first figure, the ideal values on 1 node (64 cores) are KPAR=4 and NCORE=8. It should not surprise you that NCORE is equal to the number of cores in the NUMA domain.*

// FIGURE

*The next two graphs show the combination of steps 1, 2 and 3: the green datapoint is the baseline on 1 core, the most ideal NCORE and KPAR/node combination yields the orange datapoints. The blue datapoints are always less efficient, as expected from the behavior on 1 node, but they are compatible with node-counts 2 and 10.*
  
// FIGURES
 
*In this case one should opt to use 1 node, with KPAR=4, NCORE=8 for the production runs. This yields an efficiency of 87% compared to the baseline. 2 Nodes, with KPAR=4, NCORE=8 has an efficiency of 77% and although the time-to-solution would be around 40% faster, the energy-to-solution is higher (it would use 13% more resources), which is not worth it in this case, as the total time for a relaxation calculation will be less than an hour anyway.*

##### EXAMPLE 2

*Let’s now consider a  large system NaCl with 512 atoms. Such a large supercell requires only 1 k-point, so the gamma-point only VASP executable is used: vasp_gam, which is faster and requires less memory.  An SCF calculation was performed on the UAntwerpen Tier-2 Vaughan Zen3 partition. In this case, there are 2046 valence electrons, therefore NBANDS = 1536 is chosen. Possible values of NCORE are a divisor of #cores, and NPAR(=#cores /NCORE) is a divisor of NBANDS: NCORE = {1,2,4,8,16,32,64}.*

*The following graphs show the results of the benchmark using all possible NCORE values (starting from 1 full node). For a Tier1 application, these are many unnecessary calculations, but the graphs nicely illustrate how the ideal NCORE value changes with the number of nodes used.*

// FIGURE
  
*In this case, doubling the number of cores, the ideal NCORE value doubles as well. You will not obtain better performance by lowering NCORE, hence the blue and yellow points in the graph do not need to be calculated in step 3, after you excluded them in step 2. You can even remove NCORE values lower than the ideal NCORE for a given core-count in subsequent core-counts.*

*The communication overhead in the case of NCORE=1 becomes so important with increasing NPAR that the computation time increases despite throwing more resources at the problem. (only 2 bands per core using #cores=768, NCORE=1)*

*Note that these examples using regular DFT are relatively small. Therefore, they do not scale very well beyond a few nodes, nor are they limited by memory.*
 
// FIGURE

Other interesting reads: 
[2016 Scaling tests BrENIAC], [Peter Larsson blog], [energy-to-solution]
 
### VASP + OpenMP

An OpenMP build becomes interesting when your parallelization is limited by memory constraints. One of the levels of parallelization is replaced by multi-threading instead of multi-processing (MPI tasks). At this parallelization level, the cores share access to the same memory, reducing the number of copies of data in the memory.

> ##### How do I know if my calculation is memory-bound?
>
> Memory usage estimation in VASP is quite tricky. The complex parallelization and data distribution, as well as optimizations specific to certain algorithms (and frankly lack of documentation), make it near impossible to properly estimate memory usage. Instead, it is better to recognize out-of-memory issues and understand which parameters affect it and can be reduced to remedy your calculation.
>
> ##### Recognize OOM
>
> When going out of memory (OOM), the scheduler will typically kill the job (i.e. killed externally) and throw an OOM error in stderr:
>
> ```
> slurmstepd: error: Detected 2 oom-kill event(s)… Some of your processes may have been killed by the cgroup out‑of‑memory handler.
> srun: task 0: Out Of Memory 
> ```
>
> Alternatively, VASP itself can crash with a segmentation fault: SIGSEGV
> `forrtl: error (78): process killed (SIGTERM)`. Depending how it manifests, it can be found in stdout and/or stderr
>
> Though, it can also happen that VASP hangs; it doesn’t terminate but shows no progression or no hint at what might be going wrong. If you can’t find another reason after carefully reading the OUTCAR and stderr, it’s likely an out-of-memory issue.
>
> ##### Parameters influencing memory usage
>
> The VASP wiki offers a formula for estimating how much memory one needs, however this information is outdated and does not account for parallel usage [VASP wiki]. Nonetheless it shows the most important parameters responsible for memory use. First, storage of wavefunctions is proportional to the number of (irreducible) k-points NKDIM, the number of bands NBANDS and the number of plane waves NPLWV. Secondly, large arrays such as the charge density, local potentials, etc.… are stored, on the (fine) FFT grid. The necessary memory is proportional to the dimensions of this grid: NGXF, NGYF and NGZF. 
>
> Parallelization has a large impact on the memory requirements, especially the KPAR parameter. When you double both KPAR and the number of tasks, you double the required memory, i.e. the mem-per-task remains the same. Doubling KPAR while decreasing NPAR or NCORE increases the mem-per-task. When you double either NPAR or NCORE and the number of tasks, the total memory requirement is still increased, but the mem-per-task is reduced. Increasing NPAR while decreasing NCORE has no impact on the memory needs.
> 
>The following table shows the results of memory use with different parallelization parameters in the GaAs system of example 1. The memory is reported in MiB per task and is obtained as followed:
> -	MEM_est: VASP estimates beforehand, in the OUTCAR, how much rank 0 will use:
>   `total amount of memory used by VASP MPI-rank0\s+(\S+)\. kBytes`
> -	MEM_max: at the end of the calculation, VASP reports how much the rank with the largest memory use (of all the arrays the program keeps track of) used:
>   `Maximum memory used \(kb\):\s+(\S+)\.`
> -	RSS_ave: the resident set size averaged per task according to slurm:
>   `sacct -o AveRSS,MaxRSS -j <jobid>`
> **ntasks = 16 (MiB)**
>
> | KPAR | NPAR | NCORE | MEM_est | MEM_max | RSS_ave |
> |---|---|---|---|---|---|
> | 1 | 1 | 16 | 339 | 588 | 505 |
> | 1 | 2 | 8 | 339 | 584 | 505 |
> | 1 | 8 | 2 | 339 | 585 | 505 |
> | 2 | 1 | 8 | 489 | 735 | 664 |
> | 2 | 2 | 4 | 489 | 732 | 666 |
> | 2 | 8 | 1 | 489 | 731 | 664 |
>
> **ntasks = 32 (MiB)**
>
> | KPAR | NPAR | NCORE | MEM_est | MEM_max | RSS_ave |
> |---|---|---|---|---|---|
> | 1 | 1 | 32 | 265 | 539 | 434 |
> | 1 | 2 | 16 | 265 | 541 | 434 |
> | 1 | 8 | 4 | 265 | 552 | 434 |
> | 2 | 1 | 16 | 339 | 588 | 506 |
> | 2 | 2 | 8 | 339 | 589 | 505 |
> | 2 | 8 | 2 | 339 | 588 | 505 |
>
> **ntasks = 64 (MiB)**
>
> | KPAR | NPAR | NCORE | MEM_est | MEM_max | RSS_ave |
> |---|---|---|---|---|---|
> | 1 | 1 | 64 | 227 | 454 | 334 |
> | 1 | 2 | 32 | 227 | 455 | 333 |
> | 1 | 8 | 8 | 227 | 460 | 333 |
> | 2 | 1 | 32 | 265 | 474 | 367 |
> | 2 | 2 | 16 | 265 | 469 | 367 |
> | 2 | 8 | 4 | 265 | 474 | 366 |


> ##### Memory bandwidth 
>
> On nodes with high core counts (> 64) it is reported that memory bandwidth and cache size can limit parallel efficiency [VASP wiki]. This effect is highly dependent on the node’s architecture, the parallelization parameters and the problem statement. 
>
> -	In the pure MPI version of VASP, do you experience a speed-up if you only use 75% (or even 50%) of the number of cores per node? Beware of task placement. 
>   This indicates that memory bandwidth may be a limitation.
>
> The hybrid MPI/OpenMP version of VASP could alleviate the problems due to limitations of the available memory or memory bandwidth. Contact HPC user support if there is not yet a hybrid VASP build available.

The hybrid MPI/OpenMP version of VASP still parallelizes over the k-points (**KPAR**) and electronic bands (**NPAR**) with MPI, but the work on a single band (‘Bloch orbital’) -which involves the FFTs- is spread over `$OMP_NUM_THREADS` threads instead of NCORE tasks.  Think of OMP_NUM_THREADS as essentially equal to NCORE in functionality. This means that NCORE in the INCAR file is ignored and set to 1 when you start VASP with more than 1 OpenMP thread. Because of the similarity of the parallelization to the pure-MPI version of VASP, the **strategy for the scaling tests explained before remains valid**.

To start VASP with multiple OpenMP threads, you need to slightly adjust your job-script:

```bash
...
#SBATCH --ntasks=n --cpus-per-task=c	# request n*c cores: n tasks with c threads each
...
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK	# set OMP_NUM_THREADS = c
export OMP_PLACES=cores	# bind the threads to (physical) cores
export OMP_PROC_BIND=close	# and bind them as close as possible to the core running the OpenMP master thread
export OMP_STACKSIZE=512m	# VASP needs more (than the default) memory in the private stack of the threads, else you will run into segmentation faults. At CalcUA this environment variable is already set when you load a hybrid VASP module.
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK vasp-executable >> out
```

[OpenMP in VASP: Threading and SIMD] 

### VASP-GPU

VASP offers two ports for GPUs: OpenACC (for Nvidia GPUs) and, more recently, OpenMP (for AMD or Intel GPUs) [VASP wiki]. They are similarly structured regarding the parallelization. The main difference is that the OpenACC version has more features implemented for GPU, but your choice is restricted by the available hardware.

#### PARALLELIZATION

It is important to use exactly one MPI rank (‘task’) per GPU. OpenMP threads can be used to leverage more cores for the CPU-side code, allowing you to allocate and use all CPU cores available per GPU.

Distributed parallel FFTs degrade performance when using VASP-GPU, therefore NCORE=1 will be set automatically. **KPAR, NPAR and NSIM** are the important parallelization parameters when running VASP on GPUs. If possible, choose `KPAR = #GPUs`. NCORE can’t be set, therefore NPAR is fixed by the choice of KPAR and the number of tasks. 

Default NSIM values are good for CPU-only runs, but for GPU runs they must be tuned. In the RMM-DIIS algorithm, NSIM bands are optimized at the same time. This means that you will perform matrix-matrix operations (NSIM>1) instead of matrix-vector operations (NSIM=1). Ideal values for NSIM are higher for GPU runs than they are for CPU.

#### SCALING TESTS

Because GPU nodes are much more expensive than CPU nodes, you will need to prove that your use case properly benefits from GPU resources. This means that you first perform baseline tests with 1 full CPU node.

1.	Investigate all KPAR, NPAR/NCORE combinations on 1 full CPU node to find your actual baseline.
2.	Find the optimal value for NSIM using 1 GPU with KPAR=1=NPAR. 
3.	Scale up the number of GPU’s using the largest KPAR value possible (so either KPAR=#GPUS or lower so the KPAR remains a divisor of both #k-points and #tasks) and the optimal value of NSIM.

 
##### EXAMPLE [Special thanks to Selma Mayda for sharing her benchmark results]

*In this example a molecular dynamics (MD) calculation with on-the-fly machine-learned potentials was performed.*

> The underlying workflow consists of three different computational steps:
>
> 1.	**MLFF prediction & uncertainty estimation**: based on ML forcefields, VASP predicts forces, energies and stress. It also estimates the accuracy of the prediction.
>   This step relies on task-based parallelism.
> 2.	**Standard DFT execution**: when the accuracy of the MLFF prediction is too low, VASP triggers a full DFT calculation. The resulting atomic configurations, forces and energies are written to the training database (ML_AB)
>   This step leverages GPU-offloading (with the known KPAR and NPAR parallelization)
> 3.	**MLFF fitting & retraining**: once sufficient new reference configurations are added to the database, VASP retrains and fits the potential
>   This step relies on shared-memory CPU multi-threading via BLAS/LAPACK
>
> Upon completing either a fast ML prediction or a full DFT step, VASP updates the atomic position and starts the next ionic step.
>
> Because this benchmark starts from scratch, the execution time is heavily dominated by the full DFT steps and potential retraining. In a production training run (ML_MODE=train), the prediction steps will become more prevalent as training goes on. In an actual production run (ML_MODE=run), which often simulates much larger supercells than those used during training, there will only be the prediction step, hence production runs require a separate benchmark investigation.


The system is a large CdS supercell with 432 atoms, and 1 k-point. The MD calculation was performed on the UAntwerpen Tier-2 Vaughan cluster (using both zen2 and ampere_gpu partitions).

With 3888 valence electrons, the total number of bands was set to NBANDS=2048. Therefore, any NPAR or NCORE setting that divides the number of tasks (64 per CPU node) is nicely compatible with NBANDS. KPAR=1 because of the lack of k-points.

The benchmark lasts 500 ionic steps, to obtain sufficient statistics, and is measured by ‘elapsed time’ as reported at the end of the OUTCAR file. 
 
1.	Baseline on a **full CPU node**
As there is only 1 k-point, only NCORE is varied.

| NCORE | NPAR | Elapsed time (s) | efficiency |
|---|---|---|---|
| 1 | 64 | 87660 | 0.90 |
| 2 | 32 | 85956 | 0.91 |
| 4 | 16 | 97526 | 0.81 |
| 8 | 8 | 101279 | 0.78 |
**| 16 | 4 | 78573 | 1 |**
| 32 | 2 | 84063 | 0.93 |
| 64 | 1 | 97334 | 0.81 |

2.	**1 GPU**: optimizing NSIM 
using 1 task with 16 threads (as there are 16 cores available per GPU in the ampere_gpu partition)

| NSIM | Elapsed time (s) |
|---|---|
| 1 | 46801 |
| 2 | 29430 |
| 4 | 27222 |
| 8 | 30432 |
**| 16 | 26234 |**
| 32 | CUDA OOM |
| 64 | CUDA OOM |

NSIM is found to be 16. It is not possible to increase NSIM even more on this hardware. Using only 1 GPU is 300% faster than the CPU baseline.

3.	**Scale #GPUs**
using #tasks = NPAR = #GPUs with 16 threads per task.

| #GPUs | Elapsed_time (s) | efficiency |
|---|---|---|
| 1 | 26526 | 1 |
| 2 | 14086 | 0.94 |
| 4 | 7805 | 0.85 |

A full GPU node is 10x faster than the full CPU node. It clearly demonstrates the advantage of GPU offloading over the CPU-only runs.

Despite only relying on NPAR parallelization, efficiency remains high for the full GPU node. This efficiency results allow the user the choice to use either 2 or 4 GPUs per job, depending on the size of the training job (number of ionic steps) and the number of such training jobs (different systems under investigation).
 
### —OTHER PARAMETERS—

##### IMAGES

IMAGES is used for VASP calculations in different subdirectories, e.g. in a nudged elastic band calculation. You do not need to run scaling tests with this parameter. The slowest converging calculation determines the calculation time.

##### LPLANE

LPLANE defines how data is distributed within the NCORE groups. With LPLANE=True, each core handles the FFT on 2d xy-planes (aka ‘slab decomposition’). With LPLANE=False, each core performs an FFT on a 3d section of the full grid (aka ‘pencil decomposition’). The latter will require more communication but implies a better load balance. 

The default value is LPLANE=True. Though the VASP manual only recommends this when `NGZ >= 3*NCORE`. (NGZ is the grid points along the z-axis, so this increases with the size of the z lattice parameter). To ensure a good load balance, NGZ should be a multiple of NPAR.

If your calculation does not satisfy the above requirements, see if LPLANE=False improves efficiency. Otherwise, you don’t need to investigate this parameter, as it only influences the communication pattern.

##### NSIM

NSIM is used in the most common optimization algorithms. (**ALGO=Normal**, “Blocked-Davidson” or **ALGO=Fast**, “RMM-DIIS”). This sets the number of bands that are optimized simultaneously (within the core group with NCORE cores). In the RMM-DIIS algorithm, it allows the use of **matrix-matrix** instead of matrix-vector **operations** in this optimization step. 

No need to optimize this parameter (unless you use VASP-GPU), but you can experiment with it if you use the mentioned optimization algorithms.

##### NOMEGAPAR, NTAUPAR

NOMEGAPAR and NTAUPAR are used in GW and RPA/ACFDT calculations. They define the distribution of imaginary grid-points. NTAUPAR is deduced from MAXMEM, which should automatically be set correctly. You can optimize NOMEGAPAR, though default values are usually reasonable.

##### LUSENCCL

LUSENCCL (de)activates NCCL (Nvidia collective communications library). If False, VASP will fall back to MPI-based communication for inter-GPU data exchange.
