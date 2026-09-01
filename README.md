# VASP Performance & Optimization Guide

If you are new to running VASP (on VSC Tier1 clusters), start here. [This guide](/VASP-parallel_guide.md) provides an introduction into the parallelization in VASP with examples and recommendations for your Tier1 scaling tests to help you get the most out of your compute resources.

### Sofia and VASP+OpenMP
As Tier1-Hortense will be phased out, the next iteration of this guide will be geared towards Tier1-Sofia. More focus will be placed on the hybrid VASP MPI+OpenMP version due to the limited memory-bandwidth-per-core in the Sofia compute nodes.

For those of you already writing an application for compute time **on Tier1-Sofia, I *highly* recommend testing and using the OpenMP build of VASP**.

### Feedback & Contributions
If you are an experienced user, I welcome any feedback or additional benchmark examples. Because real-world workflows vary significantly, your specific use case might require a different approach than the ones outlined here. Feel free to open an issue or reach out directly via email.
