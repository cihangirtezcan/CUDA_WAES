# CUDA_WAES

This CUDA code measures how many seconds it takes for a GPU to encrypt 2^n 256-bit Wider AES blocks. Default value of n is 40 and can be modified from the following line:

#define TWO_POWER_RANGE		40

These codes are part of our work "First Fully Pipelined High Throughput FPGA Implementation and GPU Optimization of Wider Variant of AES" submitted to Microprocessors and Microsystems by Ahmet Malal and Cihangir Tezcan.
