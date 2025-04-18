#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// MODIFICATION Preprocessor for x86 vs ARM processors
#if defined(__x86_64__) || defined(_M_X64)
    #include <emmintrin.h>  // SSE2 for x86/x86_64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>   // NEON for ARM64
#endif

#define NUM_THREADS 20

void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    
    // hawkZip parallel compression begin.
    #pragma omp parallel
    {

        // Quantizing Regular Block
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int block_num = (chunk_size+31)/32;
        int start_block = thread_id * block_num;
        int block_start, block_end;
        const float recip_precision = 0.5f/errorBound;
        int sign_ofs;
        unsigned int thread_ofs = 0; 

        // Iterate all blocks in current thread.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            float data_recip;
            int s;
            int curr_quant, max_quant=0;
            int curr_block = start_block + i;
            unsigned int sign_flag = 0;
            int temp_fixed_rate;
            
            // Prequantization, get absolute value for each data.
            for(int j=block_start; j<block_end; j++)
            {
                // Prequantization.
                data_recip = oriData[j] * recip_precision;
                s = data_recip >= -0.5f ? 0 : 1;
                curr_quant = (int)(data_recip + 0.5f) - s;
                // Get sign data.
                sign_ofs = j % 32;
                sign_flag |= (curr_quant < 0) << (31 - sign_ofs);
                // Get absolute quantization code.
                max_quant = max_quant > abs(curr_quant) ? max_quant : abs(curr_quant);
                absQuant[j] = abs(curr_quant);
            }

            // Record fixed-length encoding rate for each block.
            signFlag[curr_block] = sign_flag;
            temp_fixed_rate = max_quant==0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);
            fixedRate[curr_block] = temp_fixed_rate;
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;

            // Inner thread prefix-sum.
            thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        // Fixed-length encoding and store data to compressed data.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = signFlag[curr_block];

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag;

                // Retrieve quant data for one block.
                unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                int mask = 1;
                for(int j=0; j<temp_fixed_rate; j++)
                {
                    // Initialization.
                    tmp_char0 = 0;
                    tmp_char1 = 0;
                    tmp_char2 = 0;
                    tmp_char3 = 0;

                    
                    // Get ith bit in 0~7 quant, and store to tmp_char0.
                    /*
                    for(int k=block_start; k<block_start+8; k++)
                        tmp_char0 |= (((absQuant[k] & mask) >> j) << (7+block_start-k));
                    // Get ith bit in 8~15 quant, and store to tmp_char1.
                    for(int k=block_start+8; k<block_start+16; k++)
                        tmp_char1 |= (((absQuant[k] & mask) >> j) << (15+block_start-k));
                    // Get ith bit in 16~23 quant, and store to tmp_char2.
                    for(int k=block_start+16; k<block_start+24; k++)
                        tmp_char2 |= (((absQuant[k] & mask) >> j) << (23+block_start-k));
                    // Get ith bit in 24~31 quant, and store to tmp_char3.
                    for(int k=block_start+24; k<block_end; k++)
                        tmp_char3 |= (((absQuant[k] & mask) >> j) << (31+block_start-k));
                    */

                    for(int k=block_start; k<block_start+8; k++){
                        tmp_char0 |= (((absQuant[k] & mask) >> j) << (7+block_start-k));
                        tmp_char1 |= (((absQuant[k+8] & mask) >> j) << (7+block_start-k));
                        tmp_char2 |= (((absQuant[k+16] & mask) >> j) << (7+block_start-k));
                        tmp_char3 |= (((absQuant[k+24] & mask) >> j) << (7+block_start-k));
                    }
                    
                    // Store data to compressed data array.
                    cmpData[cmp_byte_ofs] = tmp_char0;
                    cmpData[cmp_byte_ofs+1] = tmp_char1;
                    cmpData[cmp_byte_ofs+2] = tmp_char2;
                    cmpData[cmp_byte_ofs+3] = tmp_char3;
                    cmp_byte_ofs += 4;
                    mask <<= 1;
                }
            }
        }
        
        // Return the compression data length.
        if(thread_id == NUM_THREADS - 1)
        {
            unsigned int cmpBlockInBytes = 0;
            for(int i=0; i<=thread_id; i++) {
                cmpBlockInBytes += threadOfs[i];
            }
            *cmpSize = (size_t)(cmpBlockInBytes + block_num * NUM_THREADS);
        }
    }
}

void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
    // Shared variables across threads.
    int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    
    // hawkZip parallel decompression begin.
    #pragma omp parallel
    {
        // Divide data chunk for each thread
        int thread_id = omp_get_thread_num();
        int start = thread_id * chunk_size;
        int end = start + chunk_size;
        if(end > nbEle) end = nbEle;
        int block_num = (chunk_size+31)/32;
        int block_start, block_end;
        int start_block = thread_id * block_num;
        unsigned int thread_ofs = 0;


        // Iterate all blocks in current thread.
        // Unwrap by 5, block size is 33750
        //printf("%i, %i\n", start_block, block_num);
        for(int i=start_block; i<block_num + start_block; i += 5)
        {
            // Retrieve fixed-rate for each block in the compressed data.
            int temp_fixed_rate0 = (int)cmpData[i];
            int temp_fixed_rate1 = (int)cmpData[i+1];
            int temp_fixed_rate2 = (int)cmpData[i+2];
            int temp_fixed_rate3 = (int)cmpData[i+3];
            int temp_fixed_rate4 = (int)cmpData[i+4];

            fixedRate[i]   = temp_fixed_rate0;
            fixedRate[i+1] = temp_fixed_rate1;
            fixedRate[i+2] = temp_fixed_rate2;
            fixedRate[i+3] = temp_fixed_rate3;
            fixedRate[i+4] = temp_fixed_rate4;

            int sum0 = temp_fixed_rate0 ? (32+temp_fixed_rate0*32)/8 : 0;
            int sum1 = temp_fixed_rate1 ? (32+temp_fixed_rate1*32)/8 : 0;
            int sum2 = temp_fixed_rate2 ? (32+temp_fixed_rate2*32)/8 : 0;
            int sum3 = temp_fixed_rate3 ? (32+temp_fixed_rate3*32)/8 : 0;
            int sum4 = temp_fixed_rate4 ? (32+temp_fixed_rate4*32)/8 : 0;

            thread_ofs += sum0 + sum1 + sum2 + sum3 + sum4;
            // Inner thread prefix-sum.
            /*
            thread_ofs += temp_fixed_rate0 ? (32+temp_fixed_rate0*32)/8 : 0;
            thread_ofs += temp_fixed_rate1 ? (32+temp_fixed_rate1*32)/8 : 0;
            thread_ofs += temp_fixed_rate2 ? (32+temp_fixed_rate2*32)/8 : 0;
            thread_ofs += temp_fixed_rate3 ? (32+temp_fixed_rate3*32)/8 : 0;
            thread_ofs += temp_fixed_rate4 ? (32+temp_fixed_rate4*32)/8 : 0;
            */
        }

        // Store thread ofs to global varaible, used for later global prefix-sum.
        threadOfs[thread_id] = thread_ofs;
        #pragma omp barrier

        // Exclusive prefix-sum.
        unsigned int global_ofs = 0;
        for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
        unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

        // Restore decompressed data.
        for(int i=0; i<block_num; i++)
        {
            // Block initialization.
            block_start = start + i * 32;
            block_end = (block_start+32) > end ? end : block_start+32;
            int curr_block = start_block + i;
            int temp_fixed_rate = fixedRate[curr_block];
            unsigned int sign_flag = 0;
            int sign_ofs;

            // Operation for each block, if zero block then do nothing.
            if(temp_fixed_rate)
            {
                // Retrieve sign information for one block.
                sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                            (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                            (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                            (0x000000ff & cmpData[cmp_byte_ofs++]);

                // Retrieve quant data for one block.
                unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
                for(int j=0; j<temp_fixed_rate; j++)
                {
                    // Initialization.
                    tmp_char0 = cmpData[cmp_byte_ofs];
                    tmp_char1 = cmpData[cmp_byte_ofs+1];
                    tmp_char2 = cmpData[cmp_byte_ofs+2];
                    tmp_char3 = cmpData[cmp_byte_ofs+3];
                    cmp_byte_ofs += 4;

                    /*
                    // Get ith bit in 0~7 abs quant from global memory.
                    for(int k=block_start; k<block_start+8; k++)
                        absQuant[k] |= ((tmp_char0 >> (7+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 8~15 abs quant from global memory.
                    for(int k=block_start+8; k<block_start+16; k++)
                        absQuant[k] |= ((tmp_char1 >> (15+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 16-23 abs quant from global memory.
                    for(int k=block_start+16; k<block_start+24; k++)
                        absQuant[k] |= ((tmp_char2 >> (23+block_start-k)) & 0x00000001) << j;

                    // Get ith bit in 24-31 abs quant from global memory.
                    for(int k=block_start+24; k<block_end; k++)
                        absQuant[k] |= ((tmp_char3 >> (31+block_start-k)) & 0x00000001) << j;
                    */
                    
                    for(int k=block_start; k<block_start+8; k++){
                        absQuant[k] |= ((tmp_char0 >> (7+block_start-k)) & 0x00000001) << j;
                        absQuant[k+8] |= ((tmp_char1 >> (7+block_start-k)) & 0x00000001) << j;
                        absQuant[k+16] |= ((tmp_char2 >> (7+block_start-k)) & 0x00000001) << j;
                        absQuant[k+24] |= ((tmp_char3 >> (7+block_start-k)) & 0x00000001) << j;
                    }
                }

                // De-quantize and store data back to decompression data.
                // Unwrap by 4, block of 32
                for(int i=block_start; i<block_end; i += 4)
                {
                    /*sign_ofs = i % 32;
                    if(sign_flag & (1 << (31 - i % 32)))
                        currQuant = absQuant[i] * -1;
                    else
                        currQuant = absQuant[i];
                    */
                    decData[i  ] = absQuant[i  ] * (sign_flag & (1 << (31 - (i)   % 32)) ? -1 : 1) * errorBound * 2;
                    decData[i+1] = absQuant[i+1] * (sign_flag & (1 << (31 - (i+1) % 32)) ? -1 : 1) * errorBound * 2;
                    decData[i+2] = absQuant[i+2] * (sign_flag & (1 << (31 - (i+2) % 32)) ? -1 : 1) * errorBound * 2;
                    decData[i+3] = absQuant[i+3] * (sign_flag & (1 << (31 - (i+3) % 32)) ? -1 : 1) * errorBound * 2;
                }
            }
        }
    }
}