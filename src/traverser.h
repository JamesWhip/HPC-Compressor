#include <math.h>

#define threshold 5.0f

int traverser(float* datablock, int blocks)
{


    //float max = 0.00f;
    float gradient = 0.00f;
    for(int i=0; i<blocks-1; i++)
    {
        //float diff = fabs(datablock[i+1] - datablock[i]);
        //max = diff > max ? diff : max;
        gradient += fabs(datablock[i+1] - datablock[i]);
    }


    //return (max > threshold);

    //printf("Gradient: %.2e\n", gradient);
    // if(gradient >= 5e1) {
    //     count_e1++;
    // }
    // else if (gradient >= 1e-2) {
    //     count_e2++;
    // } else if (gradient >= 1e-3) {
    //     count_e3++;
    // } else if (gradient >= 1e-4) {
    //     count_e4++;
    // } else if (gradient >= 1e-5) {
    //     count_e5++;
    // }


    if (gradient < 1e2) 
    {
        return 0;
    } 
    return 1;
}