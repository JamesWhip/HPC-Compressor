#include <math.h>

int traverser(float* datablock, int blocks)
{
    float gradient = 0.00f;
    for(int i=0; i<blocks-1; i++)
    {
        gradient += fabs(datablock[i+1] - datablock[i]);
    }

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


    if(gradient < 3e-4) 
    {
        return 3;
    }
    else if (gradient < 1.5e-3) 
    {
        return 3;
    }
    else if (gradient < 8e-3) 
    {
        return 2;

    } else if (gradient < 1e5) 
    {
        return 1;
    } 
    return 4;
}