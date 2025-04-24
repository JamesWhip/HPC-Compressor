void haar_wavelet_transform(float* data, int n, float* out) 
{
    float temp[n]; 

    for (int i = 0; i < n; i += 2) 
    {
        float avg  = (data[i] + data[i + 1]) / 2.0f;
        float diff = (data[i] - data[i + 1]) / 2.0f;
        temp[i / 2] = avg;
        temp[n / 2 + i / 2] = diff;
    }

    for (int i = 0; i < n; i++) 
    {
        out[i] = temp[i];
    }
}

void haar_wavelet_inverse(float* data, int n) 
{
    float temp[n];  

    for (int i = 0; i < n / 2; i++) 
    {
        float avg  = data[i];
        float diff = data[n / 2 + i];
        temp[2 * i]     = avg + diff;
        temp[2 * i + 1] = avg - diff;
    }

    // Copy result back into data
    for (int i = 0; i < n; i++) 
    {
        data[i] = temp[i];
    }
}