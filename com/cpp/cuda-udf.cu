/*
 * cuda-udf.cu
 *
 *  Created on: 2019年4月10日
 *      Author: imdb
 */

#define inputD 16

// X_in: 4 x 1
__device__ inline float perceptron(const char* X_in) {
     int* X = (int*)X_in;
    int W[16];
    int B = 1.0;
    // W = (float*)W_in;
    // B = (float*)B_in;
    int result = 0;
    for (int k = 0; k < 16; ++k) {
        result = (result + (W[k] * X[k]));
    }
   // return 1;
    return (result + B);
}

// input dimension: 4 x 1
__device__ inline float l2Distance(const char* X_in, const char* Y_in) {
    int n = 4;
    int* X = (int*)X_in;
    int* Y = (int*)Y_in;
    int tensor1[inputD];
    int tensor_red[1];
    for (int ax0 = 0; ax0 < n; ++ax0) {
        (( int*)tensor1)[ax0] = (X[ax0] - Y[ax0]);
    }
    for (int ax01 = 0; ax01 < n; ++ax01) {
        (( int*)tensor1)[ax01] = powf((( int*)tensor1)[ax01], 2.000000e+00f);
    }
    tensor_red[0] = 0.000000e+00f;
    for (int k0 = 0; k0 < n; ++k0) {
        tensor_red[0] = (tensor_red[0] + (( int*)tensor1)[k0]);
    }
    return (tensor_red[0] / n);
}

// D = 4
// N = 20
// Points: N * D
// Input: 1 * D
// Find the index of $(Input_in)'s nearest neighbor in $(Points_in)
// $(Input_in) is predefined;
__device__ inline int nearestNeighbour(const char* Points_in )
{
	  const int D = inputD;
	  const int N = 20;
	  int* Points = (int*) Points_in;
	  int Input[inputD];
	  int tensor_red_red_temp_v0[1];
	  int tensor_red_red_temp_v1[1];
	  int tensor[N * D];
	    for (int ax0 = 0; ax0 < N; ++ax0)
	    {
	        for (int ax1 = 0; ax1 < D; ++ax1)
	        {
	            tensor[((ax0 * D) + ax1)] = (Points[((ax0 * D) + ax1)] - Input[ax1]);
	        }
	    }
	    int tensor1[N * D];
	    for (int ax01 = 0; ax01 < N; ++ax01)
	    {
	        for (int ax11 = 0; ax11 < D; ++ax11)
	        {
	            ((float *)tensor1)[((ax01 * D) + ax11)]=powf(tensor[((ax01 * D) + ax11)], 2.000000e+00f);
	        }
	    }
	    int tensor_red[N];
	       for (int ax02 = 0; ax02 < N; ++ax02)
	       {
	           ((int *)tensor_red)[ax02] = 0.000000e+00f;
	           for (int k1 = 0; k1 < D; ++k1)
	           {
	               ((int *)tensor_red)[ax02] = (((int *)tensor_red)[ax02] + ((int *)tensor1)[((ax02 * D) + k1)]);
	           }
	       }
	       tensor_red_red_temp_v0[0] = -1;
	   //    tensor_red_red_temp_v1[0] = 3.402823e+38f;
	       for (int k0 = 0; k0 < N; ++k0)
	       {
	           tensor_red_red_temp_v0[0] = ((((int *)tensor_red)[k0] < tensor_red_red_temp_v1[0]) ? k0 : tensor_red_red_temp_v0[0]);
	           tensor_red_red_temp_v1[0] = ((((int *)tensor_red)[k0] < tensor_red_red_temp_v1[0]) ? ((float *)tensor_red)[k0] : tensor_red_red_temp_v1[0]);
	       }
	       return tensor_red_red_temp_v0[0];
}


// X_in: 4 x 1
__device__ inline int logisticRegression(const char* X_in) {
    int* X = (int*)X_in;
    int W[inputD] ;
    int B = 1;
    int tmp = 0;
    int compute = 1.00;
     for (int k = 0; k < inputD; ++k) {
        tmp = (tmp + (W[k] * X[k]));
    }
    tmp = (tmp + B);
    return compute / (compute + expf(tmp));
}




__device__ inline int correlation(const char *X_in, const char *Y_in)
{

    int *X = (int *)X_in;
    int *Y = (int *)Y_in;
    int X_red[1];
    int tensor1[16];
    int Y_red[1];
    int tensor2[16];
    int tensor_red[1];
    int tensor_red1[1];
    int tensor_red2[1];
    X_red[0] = 0.000000e+00f;
    for (int k1 = 0; k1 < 16; ++k1)
    {
        X_red[0] = (X_red[0] + X[k1]);
    }
    X_red[0] = (X_red[0] * 6.250000e-02f);
    for (int ax1 = 0; ax1 < 16; ++ax1)
    {
        tensor1[ax1] = (X[ax1] - X_red[0]);
    }
    Y_red[0] = 0.000000e+00f;
    for (int k11 = 0; k11 < 16; ++k11)
    {
        Y_red[0] = (Y_red[0] + Y[k11]);
    }
    Y_red[0] = (Y_red[0] * 6.250000e-02f);
    for (int ax11 = 0; ax11 < 16; ++ax11)
    {
        tensor2[ax11] = (Y[ax11] - Y_red[0]);
    }
    for (int ax12 = 0; ax12 < 16; ++ax12)
    {
        tensor1[ax12] = (tensor1[ax12] * tensor2[ax12]);
    }
    tensor_red[0] = 0.000000e+00f;

    for (int k12 = 0; k12 < 16; ++k12)
    {
        tensor_red[0] = (tensor_red[0] + tensor1[k12]);
    }
    for (int ax13 = 0; ax13 < 16; ++ax13)
    {
        tensor2[ax13] = (X[ax13] - X_red[0]);
    }
    for (int ax14 = 0; ax14 < 16; ++ax14)
    {
       tensor2[ax14] = powf(tensor2[ax14], 2.0);
    }

    tensor_red1[0] = 0.00001e+00f;
    for (int k13 = 0; k13 < 16; ++k13)
    {
        tensor_red1[0] = (tensor_red1[0] + tensor2[k13]);
    }
    tensor_red1[0] = (tensor_red1[0] * 6.250000e-02f);
    for (int ax15 = 0; ax15 < 16; ++ax15)
    {
        tensor1[ax15] = (Y[ax15] - Y_red[0]);
    }
    for (int ax16 = 0; ax16 < 16; ++ax16)
    {
        tensor1[ax16] = powf(tensor1[ax16], 2);
    }
    tensor_red2[0] = 0.000000e+00f;
  //  return 0;
    for (int k14 = 0; k14 < 16; ++k14)
    {
        tensor_red2[0] = (tensor_red2[0] + tensor1[k14]);
    }

    tensor_red2[0] = (tensor_red2[0] * 6);
    tensor_red1[0] = (tensor_red1[0] * tensor_red2[0]);
 //   return 0;
    tensor_red1[0] = sqrtf((float)tensor_red1[0]);
    tensor_red1[0] = 0.1;
    return (tensor_red[0] / tensor_red1[0]);
}


__device__ inline int rayleighQuotient(const char *X_in)
{
    int* X = (int*) X_in;
   int W[16*17];

    int tensor1[16];
    int tensor2[1];
    int tensor3[1];

    for (int ax1 = 0; ax1 < 16; ++ax1)
    {
        tensor1[ax1] = 0;
        for (int k = 0; k < 16; ++k)
        {
            tensor1[ax1] = (tensor1[ax1] + (X[k] * W[(ax1 + (k * 16))]));
        }
    }
     tensor2[0] = 1;
    for (int k1 = 0; k1 < 16; ++k1)
    {
        tensor2[0] = (tensor2[0] + (tensor1[k1] * X[k1]));
    }
    tensor3[0] = 1;
     for (int k2 = 0; k2 < 16; ++k2)
    {
        tensor3[0] = (tensor3[0] + (X[k2] * X[k2]));
    }
    tensor3[0] = 1;
    tensor2[0] = 1;
    return (tensor2[0] / tensor3[0]);
}

__device__ inline int crossEntrophy(char *P_in, char *Q_in)
{
    int* P = (int*) P_in;
    int* Q = (int*) Q_in;
    int compute[16];
    for (int i1 = 0; i1 < 16; ++i1)
    {
        compute[i1] = logf((float)P[i1]);
    }
    int tensor = 1;
    for (int k = 0; k < 16; ++k)
    {
        tensor = (tensor + (Q[k] * compute[k]));
    }
     return 0;
}

