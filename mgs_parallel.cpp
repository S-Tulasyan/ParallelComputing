#include <bits/stdc++.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>

//Initializing n,m
#define m 30
#define n m*m

using namespace std::chrono;
using namespace std;

//Initializing initial solution x and corresponding z
vector<double> z_star (n, -1);
vector<double> x_0 (n, 1);
vector<double> prev_z(n, 0);


double convergeSame(vector<double> x, double gamma)
{
    vector<double> z(n, 0);
    int i;
    double res = 0;

    // #pragma omp parallel for shared(z) private(i)
    for(i=0;i<n;i++)
    {
        // calculating current z vector
        z[i]=(abs(x[i])+x[i])/gamma;
    }
    // calculating error by norm of element-wise (prev_z - z) vector
    double sum_sqr=0.0;

    for(int i=0;i<n;i++)
    {
        sum_sqr += pow((prev_z[i] - z[i]),2);

    }
    double norm= sqrt(sum_sqr);
    if(norm <= 0.00001)
      {
          res=norm;

          return res;
      }
    else
      prev_z = z;
    
    return 0;
}

vector<vector<double> > cholDecomp(vector<vector<double> > L ) {
        // Warning: acts directly on given matrix!
        // cout<<"call to cholDecomp"<<endl;
        int i, j, k;
        omp_lock_t writelock;
        omp_init_lock(&writelock);

        for (j = 0; j < n; j++) {

                for (i = 0; i < j; i++){
                        L[i][j] = 0;
                }

                #pragma omp parallel for shared(L) private(k)
                for (k = 0; k < i; k++) {
                        omp_set_lock(&writelock);
                        L[j][j] = L[j][j] - L[j][k] * L[j][k]; //Critical section.
                        omp_unset_lock(&writelock);
                }

                #pragma omp single
                L[i][i] = sqrt(L[j][j]);

                #pragma omp parallel for shared(L) private(i, k)
                for (i = j+1; i < n; i++) {
                        for (k = 0; k < j; k++) {
                                L[i][j] = L[i][j] - L[i][k] * L[j][k];
                        }
                        L[i][j] = L[i][j] / L[j][j];
                }

                omp_destroy_lock(&writelock);
        }
        // cout<<"return form cholDecomp"<<endl;
        return L;
}
//Returns the vector obtained by multiplication of matrix and vector
vector<double > matrixMultiplyVector( vector<vector<double> > A, vector<double> x )
{
    vector<double > result (n);
    vector<double > temp (n);
    int i,j;
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    #pragma omp parallel shared(A, x, result) private(i, j) 
    {
      #pragma omp for
      for (i = 0; i < n; i++) {
          for (j = 0; j < n; j++) {
               result[i] += A[i][j] * x[j];
          }
      }
    }
    return result;
}

vector<double> solveLU(vector<vector<double> > A, vector<vector<double> > D, vector<vector<double> > L, vector<vector<double> > U, vector<vector<double> > omega, vector<double> q, vector<double> x)
{
  //Initializing temp_rhs and temp_lhs for simplicity of calculations
  vector<vector<double> > temp_lhs(n, vector<double> (n, 0));
  vector<vector<double> > temp_rhs(n, vector<double> (n, 0));
  vector<vector<double> > LT(n, vector<double> (n, 0));
  vector<double> diagL(n, 0);
  vector<double> result = x;
  vector<double> mod_x(n, 0);
  // double lu[n][n];
  double y[n];
  double sum =0.0;
  int i,j;
  #pragma omp parallel  shared(temp_rhs, temp_lhs, A, omega) private(i, j) 
  { 
  #pragma omp for
    for(i=0; i<n; i++){
      for(j=0; j<n; j++){
        temp_rhs[i][j] = omega[i][j] - A[i][j];
        temp_lhs[i][j] = D[i][j] + omega[i][j] - L[i][j];
      }
    }
  }
  // multiply U and x
  vector<double> g = matrixMultiplyVector(U, x);

  #pragma omp parallel for shared(x, mod_x) private(i)
  for(i=0; i<n; i++){
    mod_x[i] = fabs(x[i]);
  }
  //calculating b vector using b=(omega-A)*abs(x)-q
  vector<double> b = matrixMultiplyVector(temp_rhs, mod_x);
  #pragma omp parallel for shared(b, q) private(i)
  for(i=0; i<n; i++){
    b[i] = g[i] + b[i] - 2*q[i];
  }
  //lu decomposition method for solving temp_lhs*x=b
  temp_lhs = cholDecomp(temp_lhs);

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      // cout<<"separating transpose elem ("<<i<<","<<j<<")"<<endl;
      LT[j][i] = temp_lhs[i][j];
    }
  }

  for (int i = 0; i < n; ++i)
  {
    // cout<<"separating diag elem"<<endl;
    diagL[i] = temp_lhs[i][i];
    temp_lhs[i][i] = 1;
  }


  //   cout<<"The Cholskey decomposition = "<<endl;
  // for (int i = 0; i < n; ++i)
  // {
  //   for (int j = 0; j < n; ++j)
  //   {
  //     cout<<LT[i][j]<<"\t";
  //   }
  //   cout<<endl;
  // }

  // find solution of Ly = b
  for (int i = 0; i < n; i++)
  {
      sum = 0.0;
      for (int k = 0; k < i; k++)
          sum += temp_lhs[i][k] * y[k];
      y[i] = b[i] - sum;
      y[i] = y[i]/diagL[i];
  }
  // for (int i = 0; i < n; ++i)
  // {
  //   cout<<y[i]<<"\t";
  // }
  // cout<<endl;
  // find solution of Ux = y
  for (int i = n - 1; i >= 0; i--)
  {
      sum = 0.0;
      for (int k = i + 1; k < n; k++)
      sum += LT[i][k] * result[k];
      result[i] = (1.0 / LT[i][i]) * (y[i] - sum);
  }
  //  for (int i = 0; i < n; ++i)
  // {
  //   cout<<result[i]<<"\t";
  // }
  // cout<<endl;
  return result;
}

void LCP( vector<vector<double> > A, vector<vector<double> > D, vector<vector<double> > L, vector<vector<double> > U, vector<double> q, double gamma, double alpha)
{
  // cout<<"call to LCP"<<endl;
// Initializing omega matrix
  vector<vector<double> > omega(n, vector<double> (n, 0));
  for(int i=0; i<n; i++)
  {
    for(int j=0; j<n; j++)
    {
      omega[i][j] = (1/(2*alpha))*D[i][j];
    }
  }

  vector<double> x_k = x_0;
  int i, itr = 0;
  double res = 0;
  //starting the clock
  auto start = high_resolution_clock::now();
  do{
   //solving next x using lu decomposition method for finding next z
    vector<double> x_kplus1 = solveLU(A, D, L, U, omega, q, x_k);
    x_k = x_kplus1;
    itr++;
  }while(!convergeSame(x_k, gamma));
  //ending clock
  auto stop = high_resolution_clock::now();
  //calculating execution time
  auto duration = duration_cast<microseconds>(stop - start);
// printing execution time,iterations and error involved in solving LCP problem for particular alpha
  cout<<"alpha = "<<alpha<<"\t";
 cout<<"RES(z_k) = "<<norm<<"\t";
  cout <<"Execution time:"<< duration.count()/1000000.0<<"\t\t";
  cout<<"IT = "<<itr<<endl;
}

int main()
{
  cout<<"For N = "<<n<<endl;
  //Defining and Initializing A matrix
  double mu = 4.0;
  vector<vector<double> > A (n, vector<double> (n, 0));
  vector<vector<double> > D (n, vector<double> (n, 0));
  vector<vector<double> > L (n, vector<double> (n, 0));
  vector<vector<double> > U (n, vector<double> (n, 0));
  for(int i=0; i<n; i++)
    A[i][i] = 4.0 + mu;
  for(int i=1; i<n; i++){
    A[i][i-1] = -1.0;
    A[i-1][i] = -1.0;
  }
  for(int i=m; i<n; i+=m){
    A[i][i-1] = 0;
    A[i-1][i] = 0;
  }
  for(int i=0; i<n-m; i++){
    A[m+i][i] = -1.0;
    A[i][m+i] = -1.0;
  }
// Initializing D,L,U
  for(int i=0;i<n;i++)
   {
       for(int j=0;j<n;j++)
       {
           if(i!=j)D[i][j]=0;
           else D[i][j]=A[i][j];
           if(i<j)
           {
               L[i][j]=0;
               U[i][j]=-A[i][j];
           }
           else if(i>j)
           {
               L[i][j]=-A[i][j];
               U[i][j]=0;
           }
           else
           {
               L[i][j]=U[i][j]=0;
           }
       }
   }

//Initializing z initial and initial solution x for find actual z
  for(int i=1; i<n; i+=2){
    z_star[i] = -2;
    x_0[i] = 0;
  }
 //initialize D-L
  vector<vector<double> > D_L (n, vector<double> (n, 0)); 
  for(int i=0;i<n;i++)
   {
      for(int j=0;j<n;j++)
      {
        D_L[i][j]=D[i][j]-L[i][j];
      }
   }

//q vector obtained by multiplication of A matrix and z initial vector
  vector<double> q = matrixMultiplyVector(D_L, z_star);
//loop to find out minimum iterations for varying alpha
  double alpha = 0.4;
  while(alpha<1.0){
    //this calculates iterations and execution time for a particular alpha
    LCP(A, D, L, U, q, 2.0, alpha);
    alpha+=0.1;
  }
return 0;
}
