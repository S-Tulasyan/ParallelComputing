#include <bits/stdc++.h>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <chrono>

using namespace std::chrono;
using namespace std;
//Initializing n,m
#define m 10
#define n m*m

//Initializing initial solution x and corresponding z
vector<double> z_star (n, -1);
vector<double> x_0 (n, 1);
double prev_z[n];
double gamma_param = 2.0;

// Returns the norm of a given vector
double normL2(vector<double > x)
{
  double result = 0;
  for(int i=0; i<n; i++){
    result += x[i]*x[i];
  }
  return sqrt(result);
}

vector<vector<double> > cholDecomp(vector<vector<double> > L ) {
        // Warning: acts directly on given matrix!

        int i, j, k;

        for (j = 0; j < n; j++) {

                for (i = 0; i < j; i++){
                        L[i][j] = 0;
                }

                for (k = 0; k < i; k++) {
                        L[j][j] = L[j][j] - L[j][k] * L[j][k]; //Critical section.
                }

                L[i][i] = sqrt(L[j][j]);

                for (i = j+1; i < n; i++) {
                        for (k = 0; k < j; k++) {
                                L[i][j] = L[i][j] - L[i][k] * L[j][k];
                        }
                        L[i][j] = L[i][j] / L[j][j];
                }
        }
        return L;
}
//Returns the vector obtained by multiplication of matrix and vector
vector<double > matrixMultiplyVector( vector<vector<double> > A, vector<double> x )
{
    vector<double > result (n);
    for (int i = 0; i < n; i++) {
      double sum = 0;
        for (int j = 0; j < n; j++) {
             sum += A[i][j] * x[j];
        }
        result[i] = sum;
    }
    return result;
}

vector<double> solveLU(vector<vector<double> > A, vector<vector<double> > D,vector<vector<double> > L,
             vector<vector<double> > U,vector<vector<double> > omega, vector<double> q, vector<double> x,double alpha)
{
  //Initializing temp_rhs and temp_lhs for simplicity of calculations
  vector<vector<double> > temp_lhs(n, vector<double> (n, 0));
  vector<vector<double> > temp_rhs(n, vector<double> (n, 0));
  vector<vector<double> > LT(n, vector<double> (n, 0));
  vector<double> diagL(n, 0);
  vector<double> result = x;
  vector<double> mod_x(n, 0);
  double lu[n][n];
  double y[n];
  double sum =0.0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      temp_rhs[i][j] = omega[i][j] - alpha*A[i][j];
      temp_lhs[i][j] = D[i][j] + omega[i][j] - alpha*L[i][j];
    }
  }
  //U_D is defined for simplifying terms  U_D=((1-alpha)*D+alpha*U)
  vector<vector<double> >U_D(n,vector<double> (n,0));
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
        U_D[i][j]=(1-alpha)*D[i][j]+alpha*U[i][j];
    }
  }
  vector<double> g = matrixMultiplyVector(U_D, x);

  for(int i=0; i<n; i++){
    mod_x[i] = fabs(x[i]);
  }
  //calculating b vector using b=(omega-alpha*A)*abs(x)-alpha*gamma*q
  vector<double> b = matrixMultiplyVector(temp_rhs, mod_x);
  for(int i=0; i<n; i++){
    b[i] = g[i] + b[i] - (alpha*gamma_param*q[i]);
  }
  temp_lhs = cholDecomp(temp_lhs);

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      LT[j][i] = temp_lhs[i][j];
    }
  }

  for (int i = 0; i < n; ++i)
  {
    diagL[i] = temp_lhs[i][i];
    temp_lhs[i][i] = 1;
  }

  // find solution of Ly = b
  for (int i = 0; i < n; i++)
  {
      sum = 0.0;
      for (int k = 0; k < i; k++)
          sum += temp_lhs[i][k] * y[k];
      y[i] = b[i] - sum;
      y[i] = y[i]/diagL[i];
  }

  // find solution of Ux = y
  for (int i = n - 1; i >= 0; i--)
  {
      sum = 0.0;
      for (int k = i + 1; k < n; k++)
      sum += LT[i][k] * result[k];
      result[i] = (1.0 / LT[i][i]) * (y[i] - sum);
  }

  return result;
}
void LCP( vector<vector<double> > A, vector<vector<double> > D,
         vector<vector<double> > L, vector<vector<double> > U, vector<double> q, double alpha)
{
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
  vector<double > z_k(n, 0);
  vector<double > temp;
  //calculating z initial using assumed x initial
  /*for(int i=0; i<n; i++){
    z_k[i] = 1.0/gamma *(fabs(x_k[i]) + x_k[i]);
  }
*/
  int itr = 0;
  //calculating temp vector for finding error
  /*temp = matrixMultiplyVector(A, z_k);
  for(int i=0; i<n; i++)
    temp[i] = min((temp[i] + q[i]), z_k[i]);
  double res = normL2(temp);
  */
  //starting the clock
  auto start = high_resolution_clock::now();

   double norm;
  do
  {
   //solving next x using lu decomposition method for finding next z
    vector<double> x_kplus1 = solveLU(A, D, L, U, omega, q, x_k,alpha);
     //finding next z using next x
    for(int i=0; i<n; i++)
      z_k[i] = 1.0/gamma_param *(fabs(x_kplus1[i]) + x_kplus1[i]);
    //calculating temp vector for finding error
    double sum_sqr=0.0;
    for(int i=0;i<n;i++)
    {
        sum_sqr += pow((prev_z[i] - z_k[i]),2);

    }
    norm= sqrt(sum_sqr);
    x_k = x_kplus1;
    for(int i=0;i<n;i++)
    {
        prev_z[i]=z_k[i];
    }
    itr++;
  }while(norm > 0.00001);
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
//Initializing z initial and initial solution x for find actual z
  for(int i=1; i<n; i+=2){
    z_star[i] = -2;
    x_0[i] = 0;
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

    //initialize M=D/alpha-L
    vector<vector<double> > M (n, vector<double> (n, 0));
    double alpha=0.9;
    //loop to find out minimum iterations for varying alpha
    while(alpha<=1.0)
    {
          for(int i=0;i<n;i++)
           {
              for(int j=0;j<n;j++)
              {
                M[i][j]=D[i][j]/alpha-L[i][j];
              }
           }

           for(int i=0;i<n;i++)
           {
               prev_z[i]=(abs(x_0[i])+x_0[i])/gamma_param;
           }


    //q vector obtained by multiplication of A matrix and z initial vector
      vector<double> q = matrixMultiplyVector(M, z_star);
        //this calculates iterations and execution time for a particular alpha
        LCP(A, D, L, U, q, alpha);
        alpha+=0.1;
      }
    return 0;
}
