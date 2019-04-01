

/*
  What: Simplex in C
  AUTHOR: GPL(C) moshahmed/at/gmail.

  What: Solves LP Problem with Simplex:
    { maximize cx : Ax <= b, x >= 0 }.
  Input: { m, n, Mat[m x n] }, where:
    b = mat[1..m,0] .. column 0 is b >= 0, so x=0 is a basic feasible solution.
    c = mat[0,1..n] .. row 0 is z to maximize, note c is negated in input.
    A = mat[1..m,1..n] .. constraints.
    x = [x1..xm] are the named variables in the problem.
    Slack variables are in columns [m+1..m+n]

  USAGE:
    1. Problem can be specified before main function in source code:
      c:\> vim mosplex.c  
      Tableau tab  = { m, n, {   // tableau size, row x columns.
          {  0 , -c1, -c2,  },  // Max: z = c1 x1 + c2 x2,
          { b1 , a11, a12,  },  //  b1 >= a11 x1 + a12 x2
          { b2 , a21, a22,  },  //  b2 >= a21 x1 + a22 x2
        }
      };
      c:\> cl /W4 mosplex.c  ... compile this file.
      c:\> mosplex.exe problem.txt > solution.txt

    2. OR Read the problem data from a file:
      $ cat problem.txt
            m n
            0  -c1 -c2
            b1 a11 a12
            b2 a21 a11 
      $ gcc -Wall -g mosplex.c  -o mosplex
      $ mosplex problem.txt > solution.txt
*/

#include <vector>
#include <iostream>
using namespace std;
static const double epsilon   = 1.0e-8;
int equal(double a, double b);
vector<vector<double>> pivot_on(vector<vector<double>>tab, int row, int col);
int find_pivot_column(vector<vector<double>> tab);
int find_pivot_row(vector<vector<double>> tab, int pivot_col);
vector<vector<double>> add_slack_variables(vector<vector<double>> tab);
int find_basis_variable(vector<vector<double>> tab, int col);
vector<double> optimal_vector(vector<vector<double>> tab);
vector<double> simplex(vector<vector<double>> tab);

