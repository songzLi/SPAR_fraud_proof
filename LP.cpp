

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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include "LP.h" 
using namespace std;

//static const double epsilon   = 1.0e-8;
int equal(double a, double b) { return fabs(a-b) < epsilon; }

vector<vector<double>> pivot_on(vector<vector<double>>tab, int row, int col) {
  unsigned int i, j;
  double pivot;

  pivot = tab[row][col];
  assert(pivot>0);
  for(j=0;j<tab[0].size();j++)
    tab[row][j] /= pivot;
  assert( equal(tab[row][col], 1. ));

  for(i=0; i<tab.size(); i++) { // foreach remaining row i do
    double multiplier = tab[i][col];
    if(i==(unsigned int) row) continue;
    for(j=0; j<tab[0].size(); j++) { // r[i] = r[i] - z * r[row];
      tab[i][j] -= multiplier * tab[row][j];
    }
  }
  return tab;
}

// Find pivot_col = most negative column in mat[0][1..n]
int find_pivot_column(vector<vector<double>> tab) {
  int pivot_col = 1;
  double lowest = tab[0][pivot_col];
  for(unsigned int j=1; j<tab[0].size(); j++) {
    if (tab[0][j] < lowest) {
      lowest = tab[0][j];
      pivot_col = j;
    }
  }
  if( lowest >= 0 ) {
    return -1; // All positive columns in row[0], this is optimal.
  }
  return pivot_col;
}

// Find the pivot_row, with smallest positive ratio = col[0] / col[pivot]
int find_pivot_row(vector<vector<double>> tab, int pivot_col) {
  int pivot_row = 0;
  double min_ratio = -1;
  for(unsigned i=1;i<tab.size();i++){
    double ratio = tab[i][0] / tab[i][pivot_col];
    if ( (ratio > 0  && ratio < min_ratio ) || min_ratio < 0 ) {
      min_ratio = ratio;
      pivot_row = i;
    }
  }
  if (min_ratio == -1)
    return -1; // Unbounded.
  return pivot_row;
}

vector<vector<double>> add_slack_variables(vector<vector<double>> tab) {
  unsigned int i, j;
  for(i=0; i<tab.size(); i++) {
    for(j=1; j<tab.size(); j++)
      tab[i].push_back((i==j));
  }
  return tab;
}

void check_b_positive(vector<vector<double>> tab) {
  for(unsigned i=1; i<tab.size(); i++){
		//assert(tab[i][0] >= 0);
    }
}

// Given a column of identity matrix, find the row containing 1.
// return -1, if the column as not from an identity matrix.
int find_basis_variable(vector<vector<double>> tab, int col) {
  int xi=-1;
  for(unsigned int i=1; i < tab.size(); i++) {
    if (equal( tab[i][col],1) ) {
      if (xi == -1)
        xi=i;   // found first '1', save this row number.
      else
        return -1; // found second '1', not an identity matrix.

    } else if (!equal( tab[i][col],0) ) {
      return -1; // not an identity matrix column.
    }
  }
  return xi;
}

vector<double> optimal_vector(vector<vector<double>> tab) {			    	
  vector<double> sol(tab[0].size()-1,0);
  int xi;
  for(unsigned int j=1;j<tab[0].size();j++) { // for each column.
    xi = find_basis_variable(tab, j);
    if (xi != -1){
      sol[j-1] = tab[xi][0];
    }
  }
  return sol;
} 

vector<double> simplex(vector<vector<double>> tab) {
  int loop=0;
  tab = add_slack_variables(tab);
  check_b_positive(tab);
  while( ++loop ) {
    int pivot_col, pivot_row;
    pivot_col = find_pivot_column(tab);
    if( pivot_col < 0 ) {
      vector<double> out = optimal_vector(tab);
      return out;
      
    }
    pivot_row = find_pivot_row(tab, pivot_col);
    if (pivot_row < 0) {
      break;
    }
    tab = pivot_on(tab, pivot_row, pivot_col);
    
   // vector<double> out = optimal_vector(tab);

    if(loop > 20) {
      break;
    }
    
  }
  
  vector<double> out(tab[0].size()-1,-1);
  return out;
}
