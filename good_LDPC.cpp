#include<iostream>
#include<random>
#include<vector>
#include<algorithm>
#include<ctime>
using namespace std;
#define typical_stopping_ratio 0.124
int nchoosek(int n, int k){
	if ( k == 0)
		return 1;
	else
		return (nchoosek(n-1,k-1)*n)/k;
}
vector<vector<int>> all_combinations(vector<int> a, int k){
	int n = a.size();
	vector<vector<int>> output;
	vector<int> locations(n,0);
	for (int i =0; i<k;i++)
		locations[i] = 1;
	do{
		vector<int> current_row;
		for (int j = 0; j<n; j++){
			if( locations[j] ==  1)
				current_row.push_back(j);
		}
		output.push_back(current_row);
	}while(prev_permutation(locations.begin(), locations.end()));
	return output;
}
vector<vector<int>> create_random_LDPC(int m, int n, int c, int d){
	if (n*c - m*d != 0)
		return vector<vector<int>>(0,vector<int>(0));
	int E = n*c;
	vector<int> perm(E);
	for (int i = 0; i <E; i++)
		perm[i] = i;
	random_shuffle(perm.begin(),perm.end());
	vector<vector<int>> H(m,vector<int>(n,0));
	for (int i = 0; i<E; i++){
		int var_index = perm[i]/c;
		int check_index = i/d;
		H[check_index][var_index] =(H[check_index][var_index]+1)%2;
	}
	return H;
}
bool is_stopping_set(vector<vector<int>>H, vector<int>columns){
	int m = H.size();
	int r = columns.size();
	for (int i =0; i<m; i++){
		int counter = 0;
		for (int j = 0; j <r; j++){
			counter+=H[i][columns[j]];
		}
		if (counter ==1) 
			return false;
	}
	return true;
}
bool exists_stopping_set(vector<vector<int>>H, int stopping_number){
	int n = H[0].size();
	vector<int> a(n);
	for (int i =0; i<n; i++)
		a[i] = i;
	vector<vector<int>> all_choices = all_combinations(a,stopping_number);
	int num_choices = all_choices.size();
	for (int ell = 0; ell<num_choices; ell++){
		if (is_stopping_set(H,all_choices[ell]))
			return true;
	}
	return false;
}
int compute_stopping_distance(vector<vector<int>>H){
	int m = H.size();
	//int approx_stopping_distance = round(typical_stopping_ratio*n);
	for (int i = 2; i<m+1; i++){	
		if (exists_stopping_set(H,i))
			return i;
	}
	return m+1;			
}
int main(){
	srand(time(NULL));
	int c = 6;
	int d = 8;
	int n = 32;//make sure n is divisible by 4.
	int k = (n*(d-c))/d;
	int best_stopping_distance = 0;
	vector<vector<int>>best_H;
	while(1){
		vector<vector<int>> H = create_random_LDPC(n-k,n,c,d);
		int sd = compute_stopping_distance(H);
		if(sd > best_stopping_distance){
			best_stopping_distance = sd;
			best_H = H;
			cout <<"best H so far= "<<endl;
			for(int i=0; i<n-k;i++){
				for (int j =0; j<n;j++)
					cout<< best_H[i][j]<<" ";
				cout<<endl;
			}
			cout << endl<<"stopping distance = " <<  best_stopping_distance<<endl;
		}
	}
	return 0;
}
