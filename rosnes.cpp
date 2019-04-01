#include<iostream>
#include<random>
#include<vector>
#include<algorithm>
#include<ctime>
#include "generate_random_LDPC.h"
#include "LP.h"

vector<vector<int>> compress_rows(vector<vector<int>>H){
	int m = H.size();
	int n = H[0].size();
	vector<vector<int>> out;
	for (int i = 0; i<m;i++){
		vector<int> new_row;
		for (int j = 0; j<n; j++){
			if (H[i][j] == 1)
				new_row.push_back(j);
		}
		out.push_back(new_row);
	}
	return out;
}
vector<vector<int>>  compress_columns(vector<vector<int>>H){
	int m = H.size();
	int n = H[0].size();
	vector<vector<int>> out;
	for (int i = 0; i<n;i++){
		vector<int> new_column;
		for (int j = 0; j<m; j++){
			if (H[j][i] == 1)
				new_column.push_back(j);
		}
		out.push_back(new_column);
	}
	return out;
}
void print_matrix(vector<vector<int>>H){
	for (int i = 0; i <H.size(); i++){
		for (int j = 0; j< H[0].size(); j++)
			cout << H[i][j]<< " ";
		cout << endl;
	}
}
bool is_stopping_set(vector<vector<int>>R, vector<int>F){
	int m = R.size();
	for (int i =0; i<m; i++){
		int counter = 0;
		for (int j = 0; j <R[i].size(); j++){
			counter+=F[R[i][j]];
		}
		if (counter ==1) 
			return false;
	}
	for (int i =0; i <F.size(); i++) // return 1, unless all the elements of F are zero.
		if (F[i] == 1)
			return true;
	return false;
}
vector<vector<int>> find_H_tild(vector<vector<int>>R,vector<vector<int>>C,vector<int>F){
	vector<bool> relevant_rows(R.size(),false);
	vector<int> erased_columns;
	for (int i = 0; i < F.size(); i++){
		if (F[i] == 1){
			for (int j = 0; j<C[i].size(); j++){
				relevant_rows[C[i][j]] =true;
			}
		} else if (F[i] == 2){
				erased_columns.push_back(i);
		}		
	}
	vector<int> relevant_row_indices;
	for (int i = 0; i < R.size(); i++)
		if(relevant_rows[i] == true){
			relevant_row_indices.push_back(i);
		}
	
	vector<vector<int>> H_tild(relevant_row_indices.size(),vector<int>(erased_columns.size(),0));
	for (int i = 0; i < H_tild.size(); i++){
		int current_R_index =0;
		for (int j = 0;j < erased_columns.size(); j++){
				while((current_R_index < R[i].size()-1) && (R[i][current_R_index]<erased_columns[j]))
					current_R_index++;
				H_tild[i][j] = (R[i][current_R_index] == erased_columns[j]);
		}
	}
	return H_tild;
}
vector<double> solve_linear_program(vector<vector<int>> A, vector<int> b,vector<int> c ){
	int m = A.size();
	int n = A[0].size();
	vector<vector<double>>G(m+1,vector<double>(n+1));
	G[0][0] = 0;
	for (int j = 1; j<n+1; j++)
		G[0][j] = (double)c[j-1];
	for(int i =1; i < m+1; i++){
		G[i][0] = -(double)b[i-1];
		for (int j = 1; j <n+1; j++)
			G[i][j] = -(double)A[i-1][j-1];
	}
	vector<double> out = simplex(G);
	return out;
}
vector<int> reduce_F(vector<vector<int>>R,vector<vector<int>>C,vector<int>F){
		/*vector<vector<int>> H_tild= find_H_tild(R,C,F);
		if (H_tild.size() == 0) return F;
		vector<int>coefs(H_tild[0].size(),1);
		vector<int>bounds(H_tild.size(),1);
		vector<double> x = solve_linear_program(H_tild,bounds,coefs);

		int counter = -1;
		for ( int i = 0; i<H_tild[0].size(); i++){
			do{ counter++;}
			while((counter < F.size()) && (F[counter]!=2));
				if(equal(x[i] ,0)){				
					F[counter] =0;
			}
		}*/
		return F;
}
int choose_unconstrained(vector<vector<int>>R,vector<vector<int>>C,vector<int>F,int d){
	F = reduce_F(R,C,F);
	vector<int> erased_columns;
	vector<int> num_erasures_per_row(R.size(),0);
	for (int i = 0; i <F.size(); i++){
		if(F[i] ==2){
			erased_columns.push_back(i);
			for (int j = 0; j<C[i].size();j++){
				num_erasures_per_row[C[i][j]]++;
			}
		}
	}
	int num_erased_columns = erased_columns.size();
	vector<vector<int>>ascending_erasure_count(d, vector<int>(num_erased_columns,0));
	for(int i =0; i <num_erased_columns; i++){
		for (int j = 0; j<C[erased_columns[i]].size();j++){
			ascending_erasure_count[num_erasures_per_row[C[erased_columns[i]][j]]-1][i]++;
		}
	}	
	vector<int>contenders(num_erased_columns);
	for (int i =0; i < num_erased_columns; i++)
		contenders[i] = i;
	for(int i =0; i <d;i++){
		int largest_local = -1;
		for (int j = 0; j <contenders.size();j++){
			if ( ascending_erasure_count[i][contenders[j]] > largest_local){
				largest_local = ascending_erasure_count[i][contenders[j]];
			}
		}

		for (int j = contenders.size()-1; j>=0; j--){
			if (ascending_erasure_count[i][contenders[j]] < largest_local)
				contenders.erase(contenders.begin()+j);
		}
		if(contenders.size() ==1) break;
	}
	return erased_columns[contenders[0]];// one before last bullet point on page 4170 suggests a better way.
}
vector<int> extended_iterative_decoding(vector<vector<int>>R,vector<vector<int>>C, vector<int> F, bool *is_contradicted){
	*is_contradicted = false;
	int m = R.size();
	int n = C.size();
	vector<int> labels(m,0);
	vector<int> num_erasures(m,0);
	for (int j = 0; j<n; j++){
		if (F[j] ==1){
			for (int i = 0; i<C[j].size(); i++){
					labels[C[j][i]]++;
			}
		} else if (F[j] ==2){
			for (int i = 0; i<C[j].size(); i++){
				num_erasures[C[j][i]]++;
			}
		}		
	}
	bool event = true;
	while(event){
		event = false;
		for (int i = 0; i <m ;i++){
			if ((labels[i] <= 1)&&(num_erasures[i] ==1)){
				event = true;
				int final_value = 0;
				int erased_index = -1;
				for (int j = 0; j<R[i].size();j++)
				{
					if (F[R[i][j]] == 2)
						erased_index = R[i][j];
					else
						final_value += F[R[i][j]];
				}
				F[erased_index] = final_value%2;
				//update labels, num_erasures.
				for (int ell = 0; ell<C[erased_index].size(); ell++){
					labels[C[erased_index][ell]] += F[erased_index];
					num_erasures[C[erased_index][ell]]--;
					// detect contradiction
					if ((num_erasures[C[erased_index][ell]] == 0)&&(labels[C[erased_index][ell]] == 1))
						*is_contradicted = true;
						return F;
				}
			}
		}
	}
	return F;
}
int count_values(vector<int>F, int f){
	int num = 0;
	for (int i = 0; i<F.size(); i++)
		if (F[i] == f) 
			num++;
	return num;
}
bool rosenet_exists_stopping_set(vector<vector<int>>R,vector<vector<int>>C, int tau,int d){
	int m = R.size();
	int n = C.size();
	vector<int>F(n,2); // 2 means erasure. other possible values are zero and one.
	vector<vector<int>> L;

	L.push_back(F);
	while(L.size()){
		F = L.back();
		L.pop_back();
		if (count_values(F,2)>0){
			bool is_contradicted = true;
			vector<int> F_ = extended_iterative_decoding(R,C,F, &is_contradicted);

			int w_ = count_values(F_,1);
			if (count_values(F_,2) == 0){
				if( is_stopping_set(R,F_)&&(w_ <=tau))
					return true;		
			}
			else if ((!is_contradicted)&&(w_ <=tau)){
				vector<int> new_F1(F_);
				vector<int> new_F2(F_);
				int new_index = choose_unconstrained(R,C,F_,d);
				new_F1[new_index] = 0;
				new_F2[new_index] = 1;
				L.push_back(new_F1);
				L.push_back(new_F2);
			}
		} 
		else if((count_values(F,1)<=tau) && (is_stopping_set(R,F))) {// the second condition is unnecessary and must be removed later for efficiency.
			//print_matrix(H);
			cout <<"stopping set is "<<endl;
			for (int i = 0; i<F.size(); i++)	
				cout << F[i]<<" ";
			cout<<endl;
			return true;
		}
	}
	return false;			
}



int main(){
	srand(time(NULL));
	int c = 6;
	int d = 8;
	int n = 128;//make sure n is divisible by 4.
	int k = (n*(d-c))/d;
	int best_stopping_distance = 0;
	vector<vector<int>>best_H;
	int number_of_codes_checked = 0;
	int desired_stopping_bound = 5;
	while(1){
		vector<vector<int>> H = create_random_LDPC_1(n-k,n,c,d);//make sure to remove all-zero columns of H.
		vector<vector<int>> R = compress_rows(H);
		vector<vector<int>> C = compress_columns(H);
		bool sd = rosenet_exists_stopping_set(R,C,desired_stopping_bound,d);
		if(sd){
			cout <<"H = "<< endl;
			print_matrix(H);
			cout << "bad code" << endl;
			}
		else{
			cout << "good code" << endl;
		}
	}
	return 0;
}
