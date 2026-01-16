#include <iostream>
#include <vector>
using namespace std;

void selectionSort(vector<int>& A) {
    for (int i = 0; i < A.size() - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < A.size(); j++) {
            if (A[j] < A[min_index]) {
                min_index = j;
            }
        }
        swap(A[i], A[min_index]);   // Exchange elements
    }
}

int main() {
    vector<int> A = { 11, 22, 14, 67, 2, 9 };
    selectionSort(A);
    cout << "Sorted array: ";
    for (int i = 0; i < A.size(); i++) {
        cout << A[i] << " ";
    }
    cout << endl;
    return 0;
}
