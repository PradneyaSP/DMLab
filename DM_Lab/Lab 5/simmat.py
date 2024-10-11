'''
n : number of permutations
r : number of rows in input matrix
'''
def get_hash_functions(n):
    hash_fn = []
    print('Enter the a & b values:')
    for i in range(0,n):
        a = int(input(f'a{i}: '))
        b = int(input(f'b{i}: '))
        hash_fn.append([a,b])

    print('Hash Functions :',hash_fn)
    return hash_fn

def get_permutations(n,r):
    permutations = []
    hash_fn = get_hash_functions(n)
    for _ in range(0,n):
        p = []
        for i in range(0,r):
            p.append((hash_fn[_][0]*i + hash_fn[_][1])%r)
        permutations.append(p)
    print('Permutations :',permutations)
    return permutations

def get_signature_matrix(input_matrix,permutations):
    n = len(permutations)
    c = len(input_matrix[0])
    r = len(input_matrix) # No of rows

    sig_mat = []
    for i in range(0,n):
        temp = []
        for j in range(0,c):
            temp.append(2**50)
        sig_mat.append(temp)
    
    for i in range(0,r): # Each row in input matrix
        for j in range(0,c): # Each column in input matrix
            if input_matrix[i][j] == 1:
                for k in range(0,n): # Each permutation
                    sig_mat[k][j] = min(sig_mat[k][j],permutations[k][i])
    return sig_mat


def main():
    input_matrix = [[1,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,1],[0,0,1,0],[0,1,1,0],[0,0,0,1],[1,0,1,0],[0,1,1,1],[0,1,0,1],[0,0,1,1],[0,1,0,0]]
    c = len(input_matrix[0])
    n = 5 # No of permutations
    r = len(input_matrix) # No of rows

    permutations = get_permutations(n,r)
    sig_mat = get_signature_matrix(input_matrix,permutations)
    print('Signature Matrix:')

    for i in range (0,n):
        for j in range (0,c):
            print(sig_mat[i][j] , end=' ')
        print()


if __name__ == '__main__':
    main()