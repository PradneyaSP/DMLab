import numpy as np

def collect_documents():
    documents = []

    n = int(input("Enter the number of documents: "))

    for i in range(n):
        doc_list = []

        m = int(input(f"\nEnter the number of values for document {i+1}: "))

        for j in range(m):
            val = int(input(f"Enter value {j+1}: "))
            doc_list.append(val)

        documents.append(doc_list)

    return documents


def create_hash_functions(num_hashes, num_buckets):
    hash_functions = []
    for _ in range(num_hashes):
        a = np.random.randint(1, 100)
        b = np.random.randint(0, 100)
        hash_functions.append(lambda x, a=a, b=b, n=num_buckets: (a * x + b) % n)
    return hash_functions


def minhash_signature_matrix(documents, num_hashes):
    num_docs = len(documents)
    num_buckets = len(documents[0])
    hash_functions = create_hash_functions(num_hashes, num_buckets)

    signature_matrix = np.full((num_hashes, num_docs), np.inf)

    for doc_index, doc in enumerate(documents):
        for hash_index, hash_func in enumerate(hash_functions):
            print(
                f"Hash function {hash_index} (a={hash_func.__defaults__[0]}, b={hash_func.__defaults__[1]}):"
            )
            for value in doc:
                hashed_value = hash_func(value)
                print(f"Value: {value}, Hashed Value: {hashed_value}")
                if hashed_value < signature_matrix[hash_index, doc_index]:
                    signature_matrix[hash_index, doc_index] = hashed_value
                    print(
                        f"Updated signature_matrix[{hash_index}, {doc_index}] to {hashed_value}"
                    )

    return signature_matrix


def jaccard_similarity(signature_matrix, doc_index1, doc_index2):
    num_hashes = signature_matrix.shape[0]
    similar_hashes = np.sum(
        signature_matrix[:, doc_index1] == signature_matrix[:, doc_index2]
    )
    return similar_hashes / num_hashes


if __name__ == "__main__":

    documents = collect_documents()
    num_hashes = 5
    signature_matrix = minhash_signature_matrix(documents, num_hashes)

    length = len(documents)
    similarities = np.zeros((length, length))

    for i in range(length):
        for j in range(i, length):
            similarity = jaccard_similarity(signature_matrix, i, j)
            similarities[i, j] = similarity
            similarities[j, i] = similarity

    print(f"\nSignature Matrix:\n{signature_matrix}\n")

    print("Jaccard Similarity Matrix:")
    for i in range(length):
        for j in range(i + 1, length):
            print(f"Similarity between document {i} and {j}: {similarities[i, j]}")