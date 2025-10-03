import numpy as np

def get_matrix(prompt):
    print(f"\nEnter values for {prompt} (rows separated by newlines, values by spaces):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(list(map(float, line.strip().split())))
    return np.array(lines)

def main():
    print("Matrix Calculator")
    print("=================")
    print("Options:")
    print("1. Add matrices (A + B)")
    print("2. Subtract matrices (A - B)")
    print("3. Multiply matrices (A × B)")
    print("4. Solve Ax = B")
    print("5. Transpose of a matrix")
    print("6. Inverse of a matrix")
    print("7. Determinant of a matrix")
    print("0. Exit")

    while True:
        choice = input("\nEnter your choice (0-7): ")

        if choice == "1":
            A = get_matrix("Matrix A")
            B = get_matrix("Matrix B")
            print("\nResult (A + B):\n", A + B)

        elif choice == "2":
            A = get_matrix("Matrix A")
            B = get_matrix("Matrix B")
            print("\nResult (A - B):\n", A - B)

        elif choice == "3":
            A = get_matrix("Matrix A")
            B = get_matrix("Matrix B")
            print("\nResult (A × B):\n", A @ B)

        elif choice == "4":
            A = get_matrix("Matrix A (coefficient matrix)")
            B = get_matrix("Matrix B (right-hand side)")
            try:
                x = np.linalg.solve(A, B)
                print("\nSolution x:\n", x)
            except np.linalg.LinAlgError as e:
                print("Error solving equation:", e)

        elif choice == "5":
            A = get_matrix("Matrix A")
            print("\nTranspose of A:\n", A.T)

        elif choice == "6":
            A = get_matrix("Matrix A")
            try:
                inv = np.linalg.inv(A)
                print("\nInverse of A:\n", inv)
            except np.linalg.LinAlgError:
                print("Matrix A is not invertible.")

        elif choice == "7":
            A = get_matrix("Matrix A")
            try:
                det = np.linalg.det(A)
                print(f"\nDeterminant of A: {det}")
            except np.linalg.LinAlgError:
                print("Error computing determinant.")

        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
