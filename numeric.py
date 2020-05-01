# !/usr/bin/python
from math import sqrt


class Fraction:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.__notation = ""
        if (numerator < 0 and denominator > 0) or (numerator > 0 and denominator < 0):
            self.__numerator = abs(self.numerator)
            self.__denominator = abs(self.denominator)
            self.__notation = '-'
        else:
            self.__numerator = abs(self.numerator)
            self.__denominator = abs(self.denominator)
            
        
        self.numerator, self.denominator = self.__simplify()
        if self.__notation != "":
            self.numerator = self.numerator * (-1)
        
        self.value = self.numerator / self.denominator
        
    def __str__(self):
        return self.__notation + str(abs(self.numerator)) + '/' + str(self.denominator)
    
    def __simplify(self):
        numerator_factors = self.__find_factors(self.__numerator)
        denominator_factors = self.__find_factors(self.__denominator)
        
        removing_list = []
        for factor in numerator_factors:
            try:
                denominator_factors.remove(factor)
                removing_list.append(factor)
            except ValueError:
                pass
        
        for item in removing_list:
            numerator_factors.remove(item)
            
        return self.__findProduct(numerator_factors), self.__findProduct(denominator_factors)
                
    
    @staticmethod
    def __findProduct(the_list):
        product = 1
        for item in the_list:
            product *= item
        return product
    
    @staticmethod
    def __find_primary(index):
        if index == 1:
            return 2
        else:
            counter = 1
            number = 2
            while counter < index :
                number += 1
                is_primary = True
                for devider in range(2, number - 1):
                    if number % devider == 0:
                        is_primary = False
                        break
                if is_primary:
                    counter += 1
            return number
    
    def __find_factors(self, number):
        factors = []

        while number > 1:
            checking_primary_index = 0
            while True:
                checking_primary_index += 1
                current_primary = self.__find_primary(checking_primary_index)
                if number % current_primary == 0:
                    number = int(number / current_primary)
                    factors.append(current_primary)
                    break

        return factors

class AssignmentError(Exception):
   def __init__(self, message):
       super().__init__(message)

class Matrix:
    def __init__(self, matrix):
        """
        :param matrix: This is the given matrix
        """
        if type(matrix) is list or type(matrix) is int:
            if type(matrix) is list:
                lengths = [len(item) for item in matrix]
                if min(lengths) != max(lengths):
                    raise AssignmentError("Number of columns are different")

            # if the user requested a unit matrix, the matrix argument will be a integer
            if isinstance(matrix, int):
                # then serve him a unit matrix
                self.matrix = self.__unitMatrix(matrix)
            else:
                # otherwise set the matrix
                self.matrix = matrix

            self.dimensions = self.__getDimension()
            
            try:
                self.det = self.__determination(self.matrix)
            except Exception:
                self.det = None

        else:
            raise AssignmentError(f"{type([])} object is required, you assaigned {type(matrix)}")

    def __str__(self):
        """
        :return: This function will just print the matrix
        """
        matrix = ""
        for row in self.matrix:
            row_string = ""
            for column in row:
                row_string += (str(column) + "\t")
            matrix += (row_string + '\n')

        return matrix

    def __mul__(self, other_matrix , simplify = True):
        """
        :param other_matrix: This is the other matrix,
                             This must be a matrix object
        :return: Function will return (A x B) if possible,
                 otherwise function will return 0
        """
        try:
            if len(self.matrix[0]) == len(other_matrix.matrix):
                product = self.__generateMat(len(self.matrix))

                for k in range(len(self.matrix)):
                    for i in range(len(other_matrix.matrix[0])):
                        numeric_product = 0
                        for j, item in enumerate(self.matrix[k]):
                            numeric_product += item * other_matrix.matrix[j][i]
                        product[k].append(numeric_product)

                # return the calculated matrix as a matrix object
                return Matrix(product)
            else:
                return False
        except AttributeError:
            if simplify:
                new_matrix = []
                for row in self.matrix:
                    new_row = []
                    for column in row:
                        new_row.append(column * other_matrix)
                    new_matrix.append(new_row)
                return Matrix(new_matrix)

            if not simplify:
                new_matrix = []
                for row in self.matrix:
                    new_row = []
                    for column in row:
                        new_row.append(str(column) + 'x' + str(other_matrix))
                    new_matrix.append(new_row)
                return Matrix(new_matrix)

    def __sub__(self, other_matrix):
        """
        :param other_matrix: This is the other matrix to be subtracted
        :return: Function will return subtracted matrix if possible,
                 otherwise function will return 0
        """
        try:
            subtracted = self.__generateMat(len(self.matrix))
            for index, row in enumerate(self.matrix):
                for column, element in enumerate(row):
                    subtracted[index].append(element - other_matrix.matrix[index][column])

            # return the calculated matrix as a matrix object
            return Matrix(subtracted)
        except IndexError:
            return 0

    def __add__(self, other_matrix):
        """
                :param other_matrix: This is the other matrix to be subtracted
                :return: Function will return added matrix if possible,
                         otherwise function will return 0
                """
        try:
            added = self.__generateMat(len(self.matrix))
            for index, row in enumerate(self.matrix):
                for column, element in enumerate(row):
                    added[index].append(element + other_matrix.matrix[index][column])

            # return the calculated matrix as a matrix object
            return Matrix(added)
        except IndexError:
            return 0

    def __getDimension(self):
        """
        :return: This function will return the dimensions rows, columns respectively
        """
        rows = len(self.matrix)
        columns = len(self.matrix[0])
        return rows, columns

    @staticmethod
    def __generateMat(rows):
        """
        :param rows: Input the number of rows as an integer
        :return: This will return an empty matrix with given no of rows
        """
        try:
            matrix = []
            for _ in range(int(rows)):
                matrix.append([])
            return matrix
        except ValueError:
            return 0

    @staticmethod
    def getMatrix(*rows, byrows=False):
        """
        :docstring: This function is made for input a matrix from the user,
                    the matrix elements should separated by a space and also,
                    matrix rows should separated by a \n line
        :return: The function will return the matrix in list form
        """
        matrix = []
        if not byrows:
            while True:
                try:
                    matrix.append(list(map(int, input().split())))
                except ValueError:
                    break
            return Matrix(matrix)
        else:
            for _ in range(int(rows[0])):
                matrix.append(list(map(int, input().split())))
            return Matrix(matrix)

    def transpose(self):
        """
        :return: Function will return the transpose of the given matrix
        """
        transpose_ = self.__generateMat(len(self.matrix[0]))
        for row in self.matrix:
            for index, item in enumerate(row):
                transpose_[index].append(item)
        return Matrix(transpose_)

    def cofactor(self):
        """
        :return: This function will return the cofactor of the given matrix
        """
        signs_mat = self.__generateMat(len(self.matrix))
        cofactor = self.__generateMat(len(self.matrix))
        # This code will generate the signs matrix of the matrix
        for index, row in enumerate(self.matrix):
            for column, element in enumerate(row):
                signs_mat[index].append(pow(-1, index + column))

        # by adding signs to the original matrix you will get the cofactor
        for index, row in enumerate(self.minors().matrix):
            for column, element in enumerate(row):
                cofactor[index].append(element * signs_mat[index][column])

        return Matrix(cofactor)

    def __unitMatrix(self, length):
        """
        :param length: This is the dimension of the matrix
        :return: Function will return the unit matrix of given dimensions
        """
        matrix = self.__generateMat(length)
        for row in range(length):
            for column in range(length):
                if row == column:
                    matrix[row].append(1)
                else:
                    matrix[row].append(0)

        return matrix
    
    @staticmethod
    def __find_minor(matrix, index):
        """
        DOCSTRING: this function is to find a single minor matrix from the matrix for given mindex
        matrix: original matrix
        index: is a list [row_index, column_index]
        """
        if len(matrix) > 2:
            minor_matrix = [] # minor matrix will form inside this list

            for row_index in range(0, len(matrix)):
                if row_index != index[0]: # miss the given row
                    proccesing_row = matrix[:][row_index][:] # use slicing to prevent change in original matrix
                    # then remove the element at given index
                    proccesing_row.pop(index[1])
                    # then append it to the minor matrix
                    minor_matrix.append(proccesing_row)

            return minor_matrix
        else:
            return matrix[index[0]][index[1]]
    
    def __determination(self, matrix, propotions = None):
        # this special line is to check whether this function is called for firt time
        if propotions is None:
            propotions = []

        if len(matrix[0]) > 2:
            for item_index, master_item in enumerate(matrix[0]):
                multiplier = master_item * pow(-1, item_index) # mutiplier should switch

                current_minor = self.__find_minor(matrix[:], [0,item_index])
                proportion = multiplier * self.__determination(current_minor, [])

                propotions.append(proportion)

            return sum(propotions) # det is the sum of all propotions
        else:
            # return the det of 2 by 2 simple matrix
            single_det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
            return single_det
        
    def minors(self):
        if len(self.matrix[0]) > 2:
            minor_matrix = [] # minor matrix will grow inside this list
            for row_index, row in enumerate(self.matrix):
                single_row = []
                for column_index, _ in enumerate(row):
                    minor = self.__find_minor(self.matrix, [row_index, column_index])
                    determinant = self.__determination(minor)
                    single_row.append(determinant)
                minor_matrix.append(single_row)
                
            return Matrix(minor_matrix)
        else:
            return Matrix([[self.matrix[1][1], self.matrix[1][0]],[self.matrix[0][1], self.matrix[0][0]]])
    
    @staticmethod
    def __build_fractions_matrix(matrix, devider):
        matrix_build = []
        for row in matrix:
            new_row = []
            for column in row:
                fraction = Fraction(column, devider)
                new_row.append(fraction)
            matrix_build.append(new_row)
        return Matrix(matrix_build)

    def inverse(self, simplify = True):
        cofactor_matrix = self.cofactor() # this is the cofactor
        ct = cofactor_matrix.transpose()
        if simplify:
            return ct * (1/self.det)
        else:
            return self.__build_fractions_matrix(ct.matrix, self.det)

class Vector:
    def __init__(self, vector):
        self.__vector = self.__suitableZ(vector, calculations=True)
        self.type = "R" + str(len(vector))
        self.i = self.__vector[0]
        self.j = self.__vector[1]
        self.k = self.__vector[2]
        self.magnitude = 0
        for coordinate in self.__vector:
            self.magnitude += coordinate ** 2
        self.magnitude = sqrt(self.magnitude)
        self.unitVector = (self.i / self.magnitude, self.j / self.magnitude, self.k / self.magnitude)
        self.unitVector = self.__suitableZ(self.unitVector, calculations=False)

    def __str__(self):
        return str(self.__suitableZ(self.__vector))

    def __add__(self, other):
        x = self.i + other.i
        y = self.j + other.j
        z = self.k + other.k
        return Vector((x, y, z))

    def __neg__(self):
        return Vector((-self.i, -self.j, -self.k))

    def __sub__(self, other):
        x = self.i - other.i
        y = self.j - other.j
        z = self.k - other.k
        return Vector((x, y, z))

    @staticmethod
    def _dotProduct(vector1, vector2):
        i_product = vector1.i * vector2.i
        j_product = vector1.j * vector2.j
        k_product = vector1.k * vector2.k

        dot_product = i_product + j_product + k_product

        return dot_product

    @staticmethod
    def _crossProduct(vector1, vector2):
        i_product = vector1.j * vector2.k - vector1.k * vector2.j
        j_product = -(vector1.i * vector2.k - vector1.k * vector2.i)
        k_product = vector1.i * vector2.j - vector1.j * vector2.i

        return Vector((i_product, j_product, k_product))

    @staticmethod
    def __suitableZ(vector, calculations=True):
        if not calculations:
            if vector[2] == 0:
                converted_vector = (vector[0], vector[1])
                return converted_vector
            else:
                return vector

        else:
            if len(vector) < 3:
                converted_vector = (vector[0], vector[1], 0)
                return converted_vector
            else:
                return vector
