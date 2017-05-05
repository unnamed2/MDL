#pragma once
#ifndef matrix_h__
#define matrix_h__

#include  <stdio.h>
#include <vector>
#include <functional>

namespace MDL
{
	typedef float ElemType ;

	
	class Matrix
	{
		std::vector<float> _data_Buffer;

	public:
		
		Matrix();
		Matrix(const size_t numRows, const size_t numCols);
		Matrix(const size_t numRows, const size_t numCols, ElemType* pArray);
		Matrix(const size_t numRows, const size_t numCols, std::function<float()> _Initializer);
		Matrix& operator+=(const ElemType alpha);
		Matrix  operator+(const ElemType alpha) const;
		Matrix& AssignSumOf(const ElemType alpha, const Matrix& a);

		Matrix& operator+=(const Matrix& a);
		Matrix  operator+(const Matrix& a) const;
		Matrix& AssignSumOf(const Matrix& a, const Matrix& b);

		Matrix& operator-=(const ElemType alpha);
		Matrix  operator-(const ElemType alpha) const;
		Matrix& AssignDifferenceOf(const ElemType alpha, const Matrix& a);
		Matrix& AssignDifferenceOf(const Matrix& a, const ElemType alpha);

		Matrix& operator-=(const Matrix& a);
		Matrix  operator-(const Matrix& a) const;
		Matrix& AssignDifferenceOf(const Matrix& a, const Matrix& b);

		Matrix& operator*=(const ElemType alpha);
		Matrix  operator*(const ElemType alpha) const;
		Matrix& AssignProductOf(const ElemType alpha, const Matrix& a);

		Matrix  operator*(const Matrix& a) const;
		Matrix& AssignProductOf(const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB); // this = a * b

		Matrix& operator/=(ElemType alpha);
		Matrix  operator/(ElemType alpha) const;

		Matrix operator~() const;
		Matrix& AssignTransposeOf(const Matrix& a);

		Matrix& AssignElementProductOf(const Matrix& a, const Matrix& b);
		Matrix& ElementMultiplyWith(const Matrix& a)
		{
			return AssignElementProductOf(*this, a);
		}

		inline ElemType& operator()(const size_t row, const size_t col)
		{
			return Data()[LocateElement(row, col)];
		}
		inline const ElemType& operator()(const size_t row, const size_t col) const
		{
			return Data()[LocateElement(row, col)];
		}
		void SetValue(const ElemType v);
		void SetValue(const Matrix& deepCopyFrom)
		{
			if (this == &deepCopyFrom)
				return;

			SetValue(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), deepCopyFrom.Data());
		}
		void SetValue(const size_t numRows, const size_t numCols, const ElemType* pArray);

		float* Data() { return &_data_Buffer[0]; }

		const float* Data() const {
			return&_data_Buffer[0];
		}

		void Read(FILE* stream);
		void Write(FILE* stream) const;

		size_t GetNumRows() const { return m_numRows; }
		size_t GetNumCols() const { return m_numCols; }
		size_t GetNumElements() const { return m_numRows * m_numCols; }

		bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }


		void RequireSize(const size_t numRows, const size_t numCols, bool growOnly = true);
		void Resize(const size_t numRows, const size_t numCols, bool growOnly = true);
		void VerifySize(const size_t rows, const size_t cols);


		Matrix Apply(std::function<float(float)> func) const;

		static void ScaleAndAdd(ElemType alpha, const Matrix& a, Matrix& c);
		static void Scale(ElemType alpha, Matrix& a);
		static void Scale(ElemType alpha, const Matrix& a, Matrix& c);
		static void Multiply(const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB, Matrix& c);
		static void Multiply(const Matrix& a, const Matrix& b, Matrix& c);
		static void MultiplyAndWeightedAdd(ElemType alpha, const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB,ElemType beta, Matrix& c);

	protected:
		size_t LocateElement(const size_t i, const size_t j) const;
		size_t LocateColumn(const size_t j) const;

		size_t m_numRows;
		size_t m_numCols;
	};
}
#endif // matrix_h__