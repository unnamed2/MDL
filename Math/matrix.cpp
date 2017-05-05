#include "matrix.h"
#include <mkl.h>
#include <stdarg.h>
#include <exception>
#include <assert.h>
#pragma comment(lib,"mkl_cntk_p.lib")

std::string format_string(const char* fmt, ...)
{
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsprintf_s(buf, fmt, ap);
	va_end(ap);
	return buf;
}

#define LogicError(...) throw new std::exception(format_string(__VA_ARGS__).c_str())
#define InvalidArgument(...) LogicError(__VA_ARGS__)
#define UNUSED(x) 

#undef foreach_row
#undef foreach_column
#undef foreach_coord
#undef foreach_row_in_submat
#define foreach_row(_i, _m) for (size_t _i = 0; _i < (_m).GetNumRows(); _i++)
#define foreach_column(_j, _m) for (size_t _j = 0; _j < (_m).GetNumCols(); _j++)
#define foreach_coord(_i, _j, _m)                   \
    for (size_t _j = 0; _j < (_m).GetNumCols(); _j++) \
        for (size_t _i = 0; _i < (_m).GetNumRows(); _i++)
#define foreach_row_in_submat(_i, _istart, _iend, _m) for (size_t _i = _istart; _i < min(_iend, (_m).GetNumRows()); _i++)

// this functions returns the index of the first column element in the columnwise array representing matrix with _numRows rows
#define column_s_ind_colwisem(_colNum, _numRows) ((_numRows) * (_colNum))



namespace MDL
{

	Matrix::Matrix():m_numCols(0),m_numRows(0)
	{

	}

	Matrix::Matrix(const size_t numRows, const size_t numCols) : m_numCols(numCols), m_numRows(numRows)
	{
		_data_Buffer.assign(numRows*numCols,0.0f);
	}

	Matrix::Matrix(const size_t numRows, const size_t numCols, ElemType* pArray) : m_numCols(numCols), m_numRows(numRows)
	{
		_data_Buffer.assign(pArray, pArray + numRows * numCols);
	}

	Matrix::Matrix(const size_t numRows, const size_t numCols, std::function<float()> _Initializer) : m_numCols(numCols), m_numRows(numRows),_data_Buffer(numRows*numCols)
	{
		
		for (size_t i = 0; i < numRows * numCols; i++)
			_data_Buffer[i] = _Initializer();
	}

	Matrix& Matrix::operator+=(ElemType alpha)
	{
		return AssignSumOf(alpha, *this);
	}

	Matrix Matrix::operator+(ElemType alpha) const
	{
		Matrix c(GetNumRows(), GetNumCols());
		c.AssignSumOf(alpha, *this);
		return c;
	}

	Matrix& Matrix::AssignSumOf(const ElemType alpha, const Matrix& a)
	{
		if (a.IsEmpty())
			LogicError("AssignSumOf: Matrix a is empty.");

		auto& us = *this;
		if (this != &a)
			RequireSize(a.GetNumRows(), a.GetNumCols());

		long m = (long)GetNumRows(), n = (long)GetNumCols();
#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			// four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				us(i, j) = alpha + a(i, j);
				us(i + 1, j) = alpha + a(i + 1, j);
				us(i + 2, j) = alpha + a(i + 2, j);
				us(i + 3, j) = alpha + a(i + 3, j);
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				us(i, j) = alpha + a(i, j);
			}
		}

		return *this;
	}



	Matrix& Matrix::operator+=(const Matrix& a)
	{
		// if (a.GetNumElements() == 1)
		//    *this += a(0,0);
		// else
		ScaleAndAdd(1, a, *this);

		return *this;
	}

	//if [this] and a have same dimension then OUTPUT=[this]+a
	//if a is a column vector, add to all columns of [this]
	//if a is a row vector, add to all rows of [this]

	Matrix Matrix::operator+(const Matrix& a) const
	{
		if (GetNumElements() == 1)
		{
			Matrix c(a);
			c += (*this)(0, 0);
			return c;
		}
		else if (a.GetNumElements() == 1)
		{
			Matrix c(*this);
			c += a(0, 0);
			return c;
		}
		else
		{
			Matrix c(*this); // this implementation will introduce a copy overhead. but make resue of the code
			c += a;
			return c;
		}
	}


	Matrix& Matrix::AssignSumOf(const Matrix& a, const Matrix& b)
	{
		if (a.GetNumElements() == 1)
		{
			SetValue(b);
			(*this) += a;
		}
		else
		{
			SetValue(a);
			(*this) += b;
		}
		return *this;
	}


	Matrix& Matrix::operator-=(ElemType alpha)
	{
		return AssignDifferenceOf(*this, alpha);
	}


	Matrix Matrix::operator-(ElemType alpha) const
	{
		Matrix c(GetNumRows(), GetNumCols());
		c.AssignDifferenceOf(*this, alpha);
		return c;
	}


	Matrix& Matrix::AssignDifferenceOf(const ElemType alpha, const Matrix& a)
	{
		auto& us = *this;
		if (this != &a)
			RequireSize(a.GetNumRows(), a.GetNumCols());

		long m = (long)GetNumRows(), n = (long)GetNumCols();
#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			// four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				us(i, j) = alpha - a(i, j);
				us(i + 1, j) = alpha - a(i + 1, j);
				us(i + 2, j) = alpha - a(i + 2, j);
				us(i + 3, j) = alpha - a(i + 3, j);
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				us(i, j) = alpha - a(i, j);
			}
		}

		return *this;
	}


	Matrix& Matrix::AssignDifferenceOf(const Matrix& a, const ElemType alpha)
	{
		auto& us = *this;
		if (this != &a)
			RequireSize(a.GetNumRows(), a.GetNumCols());

		long m = (long)GetNumRows(), n = (long)GetNumCols();
#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			// four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				us(i, j) = a(i, j) - alpha;
				us(i + 1, j) = a(i + 1, j) - alpha;
				us(i + 2, j) = a(i + 2, j) - alpha;
				us(i + 3, j) = a(i + 3, j) - alpha;
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				us(i, j) = a(i, j) - alpha;
			}
		}
		return *this;
	}

	//if [this] and a have same dimension then [this]=[this]-a
	//if a is a column vector, minus it from all columns of [this]
	//if a is a row vector, minus it from all rows of [this]

	Matrix& Matrix::operator-=(const Matrix& a)
	{
		ScaleAndAdd(-1, a, *this);

		return *this;
	}

	//if [this] and a have same dimension then output=[this]-a
	//if a is a column vector, minus it from all columns of [this]
	//if a is a row vector, minus it from all rows of [this]

	Matrix Matrix::operator-(const Matrix& a) const
	{
		Matrix c(*this); // this implementation will introduce a copy overhead. but make resue of the code
		c -= a;
		return c;
	}


	Matrix& Matrix::AssignDifferenceOf(const Matrix& a, const Matrix& b)
	{
		if (this != &a)
		{
			RequireSize(a.GetNumRows(), a.GetNumCols());
			SetValue(a);
		}
		(*this) -= b;
		return *this;
	}


	Matrix& Matrix::operator*=(ElemType alpha)
	{
		Scale(alpha, *this);
		return *this;
	}


	Matrix Matrix::operator*(ElemType alpha) const
	{
		Matrix c(GetNumRows(), GetNumCols());
		Scale(alpha, *this, c);
		return c;
	}


	Matrix& Matrix::AssignProductOf(const ElemType alpha, const Matrix& a)
	{
		Scale(alpha, a, *this);
		return *this;
	}

	// [this]=a*b

	Matrix& Matrix::AssignProductOf(const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB)
	{
		if (a.GetNumElements() == 1)
		{
			if (transposeB)
				AssignTransposeOf(b);
			(*this) *= a(0, 0);
		}
		else if (b.GetNumElements() == 1)
		{
			if (transposeA)
				AssignTransposeOf(a);
			(*this) *= b(0, 0);
		}
		else
			Multiply(a, transposeA, b, transposeB, *this);

		return *this;
	}


	Matrix Matrix::operator*(const Matrix& a) const
	{
		auto& us = *this;
		if (GetNumElements() == 1)
		{
			Matrix c;
			c.AssignProductOf(us(0, 0), a);
			return c;
		}
		else if (a.GetNumElements() == 1)
		{
			Matrix c;
			c.AssignProductOf(a(0, 0), us);
			return c;
		}
		else
		{
			Matrix c;
			Multiply(*this, a, c);
			return c;
		}
	}


	Matrix& Matrix::operator/=(ElemType alpha)
	{
		(*this) *= 1 / alpha;
		return (*this);
	}


	Matrix Matrix::operator/(ElemType alpha) const
	{
		return ((*this) * (1 / alpha));
	}

	Matrix Matrix::operator~() const
	{
		Matrix c;
		c.AssignTransposeOf(*this);
		return c;
	}


	void Matrix::RequireSize(const size_t numRows, const size_t numCols, bool growOnly /*= true*/)
	{
		if (GetNumRows() != numRows || GetNumCols() != numCols)
			Resize(numRows, numCols, growOnly);
		
	}

	void Matrix::Resize(const size_t numRows, const size_t numCols, bool growOnly /*= true*/)
	{
		if (GetNumRows() == numRows && GetNumCols() == numCols)
			return;
		_data_Buffer = std::vector<float>(numRows * numCols);
		m_numRows = numRows;
		m_numCols = numCols;
	}

	void Matrix::VerifySize(const size_t rows, const size_t cols)
	{
		if (rows != GetNumRows() || cols != GetNumCols())
			LogicError("VerifySize: expected matrix size %lu x %lu, but it is %lu x %lu",
				rows, cols, GetNumRows(), GetNumCols());
	}

	Matrix Matrix::Apply(std::function<float(float)> func) const 
	{
		Matrix c = *this;
		int Total = m_numCols * m_numRows;
#pragma omp parallel for
		for (int i = 0; i < Total; i++) {
			c._data_Buffer[i] = func(c._data_Buffer[i]);
		}
		return c;
	}

	void Matrix::ScaleAndAdd(ElemType alpha, const Matrix& a, Matrix& c)
	{
		if (a.IsEmpty() || c.IsEmpty())
			LogicError("ScaleAndAdd:  one of the input matrices is empty.");

		if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
		{
			const int m = (int)a.GetNumRows();
			const int n = (int)a.GetNumCols();
			const int len = m * n;
			const int incx = 1;
			const int incy = 1;

			assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow
			if ((int)c.GetNumRows() != m || (int)c.GetNumCols() != n)
				InvalidArgument("Dimension of matrix c does not match dimension of matrix a.");

			{
#pragma warning(suppress : 4244)
				cblas_saxpy(len, alpha, reinterpret_cast<const float*>(a.Data()), incx, reinterpret_cast<float*>(c.Data()), incy);
			}
		}
		else if (a.GetNumElements() == 1) // scalar, add to all elements
		{
			ElemType v = alpha * a(0, 0);
			long m = (long)c.GetNumRows(), n = (long)c.GetNumCols();
#pragma omp parallel for
			for (long j = 0; j < n; j++)
			{
				// four-way unrolling
				for (long i = 0; i < (m & ~3); i += 4)
				{
					c(i, j) += v;
					c(i + 1, j) += v;
					c(i + 2, j) += v;
					c(i + 3, j) += v;
				}
				// handle remaining stuffs
				for (long i = m & ~3; i < m; i++)
				{
					c(i, j) += v;
				}
			}
		}
		else if (a.GetNumCols() == 1) // col vector, add it to all columns
		{
			int m = (int)c.GetNumRows();
			if (m != (int)a.GetNumRows())
				InvalidArgument("To add column vector, rows should match.");

			const ElemType* aBufPtr = a.Data();
			ElemType* cBufPtr = c.Data();
			
			{
#pragma omp parallel for
				foreach_column(j, c)
				{
#pragma warning(suppress : 4244)
					cblas_saxpy(m, alpha, reinterpret_cast<const float*>(aBufPtr), 1, reinterpret_cast<float*>(cBufPtr + c.LocateColumn(j)), 1);
				}
			}
		}
		else // row vector, add it to all rows
		{
			int m = (int)c.GetNumRows();
			int n = (int)c.GetNumCols();
			if (n != (int)a.GetNumCols())
				InvalidArgument("To add row vector, cols should match.");

			const ElemType* aBufPtr = a.Data();
			ElemType* cBufPtr = c.Data();
			
			{
#pragma omp parallel for
				foreach_row(i, c)
				{
#pragma warning(suppress : 4244)
					cblas_saxpy(n, alpha, reinterpret_cast<const float*>(aBufPtr), 1, reinterpret_cast<float*>(cBufPtr + i), m);
				}
			}
		}
	}

	void Matrix::Multiply(const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB, Matrix& c)
	{
		return MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 0.0, c);
	}

	void Matrix::Multiply(const Matrix& a, const Matrix& b, Matrix& c)
	{
		return MultiplyAndWeightedAdd(1.0, a, false, b, false, 0.0, c);
	}

	void Matrix::MultiplyAndWeightedAdd(ElemType alpha, const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB, ElemType beta, Matrix& c)
	{
		if (a.IsEmpty() || b.IsEmpty())
			return;

		int m, n, k, l;
		int lda, ldb, ldc;
		CBLAS_TRANSPOSE mklTransA;
		CBLAS_TRANSPOSE mklTransB;

		if (transposeA)
		{
			m = (int)a.GetNumCols();
			k = (int)a.GetNumRows();
			lda = k;
			mklTransA = CBLAS_TRANSPOSE::CblasTrans;
		}
		else
		{
			m = (int)a.GetNumRows();
			k = (int)a.GetNumCols();
			lda = m;
			mklTransA = CBLAS_TRANSPOSE::CblasNoTrans;
		}

		if (transposeB)
		{
			l = (int)b.GetNumCols();
			n = (int)b.GetNumRows();
			ldb = n;
			mklTransB = CBLAS_TRANSPOSE::CblasTrans;
		}
		else
		{
			l = (int)b.GetNumRows();
			n = (int)b.GetNumCols();
			ldb = l;
			mklTransB = CBLAS_TRANSPOSE::CblasNoTrans;
		}

		assert(m > 0 && k > 0 && l > 0 && n > 0); // converting from size_t to int may cause overflow
		if (k != l)
			InvalidArgument("Matrix::MultiplyAndWeightedAdd : The inner dimensions of a and b must match.");

		if (beta == 0)
			c.RequireSize(m, n);
		else
			c.VerifySize(m, n); // Can't resize if beta != 0

		ldc = (int)c.GetNumRows();

		{
#pragma warning(suppress : 4244)
				cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<const float*>(a.Data()), lda, reinterpret_cast<const float*>(b.Data()), ldb, beta, reinterpret_cast<float*>(c.Data()), ldc);
		}
		
		
	}

	size_t Matrix::LocateElement(const size_t row, const size_t col) const
	{
		assert(row < m_numRows);

		return LocateColumn(col) + row; // matrix in column-wise storage
	}

	size_t Matrix::LocateColumn(const size_t col) const
	{
		assert(col == 0 || col < GetNumCols());
		return col * m_numRows; // matrix in column-wise storage
	}

	void Matrix::SetValue(const ElemType v)
	{
		if (IsEmpty())
			LogicError("SetValue: Matrix is empty.");
		bool isFinite = std::numeric_limits<ElemType>::is_integer || std::isfinite((double)v);
		if (isFinite && v == 0)
		{
			memset(Data(), 0, sizeof(ElemType) * GetNumElements());
		}
		else
		{
			ElemType* bufPtr = Data();
			long m = (long)GetNumElements();
			// 2-way thread parallelism is sufficient for the memory bound
			// operation of just setting the values of an array.
			const unsigned SETVALUE_NUM_THREADS = 2;
			UNUSED(SETVALUE_NUM_THREADS); // in case OMP is turned off.
#pragma omp parallel for num_threads(SETVALUE_NUM_THREADS)
											  // four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				bufPtr[i] = v;
				bufPtr[i + 1] = v;
				bufPtr[i + 2] = v;
				bufPtr[i + 3] = v;
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				bufPtr[i] = v;
			}
		}
	}


	void Matrix::SetValue(const size_t numRows, const size_t numCols, const ElemType* pArray)
	{
		if (pArray == nullptr && numRows * numCols > 0)
			InvalidArgument("Invalid pArray. pArray == nullptr, but matrix is of size %d * %d = %d.", (int)numRows, (int)numCols, (int)(numRows * numCols));

		RequireSize(numRows, numCols);

		if (!IsEmpty())
		{

			ElemType* bufPtr = Data();
			auto& us = *this;
			{
#pragma omp parallel for
				foreach_column(j, us)
				{
					{
#pragma warning(suppress : 4244)
						cblas_scopy((int)numRows, reinterpret_cast<const float*>(pArray + j), (int)numCols, reinterpret_cast<float*>(bufPtr + LocateColumn(j)), 1);
					}
				}
			}

		}
	}


	void Matrix::Read(FILE* stream)
	{
		size_t bufs[2] ;
		fread(bufs, sizeof(bufs), 1, stream);
		RequireSize(bufs[0], bufs[1]);
		fread(Data(), sizeof(float)*m_numCols*m_numRows, 1, stream);
	}

	void Matrix::Write(FILE* stream) const
	{
		size_t bufs[2] = { m_numRows,m_numCols };
		fwrite(bufs, sizeof(bufs), 1, stream);
		fwrite(Data(), sizeof(float)*m_numCols*m_numRows, 1, stream);
	}

	Matrix& Matrix::AssignTransposeOf(const Matrix& a)
	{
		if (this == &a)
			LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

		if (a.IsEmpty())
			LogicError("AssignTransposeOf: Matrix a is empty.");

		RequireSize(a.GetNumCols(), a.GetNumRows());
		long n = (long)a.GetNumCols(), m = (long)a.GetNumRows();

		auto& us = *this;

#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			// four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				us(j, i) = a(i, j);
				us(j, i + 1) = a(i + 1, j);
				us(j, i + 2) = a(i + 2, j);
				us(j, i + 3) = a(i + 3, j);
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				us(j, i) = a(i, j);
			}
		}

		return *this;
	}


	Matrix& Matrix::AssignElementProductOf(const Matrix& a, const Matrix& b)
	{
		if (a.IsEmpty() || b.IsEmpty())
			LogicError("AssignElementProductOf: Matrix is empty.");

		if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
			InvalidArgument("AssignElementProductOf: The input matrix dimensions do not match.");

		auto& us = *this;
		if (this != &a)
			RequireSize(a.GetNumRows(), a.GetNumCols());

		long m = (long)GetNumRows(), n = (long)GetNumCols();
#pragma omp parallel for
		for (long j = 0; j < n; j++)
		{
			// four-way unrolling
			for (long i = 0; i < (m & ~3); i += 4)
			{
				us(i, j) = a(i, j) * b(i, j);
				us(i + 1, j) = a(i + 1, j) * b(i + 1, j);
				us(i + 2, j) = a(i + 2, j) * b(i + 2, j);
				us(i + 3, j) = a(i + 3, j) * b(i + 3, j);
			}
			// handle remaining stuffs
			for (long i = m & ~3; i < m; i++)
			{
				us(i, j) = a(i, j) * b(i, j);
			}
		}
		return *this;
	}

	void Matrix::Scale(ElemType alpha, const Matrix& a, Matrix& c)
	{
		if (a.IsEmpty())
			LogicError("Scale:  Input matrix a is empty.");

		const int m = (int)a.GetNumRows();
		const int n = (int)a.GetNumCols();

		assert(m > 0 && n > 0); // converting from size_t to int may cause overflow
		c.RequireSize(m, n);

		const ElemType* aBufPtr = a.Data();
		ElemType* cBufPtr = c.Data();

		if (alpha == 0)
		{
			memset(cBufPtr, 0, sizeof(ElemType) * c.GetNumElements());
			return;
		}

		long size = (long)c.GetNumElements();
#pragma omp parallel for
		// four-way unrolling
		for (long i = 0; i < (size & ~3); i += 4)
		{
			cBufPtr[i] = alpha * aBufPtr[i];
			cBufPtr[i + 1] = alpha * aBufPtr[i + 1];
			cBufPtr[i + 2] = alpha * aBufPtr[i + 2];
			cBufPtr[i + 3] = alpha * aBufPtr[i + 3];
		}
		// remaining elements
		for (long i = size & ~3; i < size; i++)
		{
			cBufPtr[i] = alpha * aBufPtr[i];
		}
	}

	void Matrix::Scale(ElemType alpha, Matrix& a)
	{
		if (a.IsEmpty())
			LogicError("Scale:  Input matrix a is empty.");

		const int m = (int)a.GetNumRows();
		const int n = (int)a.GetNumCols();
		const int len = m * n;
		const int incx = 1;

		assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow

		if (alpha == 0 && incx == 1)
		{
			memset(a.Data(), 0, sizeof(ElemType) * len);
		}
		else
		{
#pragma warning(suppress : 4244)
			cblas_sscal(len, alpha, reinterpret_cast<float*>(a.Data()), incx);
		}
	}
}
