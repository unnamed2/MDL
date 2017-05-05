#pragma once
#include <string>
#include <vector>
#include "../Math/matrix.h"
#include <map>
#include <stack>

namespace MDL
{
	typedef unsigned int u32;
	class Shape
	{
		std::vector<u32> dims;
	public:
		Shape() {}
		Shape(std::initializer_list<size_t> sz) :dims(sz.begin(), sz.end()) {}
		Shape(size_t* pArr, int n) :dims(pArr, pArr + n) {}

		u32& operator[](u32 idx) { return dims[idx]; }
		u32 operator[](u32 idx) const { return dims[idx]; }
		u32 rank() const{ return dims.size(); }
		u32 total_size() const{
			u32 t = 1;
			for (u32 v : dims)t *= v;
			return t;
		}
	};

	class Value
	{
		Matrix matrix;
		Shape shape;
	public:
		
		Value() {}
		Value(const Shape& shape) :matrix(shape[0],shape.total_size()/shape[0]){}
		const Matrix& getMatrix() const {return matrix;}
		Matrix& getMatrix() { return matrix; }
		const Shape& getShape() const { return shape; }
	};

	enum class VariableType
	{
		Parameter,
		Input,
		Constant,
	};

	struct BackpropState
	{
		std::stack<Value*> Outputs;
	};

	class IVariable;
	class IFunction
	{
	public:
		virtual void GetVariables(VariableType type, std::vector<IVariable*>& outputs) = 0;

		virtual bool NeedGradient() = 0;

		virtual void Forward(BackpropState& state) = 0;

		virtual void Backprop(BackpropState& state) = 0;

	};

	class IVariable : public IFunction
	{
	public:
		virtual Shape GetShape() = 0;

		virtual void SetValue(Value*) = 0;

		virtual Value* GetValue() = 0;

		virtual VariableType GetType() = 0;

		virtual void Write(FILE* stream) = 0;

		virtual void Read(FILE* stream) = 0;

		virtual std::string GetName() = 0;
	};

	IVariable* InputVarible(const Shape& shape, const std::string& name = "");
	IVariable* Parameters(const Shape& shape, const std::string& name = "");
	IVariable* Constants(const Shape& shape,float* pData, const std :: string& name);
	
	IFunction* Times(IFunction* oper1, IFunction* oper2);
	IFunction* Add(IFunction* oper1, IFunction* oper2);

	IFunction* Sigmoid(IFunction* operand);
	IFunction* ReLU(IFunction* operand);
	IFunction* LeaklyReLU(IFunction* operand, float scl = 0.1f);

	IFunction* SquareError(IFunction* operand, IFunction* label);
}