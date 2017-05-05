#include "DLframeWorks.h"

namespace MDL {

	void WriteString(const std::string& str, FILE* fp) {
		size_t s = str.length();
		fwrite(&s, sizeof(size_t), 1, fp);
		fwrite(str.c_str(), s, 1, fp);
	}

	std::string ReadString(FILE* fp) {
		size_t sz;
		char buf[1024];
		fread(&sz, sizeof(size_t), 1, fp);
		fread(buf, sz, 1, fp);
		buf[sz] = 0;
		return buf;
	}

	void WriteValue(const Value* pValue, FILE* fp) {
		if (pValue == NULL) {
			size_t f = 0;
			fwrite(&f, sizeof(size_t), 1, fp);
		}
		else {
			size_t f = pValue->getShape().rank();
			fwrite(&f, sizeof(size_t), 1, fp);
			for (size_t i = 0; i < f; i++) {
				size_t r = pValue->getShape()[i];
				fwrite(&r, sizeof(size_t), 1, fp);
			}
			pValue->getMatrix().Write(fp);
		}
	}

	Value* ReadValue(FILE* fp)
	{
		size_t f = 0;
		fread(&f, sizeof(size_t), 1, fp);
		if (f == 0)return NULL;
		size_t tmp[128];
		fread(tmp, f * sizeof(size_t), 1, fp);
		Value* pV = new Value({tmp,f});
		pV->getMatrix().Read(fp);
		return pV;
	}
	class VaribleBase : public IVariable
	{
		Shape shape;
		Value* pValue = NULL;
		std::string name;
	public:
		VaribleBase(const Shape& _shape,const std::string& _name) :shape(_shape) ,name(_name){};
		virtual Shape GetShape() override
		{
			return shape;
		}
		virtual void SetValue(Value* v) override
		{
			pValue = v;
		}


		virtual Value* GetValue() override
		{
			return pValue;
		}


		virtual VariableType GetType() override
		{
			return VariableType::Constant;
		}


		virtual void Write(FILE* stream) override
		{
			WriteString(name, stream);
			WriteValue(pValue, stream);
		}


		virtual void Read(FILE* stream) override
		{
			name = ReadString(stream);
			SetValue(ReadValue(stream));
		}


		virtual std::string GetName() override
		{
			return name;
		}


		virtual void GetVariables(VariableType type, std::vector<IVariable *>& outputs) override
		{
			if (GetType() == type)
				outputs.push_back(this);
		}


		virtual bool NeedGradient() override
		{
			return false;
		}


		virtual void Forward(BackpropState& state) override
		{
			state.Outputs.push(pValue);
		}


		virtual void Backprop(BackpropState& state) override
		{
			state.Outputs.pop();
		}

	};

	class InputVar : public VaribleBase
	{

	public:
		virtual VariableType GetType() override
		{
			return VariableType::Input;
		}

	};

	IVariable* InputVarible(const Shape& shape, const std::string& name /*= ""*/)
	{

	}


}