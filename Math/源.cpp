
#include "matrix.h"
#include <algorithm>
#include <random>
#include <chrono>
using namespace MDL;

template<size_t Row,size_t Col>
void misnt_data_reader(std::vector<std::pair<Matrix, Matrix>>& outputs)
{
	FILE* fpImg = NULL, *fpLb = NULL;
	fopen_s(&fpImg, "train-images.idx3-ubyte", "rb");
	if (fpImg == NULL )return;
	fopen_s(&fpLb, "train-labels.idx1-ubyte", "rb");
	if (fpLb == NULL)return;

	fseek(fpImg, 16, SEEK_SET);
	fseek(fpLb, 8, SEEK_SET);
	unsigned char* bufimg = new unsigned char[28 * 28 * 60000], *lb = new unsigned char[60000];
	fread(bufimg, 28 * 28 * 60000, 1, fpImg);
	fread(lb, 60000, 1, fpLb);
	for (int i = 0; i < 60000; i++) {
		Matrix ImgMat(Row, Col);
		for (int j = 0; j < 28 * 28; j++)
			ImgMat.Data()[j] = (float)bufimg[i * 28*28 + j] / 255.0f;//
		Matrix LabelDat(10, 1);
		LabelDat.SetValue(0.0f);
		LabelDat(lb[i], 0) = 1.0f;
		outputs.push_back(std::make_pair(std::move(ImgMat), std::move(LabelDat)));
	}
	delete[] bufimg;
	delete[] lb;
	fclose(fpImg);
	fclose(fpLb);
}

float Random_0_1() {
	return (float)rand() / (float)RAND_MAX * 0.001f;
}
float Sigmoid(float x)
{
	return 1.0f / (expf(-x) + 1.0f);
}
float Sigmoid_P(float x)
{
	return Sigmoid(x)*(1.0f - Sigmoid(x));
}
float LeaklyReLU(float x) {
	if (x > 0.0f)return x;
	return 0.1f * x;
}

float LeaklyReLU_P(float x) {
	if (x > 0)return 1.0f;
	return 0.1f;
}
int main()
{
	std::vector<std::pair<Matrix, Matrix>> datas;
	printf("Initializing ...");
	misnt_data_reader<28 * 28, 1>(datas);
	printf("done.\ttraining...\n");

	const int sample_per_minibatch = 25;
	const int hiddens = 100;
	const float learning_rate = 0.01f / sample_per_minibatch;

	Matrix W0(hiddens, 28 * 28, Random_0_1);
	Matrix B0(hiddens, 1, Random_0_1);
	Matrix W1(10, hiddens, Random_0_1);
	Matrix B1(10,1, Random_0_1);


	std::random_device rd;
	std::mt19937 g(rd());

	Matrix partlW0(hiddens, 28 * 28, []() {return 0.0f; });
	Matrix partlB0(hiddens, 1, []() {return 0.0f; });
	Matrix partlW1(10, hiddens, []() {return 0.0f; });
	Matrix partlB1(10, 1, []() {return 0.0f; });

	//training
	//50 
	for (int i = 0; i < 20; i++) {
		auto tp = std::chrono::system_clock::now();
		std::random_shuffle(datas.begin(), datas.begin() + 50000);
		int DataOffset = 0;
		int ErrC = 0;
		for (int j = 0; j < 50000 / sample_per_minibatch; j++) {
			
			partlB0.SetValue(0.0f);
			partlB1.SetValue(0.0f);
			partlW0.SetValue(0.0f);
			partlW1.SetValue(0.0f);
			
			for (int k = 0; k < sample_per_minibatch; k++) {
				auto & samp = datas[DataOffset + k];

				//forward pass
				Matrix A0 = W0 * samp.first + B0;
				Matrix Z0 = A0.Apply(Sigmoid);

				Matrix A1 = W1 * Z0 + B1;
				Matrix Y = A1.Apply(LeaklyReLU);

				int idx = 0;
				for (int i = 0; i < 10; i++) {
					if (Y(i, 0) > Y(idx, 0))idx = i;
				}
				if (samp.second(idx, 0) < 0.9f)ErrC++;//we got wrong

				//backward pass
				Matrix Loss = Y-samp.second;//Output Layer 's loss

				Loss.ElementMultiplyWith(A1.Apply(LeaklyReLU_P));//A1's Loss
				
				partlB1 += Loss;//B1's gradient = Loss * 1
				partlW1 += Loss * ~Z0;//W1's gradient 
			
				Loss = (~W1 * Loss); //Z0 's loss
				Loss.ElementMultiplyWith(A0.Apply(Sigmoid_P));
				partlB0 += Loss;
				partlW0 += Loss * ~samp.first;
			}
			// now update parameters
			W0 -= partlW0 * learning_rate;
			B0 -= partlB0 * learning_rate;
			W1 -= partlW1 * learning_rate;
			B1 -= partlB1 * learning_rate;
		}
		DataOffset += sample_per_minibatch;
		auto ed = std::chrono::system_clock::now();
		auto g = ed - tp;
		//do test
		//just pass forward;
		int errCount = 0;
		for (int j = 0; j < 10000; j++) {
			auto & samp = datas[50000 + j];

			//forward pass
			Matrix A0 = W0 * samp.first + B0;
			Matrix Z0 = A0.Apply(Sigmoid);

			Matrix A1 = W1 * Z0 + B1;
			Matrix Y = A1.Apply(LeaklyReLU);

			int idx = 0;
			for (int i = 0; i < 10; i++) {
				if (Y(i, 0) > Y(idx, 0))idx = i;
			}
			if (samp.second(idx, 0) < 0.9f)errCount++;//we got wrong
		}
		printf("Training %d / 20,loss %f%% on training set,loss %f%% on trst set,training cost %d ms\n", i+1,ErrC / 500.0f, errCount / 100.0f,std::chrono::duration_cast<std::chrono::duration<int,std::milli>>(g).count());
	}

	
}