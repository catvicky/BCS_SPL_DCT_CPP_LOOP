#pragma warning (disable : 4996)
#include<algorithm>
#include<iostream>
#include<cmath>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <cv.hpp>

#include"SPL_Algorithm.h"

using namespace std;
using namespace cv;


int main()
{
	double fs = 0.4;	    // ������
	const int block_size = 32;  // �ֿ��С

	// ��ȡԭͼ��
	Mat original_img = imread("Peppers.bmp", 0);	   // �ԻҶ�ͼ�����
	//imshow("ԭͼ��", original_img);
  	//waitKey();	
	//cvDestroyWindow("ԭͼ��");
 	cout << "Begin" << endl;

	int img_rows = original_img.rows;	           // ����ͼ�������
	int img_cols = original_img.cols;	           // ����ͼ�������
	original_img.convertTo(original_img, CV_32F);  // ת����ͳһ����������
	//cout << original_img << endl;

	double psnr = 0;	const int T = 6;  // �ؽ�����
	for (int i = 0; i < T; i++){
		// �õ���˹��������
		Mat Phi = Mat(int(fs * block_size * block_size),
			block_size * block_size, CV_32F);
		Phi = BCS_SPL_GenerateProjection(block_size, fs);
		cout << "Got Guassion Matrix." << endl;

		// ����
		Mat y = Mat(int(fs * block_size * block_size),
			img_cols * img_rows / (block_size * block_size), CV_32F);
		y = BCS_SPL_Encoder(original_img, Phi);
		cout << "Encoder Over." << endl;

		// ����
		int64 startTime = getTickCount();
		Mat recon_img = Mat(img_rows, img_cols, CV_32F);
		recon_img = BCS_SPL_DCT_Decoder(y, Phi, img_rows, img_cols, block_size);
		cout << "Decoder Over." << endl;
		cout << "Elipse time: " << (getTickCount() - startTime) * 1.0 / getTickFrequency()
			<< "s." << endl;
		// ��ֵ�����
		double psnrTemp = 0;
		psnrTemp = PSNR(original_img, recon_img);
		cout << i + 1 << "th. " << "PSNR = " << psnrTemp << endl;
		cout << endl;
		psnr += psnrTemp;
	}
	
	cout << "Average PSNR = " << psnr / T << endl;

	/*
	// ��� 
	double Max = 0;
	double Min = 0;
	minMaxIdx(recon_img, &Min, &Max);
	recon_img.convertTo(recon_img, CV_8U, 255.0 / (Max - Min), 
		-255.0*Min / (Max - Min));
	imshow("�ָ���ͼ��", recon_img);
	waitKey();
	cout << "PSNR = " << psnr << endl;
	cout << "End." << endl;
	*/
	getchar();
	return 0;
}






























/*
Mat im2cols(const Mat& x, const int& block_size)
{
	Mat roi(block_size, block_size, CV_32F);
	int M = x.rows, N = x.cols;

	// ������x���Ի��ֳ���������ʱ
	int MM = block_size * block_size;
	int NN = M * N / (block_size * block_size);

	// ���췵�ؾ���
	Mat ret = Mat::zeros(MM, NN, CV_32F);
	int i = 0; int j = 0;
	int iii = 0;
	for (j = 0; j < M / block_size; ++j) {		// ��
		for (i = 0; i < N / block_size; ++i) {  // ��
			roi = x(Rect(i * block_size, j * block_size, block_size, block_size));
			//cout << roi << endl;

			float temp[4096] = { 0 };	// ���64�ֿ�
			for (int ii = 0; ii < roi.rows; ii++)
			for (int jj = 0; jj < roi.cols; jj++)
				*(temp + ii*roi.cols + jj) = roi.at<float>(jj, ii);
			// �������
			for (int jjj = 0; jjj < block_size * block_size; jjj++) {
				// ������temp��ֵ�ŵ�ret��
				ret.at<float>(jjj, iii) = temp[jjj];
			}
			++iii;
			//cout << ret << endl;
		}
	}
	return ret;
}


Mat col2im(const Mat& y, const int& block_size,
	const int& num_rows, const int& num_cols)
{
	int MM = y.rows, NN = y.cols;
	Mat x = Mat::zeros(num_rows, num_cols, CV_32F);
	int blocks = sqrt(double(NN));

	int a = 0;	int b = 0;
	for (int i = 0; i < NN; ++i) {
		Mat temp = Mat::zeros(block_size, block_size, CV_32F);
		int jj = -1;
		for (int j = 0; j < MM; ++j) {
			if (j % block_size == 0)
				++jj;
			temp.at<float>(j % block_size, jj) = y.at<float>(j, i);
		}
		//cout << temp << endl;
		// ��block_size*block_size��ӵ�x
		if (i != 0 && i % blocks == 0) {
			a += block_size;	b = 0;
		}
		
		///////////
		Mat imageROI = x(Rect(b, a, block_size, block_size));
		addWeighted(imageROI, 0, temp, 1, 0., imageROI);
		////////
		//cout << x << endl;
		b += block_size;
	}
	return x;
}
//��һ��һά��������ʼ������
void InitMat(Mat& m, float* num)
{
	for (int i = 0; i<m.rows; i++)
	for (int j = 0; j<m.cols; j++)
		m.at<float>(i, j) = *(num + i*m.cols + j);
}
 
int main()
{
	const int rows_cols = 6;
	const int nums = rows_cols * rows_cols;
	float num[nums];
	for (int i = 0; i < nums; i++) {
		num[i] = i;
	}
	Mat matrix = Mat::zeros(rows_cols, rows_cols, CV_32F);
	InitMat(matrix, num);
	cout << matrix << endl;
	int block_size = 2;
	Mat xxx = Mat::zeros(block_size * block_size,
		rows_cols * rows_cols / (block_size * block_size), CV_32F);
	xxx = im2cols(matrix, block_size);
	cout << xxx << endl;
	Mat yyy = Mat::zeros(rows_cols, rows_cols, CV_32F);
	yyy = col2im(xxx, block_size, rows_cols, rows_cols);
	cout << yyy << endl;

	getchar();
}
*/