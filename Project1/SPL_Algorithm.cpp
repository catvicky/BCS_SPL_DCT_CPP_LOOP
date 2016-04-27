#pragma warning (disable : 4996)
#include<iomanip>
#include<algorithm>
#include<iostream>
#include<vector>
#include<cmath>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <cv.hpp>

#include"SPL_Algorithm.h"

using namespace std;
using namespace cv;

double PSNR(const Mat& img1, const Mat& img2)
{
	Mat s1 = Mat::zeros(img1.rows, img1.cols, CV_32F);
	absdiff(img1, img2, s1);    // |img1 - img2|
	s1 = s1.mul(s1);            // |img1 - img2|^2

	Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(img1.channels() * img1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

Mat BCS_SPL_GenerateProjection(int block_size, double fs)
{
	int N = block_size * block_size;
	int M = int(fs * N);

	// 生成高斯矩阵
	Mat mean = Mat::zeros(1, 1, CV_32F);
	Mat sigma = Mat::ones(1, 1, CV_32F);
	Mat matrix(N, N, CV_32F);
	randn(matrix, mean, sigma);

	// 对矩阵正交化处理
	Mat w = Mat(N, 1, CV_32F);
	Mat u = Mat(N, N, CV_32F);//有改动
	Mat vt = Mat(N, N, CV_32F);
	SVD::compute(matrix, w, u, vt);//u

	Mat Phi = u(Rect(0, 0, u.cols, M));
	return Phi;
}

Mat im2cols(const Mat& x, const int& block_size)
{
	Mat roi(block_size, block_size, CV_32F);
	int M = x.rows, N = x.cols;

	// 当矩阵x可以划分成整数个块时
	int MM = block_size * block_size;
	int NN = M * N / (block_size * block_size);

	// 构造返回矩阵
	Mat ret = Mat::zeros(MM, NN, CV_32F);
	int i = 0; int j = 0;
	int iii = 0;
	for (j = 0; j < M / block_size; ++j) {		// 行
		for (i = 0; i < N / block_size; ++i) {  // 列
			roi = x(Rect(i * block_size, j * block_size, block_size, block_size));
			//cout << roi << endl;

			float temp[4096] = { 0 };	// 最大64分块
			for (int ii = 0; ii < roi.rows; ii++)
			for (int jj = 0; jj < roi.cols; jj++)
				*(temp + ii*roi.cols + jj) = roi.at<float>(jj, ii);
			// 输出测试
			for (int jjj = 0; jjj < block_size * block_size; jjj++) {
				// 把数组temp的值放到ret中
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
	int blocks = int(sqrt(double(NN)));

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
		// 将block_size*block_size添加到x
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
Mat BCS_SPL_Encoder(const Mat x, const Mat Phi)
{
	int M = Phi.rows;
	int N = Phi.cols;
	int block_size = int(sqrt(double(N)));

	// 将图像x按块分割，并将分割后的块排列成列向量，重新组成新的矩阵
	Mat xx = im2cols(x, block_size);

	Mat y = Mat::zeros(M, xx.cols, CV_32F);
	y = Phi * xx;
	return y;
}

Mat DCT2D_Matrix(const int& N)
{
	Mat Psi = Mat::zeros(N * N, N * N, CV_32F);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			Mat X = Mat::zeros(N, N, CV_32F);
			X.at<float>(i, j) = 1;
			Mat x = Mat::zeros(N, N, CV_32F);
			idct(X, x);
			for (int jj = 0; jj < N; ++jj) {
				for (int ii = 0; ii < N; ++ii) {
					//cout <<  x.at<float>(ii, jj) << endl;
					Psi.at<float>(ii + jj * N, j + i * N) = 
										x.at<float>(ii, jj);
				}
			}
		}
	}
	return Psi;
}

Mat BCS_SPL_DCT_Decoder(const Mat& y, const Mat& Phi, const int& num_rows, 
	const int& num_cols, const int& block_size)
{
	int M = Phi.rows;	int N = Phi.cols;
	int Ny = y.cols;
	Mat Psi = Mat::zeros(block_size * block_size, block_size * block_size, CV_32F);
	Psi = DCT2D_Matrix(block_size);

	double lamba = 6;
	double TOL = 0.0001;
	double D_prev = 0;
	int num_factor = 0;
	int max_iterations = 200;	// 最大迭代次数

	// 对x迭代初始化
	Mat x = Mat::zeros(N, Ny, CV_32F);//////////////x的大小
	Mat PhiInv = Mat::zeros(N, M, CV_32F);
	transpose(Phi, PhiInv);
	x = PhiInv * y;

	for (int i = 0; i < max_iterations; ++i) {
		// 返回一个估计值Mat，一个均方根误差double
		double D = SPLIteration(y, x, Phi, Psi, block_size, num_rows, num_cols, lamba);

		if ((D_prev != 0) && (abs(D - D_prev) < TOL)){
			if (num_factor == 4){
				break;
			}
			lamba *= 0.6;
			++num_factor;
		}
		D_prev = D;
		//cout << "第 " << i + 1 << " 次迭代" << endl;
	}
	Mat recon_image = Mat::zeros(num_rows, num_cols, CV_32F);
	recon_image = col2im(x, block_size, num_rows, num_cols);

	return recon_image;
}

double SPLIteration(const cv::Mat& y, cv::Mat& x, const cv::Mat Phi, cv::Mat Psi,
	const int block_size, const int num_rows, const int num_cols, double lamba)
{
	// 将x变回像素域
	Mat xx = Mat::zeros(num_rows, num_cols, CV_32F);
	xx = col2im(x, block_size, num_rows, num_cols);
	
	//维纳滤波
	Mat x_hat = Mat::zeros(num_rows, num_cols, CV_32F);
	x_hat = wiener2(xx, 3, 3); // 未实现？？？？？？？	

	// 分块
	Mat x_hat1 = Mat::zeros(x.rows, x.cols, CV_32F);
	x_hat1 = im2cols(x_hat, block_size);

	Mat PhiInv = Mat::zeros(Phi.cols, Phi.rows, CV_32F);
	transpose(Phi, PhiInv);
	x_hat1 = x_hat1 + PhiInv * (y - Phi * x_hat1);/////???????

	Mat x1 = Mat::zeros(num_rows, num_cols, CV_32F);
	x1 = col2im(x_hat1, block_size, num_rows, num_cols);

	Mat PsiInv = Mat::zeros(Psi.cols, Psi.rows, CV_32F);
	transpose(Psi, PsiInv);
	Mat x_check = Mat::zeros(PsiInv.rows, x_hat1.cols, CV_32F);
	x_check = PsiInv * x_hat1;
	double t = 2.0 * log(num_rows * num_cols);
	double tt = sqrt(t);
	double med = median(x_check);
	double threshold = lamba * tt * med / 0.6745;/////////??????????
	
	for (int i = 0; i < x_check.rows; ++i){
		for (int j = 0; j < x_check.cols; ++j){
			if (abs(x_check.at<float>(i, j)) < threshold)
				x_check.at<float>(i, j) = 0;
		}
	}
	Mat x_bar = Mat::zeros(Psi.rows, x_check.cols, CV_32F);
	x_bar = Psi * x_check;

	x = x_bar + PhiInv * (y - Phi * x_bar);

	Mat x2 = Mat::zeros(num_rows, num_cols, CV_32F);
	x2 = col2im(x, block_size, num_rows, num_cols);

	return RMS(x1, x2);
}
double median(Mat x_check)
{
	double ret;

	vector<double> temp;

	for (int i = 0; i < x_check.rows; ++i){
		for (int j = 0; j < x_check.cols; ++j){
			temp.push_back(abs(x_check.at<float>(i, j)));
		}
	}

	// 排序
	sort(temp.begin(), temp.end());

	typedef vector<double>::size_type vec_sz;
	vec_sz size = temp.size();
	vec_sz mid = size / 2;

	ret = (size % 2 == 0) ? (temp[mid] + temp[mid - 1]) / 2
		: temp[mid];

	return ret;
}
double RMS(const Mat& x1, const Mat& x2)
{
	double ret;

	Mat error = Mat::zeros(x1.rows, x1.cols, CV_32F);
	error = x1 - x2;
	double sum = 0;
	int count = 0;
	for (int i = 0; i < x1.rows; ++i){
		for (int j = 0; j < x1.cols; ++j){
			sum += error.at<float>(i, j) * error.at<float>(i, j);
			count++;
		}
	}
	ret = sum / count;

	return ret;
}
Mat wiener2(Mat in, const int& m, const int& n)
{
	Mat localMean = Mat::zeros(in.rows, in.cols, CV_32F);

	Mat kern = Mat::ones(m, n, CV_32F);
	//cout << kern << endl;

	// 求均值localMean = filter2(ones(m， n), in) / （m * n);
	filter2D(in, localMean, in.depth(), kern);

	localMean = localMean / (m * n);

	// 求方差localVar = filter2(ones(m, n), in.^2) / (m * n) - localMean.^2;
	Mat localVar = Mat::zeros(in.rows, in.cols, CV_32F);
	Mat temp1 = Mat::zeros(in.rows, in.cols, CV_32F);
	multiply(in, in, temp1);

	kern = Mat::ones(m, n, CV_32F);
	filter2D(temp1, localVar, in.depth(), kern);

	Mat temp2 = Mat::zeros(in.rows, in.cols, CV_32F);
	multiply(localMean, localMean, temp2);
	localVar = localVar / (m * n) - temp2;

	// 噪声方差，即localVar的平均值
	Scalar s = sum(localVar);
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	double noise = sse / (double)(localVar.channels() * localVar.total());
	//cout << noise << endl;//

	// 计算结果
	// f = localMean + (max(0, localVar - noise) ./ ...
	//           max(localVar, noise)) .* (in - localMean);

	Mat ret = Mat::zeros(in.rows, in.cols, CV_32F);
	ret = in - localMean;
	Mat temp3 = Mat::zeros(in.rows, in.cols, CV_32F);
	temp3 = localVar - noise;
	Mat temp4 = Mat::zeros(in.rows, in.cols, CV_32F);
	max(temp3, 0, temp4);

	Mat temp5 = Mat::zeros(in.rows, in.cols, CV_32F);
	max(localVar, noise, temp5);
	divide(ret, temp5, ret);
	multiply(ret, temp4, ret);
	ret = ret + localMean;

	return ret;
}