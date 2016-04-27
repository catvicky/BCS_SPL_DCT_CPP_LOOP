#ifndef GUARD_SPL_ALGORITHM
#define GUARD_SPL_ALGORITHM

#include<opencv2\core\core.hpp>
#include<opencv2\opencv.hpp>

double PSNR(const cv::Mat& img1, const cv::Mat& img2);
cv::Mat im2cols(const cv::Mat& x, const int& block_size);
cv::Mat col2im(const cv::Mat&, const int&, const int&, const int&);

cv::Mat BCS_SPL_GenerateProjection(int block_size, double fs);
cv::Mat BCS_SPL_Encoder(const cv::Mat x, const cv::Mat Phi);

cv::Mat DCT2D_Matrix(const int& N);

cv::Mat BCS_SPL_DCT_Decoder(const cv::Mat& y, const cv::Mat& Phi, 
	const int& num_rows, const int& num_cols, const int& block_size);
double SPLIteration(const cv::Mat& y, cv::Mat& x, const cv::Mat Phi, cv::Mat Psi,
	const int block_size, const int, const int, double);
double median(cv::Mat x_check);
double RMS(const cv::Mat& x1, const cv::Mat& x2);

cv::Mat wiener2(cv::Mat in, const int& m, const int& n);

#endif
