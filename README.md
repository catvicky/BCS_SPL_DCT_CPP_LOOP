# BCS-SPL-DCT
这是一个基于二维DCT变换的图像压缩感知重构算法的C++实现程序。

参考文献：http://my.ece.msstate.edu/faculty/fowler/BCSSPL/

<!-- lang: cpp
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
	double fs = 0.4;	    // 采样率
	const int block_size = 32;  // 分块大小

	// 获取原图像
	Mat original_img = imread("Peppers.bmp", 0);	   // 以灰度图像读入
	//imshow("原图像", original_img);
  	//waitKey();	
	//cvDestroyWindow("原图像");
 	cout << "Begin" << endl;

	int img_rows = original_img.rows;	           // 输入图像的行数
	int img_cols = original_img.cols;	           // 输入图像的列数
	original_img.convertTo(original_img, CV_32F);  // 转化成统一的数据类型
	//cout << original_img << endl;

	double psnr = 0;	const int T = 6;  // 重建次数
	for (int i = 0; i < T; i++){
		// 得到高斯测量矩阵
		Mat Phi = Mat(int(fs * block_size * block_size),
			block_size * block_size, CV_32F);
		Phi = BCS_SPL_GenerateProjection(block_size, fs);
		cout << "Got Guassion Matrix." << endl;

		// 编码
		Mat y = Mat(int(fs * block_size * block_size),
			img_cols * img_rows / (block_size * block_size), CV_32F);
		y = BCS_SPL_Encoder(original_img, Phi);
		cout << "Encoder Over." << endl;

		// 解码
		int64 startTime = getTickCount();
		Mat recon_img = Mat(img_rows, img_cols, CV_32F);
		recon_img = BCS_SPL_DCT_Decoder(y, Phi, img_rows, img_cols, block_size);
		cout << "Decoder Over." << endl;
		cout << "Elipse time: " << (getTickCount() - startTime) * 1.0 / getTickFrequency()
			<< "s." << endl;
		// 峰值信噪比
		double psnrTemp = 0;
		psnrTemp = PSNR(original_img, recon_img);
		cout << i + 1 << "th. " << "PSNR = " << psnrTemp << endl;
		cout << endl;
		psnr += psnrTemp;
	}
	
	cout << "Average PSNR = " << psnr / T << endl;

	/*
	// 输出 
	double Max = 0;
	double Min = 0;
	minMaxIdx(recon_img, &Min, &Max);
	recon_img.convertTo(recon_img, CV_8U, 255.0 / (Max - Min), 
		-255.0*Min / (Max - Min));
	imshow("恢复的图像", recon_img);
	waitKey();
	cout << "PSNR = " << psnr << endl;
	cout << "End." << endl;
	*/
	getchar();
	return 0;
}
 --> 
