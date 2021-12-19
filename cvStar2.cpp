// cvStar.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#define	__HOME__
#ifdef  __HOME__
#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>

//#include <opencv2/nonfree/features2d.hpp>	->
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

//#include "opencv2/ocl/ocl.hpp"
#include <opencv2/flann/flann.hpp>
#include <opencv2/calib3d/calib3d.hpp>


//#include <opencv2/stitching/stitching.hpp>
#include "f:/opencv310/opencv/sources/modules/stitching/include/opencv2/stitching.hpp"
#pragma comment(lib, "opencv_stitching310d.lib")

//#include <opencv2/contrib/contrib.hpp>


#include <cmath>
#include <iostream>

#pragma comment(lib, "opencv_videoio310d.lib")	//	VideoCapture
#pragma comment(lib, "opencv_video310d.lib")

#pragma comment(lib, "opencv_core310d.lib")   
#pragma comment(lib, "opencv_highgui310d.lib")   
#pragma comment(lib, "opencv_imgproc310d.lib")   
#pragma comment(lib, "opencv_objdetect310d.lib") //HOGDescriptor
#pragma comment(lib, "opencv_ml310d.lib")


#pragma comment(lib, "opencv_calib3d310d.lib")

#pragma comment(lib, "opencv_imgcodecs310d.lib")
//#pragma comment(lib, "opencv_contrib310d.lib")   

#pragma comment(lib, "opencv_features2d310d.lib")   
//#pragma comment(lib, "opencv_nonfree310d.lib")   
//#pragma comment(lib, "opencv_ocl310d.lib")   
#pragma comment(lib, "opencv_flann310d.lib")

//#pragma comment(lib, "opencv_calib3d310d.lib")
//#pragma comment(lib, "opencv_contrib310d.lib")
#pragma comment(lib, "opencv_xfeatures2d310d.lib")

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#else 

#include <cmath>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;


#pragma comment(lib, "opencv_world411d.lib")

#endif



#define		VPATH		"f:/nagy4/__C_Mintak/cvStar2/cvStar2/"

//#define		DISP_W		600
//#define		DISP_W		800
#define		DISP_W		1100
//#define		DISP_W		1600
//#define		DISP_W		2400
//#define		DISP_W		4000
//#define		DISP_W		100
double sss;

/***************************************************************************************
*
*   function:		rotate
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat rotate(Mat src, double angle)
{
	Mat dst;

	// get rotation matrix for rotating the image around its center
	cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle
	cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();
	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	cv::warpAffine(src, dst, rot, bbox.size());

	return dst;
}
/***************************************************************************************
*
*   function:		FindLoc
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat ResizeProperSize(Mat src, int disp_w)
{
	double		lddisp_w = disp_w;
	int	ww = src.size().width;
	int	hh = src.size().height;
	//double sss;

	if (!disp_w) {
		return src;
	}

	if (src.size().width > src.size().height) {
		sss = src.size().width / lddisp_w;
		ww = src.size().width / sss;
		hh = src.size().height / sss;
	}
	else {
		sss = src.size().height / lddisp_w;
		ww = src.size().width / sss;
		hh = src.size().height / sss;
	}
	resize(src, src, Size(ww, hh));	//	meretezes

	return src;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Rect region_of_interest;
int		width0;
int		height0;
int		winsize = 400;
int		inproc2 = 0;
void CallBackMouseFunc(int event, int x, int y, int flags, void* userdata)
{
	if(event==EVENT_LBUTTONDOWN){
		int	X = x * sss;
		int	Y = y * sss;
		region_of_interest = Rect(
			//min( width0-1, max( 0, (int )(X-winsize/2 )-1))
			//, min( height0-1, max( 0, (int )(Y-winsize/2 )-1))
			min( width0-winsize-1, max( 0, (int )(X-winsize/2 )-1))
			, min( height0-winsize-1, max( 0, (int )(Y-winsize/2 )-1))
			, winsize
			, winsize);
		inproc2 = 1;
		return;
	}
}
/***************************************************************************************
*
*   function:		setGamma
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat setGamma(Mat src, float gamma)
{
	Mat		dst = gamma * src.clone();
return dst;
}
/***************************************************************************************
*
*   function:		addWhite
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat addWhite(Mat src, int white )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;

	for( int i = 0; i < height; i++) {
		for( int j = 0; j < width - 0; j++) {
			Vec3b intensity = src.at<Vec3b>(Point(j, i));

			int Red   = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue  = (float)intensity.val[2];

			dst.at<Vec3b>(Point(j, i))[ 0 ] = max(min( Red   + white, 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 1 ] = max(min( Green + white, 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 2 ] = max(min( Blue  + white, 255 ), 0);

		}
	}

	return dst;
}
/***************************************************************************************
*
*   function:		setKelvin
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
/*Mat setKelvin(Mat src, int val )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;

	for( int i = 0; i < height; i++) {
		for( int j = 0; j < width - 0; j++) {
			if( i > frame && i < height - frame &&  j > frame && j < width - frame ) {
				continue;
			}
			if( i == frame/2 && j == frame/2+5 ) {
				printf("");
			}
			if( i == height - 10 && j == width - 15 ) {
				printf("");
			}
			float	disti = 0;
			float	distj = 0;
			if( i < frame ) {
				disti = frame - i;
			}
			if( i > height - frame ) {
				disti = i - (height - frame);
			}
			if( j < frame ) {
				distj = frame - j;
			}
			if( j > width - frame ) {
				distj = j - (width - frame);
			}
			//float	dist = (disti + distj) / 2.0;
			float	dist = max(disti, distj);
			float	weight = 1. + (val1) * (dist / frame);
			Vec3b intensity = src.at<Vec3b>(Point(j, i));
			int Red   = weight * (float)intensity.val[0];
			int Green = weight * (float)intensity.val[1];
			int Blue  = weight * (float)intensity.val[2];

			dst.at<Vec3b>(Point(j, i))[ 0 ] = max(min( Red  , 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 1 ] = max(min( Green, 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 2 ] = max(min( Blue , 255 ), 0);

			//dst.at<Vec3b>(Point(j, i)) = (1 + val1) * (1 + weight) * intensity;
			//dst.at<Vec3b>(Point(j, i)) = (weight * (float)intensity);
		}
	}

	return dst;
}*/
/***************************************************************************************
*
*   function:		reductVignette
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat reductVignette(Mat src, int frame, float val1, float val2 )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;

	for( int i = 0; i < height; i++) {
		for( int j = 0; j < width - 0; j++) {
			if( i > frame && i < height - frame &&  j > frame && j < width - frame ) {
				continue;
			}
			if( i == frame/2 && j == frame/2+5 ) {
				printf("");
			}
			if( i == height - 10 && j == width - 15 ) {
				printf("");
			}
			float	disti = 0;
			float	distj = 0;
			if( i < frame ) {
				disti = frame - i;
			}
			if( i > height - frame ) {
				disti = i - (height - frame);
			}
			if( j < frame ) {
				distj = frame - j;
			}
			if( j > width - frame ) {
				distj = j - (width - frame);
			}
			//float	dist = (disti + distj) / 2.0;
			float	dist = max(disti, distj);
			float	weight = 1. + (val1) * (dist / frame);
			Vec3b intensity = src.at<Vec3b>(Point(j, i));
			int Red   = weight * (float)intensity.val[0];
			int Green = weight * (float)intensity.val[1];
			int Blue  = weight * (float)intensity.val[2];

			dst.at<Vec3b>(Point(j, i))[ 0 ] = max(min( Red  , 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 1 ] = max(min( Green, 255 ), 0);
			dst.at<Vec3b>(Point(j, i))[ 2 ] = max(min( Blue , 255 ), 0);

			//dst.at<Vec3b>(Point(j, i)) = (1 + val1) * (1 + weight) * intensity;
			//dst.at<Vec3b>(Point(j, i)) = (weight * (float)intensity);
		}
	}

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat setHSV(Mat src, int iR, int iG, int iB)
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;

	Mat		hsv;

	cvtColor( dst, hsv, CV_BGR2HSV);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 0; j++) {
			Vec3b intensity = hsv.at<Vec3b>(Point(j, i));

			int Red = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue = (float)intensity.val[2];

			hsv.at<Vec3b>(Point(j, i))[0] = max(min(Red + iR, 255), 0);
			hsv.at<Vec3b>(Point(j, i))[1] = max(min(Green + iG, 255), 0);
			hsv.at<Vec3b>(Point(j, i))[2] = max(min(Blue + iB, 255), 0);

		}
	}
	cvtColor(hsv, dst, CV_HSV2BGR);

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat setRGB(Mat src, int iR, int iG, int iB)
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;

	Mat		hsv;


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 0; j++) {
			Vec3b intensity = dst.at<Vec3b>(Point(j, i));

			int Red = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue = (float)intensity.val[2];

			dst.at<Vec3b>(Point(j, i))[0] = max(min(Red + iR, 255), 0);
			dst.at<Vec3b>(Point(j, i))[1] = max(min(Green + iG, 255), 0);
			dst.at<Vec3b>(Point(j, i))[2] = max(min(Blue + iB, 255), 0);

		}
	}

	return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat setEdge(Mat src, int kernel_size, int val )
{
	Mat		dst;
	int		width = src.cols;
	int		height = src.rows;

	double	delta = 0;
	Point	anchor;
	Mat		kernel;
	int		ddepth = -1;
	anchor = Point(-1, -1);

	//kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	kernel = Mat(kernel_size, kernel_size, CV_32FC1, Scalar(-(float)val/(float)(kernel_size * kernel_size)));
	kernel.at<float>(kernel_size / 2 + 1, kernel_size / 2 + 1) = val;
	//kernel.at(kernel_size / 2 + 1, kernel_size / 2 + 1) = kernel_size * kernel_size;
	//kernel.at( kernel_size * kernel_size / 2 + 1) = kernel_size * kernel_size;
	//kernel.at( Point( kernel_size / 2 + 1, kernel_size / 2 + 1) ) = kernel_size * kernel_size;
	filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
//int			fr = 0;
int			inproc = 0;
void on_trackbar(int, void*)
{
	//stream1.set(CV_CAP_PROP_POS_FRAMES, fr);

	inproc = 1;
	//printf("a1\n");
}

Mat kernel;
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int edImage( char *fileName, int rot )
{
	int		ret = 0;

	
	Mat		src;
	Mat		dst;

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );
	
	src = imread( fileName );
	if (src.empty()) {
		exit(-1);
	}

	if (rot) {
		src = rotate(src, 270);
	}


	Mat src0 = src.clone();
	width0 = src.cols;
	height0 = src.rows;
	region_of_interest = Rect((width0/2)-winsize/2, (height0/2)-winsize/2, winsize, winsize);
	Mat image_roi = src0(region_of_interest);


	if( 1 ) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}


	int		width = src.cols;
	int		height = src.rows;

	float	val1 = 10;
	float	val2 = 0;
	int		frame = width/6;
	int		framemax = min(width/2, height/2);

	float	valmax = 200;
	float	valdiv = 50;
	int		ftmp = valmax/2+3;

	int		ftmpgamma = 250;
	float	gamma = 1;
	float	gammamax = 900;

	int		blurrad = 3;
	int		blurw1 = 0;
	int		blurw2 = 0;
	int		blurw3 = 20;

	int		tmpwhite = 255;
	int		white = 0;
	int		whitemin = 0;
	int		whitemax = 510;

	int		itmphsv1 = 255;
	int		itmphsv2 = 255;
	int		itmphsv3 = 255;
	int		hsv1max = 510;
	int		hsv2max = 510;
	int		hsv3max = 510;
	int		ihsv1 = 0;
	int		ihsv2 = 0;
	int		ihsv3 = 0;

	int		itmpR = 255;
	int		itmpG = 255;
	int		itmpB = 255;
	int		Rmax = 510;
	int		Gmax = 510;
	int		Bmax = 510;
	int		iR = 0;
	int		iG = 0;
	int		iB = 0;

	int		tmpkernel_size = 0;
	int		kernel_size_max = 20;
	int		kernel_size;
	int		kernel_val = 0;
	int		kernel_val_max = 90;

	//namedWindow("Csuszkak", WINDOW_AUTOSIZE);
	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);
	namedWindow("dst1", WINDOW_AUTOSIZE);

	inproc = 1;
	for( ; ; ) {

		//	Mindig az arc keppel kezdunk
		dst = src.clone();

		
		setMouseCallback( "dst1", CallBackMouseFunc, nullptr );

		createTrackbar("frame", "Csuszkak", &frame, framemax, on_trackbar);
		createTrackbar("val1", "Csuszkak", &ftmp, valmax, on_trackbar);
		createTrackbar("blur1", "Csuszkak", &blurw1, 40, on_trackbar);
		//setTrackbarMin("blur1", "Csuszkak", 50);
		createTrackbar("blur2", "Csuszkak", &blurw2, 40, on_trackbar);
		createTrackbar("blur3", "Csuszkak", &blurw3, 40, on_trackbar);
		createTrackbar("gamma", "Csuszkak", &ftmpgamma, gammamax, on_trackbar);
		createTrackbar("white ", "Csuszkak", &tmpwhite, whitemax, on_trackbar);
		createTrackbar("hue ", "Csuszkak", &itmphsv1, hsv1max, on_trackbar);
		createTrackbar("saturation ", "Csuszkak", &itmphsv2, hsv2max, on_trackbar);
		createTrackbar("brightness ", "Csuszkak", &itmphsv3, hsv3max, on_trackbar);

		createTrackbar("Red", "Csuszkak", &itmpB, Bmax, on_trackbar);
		createTrackbar("Green", "Csuszkak", &itmpG, Gmax, on_trackbar);
		createTrackbar("Blue", "Csuszkak", &itmpR, Rmax, on_trackbar);
		

		createTrackbar("Kernel", "Csuszkak", &tmpkernel_size, kernel_size_max, on_trackbar);
		createTrackbar("Kernelval", "Csuszkak", &kernel_val, kernel_val_max, on_trackbar);


		val1 = (ftmp - valmax/2.)/valdiv;
		gamma = ftmpgamma / 100.;

		if (inproc || inproc2) {
			inproc = 0;
			inproc2 = 0;

			for (int kk = 0; kk < 2; kk++) {
				if (!kk) {
					dst = dst;
				}
				if (kk == 1) {
					image_roi = src0(region_of_interest);
					//dst = image_roi;
					dst = image_roi.clone();
				}
				dst = reductVignette(dst, frame, val1, val2);
				int		width = dst.cols;
				Mat	dstb1 = dst.clone();
				Mat	dstb2 = dst.clone();
				Mat	dstb3 = dst.clone();
				blur(dstb1, dstb1, Size(65, 65));
				blur(dstb2, dstb2, Size(15, 15));
				//blur(dstb2, dstb2, Size(32, 32));
				float 	blurwsum = blurw1 + blurw2 + blurw3;
				dst = blurw1 / blurwsum * dstb1 + blurw2 / blurwsum * dstb2 + blurw3 / blurwsum * dstb3;
				white = tmpwhite - 255;
				dst = setGamma(dst, gamma);
				dst = addWhite(dst, white);
				ihsv1 = itmphsv1 - 255;
				ihsv2 = itmphsv2 - 255;
				ihsv3 = itmphsv3 - 255;
				dst = setHSV(dst, ihsv1, ihsv2, ihsv3);
				iR = itmpR - 255;
				iG = itmpG - 255;
				iB = itmpB - 255;
				dst = setRGB(dst, iR, iG, iB);
				if (tmpkernel_size && kernel_val) {
					kernel_size = 2 * tmpkernel_size + 1;
					dst = setEdge(dst, kernel_size, kernel_val);
				}
				if (!kk) {
					imshow("dst1", dst);
				}
				if (kk == 1) {
					imshow("roi", dst);
				}
			}
			//if (1) {
			//	imshow("dst1", dst);
			//}
			//else {
			//	imshow("dst1", src);
			//}
		}


		if ((ret = waitKey(300)) >= 0) {
			//if (ret == 32) {
			//	waitKey(0);
			//}
			if (ret == 27) {
				//return 0;
				exit(1);
			}
		}
	}
	return 0;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat starEat(Mat src, int tresh )
{
	//Mat		hsvs[3];
	//vector<Mat> channels;
	Mat		gray;
	Mat		dst;
	Mat		med;

	medianBlur( src, med, 5 );

	dst = src - med;

	dst = src - ((float)tresh / 255.0)*dst;
return dst;
	cvtColor( dst, gray, CV_BGR2GRAY );

	//threshold(gray, dst, tresh, 255, CV_THRESH_TOZERO);
	//threshold(gray, dst, tresh+1, tresh, CV_THRESH_TRUNC);
	//	CV_THRESH_BINARY
	//	Csak a kisebbek maradjanak, azokkal foglalkozunk, azokat akarjuk kivonni.
	threshold(gray, dst, tresh, 255, CV_THRESH_TOZERO_INV);
	cvtColor(dst, dst, CV_GRAY2BGR );

	dst = src - ((float)tresh / 255.0) * dst;

return dst;
}
/***************************************************************************************
*
*   function:		SubsGreatAvg
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int SubsGreatAvg( char *fileName, int rot )
{
	int		ret = 0;


	Mat		src;
	Mat		dst;

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	src = imread(fileName);
	//	cv_16u
	//	https://stackoverflow.com/questions/41186294/opencv-normalization-of-16bit-grayscale-image-gives-weak-result
	//src = imread(fileName, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	
	if (src.empty()) {
		exit(-1);
	}

	if (rot) {
		src = rotate(src, 270);
	}


	Mat src0 = src.clone();
	width0 = src.cols;
	height0 = src.rows;
	region_of_interest = Rect((width0 / 2) - winsize / 2, (height0 / 2) - winsize / 2, winsize, winsize);
	Mat image_roi = src0(region_of_interest);


	if (1) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}


	int		width = src.cols;
	int		height = src.rows;


	//namedWindow("Csuszkak", WINDOW_AUTOSIZE);
	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);
	namedWindow("dst1", WINDOW_AUTOSIZE);

	int blurdim = 50;
	int blursubs = 0;
	int	gamma1 = 100;
	int	gamma2 = 100;
	int mediandim = 0;
	int stareat = 0;

	inproc = 1;
	for (;;) {

		//	Mindig az arc keppel kezdunk
		dst = src.clone();


		setMouseCallback("dst1", CallBackMouseFunc, nullptr);

		createTrackbar("gamma1", "Csuszkak", &gamma1, 800, on_trackbar);
		createTrackbar("blur dim", "Csuszkak", &blurdim, 400, on_trackbar);
		createTrackbar("blur subs", "Csuszkak", &blursubs, 200, on_trackbar);
		createTrackbar("gamma2", "Csuszkak", &gamma2, 800, on_trackbar);
		createTrackbar("mediandim", "Csuszkak", &mediandim, 10, on_trackbar);

		createTrackbar("stareat", "Csuszkak", &stareat, 255, on_trackbar);

		if (inproc || inproc2) {
			inproc = 0;
			inproc2 = 0;

			for (int kk = 0; kk < 2; kk++) {
				if (!kk) {
					dst = dst;
				}
				if (kk == 1) {
					image_roi = src0(region_of_interest);
					//dst = image_roi;
					dst = image_roi.clone();
				}
				int		width = dst.cols;

				dst *= ((float)gamma1 / 100.0);

				Mat	dstb1 = dst.clone();
				if( !blurdim ) {
					dstb1 = dst;
				} else {
					blur(dstb1, dstb1, Size(blurdim, blurdim));
					//GaussianBlur( dstb1, dstb1, Size(blurdim, blurdim), 0 );
				}

				dst = dst - ((float)blursubs / 200.0) * dstb1;
				dst *= ((float)gamma2 / 100.0);

				if (mediandim) {
					medianBlur(dst, dst, 2 * mediandim - 1);
					//blur(dst, dst, Size(mediandim, mediandim));
				}

				dst = starEat(dst, stareat);

				if (!kk) {
					imshow("dst1", dst);

					//Mat dststareat = starEat(dst, stareat);
					//imshow("stareat", dststareat);
				}
				if (kk == 1) {
					imshow("roi", dst);
				}
				imshow("src", src);
			}
		}


		if ((ret = waitKey(300)) >= 0) {
			if (ret == 27) {
				exit(1);
			}
		}
	}
return 0;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int histEqualization(char *fileName, int rot)
{
	int		ret;
	Mat		src;
	//Mat		dst;

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	src = imread(fileName);
	if (src.empty()) {
		exit(-1);
	}

	if (rot) {
		src = rotate(src, 270);
	}
//#define		DISP_W		900
	if (1) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}
	imshow("dst1", src);

	Mat		dst = src.clone();
	Mat		hsv;
	cvtColor(dst, hsv, CV_BGR2HSV);
	//cvtColor(dst, hsv, CV_BGR2YCrCb);
	Mat		hsvs[ 3 ];
	vector<Mat> channels;

	split(hsv, hsvs);

	//equalizeHist(hsvs[0], hsvs[0]);
	//equalizeHist(hsvs[1], hsvs[1]);
	equalizeHist(hsvs[2], hsvs[2]);

	channels.push_back(hsvs[0]);
	channels.push_back(hsvs[1]);
	channels.push_back(hsvs[2]);
	merge(channels, hsv);

	cvtColor(hsv, dst, CV_HSV2BGR);
	//dst = .3*dst;
	//cvtColor(hsv, dst, CV_YCrCb2BGR);

	imshow("dst2", dst);

	for (;;) {
		if (waitKey(5) >= 0)
			return 0;
	}

return 0;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int showChannels( Mat dst)
{
	int		ret;

	Mat		hsv;
	//cvtColor(dst, hsv, CV_BGR2HSV);
	//cvtColor(dst, hsv, CV_BGR2YCrCb);
	Mat		chns[ 3 ];

	split( dst, chns);

	imshow("B", chns[0]);
	imshow("G", chns[1]);
	imshow("R", chns[2]);

return 0;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat subsRGB(Mat src, int Rmin, int Gmin, int Bmin )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 0; j++) {
			Vec3b intensity = dst.at<Vec3b>(Point(j, i));

			int Red = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue = (float)intensity.val[2];


			Red   = (float)max((Red   - Rmin), 0) * 255.0 / (255. - (float)Rmin);
			Green = (float)max((Green - Gmin), 0) * 255.0 / (255. - (float)Gmin);
			Blue  = (float)max((Blue  - Bmin), 0) * 255.0 / (255. - (float)Bmin);

			dst.at<Vec3b>(Point(j, i))[0] = max(min(Red, 255), 0);
			dst.at<Vec3b>(Point(j, i))[1] = max(min(Green, 255), 0);
			dst.at<Vec3b>(Point(j, i))[2] = max(min(Blue, 255), 0);

		}
	}

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat subsHSV(Mat src, int hmin, int smin, int vmin, int vmax, int gamma )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;
	float	fgamma = (float)gamma / 100.0;
	Mat		hsv;

	cvtColor( dst, hsv, CV_BGR2HSV);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 0; j++) {
			Vec3b intensity = hsv.at<Vec3b>(Point(j, i));

			int Red = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue = (float)intensity.val[2];

/*			if( Red < hsvmin ) {
				Red = 0;
			}
			if( Red > hsvmax ) {
				Red = 0;
			}
*/			
/*			if( Blue > vmin && Blue < vmin + vmax ) {
				//Blue = Blue/2;
				//Blue = Blue-40;
				//Blue = 0;
			} else {
				//Blue = Blue * (float)(Blue - vmin) / 255.0;
			}
*/
/*			if( Red > hmin && Red < hmin + hmax ) {
				//Blue = Blue/2;
				Blue = 0;
				//Blue - 40;
			} else {
				Green *= 2;
				//Blue = Blue + 40;
				//Blue = (float)(Blue-vmin) * 255.0*(float)Blue/(255.-(float)vmin);
				Blue = (float)(Blue-hmin) * 255.0 / (255.-(float)hmin);
			}
*/
			//Blue = (float)(Blue-hmin) * 255.0 / (255.-(float)hmin);
			//Blue = (float)(Red-hmin) * 255.0 / (255.-(float)hmin);
			//Blue = (float)(Blue-hmin) * hmax / (255.-(float)hmax);
			//Blue = (float)max((Blue-hmin),0) * 255.0 / (255.-(float)hmin) * (float)hmax / 255.0 * fgamma;
			//Red = (float)max((Red - hmin), 0) * 255.0 / (255. - (float)hmin);
			Red = (Red + hmin - 128)%255;

			Green *= (smin / 100.0);

			Blue = (float)max((Blue - vmin), 0) * 255.0 / (255. - (float)vmin);
			Blue *= (float)vmax / 255.0;
			Blue += 255.0 - vmax;
			Blue *= fgamma;

			hsv.at<Vec3b>(Point(j, i))[0] = max(min(Red  , 255), 0);
			hsv.at<Vec3b>(Point(j, i))[1] = max(min(Green, 255), 0);
			hsv.at<Vec3b>(Point(j, i))[2] = max(min(Blue , 255), 0);

		}
	}
	cvtColor(hsv, dst, CV_HSV2BGR);

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int filterSpectrum(char* fileName, int rot)
{
	int		ret;
	Mat		src;
	//Mat		dst;

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	src = imread(fileName);
	//src = imread(fileName, CV_LOAD_IMAGE_ANYDEPTH);
	if (src.empty()) {
		exit(-1);
	}
	int		itype  = src.type();
	int		idepth = src.depth();

	if (rot) {
		src = rotate(src, 270);
	}

	Mat src0 = src.clone();
	width0 = src.cols;
	height0 = src.rows;
	region_of_interest = Rect((width0/2)-winsize/2, (height0/2)-winsize/2, winsize, winsize);
	Mat image_roi = src0(region_of_interest);


	if (1) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}
	imshow("dst1", src);

	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);
	namedWindow("dst1", WINDOW_AUTOSIZE);

	
	//int	hmin = 5;
	//int hminmax = 255;
	//int hmax = 150;
	//int	hmaxmax = 255;
	int	hmin = 128;
	int	smin = 100;
	int	vmin = 0;
	int	vmax = 255;

	int gamma = 100;

	int	Rmin = 0;
	int	Gmin = 0;
	int	Bmin = 0;

	//imshow("image_roi", image_roi);


	inproc = 1;
	for( ; ; ) {

		setMouseCallback( "dst2", CallBackMouseFunc, nullptr );

		createTrackbar("hmin", "Csuszkak", &hmin, 255, on_trackbar);
		createTrackbar("smin", "Csuszkak", &smin, 500, on_trackbar);
		createTrackbar("vmin", "Csuszkak", &vmin, 255, on_trackbar);
		createTrackbar("vmax", "Csuszkak", &vmax, 255, on_trackbar);

		createTrackbar("Rmin", "Csuszkak", &Bmin, 255, on_trackbar);
		createTrackbar("Gmin", "Csuszkak", &Gmin, 255, on_trackbar);
		createTrackbar("Bmin", "Csuszkak", &Rmin, 255, on_trackbar);

		createTrackbar("gamma", "Csuszkak", &gamma, 300, on_trackbar);

		//	Mindig az arc keppel kezdunk
		Mat dst = src.clone();

		if (inproc) {
			inproc = 0;

			dst = subsHSV(dst, hmin, smin, vmin, vmax, gamma );
			dst = subsRGB(dst, Rmin, Gmin, Bmin);
			if( 1 ) {
				image_roi = src0(region_of_interest);
				//dst = image_roi.clone();
				Mat dst_roi = subsHSV(image_roi, hmin, smin, vmin, vmax, gamma);
				dst_roi = subsRGB(dst_roi, Rmin, Gmin, Bmin);
				imshow("dst_roi", dst_roi);
			}

			//dst = ResizeProperSize(dst, DISP_W);

			imshow("dst2", dst);
			if (0) {
				showChannels(dst);
			}
		}
		if( inproc2 ) {
			//	Csak a poziciovaltas szamitasait kell elvegezni
			inproc2 = 0;
			image_roi = src0(region_of_interest);
			//dst = image_roi.clone();
			Mat dst_roi = subsHSV(image_roi, hmin, smin, vmin, vmax, gamma);
			dst_roi = subsRGB(dst_roi, Rmin, Gmin, Bmin);
			imshow("dst_roi", dst_roi);
		}
		if ((ret = waitKey(300)) >= 0) {
			//if (ret == 32) {
			//	waitKey(0);
			//}
			if (ret == 27) {
				return 0;
			}
		}
	}

return 0;
}
/***************************************************************************************
*
*   function:	SplineBlend
Calculate the blending value, this is done recursively.
If the numerator and denominator are 0 the expression is 0.
If the deonimator is 0 the expression is 0
*   arguments:
*   input:
*   output:
*   return:
*
***************************************************************************************/
double SplineBlend( int k, int t, vector<double> &u, double v )
{
	double	value;

	if( t == 1 ) {
		if( ( u[ k ] <= v ) && ( v < u[ k + 1 ] ) ) {
			value = 1;
		} else {
			value = 0;
		}

	} else {

		if( (u[ k + t - 1 ] == u[ k ] ) && ( u[ k + t ] == u[ k + 1 ] ) ) {

			value = 0;

		} else if( u[ k + t - 1 ] == u[ k ] ) {

			value = (double)(u[k + t] - v) / (double)(u[k + t] - u[k + 1]) * SplineBlend(k + 1, t - 1, u, v);

		} else if( u[ k + t ] == u[ k + 1 ] ) {

			value = (double)(v - u[k]) / (double)(u[k + t - 1] - u[k]) * SplineBlend(k, t - 1, u, v);

		} else {

			value = (double)( v - u[ k ] ) / ( u[ k + t - 1 ] - u[ k ] ) * SplineBlend( k, t - 1, u, v) + 
				(double)(u[k + t] - v) / (double)(u[k + t] - u[k + 1]) * SplineBlend(k + 1, t - 1, u, v);
		}
	}

	return value;
}
/***************************************************************************************
*
*   function:		SplineKnots
*   arguments:	
*	description:	Csomovektor
*   input:
*   output:
*   return:
*
***************************************************************************************/
//void SplineKnots( double *u, int n, int t )
void SplineKnots( vector<double> &u, int n, int t )
{
	int		j;

	for( j = 0; j <= n + t; j++ ) {
		if( j < t ) {
			u[ j ] = 0;
		} else if( j <= n ) {
			u[ j ] = j - t + 1;
		} else if( j > n ) {
			u[ j ] = n - t + 2;
		}
		printf("");
	}

	return;
}
/***************************************************************************************
*
*   function:
*	This returns the point "output" on the spline curve.
*	The parameter "v" indicates the position, it ranges from 0 to n-t+2
*   arguments:
*   input:
*   output:
*   return:
*
***************************************************************************************/
//void SplinePoint2D( double *u, int n, int t, double v, D2P *control, D2P *output )
//void SplinePoint2D(vector<double> &u, int n, int t, double v, vector<Point2f> control, Point2f &output)
void SplinePoint2D(vector<double> &u, int n, int t, double v, vector<Point> control, Point2f &output)
{
	int		k;
	double	b;

	output.x = 0;
	output.y = 0;

	for( k = 0; k <= n; k++ ) {
		b = SplineBlend( k, t, u, v );
		output.x += control[ k ].x * b;
		output.y += control[ k ].y * b;
	}

	return;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int XX = -1;
int YY = -1;
void CallBackCurveMouseFunc(int event, int x, int y, int flags, void* userdata)
{
	if(event==EVENT_LBUTTONDOWN){
		XX = x;
		YY = y;
/*		int	X = x * sss;
		int	Y = y * sss;
		region_of_interest = Rect(
			//min( width0-1, max( 0, (int )(X-winsize/2 )-1))
			//, min( height0-1, max( 0, (int )(Y-winsize/2 )-1))
			min( width0-winsize-1, max( 0, (int )(X-winsize/2 )-1))
			, min( height0-winsize-1, max( 0, (int )(Y-winsize/2 )-1))
			, winsize
			, winsize);
		inproc2 = 1;
*/		return;
	}
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
//Mat calcCurve(Mat mCurve, vector<Point2f> &control, vector<int> &weight)
Mat calcCurve(Mat mCurve, vector<Point> &control, vector<int> &weight)
{
	Mat dst = mCurve.clone();
	dst.setTo(Scalar(0, 0, 0));


	int		i100 = 100;

	int		width = mCurve.cols;
	int		height = mCurve.rows;

	Scalar	white(255, 255, 255);

	vector<double>	u;
	//int				nx = 3;
	int				nx = control.size() - 1;
	int				t = 3;
	u.resize(nx + t + 1, 0);
	SplineKnots(u, nx, t);

	if (XX>=0 && YY>=0) {
		XX = XX / (double)width * 100.0;
		YY = 100.0 - YY / (double)height * 100.0;
		double	dist = 0;
		double	distmax = 100000;
		int		idx = -1;
		for (int i = 1; i < nx; i++) {
			dist = sqrt(
				(control[i].x - XX) * (control[i].x - XX)
				+ (control[i].y - YY) * (control[i].y - YY)
				);
			if (dist < distmax) {
				distmax = dist;
				idx = i;
			}
		}
		control[idx].x = XX;
		control[idx].y = YY;
		XX = -1;
		YY = -1;
	}
	control[0].y = control[1].y;
	control[nx].y = control[nx-1].y;

	double increment = ( nx - t + 2 ) / (double)( i100 - 1 );

	for (int i = 0; i <= nx; i = i + 1) {
		line(dst
			, Point(control[i].x / 100.0 * (double)width, height - control[i].y / 100.0 * (double)height - 3)
			, Point(control[i].x / 100.0 * (double)width, height - control[i].y / 100.0 * (double)height + 3)
			, white);
		line(dst
			, Point(control[i].x / 100.0 * (double)width-3, height - control[i].y / 100.0 * (double)height)
			, Point(control[i].x / 100.0 * (double)width+3, height - control[i].y / 100.0 * (double)height)
			, white);
	}


	Point2f pAct;
	Point2f pLast;
	for( int i = 0; i < i100; i = i + 1 ) {
		double	v = i * increment;
		SplinePoint2D( u, nx, t, v, control, pAct );
		if (pAct.x < 0) {
			pAct.x = 0;
		}
		if (pAct.x > 255){
			pAct.x = 255;
		}
		if (pAct.y < 0) {
			pAct.y = 0;
		}
		if (pAct.y > 255) {
			pAct.y = 255;
		}
		pAct.x = 0 + pAct.x / 100.0 * (double)width;
		//pAct.x = 0 + (double)i / 100.00 * (double)width;
		pAct.y = height - pAct.y / 100.0 * (double)height;
		if (i > 0) {
			line( dst, pLast, pAct, white);
		}
		pLast = pAct;
	}
	pAct.x = control[nx].x / 100.0 * (double)width;
	pAct.y = height - control[nx].y / 100.0 * (double)height;
	line(dst, pLast, pAct, white);

	Mat		hsv;


	for( int i = 0; i < 256; i = i + 1 ) {
		double v = (double)i / 255.0 * 100.0 * increment;
		Point2f pAct;
		SplinePoint2D( u, nx, t, v, control, pAct );
		weight[i] = (int)(pAct.y / 100.0 * 255.0);
		if (weight[i] < 0) {
			weight[i] = 0;
		}
		if (weight[i] > 255) {
			weight[i] = 255;
		}
	}

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
Mat DoCurve( Mat src, vector<int>weight )
{
	Mat	dst = src.clone();
	int		width = src.cols;
	int		height = src.rows;


	//dst.zeros(height, width, CV_8UC3 );
	//dst.setTo(Scalar(0, 0, 0));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 0; j++) {
			Vec3b intensity = dst.at<Vec3b>(Point(j, i));

			int Red = (float)intensity.val[0];
			int Green = (float)intensity.val[1];
			int Blue = (float)intensity.val[2];

			Red   = weight[ Red   ];
			Green = weight[ Green ];
			Blue  = weight[ Blue  ];

			dst.at<Vec3b>(Point(j, i))[0] = max(min(Red  , 255), 0);
			dst.at<Vec3b>(Point(j, i))[1] = max(min(Green, 255), 0);
			dst.at<Vec3b>(Point(j, i))[2] = max(min(Blue , 255), 0);

		}
	}

return dst;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int CurveFilter(char* fileName, int rot)
{
	int		ret;
	Mat		src;
	//Mat		dst;

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	src = imread(fileName);
	//src = imread(fileName, CV_LOAD_IMAGE_ANYDEPTH);
	if (src.empty()) {
		exit(-1);
	}
	int		itype  = src.type();
	int		idepth = src.depth();

	if (rot) {
		src = rotate(src, 270);
	}

	Mat src0 = src.clone();
	width0 = src.cols;
	height0 = src.rows;
	region_of_interest = Rect((width0/2)-winsize/2, (height0/2)-winsize/2, winsize, winsize);
	Mat image_roi = src0(region_of_interest);


	if (1) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}
	imshow("dst1", src);

	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);
	namedWindow("dst1", WINDOW_AUTOSIZE);


	namedWindow("Curve", WINDOW_AUTOSIZE);
	//resizeWindow("Curve", 600, 600);


	int gamma = 100;

	int	x0 = 0;
	int	y0 = 0;
	int	x1 = 0;
	int	y1 = 0;
	int	x2 = 33;
	int	y2 = 13;
	int x3 = 66;
	int	y3 = 66;
	int	x4 = 100;
	int	y4 = 100;
	int	x5 = 100;
	int	y5 = 100;



	Mat	mCurve(400, 400, CV_8UC3, Scalar(0,0,0));
	//mCurve.zeros(200,200,CV_8UC3);
	
	setMouseCallback( "Curve", CallBackCurveMouseFunc, nullptr );
	vector<int>weight;
	weight.resize(256, 0);

	//vector<Point2f> control;
	vector<Point> control;

	control.push_back(Point2f(x0, y0));
	control.push_back(Point2f(x1, y1));
	control.push_back(Point2f(x2, y2));
	control.push_back(Point2f(x3, y3));
	control.push_back(Point2f(x4, y4));
	control.push_back(Point2f(x5, y5));




	//imshow("image_roi", image_roi);


	inproc = 1;
	for( ; ; ) {

		setMouseCallback( "dst2", CallBackMouseFunc, nullptr );

		createTrackbar("gamma", "Csuszkak", &gamma, 300, on_trackbar);

/*		createTrackbar("x1", "Csuszkak", &x1, 100, on_trackbar);
		createTrackbar("y1", "Csuszkak", &y1, 100, on_trackbar);
		createTrackbar("x2", "Csuszkak", &x2, 100, on_trackbar);
		createTrackbar("y2", "Csuszkak", &y2, 100, on_trackbar);

		createTrackbar("x3", "Csuszkak", &x3, 100, on_trackbar);
		createTrackbar("y3", "Csuszkak", &y3, 100, on_trackbar);
		createTrackbar("x4", "Csuszkak", &x4, 100, on_trackbar);
		createTrackbar("y4", "Csuszkak", &y4, 100, on_trackbar);
*/
		createTrackbar("x1", "Csuszkak", &control[1].x, 100, on_trackbar);
		createTrackbar("y1", "Csuszkak", &control[1].y, 100, on_trackbar);
		createTrackbar("x2", "Csuszkak", &control[2].x, 100, on_trackbar);
		createTrackbar("y2", "Csuszkak", &control[2].y, 100, on_trackbar);

		createTrackbar("x3", "Csuszkak", &control[3].x, 100, on_trackbar);
		createTrackbar("y3", "Csuszkak", &control[1].y, 100, on_trackbar);
		createTrackbar("x4", "Csuszkak", &control[4].x, 100, on_trackbar);
		createTrackbar("y4", "Csuszkak", &control[4].y, 100, on_trackbar);

		if (control[1].x >= control[2].x) {
			control[1].x = control[2].x - 1;
		}
		if (control[2].x >= control[3].x) {
			control[2].x = control[3].x - 1;
		}
		if (control[3].x >= control[4].x) {
			control[4].x = control[4].x - 1;
		}


		Mat dst = src.clone();
		mCurve = calcCurve( mCurve, control, weight);
		dst = DoCurve( dst, weight );

		imshow("Curve", mCurve);
		imshow("dst", dst);


		//	Mindig az arc keppel kezdunk

		if (inproc) {
/*			inproc = 0;

			dst = subsHSV(dst, hmin, smin, vmin, vmax, gamma );
			dst = subsRGB(dst, Rmin, Gmin, Bmin);
			if( 1 ) {
				image_roi = src0(region_of_interest);
				Mat dst_roi = subsHSV(image_roi, hmin, smin, vmin, vmax, gamma );
				dst_roi = subsRGB(dst_roi, Rmin, Gmin, Bmin);
				imshow("dst_roi", dst_roi);
			}

			//dst = ResizeProperSize(dst, DISP_W);

			imshow("dst2", dst);
			if (0) {
				showChannels(dst);
			}
*/		}
		if( inproc2 ) {
/*			//	Csak a poziciovaltas szamitasait kell elvegezni
			inproc2 = 0;
			image_roi = src0(region_of_interest);
			Mat dst_roi = subsHSV(image_roi, hmin, smin, vmin, vmax, gamma );
			dst_roi = subsRGB(dst_roi, Rmin, Gmin, Bmin);
			imshow("dst_roi", dst_roi);
*/		}
		if ((ret = waitKey(300)) >= 0) {
			//if (ret == 32) {
			//	waitKey(0);
			//}
			if (ret == 27) {
				return 0;
			}
		}
	}

	return 0;

}
/***************************************************************************************
*
*   function:		CallBackGombsorMouseFunc
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int Xgomb = -1;
int Ygomb = -1;
void CallBackGombsorMouseFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN){
		XX = x;
		YY = y;
		return;
	}
}
int pointInRect(Rect rct, int x, int y)
{
	if(    x >= rct.x && x <= (rct.x + rct.width )
		&& y >= rct.y && y <= (rct.y + rct.height)
	) {
		return 1;
	} else {
		return 0;
	}
}
/***************************************************************************************
*
*   function:		cv_16UNormalization
*   arguments:
*	description:	histEqualization
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int cv_16UNormalization(char *fileName, int rot)
{
	int		ret = 0;
	Mat		src;
	Mat		dst;
	Mat		dst2;

	int		imtype = CV_32FC3;
	//int		imtype = CV_8UC3;
	Mat		canvas;

	Mat	mgombsor(400, 400, CV_8UC3, Scalar(64, 0, 0));
	Rect buttonCrop = Rect(0, 0, 100, 40);
	mgombsor(buttonCrop ) = Vec3b(200, 200, 200);
	putText(mgombsor(buttonCrop), "Crop", Point(buttonCrop.width*0.35, buttonCrop.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));

	setMouseCallback("gombsor", CallBackGombsorMouseFunc, nullptr);
	imshow("gombsor", mgombsor);

	//mCurve.zeros(200,200,CV_8UC3);


	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);

	//canvas

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	//src = imread(fileName);
	//	cv_16u
	//	https://stackoverflow.com/questions/41186294/opencv-normalization-of-16bit-grayscale-image-gives-weak-result
	//	https://stackoverflow.com/questions/17345967/normalize-pixel-values-between-0-and-1
	//	https://opencv.programmingpedia.net/en/tutorial/1957/pixel-access
	//	https://arato.inf.unideb.hu/szeghalmy.szilvia/kepfeld/diak/szsz_ocv_gyak2.pdf
	//src = imread(fileName);
	src = imread(fileName, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

	if (src.empty()) {		
		exit(-1);
	}

	//	3x32bitess float
	src.convertTo(src, CV_32FC3);
	//	Nem kell leskalazni, bar az is bizonyos mostmar, hogy akkor is marad benne ertekes adat
	//src.convertTo(src, CV_32FC3, 1.0 / 255.0);

	if (rot) {
		src = rotate(src, 270);
	}


	if (1) {
		src = ResizeProperSize(src, DISP_W);
		//src = ResizeProperSize(src, DISP_W*2);
	}
	imshow("dst1", src);


	//normalize(src, dst, 0, 65535.0, cv::NORM_MINMAX, CV_16U);
	//normalize(src, dst, 0, NULL, cv::NORM_MINMAX, CV_32F);

	//equalizeHist(src, dst);
	//dst = normalize(src, None, 0, 1, NORM_MINMAX, cv2.CV_32F);
	//normalize(src, dst, 0.0, 1.0, NORM_MINMAX, imtype);
	//normalize(src, dst, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
	//normalize(src, dst, 0.0, 255.0, NORM_MINMAX, imtype);

	int		width = src.cols;
	int		height = src.rows;

	//float gain = 0.30;
	//float gain = 6.30;
	float gain = 1.00;

	int igain = 100;
	int igainmax = 300;
	createTrackbar("frame", "Csuszkak", &igain, igainmax, on_trackbar);

	dst = src.clone();
	dst2 = src.clone();
	Vec3f  fact;
	{
		Vec3f  intensity = src.at<Vec3f>(src.rows / 2, src.cols /2);
		float Red = (float)intensity.val[0];
		float Green = (float)intensity.val[1];
		float Blue = (float)intensity.val[2];
		//fact[0] = 1.0 / Red * gain;
		//fact[1] = 1.0 / Green * gain;
		//fact[2] = 1.0 / Blue * gain;

		fact[0] = 0.0;
		fact[1] = 0.0;
		fact[2] = 0.0;

		if( !(Red + Green + Blue)) {
			fact[0] = 1.0 * gain;
			fact[1] = 1.0 * gain;
			fact[2] = 1.0 * gain;
		}
		//	A reszuk a teljes aranyaban
		if( Red ) {
			fact[0] = (Red + Green + Blue) / Red * gain;
		}
		if( Green ) {
			fact[1]	= (Red + Green + Blue) / Green * gain;
		}
		if( Blue ) {
			fact[2] =  (Red + Green + Blue) / Blue * gain;
		}

	}

	float rmax = 0;
	float gmax = 0;
	float bmax = 0;
	if( 1 ) {
		for (int i = 0; i < dst.cols; i++) {
			for (int j = 0; j < dst.rows - 0; j++) {
				Vec3f  intensity = dst.at<Vec3f>(j, i);

				float Red = fact[ 0 ] * (float)intensity.val[0];
				float Green = fact[ 1 ] * (float)intensity.val[1];
				float Blue = fact[ 2 ] * (float)intensity.val[2];

				if( rmax < Red ) {
					rmax = Red;
				}
				if( gmax < Green ) {
					gmax = Green;
				}
				if( bmax < Blue ) {
					bmax = Blue;
				}
					
				Vec3f  intensity2 = { Red, Green, Blue };

				dst.at<Vec3f>(j, i) = intensity2;
				//dst.at<Vec3f>(j, i)[0] = Red * 3.0;
				//dst.at<Vec3f>(j, i)[1] = Green * 3.0;
				//dst.at<Vec3b>(j, i)[2] = Blue * 3.0;

			}
		}
	}
	Mat	dst0 = dst.clone();
	Mat	dstmask = dst.clone();

	float	fmax = 0.125;
	float	fmax2 = 30;
	float	fsub = 0.102;
	if( 1 ) {
		for (int i = 0; i < dst.cols; i++) {
			for (int j = 0; j < dst.rows - 0; j++) {
				Vec3f  intensity = dst.at<Vec3f>(j, i);

				float Red = (float)intensity.val[0];
				float Green = (float)intensity.val[1];
				float Blue = (float)intensity.val[2];

				//
				//	MASK
				//
				if( Red < fmax && Green < fmax && Blue < fmax   &&   Red < 1.1*Green && Red < 1.1*Blue) {
					dstmask.at<Vec3f>(j, i) = {1,1,1};
					//dstmask.at<Vec3f>(j, i) = {Red, Green, Blue};
				} else {
					dstmask.at<Vec3f>(j, i) = {0,0,0};
				}

				//if( Red < fmax && Green < fmax && Blue < fmax   &&   Red < 1.1*Green && Red < 1.1*Blue) {
				//if( Red + Green + Blue < 3 * fmax ) {
					Red = fmax2 * ((float)intensity.val[0] - fsub);
					Green = fmax2 * ((float)intensity.val[1]- fsub);
					Blue = fmax2 * ((float)intensity.val[2] - fsub);
					/*
					Red = 1;//fmax2 * ((float)intensity.val[0] - fsub);
					Green = 1;//fmax2 * ((float)intensity.val[1]- fsub);
					Blue = 1;//fmax2 * ((float)intensity.val[2] - fsub);
					*/
				//} else {
					if( 0 ) {
						Red = 0;
						Green = 0;
						Blue = 0;
					}
				//}

				Vec3f  intensity2 = { Red, Green, Blue };

				dst.at<Vec3f>(j, i) = intensity2;
				//dst.at<Vec3f>(j, i)[0] = Red * 3.0;
				//dst.at<Vec3f>(j, i)[1] = Green * 3.0;
				//dst.at<Vec3b>(j, i)[2] = Blue * 3.0;


			}
		}
	}


	//
	//	EDGE
	//
	Mat	dst3;
	Mat	dst4;
	Mat dst5 = dst.clone();// = dstmask * dst;

	GaussianBlur( dst , dst3, cv::Size(0, 0), 3 );
	//addWeighted( dst, 2.5, dst3, -1.5, 0, dst4 );
	addWeighted( dst, 1.0, dst3, -1.0, 0, dst4 );
	double	fmin = 0.05;
	if( 1 ) {
		for (int i = 0; i < dst4.cols; i++) {
			for (int j = 0; j < dst4.rows - 0; j++) {
				Vec3f  intensity = dst4.at<Vec3f>(j, i);
				Vec3f  intensity_dst = dst.at<Vec3f>(j, i);
				//if( intensity.val[0] < 0.99  &&  intensity.val[1] < 0.99  &&  intensity.val[2] < 0.99 ) {
				if( intensity.val[0] > fmin  ||  intensity.val[1] > fmin  ||  intensity.val[2]  > fmin ) {
					dst4.at<Vec3f>(j, i) = {1,1,1};
					dst5.at<Vec3f>(j, i) = {0,0,0};
				} else {
					dst4.at<Vec3f>(j, i) = {0,0,0};
					dst5.at<Vec3f>(j, i) = intensity_dst;
				}
			}
		}
	}
	Mat		ones( dst.size(), CV_8UC3 );
	ones.setTo( 255 );
	ones.convertTo( ones, CV_32FC3);
	//dst4 = ones - dst4;
	//dst4 = {1,1,1} - dst4;
	//cvtColor( dst4, dst4, COLOR_BGR2GRAY );
	//dst3.convertTo( dst3, CV_8UC1);
	//threshold( dst4, dst4, 1,255,THRESH_BINARY );



	//cvtColor( dstmask, dstmask, COLOR_BGR2GRAY );
	//threshold( dstmask, dstmask, 100,255,THRESH_BINARY );

	//dstmask.convertTo(dstmask, CV_32FC1);
	//dstmask = 1 - dstmask;


	//dst5 = dstmask * dst;
	//bitwise_and( dstmask, dst, dst5 );

	erode( dstmask, dstmask, getStructuringElement( MORPH_ELLIPSE, Size(4, 4) ) );
	if( 1 ) {
		for (int i = 0; i < dst.cols; i++) {
			for (int j = 0; j < dst.rows - 0; j++) {
				//printf("\ni=%d, j=%d", i, j);
				//if( i>=dst.cols && j>=dst.rows) {
				//	printf("");
				//}
				Vec3f  intensity_mask = dstmask.at<Vec3f>(j, i);
				//float  intensity_mask = dstmask.at<float*>(j, i);
				Vec3f  intensity_dst = dst.at<Vec3f>(j, i);

				if( intensity_mask.val[0]  &&  intensity_mask.val[1] && intensity_mask.val[2] ) {
					//intensity_dst = 
//					dst5.at<Vec3f>(j, i) = intensity_dst;
				} else {
					dst5.at<Vec3f>(j, i) = {0,0,0};
				}			
			}
		}
	}
	dstmask = 1 - dstmask;
	cvtColor( dstmask, dstmask, COLOR_BGR2GRAY );


/*
	Mat		dstnot;
	//bitwise_not( dst, dstnot );
	//dstnot = 1 - dst;
	//bitwise_not( dstnot, dstnot );
	//dstnot = 1 - dstnot;
	//bitwise_not( dstnot, dst );
	dstnot = dst.clone();
	bitwise_xor( dstnot, 1, dstnot );
	bitwise_and( dstnot, 1, dstnot );
	//dst = 1 - dst;
	//bitwise_not( dstnot, dstnot );
	dstnot = 1 - dstnot;
	//bitwise_and( dstnot, dst, dst );
	//dst = dstnot * dst;
	dst = dstnot.clone(); 

	//erode( dst, dst, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
*/
/*
	//erode( dst, dst, getStructuringElement( MORPH_ELLIPSE, Size(4, 4) ) );
	erode( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	erode( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	erode( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	erode( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
	dilate( dst2, dst2, getStructuringElement( MORPH_ELLIPSE, Size(2, 2) ) );
*/
	//bitwise_not( dst, dst );
	//dst = 1 & dst;
	//bitwise_xand( dst, dst2, dst );

	if( 0 ) {
		for (int i = 0; i < dst.cols; i++) {
			for (int j = 0; j < dst.rows - 0; j++) {
				Vec3f  intensity = dst.at<Vec3f>(j, i);

				float Red = fact[ 0 ] * (float)intensity.val[0];
				float Green = fact[ 1 ] * (float)intensity.val[1];
				float Blue = fact[ 2 ] * (float)intensity.val[2];

				if( rmax < Red ) {
					rmax = Red;
				}
				if( gmax < Green ) {
					gmax = Green;
				}
				if( bmax < Blue ) {
					bmax = Blue;
				}
					
				Vec3f  intensity2 = { Red, Green, Blue };

				dst.at<Vec3f>(j, i) = intensity2;
				//dst.at<Vec3f>(j, i)[0] = Red * 3.0;
				//dst.at<Vec3f>(j, i)[1] = Green * 3.0;
				//dst.at<Vec3b>(j, i)[2] = Blue * 3.0;

			}
		}
	}

	for (;;) {

		imshow("dst0", dst0 );
		imshow("dst", dst );
		imshow("dst3", dst3 );
		imshow("dst4", dst4 );
		imshow("dst5", dst5 );
		imshow("dstmask", dstmask );

		if ((ret = waitKey(30)) >= 0) {
			if (ret == 27) {
				return 0;
			}
		}

	}

return 0;
}
/***************************************************************************************
*
*   function:		
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int whiteBalance( Mat src, Mat &dst, double gain, double &rmin, double &gmin, double &bmin, double &rmax, double &gmax, double &bmax )
{
	dst = src.clone();
	//	WhiteBalance faktorok szamitasa
	Vec3f  fact;
	{
		Vec3f  intensity = src.at<Vec3f>(src.rows / 2, src.cols /2);
		float Red = (float)intensity.val[0];
		float Green = (float)intensity.val[1];
		float Blue = (float)intensity.val[2];
		fact[0] = 0.0;
		fact[1] = 0.0;
		fact[2] = 0.0;
		if( !(Red + Green + Blue)) {
			fact[0] = 1.0 * gain;
			fact[1] = 1.0 * gain;
			fact[2] = 1.0 * gain;
		}
		//	A reszuk a teljes aranyaban
		if( Red ) {
			fact[0] = (Red + Green + Blue) / Red * gain;
		}
		if( Green ) {
			fact[1]	= (Red + Green + Blue) / Green * gain;
		}
		if( Blue ) {
			fact[2] =  (Red + Green + Blue) / Blue * gain;
		}

	}

	//	WhiteBalance vegrahajtasa, kozben akkor mar megkeresem a minimum es 
	//	maximum ertekeket (de ez nem szinkomponensenkent kellene csinalni valszeg) 
	rmin = 1;
	gmin = 1;
	bmin = 1;
	rmax = 0;
	gmax = 0;
	bmax = 0;
	if( 1 ) {
		for (int i = 0; i < dst.cols; i++) {
			for (int j = 0; j < dst.rows - 0; j++) {
				Vec3f  intensity = dst.at<Vec3f>(j, i);

				float Red = fact[ 0 ] * (float)intensity.val[0];
				float Green = fact[ 1 ] * (float)intensity.val[1];
				float Blue = fact[ 2 ] * (float)intensity.val[2];

				if( rmax < Red ) {
					rmax = Red;
				}
				if( gmax < Green ) {
					gmax = Green;
				}
				if( bmax < Blue ) {
					bmax = Blue;
				}

				if( rmin > Red ) {
					rmin = Red;
				}
				if( gmin > Green ) {
					gmin = Green;
				}
				if( bmin > Blue ) {
					bmin = Blue;
				}

				Vec3f  intensity2 = { Red, Green, Blue };

				dst.at<Vec3f>(j, i) = intensity2;
			}
		}
	}


return 1;
}
/***************************************************************************************
*
*   function:		
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int starMaskTresh( Mat src, Mat &dst, Mat &dstmask, double thresval, double thresmul, double fmul, double fsub )
{
	for (int i = 0; i < src.cols; i++) {
		for (int j = 0; j < src.rows - 0; j++) {
			Vec3f  intensity = src.at<Vec3f>(j, i);

			float Red = (float)intensity.val[0];
			float Green = (float)intensity.val[1];
			float Blue = (float)intensity.val[2];

			//
			//	MASK
			//
			if( 
				( Red > thresval 
				|| Green > thresval 
				|| Blue > thresval
				)
			//	Ezt nem tudtam mire hasznalni. A celja az lett volna, hogy az eles, egyszinu halokat
			//	megtalaljam.
			//	||
			//	(   Red > thresmul * Green 
			//	 || Red > thresmul * Blue
			//	 || Green > thresmul * Red
			//	 || Green > thresmul * Blue
			//	 || Blue > thresmul * Red
			//	 || Blue > thresmul * Green
			//	) 
			) {
				dstmask.at<Vec3f>(j, i) = {0,0,0};
			} else {
				dstmask.at<Vec3f>(j, i) = {1,1,1};
			}

			//if( Red < fmax && Green < fmax && Blue < fmax   &&   Red < 1.1*Green && Red < 1.1*Blue) {
			//if( Red + Green + Blue < 3 * fmax ) {
				Red = fmul * ((float)intensity.val[0] - fsub);
				Green = fmul * ((float)intensity.val[1]- fsub);
				Blue = fmul * ((float)intensity.val[2] - fsub);
			//} else {
				if( 0 ) {
					Red = 0;
					Green = 0;
					Blue = 0;
				}
			//}

			Vec3f  intensity2 = { Red, Green, Blue };

			dst.at<Vec3f>(j, i) = intensity2;
		}
	}
return 1;
}
/***************************************************************************************
*
*   function:		
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int get3x3Avegrage( Mat src, int x, int y, Vec3f &intensity ) {
	Vec3f	intensity_tmp0;
	Vec3f	intensity_tmp1;
	int		tmp0; 
	int		isav = 4;

	intensity.val[0] = 0;
	intensity.val[1] = 0;
	intensity.val[2] = 0;

	tmp0 = 0;
	for( int i = max(0,x-isav); i < min( x+isav, src.cols); i++ ) {
		for( int j = max(0,y-isav); j < min( y+isav, src.rows); j++ ) {
			intensity_tmp0 = src.at<Vec3f>(j, i);
			if( intensity_tmp0.val[0]  ||  intensity_tmp0.val[1] || intensity_tmp0.val[2] ) {
				intensity_tmp1.val[ 0 ] = intensity_tmp1.val[ 0 ] + intensity_tmp0.val[ 0 ];
				intensity_tmp1.val[ 1 ] = intensity_tmp1.val[ 1 ] + intensity_tmp0.val[ 1 ];
				intensity_tmp1.val[ 2 ] = intensity_tmp1.val[ 2 ] + intensity_tmp0.val[ 2 ];
				tmp0++;
			}
		}
	}
	if( tmp0 ) {
		//intensity_tmp1 /= tmp0;
		intensity.val[0] = intensity_tmp1.val[0] / (double)tmp0;
		intensity.val[1] = intensity_tmp1.val[1] / (double)tmp0;
		intensity.val[2] = intensity_tmp1.val[2] / (double)tmp0;
	}								
	//intensity_x0 = src.at<Vec3f>(j, m);
	if( intensity.val[0]  ||  intensity.val[1] || intensity.val[2] ) {
		return 1;
	}
return 0;
}
/***************************************************************************************
*
*   function:		replaceHole
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int replaceHole( Mat src, Mat dstIsHole, Mat &dst, int iavg )
{
	int		istart = 0;
	int		jstart = 0;
	for (int i = istart; i < dst.cols; i++) {
		for (int j = jstart; j < dst.rows - 0; j++) {
			Vec3f  intensity_IsHole = dstIsHole.at<Vec3f>(j, i);
			if( intensity_IsHole.val[0]  ||  intensity_IsHole.val[1] || intensity_IsHole.val[2] ) {
				int x0 = -1, x1 = -1, y0 = -1, y1 = -1;
				Vec3f  intensity_x0;// = src.at<Vec3f>(j, i);
				Vec3f  intensity_x1;
				Vec3f  intensity_y0;
				Vec3f  intensity_y1;
				Vec3f  intensity_tmp0;
				int		tmp0; 
				Vec3f  intensity_tmp1;
				int		isav = 60;

				if( i == 100 && j ==117 ) {
					printf("");
				}

				for( int m = i; m < i + isav && m < dst.cols; m++ ) {
					intensity_x0 = src.at<Vec3f>(j, m);
					if( intensity_x0.val[0]  ||  intensity_x0.val[1] || intensity_x0.val[2] ) {
						x0 = m;
						if( iavg ) {
							get3x3Avegrage( src, m, j, intensity_x0 );
						}
						break;
					}
				}
				for( int m = i; m >= i - isav && m >= 0; m-- ) {
					intensity_x1 = src.at<Vec3f>(j, m);
					if( intensity_x1.val[0]  ||  intensity_x1.val[1] || intensity_x1.val[2] ) {
						x1 = m;
						if( iavg ) {
							get3x3Avegrage( src, m, j, intensity_x1 );
						}
						break;
					}
				}
				for( int n = j; n < j + isav && n < dst.rows; n++ ) {
					intensity_y0 = src.at<Vec3f>(n, i);
					if( intensity_y0.val[0]  ||  intensity_y0.val[1] || intensity_y0.val[2] ) {
						y0 = n;
						if( iavg ) {
							get3x3Avegrage( src, i, n, intensity_y0 );
						}
						break;
					}
				}
				for( int n = j; n > j - isav && n >= 0; n-- ) {
					intensity_y1 = src.at<Vec3f>(n, i);
					if( intensity_y1.val[0]  ||  intensity_y1.val[1] || intensity_y1.val[2] ) {
						y1 = n;
						if( iavg ) {
							get3x3Avegrage( src, i, n, intensity_y1 );
						}
						break;
					}
				}
				if( (x1-x0) && (y1 - y0) ) {
					float	dx = (i - x0) / (x1 - x0);
					float	dy = (j - y0) / (y1 - y0);
					Vec3f  intensity_x;
					Vec3f  intensity_y;
					if( x0 != -1 && x1 != -1) {
						intensity_x = (dx * intensity_x0 + (1-dx) * intensity_x1);
					}
					if( y0 != -1 && y1 != -1) {
						intensity_y = (dy * intensity_y0 + (1-dy) * intensity_y1);
					}
					if( x0 != -1 && x1 != -1 && y0 != -1 && y1 != -1) {
						Vec3f  intensity = (intensity_x + intensity_y) / 2.0;
						dst.at<Vec3f>(j, i) = intensity;
					} else {
						if( x0 != -1 && x1 != -1) {
							Vec3f  intensity = intensity_x;
							dst.at<Vec3f>(j, i) = intensity;
						}
						if( y0 != -1 && y1 != -1) {
							Vec3f  intensity = intensity_y;
							dst.at<Vec3f>(j, i) = intensity;
						}
					}
				} else {
				}
			}
		}
	}
return 1;
}
/***************************************************************************************
*
*   function:		cv_16UNormalization
*   arguments:
*	description:	histEqualization
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int cv_16UNormalization2(char *fileName, int rot)
{
	int		ret = 0;
	Mat		src;

	int		imtype = CV_32FC3;
	//int		imtype = CV_8UC3;
	Mat		canvas;

	Mat	mgombsor(400, 400, CV_8UC3, Scalar(64, 0, 0));
	Rect buttonCrop = Rect(0, 0, 100, 40);
	mgombsor(buttonCrop ) = Vec3b(200, 200, 200);
	putText(mgombsor(buttonCrop), "Crop", Point(buttonCrop.width*0.35, buttonCrop.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));

	setMouseCallback("gombsor", CallBackGombsorMouseFunc, nullptr);
	imshow("gombsor", mgombsor);

	//mCurve.zeros(200,200,CV_8UC3);


	namedWindow("Csuszkak", WINDOW_NORMAL);
	resizeWindow("Csuszkak", 600, 600);

	//canvas

	//src = imread( fileName, CV_LOAD_IMAGE_GRAYSCALE );
	//src = imread( fileName, IMREAD_GRAYSCALE  );

	//src = imread(fileName);
	//	cv_16u
	//	https://stackoverflow.com/questions/41186294/opencv-normalization-of-16bit-grayscale-image-gives-weak-result
	//	https://stackoverflow.com/questions/17345967/normalize-pixel-values-between-0-and-1
	//	https://opencv.programmingpedia.net/en/tutorial/1957/pixel-access
	//	https://arato.inf.unideb.hu/szeghalmy.szilvia/kepfeld/diak/szsz_ocv_gyak2.pdf
	//src = imread(fileName);
	src = imread(fileName, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

	if (src.empty()) {		
		exit(-1);
	}

	//	3x32bitess float
	src.convertTo(src, CV_32FC3);
	//	Nem kell leskalazni, bar az is bizonyos mostmar, hogy akkor is marad benne ertekes adat
	//src.convertTo(src, CV_32FC3, 1.0 / 255.0);

	if (rot) {
		src = rotate(src, 270);
	}


	if (1) {
		src = ResizeProperSize(src, DISP_W);
	}


	int		width = src.cols;
	int		height = src.rows;

	int		igain = 1000;
	int		igainmax = 10000;
	double	gain = (double)igain / (double)1000.0;


	Mat		dst0 = src.clone();
	Mat		dst;
	Mat		dst2;
	double	rmax = 1, gmax = 1, bmax = 1;
	double	rmin = 1, gmin = 1, bmin = 1;


	//	WhiteBalance a kozepso ponttal
	whiteBalance( src, dst0, 1, rmin, gmin, bmin, rmax, gmax, bmax );
	dst0 = dst0 * 2;



	double	colmax = max( rmax, max( gmax, bmax ) );
	double	colmin = min( rmin, min( gmin, bmin ) );

	//int		ithresval = 200;
	//int		ithresvalmax = 1000;
	int		ithresvalmax = 1000.0 * colmax * 1.05;
	int		ithresval = ithresvalmax;
	
	double	thresval = (double)ithresval / (double)1000.0;

//	int		ithresmul = 300;
//	int		ithresmulmax = 6000;
//	double	thresmul = (double)ithresmul / (double)1000.0;//1.1;

	//int		ifmul = 30000;
	//int		ifmul = colmax * 1000;
	//int		ifmul = 25959;
	int		ifmul = 2000;
	int		ifmulmax = 100000;
	double	fmul = (double)ifmul/1000.0;//30;

	//int		ifsub = 102;
	//int		ifsub = 209;
	int		ifsub = 0;
	int		ifsubmax = 300;
	//int		ifsubmax = colmin * 1000;	
	double	fsub = (double)ifsub / 1000.0;//0.102;

	int		ierode = 0;
	int		ierodemax = 15;

	int		iedge = 0;
	//int		iedge = 1500;
	int		iedgemax = 3000;
	int		fedge = (double)iedge / 1000;

	int		iedgedilate = 0;
	int		iedgedilatemax = 15;


	for( ; ; ) {

		//	Alap gain minden elott
		createTrackbar("gain", "Csuszkak", &igain, igainmax, on_trackbar);
		gain = (double)igain / (double)1000.0;

		//	A nagy csillagokta a threshold erteke
		createTrackbar("STAR Thres", "Csuszkak", &ithresval, ithresvalmax, on_trackbar);
		thresval = (double)ithresval / (double)1000.0;

		//createTrackbar("thresmul", "Csuszkak", &ithresmul, ithresmulmax, on_trackbar);
		//thresmul = (double)(ithresmul + 1000.0) / (double)1000.0;

		//	A "gorbezes" szorzoja elott egy alapertek kivonas
		createTrackbar("ValSub", "Csuszkak", &ifsub, ifsubmax, on_trackbar);
		fsub = (double)ifsub/1000.0;

		//	A "gorbezeshez" egy szorzo
		createTrackbar("ValMul", "Csuszkak", &ifmul, ifmulmax, on_trackbar);
		fmul = (double)ifmul/1000.0;

		//	A nagyobb csillagok maszkjat ennyi pixellel noveljuk
		//	Megteveszto a neve, mert egy negativ maszkkal dolgozom itt eloszor
		//	Konnyen atirhato
		createTrackbar("STAR Dlte", "Csuszkak", &ierode, ierodemax, on_trackbar);
		//	A kovetkezo lepes az lenne, hogy kikeressuk a tenyleg nagy csillagok blobjat es
		//	azokra egy kulon nagyobb dilate legyen kiadhato, mert most a kozepes csillagok maszkaja 
		//	tul sokkal no, alig marad a kornyezetben helyettesitoertek

		//	A kisebb csillagokat "elkeresessel" talaljuk meg. 
		//	Az fedge erteke 1.0 es 2.0 kozott a legjobb
		createTrackbar("star edge", "Csuszkak", &iedge, iedgemax, on_trackbar);
		fedge = (double)iedge/1000.0;

		//	A kisebb csillagok is haloval terheltek, noveljuk a maszkjukat
		createTrackbar("star dlte", "Csuszkak", &iedgedilate, iedgedilatemax, on_trackbar);



		dst = dst0.clone();
		dst2 = src.clone();
		Mat	dst0 = dst.clone();
		dst = dst * gain;


		Mat	dstmask = dst.clone();
		starMaskTresh( dst, dst, dstmask, thresval, 1, fmul, fsub );


		//
		//	EDGE
		//	A kis csillagokat nem sima thresholddal tuntetjuk el, 
		//	hanem a (kep - gauss) > szam alapjan
		//
		Mat	dst3;
		Mat	dstEdge;
		Mat dstHole = dst.clone();// = dstmask * dst;

		Mat dstIsHole( dst.size(), CV_32FC3);
		dstIsHole.zeros(dst.rows, dst.cols, CV_32FC3 );


		GaussianBlur( dst , dst3, cv::Size(0, 0), 3 );
		addWeighted( dst, fedge, dst3, -fedge, 0, dstEdge );
		if( iedgedilate ) {
			dilate( dstEdge, dstEdge, getStructuringElement( MORPH_ELLIPSE, Size(iedgedilate, iedgedilate) ) );
		}
		double	fmin = 0.05;
		if( 1 ) {
			for (int i = 0; i < dstEdge.cols; i++) {
				for (int j = 0; j < dstEdge.rows - 0; j++) {
					Vec3f  intensity = dstEdge.at<Vec3f>(j, i);
					Vec3f  intensity_dst = dst.at<Vec3f>(j, i);
					if( intensity.val[0] > fmin  ||  intensity.val[1] > fmin  ||  intensity.val[2]  > fmin ) {
						dstEdge.at<Vec3f>(j, i) = {1,1,1};
						dstHole.at<Vec3f>(j, i) = {0,0,0};
						//dstIsHole.at<Vec3f>(j, i) = {1,1,1};
						dstIsHole.at<Vec3f>(j, i)[0] = 1.0;
						dstIsHole.at<Vec3f>(j, i)[1] = 1.0;
						dstIsHole.at<Vec3f>(j, i)[2] = 1.0;
					} else {
						dstEdge.at<Vec3f>(j, i) = {0,0,0};
						//dstHole.at<Vec3f>(j, i) = intensity_dst;
						dstIsHole.at<Vec3f>(j, i)[0] = 0.0;
						dstIsHole.at<Vec3f>(j, i)[1] = 0.0;
						dstIsHole.at<Vec3f>(j, i)[2] = 0.0;

					}
				}
			}
		}

		

		if( ierode ) {
			erode( dstmask, dstmask, getStructuringElement( MORPH_ELLIPSE, Size(ierode, ierode) ) );
		}
		if( 1 ) {
			for (int i = 0; i < dst.cols; i++) {
				for (int j = 0; j < dst.rows - 0; j++) {
					Vec3f  intensity_mask = dstmask.at<Vec3f>(j, i);
					Vec3f  intensity_dst = dst.at<Vec3f>(j, i);
					if( intensity_mask.val[0]  &&  intensity_mask.val[1] && intensity_mask.val[2] ) {
					} else {
						dstHole.at<Vec3f>(j, i) = {0,0,0};
						//dstIsHole.at<Vec3f>(j, i) = {1,1,1};
						dstIsHole.at<Vec3f>(j, i)[0] = 1.0;
						dstIsHole.at<Vec3f>(j, i)[1] = 1.0;
						dstIsHole.at<Vec3f>(j, i)[2] = 1.0;
						float cc = dstIsHole.at<Vec3f>(j, i)[0];
						printf("");
					}			
				}
			}
		}
		dstmask = 1 - dstmask;
		cvtColor( dstmask, dstmask, COLOR_BGR2GRAY );


		//	Egy konturt kepzek a csillagok korul, ez lesz a legkozelebbi ertek, amivel helyettesiteni
		//	lehet.
		//	Csillagkicsinyiternel ez a kontur (csak erode-dal elott) lesz az a maszk, aminek az erteket
		//	a kornyezettel kell lecserelni
		Mat dstIsHoleDilate = dstIsHole.clone();
		dilate( dstIsHoleDilate, dstIsHoleDilate, getStructuringElement( MORPH_ELLIPSE, Size(5, 5) ) );
		Mat dstIsHoleContour( dst.size(), CV_32FC3);
		dstIsHoleContour.zeros(dst.rows, dst.cols, CV_32FC3 );
		bitwise_xor( dstIsHole, dstIsHoleDilate, dstIsHoleContour );
		if( 1 ) {
			for (int i = 0; i < dst.cols; i++) {
				for (int j = 0; j < dst.rows - 0; j++) {
					Vec3f  intensity_HoleC = dstIsHoleContour.at<Vec3f>(j, i);
					Vec3f  intensity_dst = dst.at<Vec3f>(j, i);
					if( intensity_HoleC.val[0]  ||  intensity_HoleC.val[1] || intensity_HoleC.val[2] ) {
						dstIsHoleContour.at<Vec3f>(j, i) = intensity_dst;
					}
				}
			}
		}

//
//        .....
//       .     .
//      . +     .
//     .         .
//     .       + .
//     .    +    .
//      .       .
//       .     .
//        .....
//
//	Minden pontot az x es az y koordinatajan levo legkozelebbi pontok tavolsagaryanos 
//	szinertek atlagakent hatarozunk meg.
//	Vagy ugyanez, de csak hsv[v] atlagot allitunk be, igy a szine megmarad, csak a 
//	csillagmagon kivul az eredetileg beszurodo szine marad meg.
//	Ezeket blobokkent keressuk a dstHole layeren.
//
//	Jobb megoldas lenne, ha nem a kereszt alaku pontokat vennem, hanem az osszes legkozelebbi
//	pont tavolsagaranyos atlagat, de az joval lassabb es meg nem is tartok ott.

		Mat dstRes = dstHole.clone();
		Mat	dstResAvg = dstHole.clone();
		replaceHole( dstHole, dstIsHole, dstRes, 0 );
		replaceHole( dstHole, dstIsHole, dstResAvg, 1 );


//	PARABOLOID
//	Hatterszint valasztunk tobb helyen a gradienssel es 
//	paraboloidot illesztunk ra. A paraboloid erteke lesz a szorzo, 
//	amivel osztjuk az eredeti keppont szineket.
//	Lehet, hogy ekkor mar a whitebalance nem is kell.
//	CSILLAGSZINEK
//	Valahogy vissza kellene csempeszni az eredeti csillagokat az eredeti szinukkel is?



		imshow("src", src );
		imshow("dst", dst );
		imshow("dstmask", dstmask );
		imshow("dstHole", dstHole );
		imshow("dstIsHole", dstIsHole );
		//imshow("dstIsHoleDilate", dstIsHoleDilate );
		imshow("dstIsHoleContour", dstIsHoleContour );
		//imshow("dstDilate", dstDilate );
		imshow("dstRes", dstRes );
		imshow("dstResAvg", dstResAvg );
		
		if ((ret = waitKey(30)) >= 0) {
			if (ret == 27) {
				return 0;
			}
		}

	}

return 0;
}
/***************************************************************************************
*
*   function:		main
*   arguments:
*	description:
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int main()
{
	int		ret = 0;
	int		rot = 0;

	//char fileName[100] = VPATH"grey.bmp";
	//char fileName[100] = VPATH"Autosave008.tif";
	//char fileName[100] = VPATH"SAM_1312.JPG";
	//char fileName[100] = VPATH"raw01.JPG";
	//char fileName[100] = VPATH"1.JPG";
	//char fileName[100] = VPATH"3.jpg";
	//char fileName[100] = VPATH"SAM_1177.jpg";
	//char fileName[100] = VPATH"SAM_1206.JPG";
	//char fileName[100] = VPATH"SAM_1206.JPG";
	//char fileName[100] = VPATH"SAM_1215.JPG";
	//char fileName[100] = VPATH"SAM_1282.JPG";
	//char fileName[100] = VPATH"SAM_1313.JPG";
	//char fileName[100] = VPATH"SAM_1328.JPG";
	//char fileName[100] = VPATH"SAM_1633.JPG";
	
	
	//char fileName[100] = VPATH"C7150_7174.jpg";
	
	
	//char fileName[100] = VPATH"4_00.jpg";
	//char fileName[100] = VPATH"5_02.jpg";
	//char fileName[100] = VPATH"8.JPG";
	//char fileName[100] = VPATH"9.JPG";
	//char fileName[100] = VPATH"04.tif"; rot = 1;
	//char fileName[100] = VPATH"SAM_1206.JPG";
	//char fileName[100] = VPATH"SAM_1236.JPG";
	
	//char fileName[100] = VPATH"SAM_1332.jpg";
	//char fileName[100] = VPATH"SAM_1334.jpg";
	//char fileName[100] = VPATH"CSM30799.jpg";
	//char fileName[100] = VPATH"test32bit_red.tiff";
	//char fileName[100] = VPATH"1.TIF"; rot = 1;	// frame:95 Val1:105 gamma:113 blur1:26 blur2:32 blur3:40 white 189
	//char fileName[100] = VPATH"SAM_0165.JPG"; rot = 0;	// frame:95 Val1:105 gamma:113 blur1:26 blur2:32 blur3:40 white 189
	//char fileName[100] = VPATH"SAM_0117.JPG"; rot = 0;	// frame:95 Val1:105 gamma:113 blur1:26 blur2:32 blur3:40 white 189
	//char fileName[100] = VPATH"CSM30799.tif"; rot = 0;	// frame:95 Val1:105 gamma:113 blur1:26 blur2:32 blur3:40 white 189
	//char fileName[100] = VPATH"1.tif"; rot = 1;
	//char fileName[100] = VPATH"C6641_6801.TIF"; rot = 0;
	//char fileName[100] = VPATH"HeartSoul2.TIF"; rot = 0;
	//char fileName[100] = VPATH"padlas.TIF"; rot = 0;
	//char fileName[100] = VPATH"190831_4_a06.jpg"; rot = 0;
	//char fileName[100] = VPATH"C7150_7174.jpg"; rot = 0;
	//char fileName[100] = VPATH"190831_4_as001.jpg"; rot = 0;
	//char fileName[100] = VPATH"10_01.jpg"; rot = 0;
	//char fileName[100] = VPATH"CSM30809.jpg"; rot = 0;

	//char fileName[100] = VPATH"401.tif"; rot = 0;
	//char fileName[100] = VPATH"402.tif"; rot = 0;
	//char fileName[100] = VPATH"403.tif"; rot = 0;
	//char fileName[100] = VPATH"404.tif"; rot = 0;
	//char fileName[100] = VPATH"405.tif"; rot = 0;
	//char fileName[100] = VPATH"406.tif"; rot = 0;
	//char fileName[100] = VPATH"407.tif"; rot = 0;
	//char fileName[100] = VPATH"407_01.tif"; rot = 0;

	//char fileName[100] = VPATH"sp01.tif"; rot = 0;
	//char fileName[100] = VPATH"sp02.tif"; rot = 0;

	
	//char fileName[100] = VPATH"practice.tif"; rot = 1;

	//char fileName[100] = VPATH"501.tif"; rot = 0;
	//char fileName[100] = VPATH"502_laggon_01.TIF"; rot = 0;
	//char fileName[100] = VPATH"503.tif"; rot = 1;
	//char fileName[100] = VPATH"504_7777_7826.TIF"; rot = 1;
	//char fileName[100] = VPATH"505.tif"; rot = 0;
	//char fileName[100] = VPATH"505_8001_8356_p1.TIF"; rot = 0;
	//char fileName[100] = VPATH"506_8361_8434_p1.TIF"; rot = 0;
	//char fileName[100] = VPATH"507_8492_8513.TIF"; rot = 0;
	//char fileName[100] = VPATH"508_8514_8532.TIF"; rot = 0;
	//char fileName[100] = VPATH"510_8554_8571.TIF"; rot = 0;
	//char fileName[100] = VPATH"511_8572_8587.TIF"; rot = 0;
	//char fileName[100] = VPATH"512_8588_8602.TIF"; rot = 0;
	//char fileName[100] = VPATH"Autosave.jpg"; rot = 0;
	//char fileName[100] = VPATH"Autosave001.jpg"; rot = 0;
	//char fileName[100] = VPATH"Autosave002.jpg"; rot = 0;
	//char fileName[100] = VPATH"Autosave001.jpg"; rot = 0;
	//char fileName[100] = VPATH"AutoRGBAlign001.TIF"; rot = 0;
	//char fileName[100] = VPATH"20200414_0415_NothAmerica_Pont.png"; rot = 0;
	//char fileName[100] = VPATH"20200414_0415_NothAmerica_Pont.tif"; rot = 0;
	//char fileName[100] = VPATH"20200820_Bp_Adnromeda2.tif"; rot = 0;
	//char fileName[100] = VPATH"20200820_Bp_Triangulum1.tif"; rot = 0;
	//char fileName[100] = VPATH"20211028_Mc_Soul_200mm_LDFB3_485min.tif"; rot = 0;
	//char fileName[100] = VPATH"20211028_Mc_Iris_200mm_03h20m.tif"; rot = 0;
	//char fileName[100] = VPATH"20211028_Mc_FlamingStar_200mm_LDFB2_4h03m.tif"; rot = 0;
	//char fileName[100] = VPATH"20211028_Mc_Pacman_200mm_LDFB_1h12m.tif"; rot = 0;
	//char fileName[100] = VPATH"20211008_Elephant300mm35min.tif"; rot = 0;
	char fileName[100] = VPATH"20211002_Mc_California_LDFB_1h17_01.tif"; rot = 0;

	
	



/*
char *pszTxt = NULL;
//pszTxt = (char *)malloc( 16 );
pszTxt = (char *)realloc( pszTxt, 16 );
memset( pszTxt, 0, sizeof( pszTxt ) );
for( int i = 0; i < 16; i++ ) {
	pszTxt[ i ] = '1';
}
pszTxt = (char *)realloc( pszTxt, 32 );
for( int i = 16; i < 32; i++ ) {
	pszTxt[ i ] = '2';
}
pszTxt = (char *)realloc( pszTxt, 16 );
free( pszTxt );
*/	
	



	if (0) {
		edImage(fileName, rot);
	}
	if (0) {
		histEqualization(fileName, rot);
	}
	if( 0 ) {
		filterSpectrum(fileName, rot);
	}
	if( 0 ) {
		CurveFilter(fileName, rot);
	}
	if (0) {
		SubsGreatAvg(fileName, rot);
	}
	if (0) {
		cv_16UNormalization(fileName, rot);
	}
	if (1) {
		cv_16UNormalization2(fileName, rot);
	}
	return 0;
}
