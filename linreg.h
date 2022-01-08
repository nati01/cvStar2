#include	"stdafx.h"
#include	<math.h>
#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;

int fitParaboloid( vector<Point3d> &control, Mat &MxTheta );
int srcMinusParaboloid( Mat src8UC1, Mat MxTheta, Mat &dst28UC1 );
int calcParaboloidScreen( Mat MxTheta, Mat &dst28UC1 );



void dszamit( double * param , double * d );
int Gmeghat(double *G, double *moditt, int Sr, int *SZR, int *R, int RR);
void GTd( double *G, double *dd, double *Gd, int HanySor, int HanyOszlop );
void GTG( double *G, double *A, int HanySor, int HanyOszlop );
void GTGGTdIRLS(double *G , double *Gd, double *A, double *ddelozo, double *dd, int HanySor, int HanyOszlop);
int Ermego( double *a, double *b, double *x, int Meret );
int LU(double *a, int *indx, int Meret );
int LybUxy(double *a, double *x, double *b, int *prow, int Meret);
