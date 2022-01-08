#include	"stdafx.h"

/*
	https://math.stackexchange.com/questions/260856/algorithm-to-get-inverse-parabola-fitting
	https://www.efunda.com/math/leastsquares/lstsqr2dcurve.cfm
	https://www.sci.utah.edu/~balling/FEtools/doc_files/LeastSquaresFitting.pdf
	https://stackoverflow.com/questions/11015119/inverse-matrix-opencv-matrix-inv-not-working-properly
	https://stackoverflow.com/questions/43310433/difference-between-cvmatt-and-cvtranspose
	https://stackoverflow.com/questions/10936099/matrix-multiplication-in-opencv
	https://people.math.harvard.edu/~knill/teaching/math19b_2011/handouts/lecture20.pdf

*/


#include	<stdlib.h>
#include	<math.h>

#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>


#include	"linreg.h"
#include	<cmath>

using namespace cv;
using namespace std;


/*
m = 4	- adatok szma
n = 1+4 = 5
X[m X (m+1)] = [
	x0        x1      x2    x3    x4   | y[1..4] = 
	--------------------------------     -------
	 1      2104       5     1    45   | 460
	 1      1416       3     2    40   | 232
	 1      1534       3     2    30   | 315
	 1       852       2     1    36   | 178
	 ]

Theta = (X'*X)^-1 * X' * y
? Theta[ 1x5 ]
?       [4x4] *    [5x4] * [1x4]
?       [5x5] *    [4x5] * [1x4]


Paraboloid:
z = p1*x^2 + p2*x*y + p3*y^2 + p4*x + p5*y + p6

*/

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
int FillParabloidX( Mat )
{
	vector<Mat> channels;
	vector<int>weight;
	weight.resize(256, 0);
	vector<Point> control;

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
int setParaboloidPoints( vector<Point3d> &control )
{
	
	//control.push_back(Point3d(400, 400, 20));
	control.push_back(Point3d(400, 400, 0));

	control.push_back(Point3d(200, 400, 40));	//	--.--
	control.push_back(Point3d(600, 400, 45));

												//	|
	control.push_back(Point3d(400, 200, 34));	//	.
	control.push_back(Point3d(400, 600, 51));	//	|

	control.push_back(Point3d(100, 100, 151));	//	\./
	control.push_back(Point3d(700, 100, 167));

	control.push_back(Point3d(100, 700, 141));	//	 .
	control.push_back(Point3d(700, 700, 167));	//	/ \	

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
int setParaboloidMxAndy( vector<Point3d> control, Mat &Mx, Mat &y )
{
	Mx = Mat( control.size(), 6, CV_64F);	//	es NEM CV_32FC3
	y = Mat( control.size(), 1, CV_64F );
	for( int i = 0; i < control.size(); i++ ) {
		//	z = p1*x^2 + p2*x*y + p3*y^2 + p4*x + p5*y + p6
		Mx.at<double>(i,0) = 1;
		Mx.at<double>(i,1) = control[ i ].x * control[ i ].x;
		Mx.at<double>(i,2) = control[ i ].x * control[ i ].y;
		Mx.at<double>(i,3) = control[ i ].y * control[ i ].y;
		Mx.at<double>(i,4) = control[ i ].x;
		Mx.at<double>(i,5) = control[ i ].y;
		y.at<double>( i, 0 ) = control[ i ].z;
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
int calcParaboloid( Mat MxTheta, Point3d &pt )
{
	double	y;
	y = 0;
	y += MxTheta.at<double>(0,0) * 1.0;
	y += MxTheta.at<double>(1,0) * pt.x * pt.x;
	y += MxTheta.at<double>(2,0) * pt.x * pt.y;
	y += MxTheta.at<double>(3,0) * pt.y * pt.y;
	y += MxTheta.at<double>(4,0) * pt.x;
	y += MxTheta.at<double>(5,0) * pt.y;

	pt.z = y;

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
double calcParaboloid( Mat MxTheta, Point &pt )
{
	double	y;
	y = 0;
	y += MxTheta.at<double>(0,0) * 1.0;
	y += MxTheta.at<double>(1,0) * pt.x * pt.x;
	y += MxTheta.at<double>(2,0) * pt.x * pt.y;
	y += MxTheta.at<double>(3,0) * pt.y * pt.y;
	y += MxTheta.at<double>(4,0) * pt.x;
	y += MxTheta.at<double>(5,0) * pt.y;


return y;
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
int calcParaboloidScreen( Mat MxTheta, Mat &dst28UC1 )
{
	Mat dst8UC1loc = dst28UC1.clone();

	for (int i = 0; i < dst28UC1.cols; i++) {
		for (int j = 0; j < dst28UC1.rows; j++) {
			float	intensity = calcParaboloid( MxTheta, Point(i, j) );
			dst8UC1loc.at<float>( j, i ) = intensity;
		}
	}

	dst28UC1 = dst8UC1loc.clone();
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
int srcMinusParaboloid( Mat src8UC1, Mat MxTheta, Mat &dst28UC1 )
{
	Mat dst8UC1loc = src8UC1.clone();

	for (int i = 0; i < src8UC1.cols; i++) {
		for (int j = 0; j < src8UC1.rows; j++) {
			float	intensity = src8UC1.at<float>(j, i);
			double	lfminus = calcParaboloid( MxTheta, Point(i, j) );
			if( 0 ) {
				intensity = max( 0.0, intensity - lfminus/2.0 );
			} else {
				intensity = intensity / lfminus;
			}
			dst8UC1loc.at<float>( j, i ) = intensity;
		}
	}

	dst28UC1 = dst8UC1loc.clone();

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
int printMx( Mat Mx, char *szMxName )
{
	printf("\n%s:\n", szMxName );
	for( int i = 0; i < Mx.rows; i++ ) {
		for( int j = 0; j < Mx.cols; j++ ) {
			printf( "%21.9f ", Mx.at<double>(i, j));
		}
		printf("\n");
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
int fitParaboloid( vector<Point3d> &control, Mat &MxTheta )
{
//	vector<Point3d> control;
//	setParaboloidPoints( control );

	Mat	Mx;
	Mat	y;
	setParaboloidMxAndy( control, Mx, y );
	//printMx( Mx, "Mx" );

	Mat	MxT;
	MxT = Mx.clone();
	transpose( Mx, MxT );

	//printMx( MxT, "MxT" );
	Mat	MxA = Mat( control.size(), control.size(), CV_64F );

	MxA = MxT * Mx;
	//printMx( MxA, "MxA" );

	Mat		MxAInv;
	invert( MxA, MxAInv );
	//printMx( MxAInv, "MxAInv" );

	Mat		MxAInvMxT;
	MxAInvMxT = MxAInv * MxT;
	//printMx( MxAInvMxT, "MxAInvMxT" );

	//Mat		MxTheta;
	MxTheta = MxAInvMxT * y;
	printMx( MxTheta, "MxTheta" );

	if( 0 ) {
		for( int i = 0; i < control.size(); i++ ) {
			Point3d		pt(control[i].x, control[i].y, 0 );
			calcParaboloid( MxTheta, pt );
			printf( "\n%13.5lf - %13.5lf, = %13.5lf ", control[i].z, pt.z, abs( control[i].z -  pt.z) );
		}
	}
return 1;
}




void dszamit( double * param , double * d )
{
/*
int z;

double 	fi,	sx0, 	sw,	vcy,	vma1,	vma2,	vma3 ;
double	sp,	de,	tg,	fin, 	at,	rx,	rt ;
double 	spsd,	spsh,	al,	pmf,	dechl,	be,	demfl;
double	decyl, 	dema1,	dema2, 	dema3,	tgmf, 	tgch, 	tgcy;
double	tgma1, 	tgma2, 	tgma3, 	b, 	bc, 	bcor, 	finmf;
double	fincy,	finma1,	finma2,	finma3,	vma,	atma,	atma1;
double	atma2,	atma3,	atch,	dech,	atoil,	atgas,	atcy;
double	atmf,	rcy,	bm,	rmf,	rw,	bn,	ba;
double	rsh,	atvma2;
double 	se=0;

for(z = 0; z < Z; z++)
	{
	fi 	= param[z*M + 0];
	sx0 	= param[z*M + 1];
	sw 	= param[z*M + 2];
	vcy	= param[z*M + 3];
	vma1	= param[z*M + 4];
	vma2	= param[z*M + 5];
	vma3	= param[z*M + 6];



//------------------------------------------------------------------------------
		//A REGI SZELVENYEK KONSTANSAI
	pmf = .000116;
//---   SP ---
	rmf = 1; rw = .5; spsh = 0;
	spsd = -70.7 * (log(rmf / rw) / log(10));
//--- TG ---
	tgmf = 0; tgch = 0; tgma1 = 45; tgcy = 100; tgma2 = .5; tgma3 = .5;
//--- FIN ---
	finmf = 1; finma1 = -.04; fincy = .38; finma2 = -.001; finma3 = -.001;
//--- AT ---
	atmf = 620; atma1 = 180; atcy = 250; atvma2 = 150; atma2 = 1; atma3 = 1;
		atoil = 1200; atgas = 2000;
//--- DE ---
	demfl = 1; dema1 = 2.65; decyl = 2.5; dema2 = 2.71; dema3 = .1;
	dechl = .7; dech = .7; be = .807;
//--- RX0 ---
	rmf = 1; rcy = 5; bn = 2; bm = 2; ba = 1;
//--- RT ---
	rw = .5; rsh = 5; bn = 2; bm = 2; ba = 1;



//--- SP ---
	sp = spsd - vcy * (spsd - spsh);
//--- DE ---
	al = 1.11 - .15 * pmf;
	if(dechl <= .275)
		be = 1.24 * dechl;
	else
		be = 1.11 * dechl + .03;
	de = fi * demfl + vcy * decyl + vma1 * dema1 + vma2 * dema2 + vma3 * dema3;
	de = de - fi*1.07 * (1 - sx0) * (al*demfl - be);
//--- TG ---
	tg = fi * (tgmf * sx0 * demfl + tgch * (1 - sx0) * dechl);
	tg = tg + vcy * tgcy * decyl;
	tg = tg + vma1 * tgma1 * dema1 + vma2 * tgma2 * dema2 + vma3 * tgma3 * dema3;
	tg = tg / de;
//--- FIN ---
	if(dechl <= .275)
		b = 2.2 * dechl;
	else
		b = dechl + .3;
	bc = 2 * fi * (1 - sx0) * (1 - b) * (1 - (1 - sx0) * (1 - b));
	bcor = 1 - b / demfl / (1 - pmf);

	fin = fi * finmf + vcy * fincy + vma1 * finma1 + vma2 * finma2 + vma3 * finma3;
	fin = fin - fi * ((1 - sx0) * bcor + bc);
//--- AT ---
	vma = vma1 + vma2 + vma3;
	if(vma > .01 )
		atma = (vma1 * atma1 + vma2 * atma2 + vma3 * atma3) / vma;
	else
		atma = 0;
	atch = 1.11 * ((dech - .05) * atoil + (.95 - dech) * atgas);
	at = fi * (sx0 * atmf + (1 - sx0) * atch) + vcy * atcy + vma * atma;
// ----RX----
if(vcy>0) se = exp( (1 - vcy/2)*log(vcy) );
else se = 0;
	rx = (  se /  sqrt(rcy) + exp((bm/2)*log(fi))/sqrt(rmf)  ) * sx0;
	rx = 1 / (rx * rx);
// ----RT----

	rt = (  se /  sqrt(rcy) + exp((bm/2)*log(fi))/sqrt(rw)   ) * sw;
	rt = 1 / (rt * rt);
					// ** a^b = exp(b*log(a)) ** 
	d[z*N + 0] = sp;
	d[z*N + 1] = de;
	d[z*N + 2] = tg;
	d[z*N + 3] = fin;
	d[z*N + 4] = at;
	d[z*N + 5] = rx;
	d[z*N + 6] = rt;
	}
*/
}


//------------------------------------------------------------------------------

int Gmeghat(double *G, double *moditt, int Sr, int *SZR, int *R, int RR)
{			  // a fix parameter 0 oszlopat kihagyja a matrixbol
	int		Z = 4;	//???	Retegek szama
	int		N = 7;	//???
	int		M = 7;	//???
	int		mfix[7]={0,0,0,0,0,1,1};	// a meglevo szelvenytipusok 1-esek
	int		mik[ 7 ]        = { 0, 1, 2, 3, 4, 5, 6};	// parameterek sorrendje a fixek nelkul (M=7 a dimenzio)
	int		sorrend[7]={0,1,2,3,4,5,6};
	int 	L = 1;		// (max.7) a nem fix parameterek szama lesz

	double 	h = .0001;
	double 	*d1, *d2, *modmost;

	int 	k, i, j, z, r;

	d1 = (double *)malloc(Z * N * sizeof(double) );
	d2 = (double *)malloc(Z * N * sizeof(double) );
	modmost = (double *)malloc(Z * M * sizeof(double) );

	dszamit( moditt, d1 );

	for(i = 0; i < RR*Sr*Z*L; i++) G[i]=0;

	for(i = 0 ; i < L ; i++) {

		if( mfix[i] == 1)
			continue;

		for(z = 0; z < Z; z++) {
			for(j = 0 ; j < M ; j++) {
				if(mik[i] == j)
					modmost[z*M + j] = moditt[z*M + j] + h ;
				else
					modmost[z*M + j] = moditt[z*M + j];
			}
		}
		dszamit( modmost , d2 );
		for(z = 0; z < Z; z++) {
			for(r=0;r<R[z];r++) {
				for( k = 0 ; k < Sr ; k++ ) {
					if(moditt[z*M + i] > .001) {

						G[ SZR[z]*Sr*Z*L + r*Sr*Z*L + k*Z*L + z*L + i ] =

						(d2[ z*N + sorrend[k] ] - d1[ z*N + sorrend[k] ]) / h /

						d1[ z*N + sorrend[k] ] * moditt[z*M + mik[i]];
							//data[SZR[z]*Sr + r*Sr + k] * moditt[z*M + mik[i]];

					} else {

						G[ SZR[z]*Sr*Z*L + r*Sr*Z*L + k*Z*L + z*L + i ] =

						(d2[ z*N + sorrend[k] ] - d1[ z*N + sorrend[k] ]) / h /

						d1[ z*N + sorrend[k] ] * .001;
							//data[SZR[z]*Sr + r*Sr + k] * .001;
					}

				}
			}
		}
	}

	free(d1);
	free(d2);
	free(modmost);
return 1;
}
/***************************************************************************************
*
*   function:		GTd
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
void GTd( double *G, double *dd, double *Gd, int HanySor, int HanyOszlop )
{
	int	kk, i;

	for( i = 0; i < HanyOszlop; i++ ) Gd[i] = 0;

	for( kk = 0 ; kk < HanySor ; kk++ ) {
		for( i = 0; i < HanyOszlop; i++ ) {
			Gd[i] += G[kk*HanyOszlop + i] * dd[kk];
		}
	}
}
/***************************************************************************************
*
*   function:		GTG
*   arguments:
*	description:	
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
void GTG( double *G, double *A, int HanySor, int HanyOszlop )
{
	int	i, j, kk;

	for( i = 0; i < HanyOszlop; i++ ) for( j = 0; j < HanyOszlop; j++ ) A[ i*HanyOszlop + j ] = 0;

	for( i = 0; i < HanyOszlop; i++ ) {
		for( j = 0; j < HanyOszlop; j++ ) {
			for( kk = 0; kk < HanySor; kk++ ) {
				A[ i * HanyOszlop + j ] += G[ kk * HanyOszlop + i ] 
					* G[ kk * HanyOszlop + j ]; // GT*G
			}
		}
	}
}
//------------------------------------------------------------------------------
void GTGGTdIRLS(double *G , double *Gd, double *A, double *ddelozo, double *dd, int HanySor, int HanyOszlop)
{
	int	i , j , kk ;
	double  e2=.001;	// îý


	for(i = 0; i < HanyOszlop; i++ ) Gd[i] = 0;
	for(i = 0; i < HanyOszlop; i++ ) for(j = 0; j < HanyOszlop; j++) A[i*HanyOszlop + j] = 0;


	for(kk = 0 ; kk < HanySor ; kk++ ) {
		for(i = 0 ; i < HanyOszlop ; i++ ) {
			Gd[i] += G[kk * HanyOszlop + i] *
					e2 / (e2 + fabs(ddelozo[kk]) ) *
					dd[kk];
			// GTWd

			for(j = 0 ; j < HanyOszlop ; j++ ) {
				A[i * HanyOszlop + j] += G[kk * HanyOszlop + i] *
					e2 / (e2 + fabs(ddelozo[kk]) ) *
					G[kk * HanyOszlop + j]; // GT*W*G

			}
		}
	}
}
/***************************************************************************************
*
*   function:		karot.arj: wjpkor1.cpp: Ermego
*   arguments:
*	description:	A Gt*G*m = G*d egyenletrendszer megoldasa
*						a*x = b
*					Meret: Meret*Meret -es mx
*	globals:
*	side effect:
*   return:
*
***************************************************************************************/
int Ermego( double *a, double *b, double *x, int Meret )
{

	int 		i, j, k, ixm;
	int			*prow;
	double 		m, aijxj = 0, xm;

	prow = (int *)malloc(Meret*sizeof(int));

	for (i = 0; i < (Meret); i++) prow[i] = i;


	for (k = 0; k < (Meret - 1); k++) {
		xm = 0;
		//	foelem kivalasztas
		for (i = k; i < Meret; i++)	{
			if (xm < fabs(a[prow[i] * Meret + k])) {
				xm = fabs(a[prow[i] * Meret + k]);
				ixm = i;
			}
		}

		// sorok mutatoinak csereje
		if (ixm > k) {
			j = prow[k]; prow[k] = prow[ixm]; prow[ixm] = j;
		}

		for (i = (k + 1); i < Meret; i++) {
			m = a[prow[i] * Meret + k] / a[prow[k] * Meret + k];
			for (j = k/*+1*/; j < Meret; j++) {
				a[prow[i] * Meret + j] =
					a[prow[i] * Meret + j] - m * a[prow[k] * Meret + j];
			}
			b[prow[i]] = b[prow[i]] - m * b[prow[k]];
		}
	}

	x[Meret - 1] = b[prow[Meret - 1]] / a[prow[Meret - 1] * Meret + (Meret - 1)];

	for (i = (Meret - 2); i >= 0; i--) {
		aijxj = 0;
		for (j = (i + 1); j < Meret; j++) {
			aijxj += a[prow[i] * Meret + j] * x[j];
		}
		x[i] = (b[prow[i]] - aijxj) / a[prow[i] * Meret + i];
	}

return 1;
}
//------------------------------------------------------------------------------
/*
double detav(double huge* para , double huge* mdat , int Sr, int huge* SZR , int huge* R, int RR)
{

double	adattav = 0, ddd = 0;
int 	r , k, z;
HGLOBAL	hszdat;
double	huge *szdat;

hszdat=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * N * sizeof(double));
	if(!hszdat)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return -1;//(FALSE);
	}
szdat=(double huge*)GlobalLock(hszdat);

dszamit(para, szdat);

for(z = 0; z < Z ; z++)
	{

	for(k = 0; k < Sr; k++)
		{

		for(r = 0; r < R[z] ; r++)
			{

			ddd = mdat[ SZR[z]*Sr + r*Sr + k ] -

				szdat[ z*N + sorrend[k] ];


			if( IRLS != 1 )
				{
				adattav += ddd*ddd /

					( mdat[ SZR[z]*Sr + r*Sr + k ] *

					  mdat[ SZR[z]*Sr + r*Sr + k ] );
				}
			else
				adattav += fabs(  ddd /

					mdat[ SZR[z]*Sr + r*Sr + k ] );

			}
		}
	}

if( IRLS == 1 )

	adattav = adattav/Sr/RR;
else
	adattav = sqrt( adattav/Sr/RR );

GlobalUnlock(hszdat);
GlobalFree(hszdat);

return adattav;
}
*/
//------------------------------------------------------------------------------
/*
BOOL parkeres(double huge* mitt, double huge* adat, int Sr, int huge* SZR, int huge* R, int RR )
{

int 	i, k, r, z, lep;

double	gr = 0, ta = 0, dtav = 0;

HGLOBAL	hG,hdd,hddelozo,hd2,hA,hGd,hdm,hworkm;
double huge *G, huge *dd, huge*ddelozo, huge*d2, huge*A, huge*Gd, huge*dm, huge*workm;

hG=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, RR * Sr * Z * L * sizeof(double));
	if(!hG)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
G=(double huge*)GlobalLock(hG);
hA=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * L * Z * L * sizeof(double));
	if(!hA)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
A=(double huge*)GlobalLock(hA);

hdd=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, RR * Sr * sizeof(double));
	if(!hdd)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
dd=(double huge*)GlobalLock(hdd);
hddelozo=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, RR * Sr * sizeof(double));
	if(!hddelozo)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
ddelozo=(double huge*)GlobalLock(hddelozo);
hd2=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * N * sizeof(double));
	if(!hd2)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
d2=(double huge*)GlobalLock(hd2);
hGd=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * L * sizeof(double));
	if(!hGd)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
Gd=(double huge*)GlobalLock(hGd);
hdm=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * L * sizeof(double));
	if(!hdm)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
dm=(double huge*)GlobalLock(hdm);
hworkm=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * M * sizeof(double));
	if(!hworkm)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
workm=(double huge*)GlobalLock(hworkm);

for(i = 0; i < Z*M; i++) workm[i] = mitt[i];


for(lep = 0; lep < LEPES; lep++ )
	{

	dtav=0;

	dszamit(mitt, d2);

	// elteres vektor kepzese normalva

	for(k = 0; k < Sr; k++)
		{
		for(z = 0; z < Z; z++)
			{
			for(r = 0; r < R[z]; r++)
				{

					dd[ SZR[z]*Sr + r*Sr + k ] =

					(adat[ SZR[z]*Sr + r*Sr + k] -

					d2[ z*N + sorrend[k] ]) /

					d2[ z*N + sorrend[k] ];

					//adat[ SZR[z]*Sr + r*Sr + k];

				}
			}
		}


	dtav = detav(workm, adat, Sr, SZR, R, RR);
	if(dtav == -1) return FALSE;


// kilepesi feltetelek

	if( dtav < 1e-3 ) break;

//	if( (gr>2)&&(ta/dtav <1.00001) ) break;

	ta=dtav;gr++;



// a szamitas

	// a derivalt matrix meghatarozasa a normalassal
	if(!Gmeghat(G, workm, Sr, SZR, R, RR ))return FALSE;
							//Gkiir(G,dd,Sr, RR);

	if( (lep == 0) || (IRLS == 0) )
		{

		// a G*d eloallitasa
		GTd(G , dd , Gd , RR*Sr, Z*L);

		// a transzp. G matrix es a G matrix szorzata
		GTG(G , A, RR*Sr, Z*L);
		}

	else
		// a transzp.G mx az R diagsulymx es a G mx szorzata
		GTGGTdIRLS(G, Gd, A, ddelozo, dd, RR*Sr, Z*L);


	//csillapitas
	for(i = 0 ; i < Z*L ; i++ ) A[i*Z*L + i] += 0.1/(lep*lep+1);

	// a Gt*G*m = G*d egyenletrendszer megoldasa
	if(!(Ermego(A , Gd , dm , Z*L )))return FALSE;


// dm-mel modositas
	for(z = 0; z < Z; z++)
		{

		for(i = 0; i < L; i++)
			{

			if( workm[ z*M + mik[i] ] > .001 )

				workm[ z*M + mik[i] ] = workm[ z*M + mik[i] ] +

					dm[ z*L + i]  *workm[ z*M + mik[i] ];

			else

				workm[ z*M + mik[i] ] = workm[ z*M + mik[i] ] +

					dm[z*M + i] * .001;

			if( workm[ z*M + mik[i] ] > 1 )

				workm[ z*M + mik[i] ] = 1;

			if( workm[ z*M + mik[i] ] < 0 )


				workm[ z*M + mik[i] ] = 10e-16;
			}
		}

	if( (dtav = detav(workm, adat, Sr, SZR, R, RR)) == -1) return FALSE;
	if(ta>dtav)for(i=0;i<Z*M;i++)mitt[i]=workm[i];

	for(k = 0; k < RR*Sr; k++) ddelozo[k] = dd[k];

	}

GlobalUnlock(hG);
GlobalFree(hG);
GlobalUnlock(hdd);
GlobalFree(hdd);
GlobalUnlock(hd2);
GlobalFree(hd2);
GlobalUnlock(hA);
GlobalFree(hA);

GlobalUnlock(hGd);
GlobalFree(hGd);
GlobalUnlock(hdm);
GlobalFree(hdm);
GlobalUnlock(hddelozo);
GlobalFree(hddelozo);
GlobalUnlock(hworkm);
GlobalFree(hworkm);

return TRUE;
}
*/
//------------------------------------------------------------------------------
int LU(double *a, int *indx, int Meret )	// a*x = b
{						// Meret: Meret*Meret -es mx

	int			i , j , k, ixm;
	int			*prow;
	double		m , xm;

	prow = (int *)malloc( Meret * sizeof(int) );

	for( i = 0 ; i < (Meret) ; i++ )prow[i] = i;

	for( k = 0 ; k < (Meret - 1) ; k++ ) {
		xm = 0;
		//foelem kivalasztas
		for(i = k ; i < Meret ; i++) {
			if( xm < fabs( a[ prow[i]*Meret + k ] )	) {
				xm = fabs( a[ prow[i]*Meret + k ] );
				ixm = i;
			}
		}
		if(xm == 0) {
			printf("\nSzingularis matrix");
			//{MessageBox(localhwnd,"Szingularis matrix","Figyelmeztetés",MB_OK);return FALSE;}
		}
		// sorok mutatoinak csereje
		if( ixm > k ) { j = prow[k] ; prow[k] = prow[ixm] ; prow[ixm] = j; }

		for(i = (k + 1) ; i < Meret ; i++ ) {
			if( a[prow[i]*Meret+k] == 0 ) continue;

			m = a[ prow[i]*Meret + k ] / a[prow[k]*Meret + k ] ;

			for(j = k ; j < Meret ; j++ ) {
				a[prow[i]*Meret + j] =
					a[ prow[i]*Meret + j ] - m * a[ prow[k]*Meret + j ];
			}
			a[ prow[i]*Meret + k ] = m;
		}
	}
	for(i=0;i<Meret;i++)indx[i] = prow[i];

	free(prow);

return 1;
}

//------------------------------------------------------------------------------
int LybUxy(double *a, double *x, double *b, int *prow, int Meret)
{                                        //!!! x volt
double	*y;
int		i,j;
double	aijxj = 0;

	y = (double *)malloc(Meret * sizeof(double));

	y[prow[0]] = b[prow[0]];		// Ly=b
	for(i = 1; i < Meret; i++) {
		aijxj = 0;

		for(j = 0; j < i; j++)
			aijxj += a[prow[i]*Meret + j] * y[prow[j]];

		y[prow[i]] = b[prow[i]] - aijxj;
	}

	x[Meret-1] = y[ prow[Meret-1] ] / a[ prow[Meret-1]*Meret + (Meret-1) ];
	for(i = (Meret - 2) ; i >= 0 ; i-- ) {
		aijxj = 0;

		for(j = (i + 1) ; j < Meret ; j++ )
			aijxj += a[ prow[i]*Meret + j ] * x[j];

		x[i] = (y[prow[i]] - aijxj)/ a[ prow[i]*Meret + i ];
	}

	free(y);

return 1;
}
//------------------------------------------------------------------------------
int Invert( double *A, double *Ain, int Meret)
{
	int 		i, j, n = Meret;
	double		*b, *x, *a;
	int			*indx;

	a = (double *)malloc(n * n * sizeof(double));
	indx = (int *)malloc(n * sizeof(int));
	b = (double *)malloc(n * sizeof(double));
	x = (double *)malloc(n * sizeof(double));

	for(i=0;i<n;i++) {
		x[i]=0;b[i]=0;indx[i]=0;
		for(j=0;j<n;j++)
			a[i*n+j] = A[i*n+j];
	}

	if( !(LU(a,indx,n) )) return 0;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
			b[j]=0;
		b[i]=1;
		if( !(LybUxy( a, x, b, indx, n))) return 0;
		for(j=0;j<n;j++)
			Ain[j*n+i] = x[j];
	}
	free(a);
	free(a);

return 1;
}
//------------------------------------------------------------------------------
/*
BOOL mpinverz(double huge* BecsPar, double huge* data)
{

int 	i, k; 		// ciklusvaltozok

int	z,r;

HGLOBAL	hparam,hdata1;
double 	huge *param;			// pointer a parametervektorhoz
double	huge *data1;


FILE *filkipar;			// kimeno parameterek file azonositoja

Z = 1;	// !!!!!!!!!!!!!!!!! melysegpontonkent

hparam=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, Z * M * sizeof(double));
	if(!hparam)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
param=(double huge*)GlobalLock(hparam);
hdata1=GlobalAlloc(LMEM_MOVEABLE | LMEM_ZEROINIT, M * sizeof(double));
	if(!hdata1)
	{
		MessageBox(localhwnd,"Nincs több memória!!!","Figyelmeztetés",MB_OK|MB_ICONSTOP);
		return (FALSE);
	}
data1=(double huge*)GlobalLock(hdata1);



		//---------------------------------
Gosz();

if( !(filkipar = fopen( "param.xxx" , "w+t")) )
	{
	MessageBox(localhwnd,"Nem lehet megnyitni az eredmenyfile-t","Figyelmeztetés",MB_OK);
	return FALSE;
	}


Z=1;
R[0]=1;


SZR[0] = 0;
if( Z > 1 )
	{

	for(z = 1; z < Z; z++)

		SZR[z] = SZR[z - 1] + R[z - 1];
	}


for(r=0;r<RR;r++)
	{

	for(z = 0; z < Z; z++)
		for(k = 0; k < M; k++)
			param[z*M + k]=startmp[z*7+k];


	for(k=0;k<Sr;k++)
		data1[k]=data[r*Sr+k];

	switch(alg)
		{
		case SA: SimAnn(param, data1, SZR, R, Sr, RR);break;
		case LS: parkeres(param, data1, Sr, SZR, R, RR);break;
		case IR: IRLS = 1;parkeres(param, data1, Sr, SZR, R, RR); break;
		}

	IRLS = 0;
	for(z = 0; z < Z; z++)
		{
		fprintf(filkipar,"%5.2lfm ",Mkoor[r]);

		for(i=0;i<M;i++)
			{
			fprintf(filkipar,"%8.3lf",param[z*M + i]);
			}
		}


	fprintf(filkipar," %5.2lf%c\n",100*detav(param, data1, Sr, SZR, R, RR),'%');

	}

fflush(filkipar);
fclose(filkipar);

GlobalUnlock(hparam);
GlobalFree(hparam);
GlobalUnlock(hdata1);
GlobalFree(hdata1);

return TRUE;
}
*/
