// This file is for image processing

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#define PRINT_LN(val) \
        {\
            std::cout << val << std::endl;\
        }\


#define PRINT(_val) \
        {\
            std::cout << _val << " ";\
        }\


#define CHECK_IMAGE_EMPTY(src, filename) \
    if (src.empty()) {                   \
        std::cerr << "Can't open image [" << filename << "]" << std::endl; \
        return EXIT_FAILURE;              \
    } \

#define w 400

using namespace cv;
using namespace std;


// pattern
namespace ex::theory
{
    void LineDrawHelper(Mat img, Point s, Point e)
    {

    }

    int Test(const char * picPath, const char * picPath2)
    {

        Mat atom_image = Mat::zeros(w,w, CV_8UC3);

        LineDrawHelper( atom_image, Point( 0, w/2 ), Point( w, w/2));

        namedWindow("Drawing_Atom", WINDOW_AUTOSIZE);
        imshow("Drawing_Atom", atom_image);
        waitKey(0);


        return 0;
    }
}


// general learning is, drawing
// line, ellopse, rectange, circle, filled polygon
namespace ex1::theory
{
    void LineDrawHelper(Mat img, Point s, Point e)
    {
        int thickness = 10;
        int lineType = LINE_8;
        line( img,
              s,
              e,
              Scalar( 255, 255, 255 ),
              thickness,
              lineType );

    }
    void EllipseDrawHelper( Mat img, double angle )
    {
        int thickness = 2;
        int lineType = 8;
        ellipse( img,
                 Point( w/2, w/2 ),
                 Size( w/4, w/16 ),
                 angle,
                 0,
                 360,
                 Scalar( 255, 0, 0 ),
                 thickness,
                 lineType );
    }
    void FilledCircleDrawHelper( Mat img, Point center )
    {
        circle( img,
                center,
                w/32,
                Scalar( 0, 0, 255 ),
                FILLED,
                LINE_8 );
    }

    int Test(const char * picPath, const char * picPath2)
    {

        Mat atom_image = Mat::zeros(w,w, CV_8UC3);

        LineDrawHelper( atom_image, Point( 0, w/2 ), Point( w, w/2));
        FilledCircleDrawHelper( atom_image, Point( w/2, w/2 ));
        EllipseDrawHelper( atom_image, 30);


        namedWindow("Drawing_Atom", WINDOW_AUTOSIZE);
        imshow("Drawing_Atom", atom_image);
        waitKey(0);


        return 0;
    }
}

// Random generator and text with opencv
//
namespace ex2::theory
{
    void LineDrawHelper(Mat img, Point s, Point e)
    {


    }

    int Test(const char * picPath, const char * picPath2)
    {


        // generally i prefer to make it as commented
//        Mat atom_image = Mat::zeros(w,w, CV_8UC3);
//
//        LineDrawHelper( atom_image, Point( 0, w/2 ), Point( w, w/2));
//
//        namedWindow("Drawing_Atom", WINDOW_AUTOSIZE);
//        imshow("Drawing_Atom", atom_image);
//        waitKey(0);


        return 0;
    }
}

// it is a probability distribution in which every value between an interval form a to b is equally likely to occur
// P(x1 < X < x2) = (x2 - x1) / (b - a)

// example
// if a dolphins are uniformly distributes between 100 - 150
// if we select a random dolphin, we can use a formula
// to determine the probablity that the chosen dolphin wil weigh between 120 and 130 pounds

float UniformDistribution(uint dmax, uint dmin, uint maxPossibleVal, uint minPossibleVal)
{
    return ((float)(dmax - dmin)/(float)(maxPossibleVal - minPossibleVal));
}



int main()
{
    cout << UniformDistribution(130, 120, 150, 100) << endl;

    return 0;
}