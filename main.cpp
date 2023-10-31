// This file is for Core tutorials

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#define PRINT(val) \
        {\
            std::cout << val << std::endl;\
        }\


#define CHECK_IMAGE_EMPTY(src, filename) \
    if (src.empty()) {                   \
        std::cerr << "Can't open image [" << filename << "]" << std::endl; \
        return EXIT_FAILURE;              \
    } \


using namespace cv;
using namespace std;


// Default opencv general example
namespace ex1
{
    void Runner(const char* pic1Path)
    {
        Mat image;
        image = imread(pic1Path, IMREAD_COLOR );
        if ( !image.data )
        {
            printf("No image data \n");
            return;
        }

        namedWindow("Display Image", WINDOW_AUTOSIZE );
        imshow("Display Image", image);
        waitKey(0);
    }
}

// How to use mat
namespace ex2
{
    void Runner(const char* pic1Path)
    {
        auto e1 = [=]()
            {
                Mat A, C; // creating header parts
                A = imread(pic1Path, IMREAD_COLOR);

                Mat B(A); // using copy constructor

                C = A;  // assignment operator

                Mat D(A ,Rect(10,10,100,100));
                Mat E = A(Range::all(), Range(1,3));


                Mat F = A.clone();
                Mat G;
                A.copyTo(G);

            };

        auto e2 = [=]()
        {
            Mat M(2,2, CV_8UC3, Scalar(0,0,255));
            std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;

        };

        auto e3 = [=]()
        {
            Mat M;
            M.create(4,4, CV_8UC(2));
//            std::cout << "M = " << std::endl << " " << M << std::endl << std::endl;

            Mat E = Mat::eye(4, 4, CV_64F);
//            cout << "E = " << endl << " " << E << endl << endl;
            Mat O = Mat::eye(4, 4, CV_32F);

//            cout << "O = " << endl << " " << O << endl << endl;
            Mat Z = Mat::zeros(3,3, CV_8UC1);
//            cout << "Z = " << endl << " " << Z << endl << endl;

            Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//            cout << "C = " << endl << " " << C << endl << endl;

            C = (Mat_<double>({0, -1, 0, -1, 5, -1, 0, -1, 0})).reshape(3);
//            cout << "C = " << endl << " " << C << endl << endl;

            Mat RowClone = C.row(1).clone();
//            cout << "RowClone = " << endl << " " << RowClone << endl << endl;

        };

        e1();
        e2();
        e3();

    }
}

/*
 * how to scan images lookup tables and time measurement
 * a simple color reduction method, by using unsigned char C and C++ type for matrix item storing.
 * we may work 16 million colors this is heavy but we can do less work to get the same final result
 * by doing color space reduction
*/
/*
 * Mask Operations On Matrices
 * it is quite simple, recalculate each pix val in an image according to a mask matrix
 * also as kernel.
 * general formula, we just changing color matrix values
 * I(i,j)=5∗I(i,j)−[I(i−1,j)+I(i+1,j)+I(i,j−1)+I(i,j+1)]
 * ⟺I(i,j)∗M,where M=i∖j−10+1−10−100−15−1+10−10
 */
namespace ex3
{
    void Sharper(const Mat& image, Mat& result);

    int Test(const char* pic1Path) {
        cvflann::StartStopTimer timer;

        PRINT("ok")

        Mat src, dst, dst2;

        src = imread(pic1Path, IMREAD_COLOR);
        CHECK_IMAGE_EMPTY(src, pic1Path)

        namedWindow("INPUT", WINDOW_AUTOSIZE);
        namedWindow("OUTPUT", WINDOW_AUTOSIZE);

        imshow("INPUT", src);

        timer.start();
        Sharper(src, dst);
        timer.stop();
        PRINT(timer.value)

        imshow("OUTPUT", dst);
        timer.reset();
        waitKey();

        Mat kernel = (Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

        timer.start();
        filter2D( src, dst2, src.depth(), kernel );
        timer.stop();

        PRINT(timer.value)

        imshow( "Output", dst2 );

        waitKey();

        return EXIT_SUCCESS;

    }

    void Sharper(const Mat& image, Mat& result)
    {
        const int chan = image.channels();
        result.create(image.size(), image.type());

        for (int i = 1; i < image.rows - 1; ++i) {
            const auto* pre = image.ptr<uchar>(i - 1);
            const auto* curr = image.ptr<uchar>(i);
            const auto* next = image.ptr<uchar>(i + 1);

            auto* output = result.ptr<uchar>(i);


            for(int j= chan; j < chan * (image.cols - 1); ++j)
            {
                output[j] = saturate_cast<uchar>
                        (5 * curr[j] - curr[j - chan] - curr[j + chan] - pre[j] - next[j]);
            }

            result.row(0).setTo(Scalar(0));
            result.row(result.rows-1).setTo(Scalar(0));
            result.col(0).setTo(Scalar(0));
            result.col(result.cols-1).setTo(Scalar(0));

        }

    }

}




int main()
{

    const char * pic1Path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/pic1.jpg";

    //ex1::Runner(pic1Path);
    ex3::Test(pic1Path);

    return 0;
}