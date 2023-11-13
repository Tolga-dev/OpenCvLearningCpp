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


using namespace cv;
using namespace std;


namespace ex1::theory
{
    Mat DiscreteFourierTransform(Mat& img, Mat& img2)
    {


        return img;
    }

    int Test(const char * picPath, const char * picPath2)
    {
        Mat img = imread(picPath);
        Mat img2 = imread(picPath2);

        CHECK_IMAGE_EMPTY(img, picPath)

        DiscreteFourierTransform(img, img2);

        return 0;
    }
}




int main()
{

    const char * pic1Path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/pic1.jpg";
    const char * pic2Path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/pic2.png";

    ex1::theory::Test(pic1Path, pic2Path);

    return 0;
}