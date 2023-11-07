// This file is for Core tutorials

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
    namespace MaskOperationOnMatrices
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
                    output[j] = saturate_cast<uchar>
                            (5 * curr[j] - curr[j - chan] - curr[j + chan] - pre[j] - next[j]);

                result.row(0).setTo(Scalar(0));
                result.row(result.rows-1).setTo(Scalar(0));
                result.col(0).setTo(Scalar(0));
                result.col(result.cols-1).setTo(Scalar(0));

            }

        }

    }
    namespace OperationWithImage
    {
        Mat MemoryManagementAndReferenceCounting(Mat& img)
        {
            std::vector<Point3f> points = {{0, 0, 0},{0, 0, 0},{0, 0, 0}};

            Mat pointsMat = Mat(points).reshape(1); // example of creating two matrices without copying data

            for (int i = 0; i <= getElemSize(pointsMat.rows); i++ )
                for (int j = 0; j <= getElemSize(pointsMat.cols); j++ )
                {
                    cout << pointsMat.col(i) << " ";
                }

            pointsMat = img.clone(); // cloning example

            return img;
        }
        Mat PrimitiveOperations(Mat& img)
        {
            Rect r(10, 10, 100, 100);
            Mat smallImg = img(r); // selecting region

            Mat grey;
            cvtColor(img, grey, COLOR_BGR2GRAY); // making gray

            namedWindow("CONVERT_GRAY", WINDOW_NORMAL);
            namedWindow("MAKE_SMALL", WINDOW_NORMAL);

            imshow( "CONVERT_GRAY", grey);
            imshow( "MAKE_SMALL", smallImg);
            waitKey();

            return img;
        }

        int Test(const char * picPath)
        {
            Mat img = imread(picPath);

            CHECK_IMAGE_EMPTY(img, picPath)

            // if you want to create a gray one
            Mat imgGray = imread(picPath, IMREAD_GRAYSCALE);

            // check your types
//            cout << typeid(imgGray).name() << " " << typeid(Scalar).name() << endl;

            //  to save an image to a file
            // imwrite(picPath, img);

            // in order to get pixel intensity value, you have to know the type of image and the number of chans
            for (int i = 0; i <= getElemSize(img.rows); i++ )
                for (int j = 0; j <= getElemSize(img.cols); j++ )
                {
//                    cout << (static_cast<Scalar>(img.at<uchar>(Point(i, j)))).val[0] << " ";

                    Vec3f intense = img.at<Vec3f>(Point(i, j));
//                    cout << intense.val[0] << " ";
//                    cout << intense.val[1] << " ";
//                    cout << intense.val[2] << endl;

                    img.at<Vec3f>(Point(i, j)) = {static_cast<float>(i),static_cast<float>(j),0};

                }
//            img = MemoryManagementAndReferenceCounting(img);
            img = PrimitiveOperations(img);
//            show the changed pic
//            namedWindow("OUTPUT", WINDOW_NORMAL);
//            imshow( "OUTPUT", img);
//            waitKey();

            return 0;
        }
    }
}

/*
 * what is linear blending and why it is useful
 * how to add two images
 *
 * linear blend operator
 * g(x) = (1 - a)* f(x) + af(x)
 *
 */
namespace ex4::AddingTwoImages
    {

        Mat PrimitiveOperations(Mat& img, Mat& img2)
        {
            double alpha, beta;

            Mat src1 = img.clone(), src2 = img2.clone(), dst;

            alpha = 0.3; // random assigned

            beta = ( 1.0 - alpha );
            addWeighted( src1, alpha, src2, beta, 0.0, dst);

            imshow( "Linear Blend", dst );
            waitKey(0);

            return img;
        }

        int Test(const char * picPath, const char * picPath2)
        {
            Mat img = imread(picPath);
            Mat img2 = imread(picPath2);

            CHECK_IMAGE_EMPTY(img, picPath)

            img = PrimitiveOperations(img, img2);

            return 0;
        }
    }

/*
 * changing the contrast and brightness of an image
 * access pix vals, initialize a matrix with zeros
 * cv::saturate_cast
 */


namespace ex5::theory
    {
        Mat BrightnessOperations(Mat& img, Mat& img2)
        {
            Mat newImage = Mat::zeros(img.size(), img.type());
            double alpha = 1.0; // contrast control
            int beta = 40; // brightness control 0 - 100

            int chans = img.channels();

            for( int y = 0; y < img.rows; y++ ) {
                for( int x = 0; x < img.cols; x++ ) {
                    for( int c = 0; c < chans; c++ ) {
                        newImage.at<Vec3b>(y,x)[c] =
                                saturate_cast<uchar>( alpha*img.at<Vec3b>(y,x)[c] + beta );
                    }
                }
            }

            imshow("Original Image", img);
            imshow("New Image", newImage);

            waitKey();

            return img;
        }

        int Test(const char * picPath, const char * picPath2)
        {
            Mat img = imread(picPath);
            Mat img2 = imread(picPath2);

            CHECK_IMAGE_EMPTY(img, picPath)

            BrightnessOperations(img, img2);

            return 0;
        }
    }

// this is empty for now!
/*
 * fourier transform
 * and functions
 */
namespace ex6::theory
{
    // fourier transform will decompose an image into its sinus and cosines components
    //
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


/*
 * how to print and read text entries to a file and opencv using yalm or xml files
 * how to do same for opencv data structures
 * how to do this for ur data structures
 * usage of opencv data structures
 */
namespace ex7::theory
{
    class DataManager
    {
    public:
        DataManager() : A(0), X(0), id()
        {}
        explicit DataManager(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
        {}

        void write(FileStorage& fs) const                        //Write serialization for this class
        {
            fs << "{" << "A" << A << "X" << X << "id" << id << "}";
        }
        void read(const FileNode& node)                          //Read serialization for this class
        {
            A = (int)node["A"];
            X = (double)node["X"];
            id = (string)node["id"];
        }

        int A;
        double X;
        string id;
    };

    static void write(FileStorage& fs, const std::string&, const DataManager& x)
    {
        x.write(fs);
    }
    static void read(const FileNode& node, DataManager& x, const DataManager& default_value = DataManager()){
        if(node.empty())
            x = default_value;
        else
            x.read(node);
    }
    static ostream& operator<<(ostream& out, const DataManager& m)
    {
        out << "{ id = " << m.id << ", ";
        out << "X = " << m.X << ", ";
        out << "A = " << m.A << "}";
        return out;
    }


    int Test(const char * opencv_path = "", const char * picPath2 = "")
    {
        string filename = opencv_path;

        {
            Mat R = Mat_<uchar>::eye(3,3);
            Mat T = Mat_<double>::eye(3,1);

            DataManager manager(1);

            FileStorage fs(opencv_path, FileStorage::WRITE);

            fs << "iterationNr" << 100;
            fs << "strings" << "[";                              // text - string sequence
            fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
            fs << "]";                                           // close sequence
            fs << "Mapping";                              // text - mapping
            fs << "{" << "One" << 1;
            fs <<        "Two" << 2 << "}";
            fs << "R" << R;                                      // cv::Mat
            fs << "T" << T;
            fs << "MyData" << manager;                                // your own data structures
            fs.release();                                       // explicit close
            cout << "Write Done." << endl;
        }
        {
            cout << endl << "Reading: " << endl;
            FileStorage fs;
            fs.open(filename, FileStorage::READ);
            int itNr;
            itNr = (int) fs["iterationNr"];

            cout << itNr;

            if (!fs.isOpened())
            {
                cerr << "Failed to open " << filename << endl;
                return 1;
            }
            FileNode n = fs["strings"];                         // Read string sequence - Get node
            if (n.type() != FileNode::SEQ)
            {
                cerr << "strings is not a sequence! FAIL" << endl;
                return 1;
            }
            FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
            for (; it != it_end; ++it)
                cout << (string)*it << endl;
            n = fs["Mapping"];                                // Read mappings from a sequence
            cout << "Two  " << (int)(n["Two"]) << "; ";
            cout << "One  " << (int)(n["One"]) << endl << endl;
            DataManager m;
            Mat R, T;
            fs["R"] >> R;                                      // Read cv::Mat
            fs["T"] >> T;
            fs["MyData"] >> m;                                 // Read your own structure_
            cout << endl
                 << "R = " << R << endl;
            cout << "T = " << T << endl << endl;
            cout << "MyData = " << endl << m << endl << endl;
            cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
            fs["NonExisting"] >> m;
            cout << endl << "NonExisting = " << endl << m << endl;
        }

        cout << endl
             << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

        return 0;
    }
}


// a program to perform convolution operation over an image
// race conditions
// it is occured when more than on e thread try to write or read andwrite to a particular memory block
// simultaneously.

// multiple threads can read from data but only one data can write on that blocks

// algorithms in which multiple thread may write to a single memory location
//
namespace ex8::theory
{
    // parallel implementation


    // parallel frameworks
    //
    void conv_seq(Mat src, Mat &dst, Mat kernel)
    {
        int rows = src.rows, cols = src.cols;
        dst = Mat(rows, cols, src.type());
        // Taking care of edge values
        // Make border = kernel.rows / 2;
        int sz = kernel.rows / 2;
        copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);
        for (int i = 0; i < rows; i++)
        {
            uchar *dptr = dst.ptr(i);
            for (int j = 0; j < cols; j++)
            {
                double value = 0;
                for (int k = -sz; k <= sz; k++)
                {
                    // slightly faster results when we create a ptr due to more efficient memory access.
                    uchar *sptr = src.ptr(i + sz + k);
                    for (int l = -sz; l <= sz; l++)
                    {
                        value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                    }
                }
                dptr[j] = saturate_cast<uchar>(value);
            }
        }
    }

    Mat Parallelize(Mat& img, Mat& img2)
    {
        

        return img;
    }



    int Test(const char * picPath, const char * picPath2)
    {
        Mat img = imread(picPath);
        Mat img2 = imread(picPath2);

        CHECK_IMAGE_EMPTY(img, picPath)

        Parallelize(img, img2);

        return 0;
    }
}




int main()
{

    const char * pic1Path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/pic1.jpg";
    const char * pic2Path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/pic2.png";
    const char * opencv_xml_path = "/home/xamblot/CLionProjects/OpencvLearningCpp/asset/opencv.xml";

    //ex1::Runner(pic1Path);
    ex7::theory::Test(opencv_xml_path);

    return 0;
}