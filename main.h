#ifndef MAIN_H
#define MAIN_H

#include <QApplication>
#include <QMainWindow>
#include <QImage>
#include <QHBoxLayout>
#undef slots
#include "torch/torch.h"
#include "torch/jit.h"
#include "torch/nn.h"
#include "torch/script.h"
#define slots Q_SLOTS

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

bool LoadImage(std::string file_name, cv::Mat &image);
bool LoadImageNetLabel(std::string file_name, std::vector<std::string> &labels);
cv::Mat imgClassifier(QString modelPath, QString labPath, QString img_path, torch::Device device);
cv::Mat QImageToMat(QImage image);
QImage MatToQImage(cv::Mat InputMat);
int detectResult(QImage& qimg);


class MyPicture : public QMainWindow
{
    Q_OBJECT
public:
    explicit MyPicture::MyPicture(QWidget *parent = 0);
    void resizeEvent(QResizeEvent *event);

private:
    QWidget * widget;
    QHBoxLayout * hboxlayout;
    QImage image;
};


#endif
