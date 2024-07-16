#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#define pi 3.14159265358979323846

typedef signed char int8_t;
typedef unsigned int uint32_t;

/***
CLASSES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
           'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
           'soccer ball field', 'swimming pool']
***/

typedef struct
{
    int classId;
    float score;
    float x;
    float y;
    float w;
    float h;
    float angle;
} CSXYWHR;

// yolov8
class GetResultRectYolov8obb
{
public:
    GetResultRectYolov8obb();

    ~GetResultRectYolov8obb();

    int GenerateMeshgrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

    float Sigmoid(float x);

private:
    std::vector<float> Meshgrid;

    const int ClassNum = 15;
    int HeadNum = 3;

    int InputWidth = 640;
    int InputHeight = 640;
    int Strides[3] = {8, 16, 32};
    int MapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    std::vector<float> RegDFL;

    int RegNum = 16;
    float RegDeq[16] = {0};

    float NMSThresh = 0.5;
    float ObjectThresh = 0.27;
};

#endif
