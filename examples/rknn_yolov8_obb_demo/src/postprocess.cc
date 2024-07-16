#include "postprocess.h"
#include <algorithm>
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float FastExp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static inline void GetCovarianceMatrix(CSXYWHR &Box, float &A, float &B, float &C)
{
    float a = Box.w;
    float b = Box.h;
    float c = Box.angle;

    float cos1 = cos(c);
    float sin1 = sin(c);
    float cos2 = pow(cos1, 2);
    float sin2 = pow(sin1, 2);

    A = a * cos2 + b * sin2;
    B = a * sin2 + b * cos2;
    C = (a - b) * cos1 * sin1;
}

static inline float probiou(CSXYWHR &obb1, CSXYWHR &obb2)
{
    float eps = 1e-7;

    float x1 = obb1.x;
    float y1 = obb1.y;
    float x2 = obb2.x;
    float y2 = obb2.y;
    float a1 = 0, b1 = 0, c1 = 0;
    GetCovarianceMatrix(obb1, a1, b1, c1);

    float a2 = 0, b2 = 0, c2 = 0;
    GetCovarianceMatrix(obb2, a2, b2, c2);

    float t1 = (((a1 + a2) * pow((y1 - y2), 2) + (b1 + b2) * pow((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.25;
    float t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.5;

    float temp1 = (a1 * b1 - pow(c1, 2));
    temp1 = temp1 > 0 ? temp1 : 0;

    float temp2 = (a2 * b2 - pow(c2, 2));
    temp2 = temp2 > 0 ? temp2 : 0;

    float t3 = log((((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2)) / (4 * sqrt((temp1 * temp2)) + eps) + eps)) * 0.5;

    float bd = 0;
    if ((t1 + t2 + t3) > 100)
    {
        bd = 100;
    }
    else if ((t1 + t2 + t3) < eps)
    {
        bd = eps;
    }
    else
    {
        bd = t1 + t2 + t3;
    }

    float hd = sqrt((1.0 - exp(-bd) + eps));
    return 1 - hd;
}

static inline void xywhr2xyxyxyxy(float x, float y, float w, float h, float angle,
                                  float &pt1x, float &pt1y, float &pt2x, float &pt2y,
                                  float &pt3x, float &pt3y, float &pt4x, float &pt4y)
{
    float cos_value = cos(angle);
    float sin_value = sin(angle);

    float vec1x = w / 2 * cos_value;
    float vec1y = w / 2 * sin_value;
    float vec2x = -h / 2 * sin_value;
    float vec2y = h / 2 * cos_value;

    pt1x = x + vec1x + vec2x;
    pt1y = y + vec1y + vec2y;

    pt2x = x + vec1x - vec2x;
    pt2y = y + vec1y - vec2y;

    pt3x = x - vec1x - vec2x;
    pt3y = y - vec1y - vec2y;

    pt4x = x - vec1x + vec2x;
    pt4y = y - vec1y + vec2y;
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/****** yolov8 ****/
GetResultRectYolov8obb::GetResultRectYolov8obb()
{
}

GetResultRectYolov8obb::~GetResultRectYolov8obb()
{
}

float GetResultRectYolov8obb::Sigmoid(float x)
{
    return 1 / (1 + FastExp(-x));
}

int GetResultRectYolov8obb::GenerateMeshgrid()
{
    int ret = 0;
    if (HeadNum == 0)
    {
        printf("=== yolov8 Meshgrid  Generate failed! \n");
    }

    for (int index = 0; index < HeadNum; index++)
    {
        for (int i = 0; i < MapSize[index][0]; i++)
        {
            for (int j = 0; j < MapSize[index][1]; j++)
            {
                Meshgrid.push_back(float(j + 0.5));
                Meshgrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== yolov8 Meshgrid  Generate success! \n");

    return ret;
}

int GetResultRectYolov8obb::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    if (Meshgrid.empty())
    {
        ret = GenerateMeshgrid();
    }

    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0, angle = 0;
    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_ang = 0;
    float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_ang = 0;

    float sfsum = 0;
    float locval = 0;
    float locvaltemp = 0;

    CSXYWHR Temp;
    std::vector<CSXYWHR> detectRects;

    for (int index = 0; index < HeadNum; index++)
    {
        int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
        int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
        int8_t *ang = (int8_t *)pBlob[HeadNum * 2 + index];

        quant_zp_reg = qnt_zp[index * 2 + 0];
        quant_zp_cls = qnt_zp[index * 2 + 1];
        quant_zp_ang = qnt_zp[HeadNum * 2 + index];

        quant_scale_reg = qnt_scale[index * 2 + 0];
        quant_scale_cls = qnt_scale[index * 2 + 1];
        quant_scale_ang = qnt_scale[HeadNum * 2 + index];

        for (int h = 0; h < MapSize[index][0]; h++)
        {
            for (int w = 0; w < MapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == ClassNum)
                {
                    cls_max = Sigmoid(DeQnt2F32(cls[0 * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                    cls_index = 0;
                }
                else
                {
                    for (int cl = 0; cl < ClassNum; cl++)
                    {
                        cls_val = cls[cl * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w];

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    cls_max = Sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
                }

                if (cls_max > ObjectThresh)
                {
                    RegDFL.clear();
                    for (int lc = 0; lc < 4; lc++)
                    {
                        sfsum = 0;
                        locval = 0;
                        for (int df = 0; df < RegNum; df++)
                        {
                            locvaltemp = exp(DeQnt2F32(reg[((lc * RegNum) + df) * MapSize[index][0] * MapSize[index][1] + h * MapSize[index][1] + w], quant_zp_reg, quant_scale_reg));
                            RegDeq[df] = locvaltemp;
                            sfsum += locvaltemp;
                        }
                        for (int df = 0; df < RegNum; df++)
                        {
                            locvaltemp = RegDeq[df] / sfsum;
                            locval += locvaltemp * df;
                        }

                        RegDFL.push_back(locval);
                    }

                    angle = (Sigmoid(DeQnt2F32(ang[h * MapSize[index][1] + w], quant_zp_ang, quant_scale_ang)) - 0.25) * pi;
                    xmin = RegDFL[0];
                    ymin = RegDFL[1];
                    xmax = RegDFL[2];
                    ymax = RegDFL[3];
                    float cos1 = cos(angle);
                    float sin1 = sin(angle);

                    float fx = (xmax - xmin) / 2;
                    float fy = (ymax - ymin) / 2;

                    float cx = ((fx * cos1 - fy * sin1) + Meshgrid[gridIndex + 0]) * Strides[index];
                    float cy = ((fx * sin1 + fy * cos1) + Meshgrid[gridIndex + 1]) * Strides[index];
                    float cw = (xmin + xmax) * Strides[index];
                    float ch = (ymin + ymax) * Strides[index];
                    Temp = {cls_index, cls_max, cx, cy, cw, ch, angle};

                    detectRects.push_back(Temp);
                }
            }
        }
    }

    std::sort(detectRects.begin(), detectRects.end(), [](CSXYWHR &Rect1, CSXYWHR &Rect2) -> bool
              { return (Rect1.score > Rect2.score); });

    std::cout << "NMS Before num :" << detectRects.size() << std::endl;
    for (int i = 0; i < detectRects.size(); ++i)
    {
        for (int j = i + 1; j < detectRects.size(); ++j)
        {
            if (detectRects[j].classId != -1)
            {
                if (probiou(detectRects[i], detectRects[j]) > NMSThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }

    for (int i = 0; i < detectRects.size(); i++)
    {
        float classid = detectRects[i].classId;
        if (-1 == classid)
        {
            continue;
        }
        float score = detectRects[i].score;
        float cx = detectRects[i].x;
        float cy = detectRects[i].y;
        float cw = detectRects[i].w;
        float ch = detectRects[i].h;
        float angle = detectRects[i].angle;

        float bw_ = cw > ch ? cw : ch;
        float bh_ = cw > ch ? ch : cw;

        float bt = cw > ch ? (angle - float(int(angle / pi)) * pi) : ((angle + pi / 2) - float(int((angle + pi / 2) / pi)) * pi);
        float pt1x = 0, pt1y = 0, pt2x = 0, pt2y = 0, pt3x = 0, pt3y = 0, pt4x = 0, pt4y = 0;
        xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y);

        DetectiontRects.push_back(classid);
        DetectiontRects.push_back(score);
        DetectiontRects.push_back(pt1x / InputWidth);
        DetectiontRects.push_back(pt1y / InputHeight);
        DetectiontRects.push_back(pt2x / InputWidth);
        DetectiontRects.push_back(pt2y / InputHeight);
        DetectiontRects.push_back(pt3x / InputWidth);
        DetectiontRects.push_back(pt3y / InputHeight);
        DetectiontRects.push_back(pt4x / InputWidth);
        DetectiontRects.push_back(pt4y / InputHeight);
    }

    return ret;
}
