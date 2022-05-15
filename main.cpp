#include<windows.h>
#include<string.h>
#include <iostream>
#include <cstdlib>
#include <cmath> // log10()
// SIMD
#include <xmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <time.h>
#include <iomanip> // std::setprecision()
using namespace std;

#define sqr(x) (x*x)
#define vecDiv(x,y) {for(double& i: x) i/=y;}

int MYSCREEN_WIDTH = 1920;
int MYSCREEN_HEIGHT = 1080;
HINSTANCE GlobalHInst;
bool DitherArg = true;
bool OrderedDitherArg = false;
bool UseSIMD = false;
BYTE BayerMatrixSizeArg = 16;
BYTE BWBoundArg = 128;
BYTE src[1920*1080*4]; // 存储原图像，用以计算生成图像的效果
const int DEPTH = 255; // 色彩空间深度
//int frame = 0; // 帧
//double frames_per_second;
//clock_t start, current;

int psnr_sum = 0;
LRESULT CALLBACK WndProc( HWND, UINT, WPARAM, LPARAM ) ;

const BYTE BayerMatrix_1[1][1] = {BWBoundArg};

//const BYTE BayerMatrix_2[2][2] = {0, 2,
//                                  3, 1};

const BYTE BayerMatrix_2[2][2] = {0, 128,
                                  192, 64};

//const BYTE BayerMatrix_4[4][4] = {0, 8, 2, 10,
//                                  12, 4, 14, 6,
//                                  3, 11, 1, 9,
//                                  15, 7, 13, 5};

const BYTE BayerMatrix_4[4][4] = {0, 128, 32, 160,
                                  192, 64, 224, 96,
                                  48, 176, 16, 144,
                                  240, 112, 208, 80};

//const BYTE BayerMatrix_8[8][8] = {0, 32, 8, 40, 2, 34, 10, 42,
//                                  48, 16, 56, 24, 50, 18, 58, 26,
//                                  12, 44, 4, 36, 14, 46, 6, 38,
//                                  60, 28, 52, 20, 62, 30, 54, 22,
//                                  3, 35, 11, 43, 1, 33, 9, 41,
//                                  51, 19, 59, 27, 49, 17, 57, 25,
//                                  15, 47, 7, 39, 13, 45, 5, 37,
//                                  63, 31, 55, 23, 61, 29, 53, 21};

const BYTE BayerMatrix_8[8][8] = {0, 128, 32, 160, 8, 136, 40, 168,
                                  192, 64, 224, 96, 200, 72, 232, 104,
                                  48, 176, 16, 144, 56, 184, 24, 152,
                                  240, 112, 208, 80, 248, 120, 216, 88,
                                  12, 140, 44, 172, 4, 132, 36, 164,
                                  204, 76, 236, 108, 196, 68, 228, 100,
                                  60, 188, 28, 156, 52, 180, 20, 148,
                                  252, 124, 220, 92, 244, 116, 212, 84};

const BYTE BayerMatrix_16[16][16] ={ 0, 128, 32, 160, 8, 136, 40, 168, 2, 130, 34, 162, 10, 138, 42, 170,
                                     192, 64, 224, 96, 200, 72, 232, 104, 194, 66, 226, 98, 202, 74, 234, 106,
                                     48, 176, 16, 144, 56, 184, 24, 152, 50, 178, 18, 146, 58, 186, 26, 154,
                                     240, 112, 208, 80, 248, 120, 216, 88, 242, 114, 210, 82, 250, 122, 218, 90,
                                     12, 140, 44, 172, 4, 132, 36, 164, 14, 142, 46, 174, 6, 134, 38, 166,
                                     204, 76, 236, 108, 196, 68, 228, 100, 206, 78, 238, 110, 198, 70, 230, 102,
                                     60, 188, 28, 156, 52, 180, 20, 148, 62, 190, 30, 158, 54, 182, 22, 150,
                                     252, 124, 220, 92, 244, 116, 212, 84, 254, 126, 222, 94, 246, 118, 214, 86,
                                     3, 131, 35, 163, 11, 139, 43, 171, 1, 129, 33, 161, 9, 137, 41, 169,
                                     195, 67, 227, 99, 203, 75, 235, 107, 193, 65, 225, 97, 201, 73, 233, 105,
                                     51, 179, 19, 147, 59, 187, 27, 155, 49, 177, 17, 145, 57, 185, 25, 153,
                                     243, 115, 211, 83, 251, 123, 219, 91, 241, 113, 209, 81, 249, 121, 217, 89,
                                     15, 143, 47, 175, 7, 135, 39, 167, 13, 141, 45, 173, 5, 133, 37, 165,
                                     207, 79, 239, 111, 199, 71, 231, 103, 205, 77, 237, 109, 197, 69, 229, 101,
                                     63, 191, 31, 159, 55, 183, 23, 151, 61, 189, 29, 157, 53, 181, 21, 149,
                                     255, 127, 223, 95, 247, 119, 215, 87, 253, 125, 221, 93, 245, 117, 213, 85 };

BYTE errDiffusion(BYTE err, int dist){
    BYTE res;
    switch (dist)
    {
        case 1:
            res = (err>>3) + (err>>6) + (err>>7);
            break;
        case 2:
            res = (err>>3) - (err>>6) - (err>>7);
            break;
        case 3:
            res = (err>>4);
            break;
        case 4:
            res = (err>>6) + (err>>7);
            break;
        default:
            res = 0;
            break;
    }
    return res;
}

BYTE CHARToBYTE(TCHAR *ch, int len){
    BYTE res = 0;
    for(int i=0; i<len; i++){
        res *= 10;
        res += ch[i]-'0';
    }
    return res;
}

int CHARToInt(TCHAR *ch, int len){
    int res = 0;
    for(int i=0; i<len; i++){
        res *= 10;
        res += ch[i]-'0';
    }
    return res;
}

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow )
{
    static TCHAR szAppName[] = TEXT( "DrawDemo" ) ;
    HWND hwnd;
    MSG msg ;
    WNDCLASS wndclass ;

    GlobalHInst = hInstance;

    wndclass.lpfnWndProc    = WndProc ;
    wndclass.style            = CS_HREDRAW | CS_VREDRAW ;
    wndclass.hInstance        = hInstance ;
    wndclass.lpszClassName    = szAppName ;
    wndclass.hbrBackground    = (HBRUSH) GetStockObject( WHITE_BRUSH ) ;
    wndclass.cbClsExtra        = 0 ;
    wndclass.cbWndExtra        = 0 ;
    wndclass.hCursor        = LoadCursor( NULL, IDC_ARROW ) ;
    wndclass.hIcon            = LoadIcon( NULL, IDI_APPLICATION ) ;
    wndclass.lpszMenuName    = NULL ;

    if( !RegisterClass( &wndclass ) )
    {
        MessageBox( NULL, TEXT( " " ), TEXT( " " ), MB_OK  ) ;
        return 0 ;
    }

    hwnd = CreateWindow(szAppName, TEXT( "Demo" ), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 500, 500, NULL, NULL, hInstance, NULL ) ;

    ShowWindow( hwnd, iCmdShow ) ;
    UpdateWindow( hwnd ) ;
//    start = clock();

    while( GetMessage( &msg, NULL, 0, 0 ) )
    {
        TranslateMessage( &msg ) ;
        DispatchMessage( &msg ) ;
    }

    return msg.wParam ;
}

void dither_jarvis(int W, int H, BYTE *buff, int bmW, int bmH, BYTE *bayerMatrix) {
    // reverse up && down && count y (rgba -> rgby)
    int index;
    BYTE r, g, b, y;
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            index = (i * W + j) * 4;
            r = buff[index];
            g = buff[index+1];
            b = buff[index+2];
            buff[index+3] = (r>>2) + (g>>1) + (b>>2) + (r&0x1) + (g&0x1) + (b&0x1) - (r&g&b&0x1);
        }
    }

    BYTE IQxy, IRxy, tmperr;
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            index = (i*W+j)*4;
            y = buff[index+3];
            IQxy = y > bayerMatrix[(i%bmH)*bmW+j%bmW] || y==0xff ? 1 : 0;
            if(IQxy) {
                buff[index] = 0xff;
                buff[index+1] = 0xff;
                buff[index+2] = 0xff;
            } else {
                buff[index] = 0;
                buff[index+1] = 0;
                buff[index+2] = 0;
//                printf("0 ");
            }
            buff[index+3] = 0;
            IRxy = IQxy ? 255-y : y;
            for(int r=i; r<=i+2 && r<H; r++){
                for(int c=j; c<=j+2 && c<W; c++){
                    if(r==i && c==j) continue;
                    index = (r*W+c)*4;
                    y = buff[index+3];
                    tmperr = errDiffusion(IRxy, r-i+c-j);
                    if(IQxy) buff[index+3] = tmperr>y ? 0 : y-tmperr;
                    else buff[index+3] = 255-tmperr<y ? 255 : y+tmperr;
                }
            }
            for(int r=i+1; r<=i+2 && r<H; r++){
                for(int c=j-1; c>=j-2 && c>=0; c--){
                    index = (r*W+c)*4;
                    y = buff[index+3];
                    tmperr = errDiffusion(IRxy, r-i+j-c);
                    if(IQxy) buff[index+3] = tmperr>y ? 0 : y-tmperr;
                    else buff[index+3] = 255-tmperr<y ? 255 : y+tmperr;
                }
            }
        }
//        printf("\n");
    }
//    printf("=================\n");
}

void black_white(int W, int H, BYTE *buff, int bmW, int bmH, const BYTE *bayerMatrix) {
    int index;
    BYTE r, g, b, y;
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            index = (i * W + j) * 4;
            r = buff[index];
            g = buff[index+1];
            b = buff[index+2];
            y = (r>>2) + (g>>1) + (b>>2) + (r&0x1) + (g&0x1) + (b&0x1) - (r&g&b&0x1);
            y = y > bayerMatrix[(i%bmH)*bmW+j%bmW] || y==0xff ? 0xff : 0;
            buff[index] = y;
            buff[index+1] = y;
            buff[index+2] = y;
            buff[index+3] = 0;
        }
    }
}

void black_white_simd(int W, int H, BYTE *buff, int bmW, int bmH, const BYTE *bayerMatrix){
    int index = 0;
    BYTE r, g, b, y;
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            r = buff[index];
            g = buff[index+1];
            b = buff[index+2];
            y = (r>>2) + (g>>1) + (b>>2) + (r&0x1) + (g&0x1) + (b&0x1) - (r&g&b&0x1);
            *(unsigned int*)(buff + index) = y > bayerMatrix[(i%bmH)*bmW+j%bmW] || y==0xff ? 0x00ffffff : 0;
            index += 4;
        }
    }
}

// PSNR/峰值信噪比 (Peak Signal-to-Noise Ratio)
// 越大越好
// MSE = (sum_{i,j}((source(i,j)-dst(i,j))^2))/(mn)
// PSNR = 10*log_10(DEPTH^2/MSE)
double psnr(int W, int H, const BYTE* source, const BYTE* dst){
    double mse[3] = {0,0,0};
    // 计算三通道的MSE，并求平均值作为最终的MSE
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            int index = (i*W+j)*4;
            for(int k=0; k<3; k++){
                mse[k] += ((int)source[index + k] - (int)dst[index + k])
                          *((int)source[index + k] - (int)dst[index + k]);
            }

        }
    }
    for(double & i : mse){
        i /= W*H;
    }
    double final_mse = (mse[0]+mse[1]+mse[2])/3;
    double psnr = 10*log10(DEPTH*DEPTH/final_mse);
    return psnr;
}

// SSIM/结构相似性 (Structural Similarity)
// Range: [-1, 1]
// When ssim == 1, two images are the same.
// l = (2mu_x*mu_y+c1)/(mu_x^2+mu_y^2+c1)
// c = (2sigma_x*sigma_y+c2)/(sigma_x^2+sigma_y^2+c2)
// s = (sigma_xy + c3)/(sigma_x + sigma_y + c3)
// Thus, s = 1.
// SSIM = l^alpha * c^beta * s^gamma
//      = l^alpha * c^beta
double ssim(int W, int H, const BYTE* source, const BYTE* dst){
    // c1 = (0.01*DEPTH)^2, c2 = (0.03*DEPTH)^2
    const double c1 = 6.5025, c2 = 58.5225;
    // 类似PSNR，计算三通道分别的SSIM并求平均值
    double mu_x[3], mu_y[3], sigma_x[3], sigma_y[3], sigma_xy[3];
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            int index = (i*H+j)*4;
            for(int k=0; k<3; k++){
                mu_x[k] += (int) source[index+k];
                mu_y[k] += (int) dst[index+k];
            }
        }
    }
    const int pixel_num = H*W;
    vecDiv(mu_x, pixel_num);
    vecDiv(mu_y, pixel_num);
//    for(double& i: mu_x){
//        i /= pixel_num;
//    }
//    for(double& i: mu_y){
//        i /= pixel_num;
//    }
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            int index = (i*H+j)*4;
            for(int k=0; k<3; k++){
                sigma_x[k] += (((int)source[index+k])-mu_x[k])*(((int)source[index+k])-mu_x[k]);
                sigma_y[k] += (((int)source[index+k])-mu_y[k])*(((int)source[index+k])-mu_y[k]);
                sigma_xy[k] += (((int)source[index+k])-mu_x[k])*(((int)source[index+k])-mu_y[k]);
            }
        }
    }
    // 为了得到无偏估计，除以pixel_num-1而非pixel_num
    vecDiv(sigma_x, pixel_num-1);
    vecDiv(sigma_y, pixel_num-1);
    vecDiv(sigma_xy, pixel_num-1);
//    for(double& i: sigma_x){
//        i /= pixel_num-1;
//    }
//    for(double& i: sigma_y){
//        i /= pixel_num-1;
//    }
//    for(double& i: sigma_xy){
//        i /= pixel_num-1;
//    }
    double l[3], c[3], ssim[3];
    for(int i=0; i<3; i++){
        l[i] = (2*mu_x[i]*mu_y[i]+c1)/(sqr(mu_x[i])+sqr(mu_y[i])+c1);
        c[i] = (2*sigma_x[i]*sigma_y[i]+c2)/(sqr(sigma_x[i])+sqr(sigma_y[i])+c2);
        // To simplify, let alpha=beta=gamma=1.
        ssim[i] = l[i]*c[i];
    }
    return (ssim[0]+ssim[1]+ssim[2])/3;
}

void processScreenShootPixels()
{
    clock_t start = clock();
    // copy screen to bitmap, get pixels of screen
    HDC     src_hdc = GetDC(NULL);
    HDC     src_mdc = CreateCompatibleDC(src_hdc);
    BITMAPINFO bitmap;
    bitmap.bmiHeader.biSize = sizeof(bitmap.bmiHeader);
    bitmap.bmiHeader.biWidth = MYSCREEN_WIDTH;
    bitmap.bmiHeader.biHeight = MYSCREEN_HEIGHT;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount = 32;
    bitmap.bmiHeader.biCompression = BI_RGB;
    bitmap.bmiHeader.biSizeImage = MYSCREEN_WIDTH * 4 * MYSCREEN_HEIGHT;
    bitmap.bmiHeader.biClrUsed = 0;
    bitmap.bmiHeader.biClrImportant = 0;
    BYTE *pixels;
    HBITMAP src_hbitmap = CreateDIBSection(src_mdc, &bitmap, DIB_RGB_COLORS, (void **)(&pixels), NULL, 0);
    HGDIOBJ src_mdc_hbitmap_obj = SelectObject(src_mdc, src_hbitmap);
    BitBlt(src_mdc, 0, 0, MYSCREEN_WIDTH, MYSCREEN_HEIGHT, src_hdc, 1, 1, SRCCOPY);

    // clean up, but do not clean src_bitmap
    SelectObject(src_mdc, src_mdc_hbitmap_obj);
    DeleteDC(src_mdc);
    ReleaseDC(NULL, src_hdc);


    BYTE *bayerMatrix = (BYTE *)malloc(sizeof(BYTE));;
    memcpy(bayerMatrix, BayerMatrix_1, sizeof(BYTE));
    int bayerMatrixSize = 1;
    if(OrderedDitherArg) {
        switch (BayerMatrixSizeArg) {
            case 1:
            {
                bayerMatrixSize = 1;
                bayerMatrix = (BYTE *)malloc(sizeof(BYTE));
                memcpy(bayerMatrix, BayerMatrix_1, sizeof(BYTE));
                break;
            }
            case 2:
            {
                bayerMatrixSize = 2;
                bayerMatrix = (BYTE *)malloc(2*2*sizeof(BYTE));
                memcpy(bayerMatrix, BayerMatrix_2, 2*2*sizeof(BYTE));
                break;
            }
            case 4:
            {
                bayerMatrix = (BYTE *)malloc(4*4*sizeof(BYTE));
                memcpy(bayerMatrix, BayerMatrix_4, 4*4*sizeof(BYTE));
                bayerMatrixSize = 4;
                break;
            }
            case 8:
            {
                bayerMatrix = (BYTE *)malloc(8*8*sizeof(BYTE));
                memcpy(bayerMatrix, BayerMatrix_8, 8*8*sizeof(BYTE));
                bayerMatrixSize = 8;
                break;
            }
            default:
            {
                bayerMatrix = (BYTE *)malloc(16*16*sizeof(BYTE));
                memcpy(bayerMatrix, BayerMatrix_16, 16*16*sizeof(BYTE));
                bayerMatrixSize = 16;
                break;
            }
        }
    }
//    BYTE src[MYSCREEN_WIDTH*MYSCREEN_HEIGHT*4];
//    for(int i=0; i<MYSCREEN_WIDTH*MYSCREEN_HEIGHT*4; i++){
//        src[i]=pixels[i];
//    }
    memcpy(src, pixels, sizeof(src));

    if(DitherArg) {
        dither_jarvis(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, pixels, bayerMatrixSize, bayerMatrixSize, bayerMatrix);
    } else {
        if(UseSIMD){
            black_white_simd(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, pixels, bayerMatrixSize, bayerMatrixSize, bayerMatrix);
        }
        else {
            black_white(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, pixels, bayerMatrixSize, bayerMatrixSize, bayerMatrix);
        }
    }
//    frame++;
    cout << "PSNR: " << psnr(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, src, pixels) << endl;
    cout << "SSIM: " << ssim(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, src, pixels) << endl;
//    cout << "FRAME: " << frame << endl;
    // The two seems working well

    // draw pixels in extend screen
    HDC     dst_hdc = GetDC(NULL);
    HDC     dst_mdc = CreateCompatibleDC(dst_hdc);
    HBITMAP dst_hbitmap = CreateBitmap(MYSCREEN_WIDTH, MYSCREEN_HEIGHT, 1, 32, pixels);
    HGDIOBJ dst_mdc_hbitmap_obj = SelectObject(dst_mdc, dst_hbitmap);
    StretchBlt (dst_hdc, MYSCREEN_WIDTH, MYSCREEN_HEIGHT, MYSCREEN_WIDTH, -MYSCREEN_HEIGHT, dst_mdc, 0, 0, MYSCREEN_WIDTH, MYSCREEN_HEIGHT, SRCCOPY);

    // clean up
    SelectObject(dst_mdc, dst_mdc_hbitmap_obj);
    DeleteDC(dst_mdc);
    ReleaseDC(NULL, dst_hdc);
    DeleteObject(dst_hbitmap);
    DeleteObject(src_hbitmap);

    clock_t current = clock();
    double time_taken = double(current - start) / double(CLOCKS_PER_SEC);
    double frames_per_second = 1 / time_taken;
    cout << "Frames per second: " << fixed << frames_per_second << setprecision(4) << endl;
}

LRESULT CALLBACK WndProc( HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    HFONT hFont;
    static HWND screenWidthText, screenWidthEdit;
    static HWND screenHeightText, screenHeightEdit;
    static HWND ditherText, ditherYes, ditherNo;
    static HWND orderedDitherText, orderedDitherYes, orderedDitherNo;
    static HWND bayerMatrixSizeText, bayerMatrixSizeEdit;
    static HWND boundText, boundEdit;
    static HWND simdText, simdYes, simdNo;
    int wmId, wmEvent;
    TCHAR screenWidth[5], screenHeight[5], bwBoundInput[5], bayerMatrixSizeInput[5];
    switch( message ){
        case WM_CREATE:
        {
            hFont = CreateFont(-14, -7, 0, 0, 400, FALSE, FALSE, FALSE,DEFAULT_CHARSET, OUT_CHARACTER_PRECIS, CLIP_CHARACTER_PRECIS, DEFAULT_QUALITY, FF_DONTCARE, TEXT("微软雅黑"));

            screenWidthText = CreateWindow(TEXT("static"), TEXT("屏幕宽："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT, 10, 10, 150, 30, hwnd, (HMENU)1, GlobalHInst, NULL);
            screenWidthEdit = CreateWindow(TEXT("edit"), TEXT("1920"), WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 165, 10, 100, 30, hwnd, (HMENU)2, GlobalHInst, NULL);

            screenHeightText = CreateWindow(TEXT("static"), TEXT("屏幕高："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT, 10, 40, 150, 30, hwnd, (HMENU)3, GlobalHInst, NULL);
            screenHeightEdit = CreateWindow(TEXT("edit"), TEXT("1080"), WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 165, 40, 100, 30, hwnd, (HMENU)4, GlobalHInst, NULL);

            ditherText = CreateWindow(TEXT("static"), TEXT("Dither："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT,10, 70, 150, 30, hwnd, (HMENU)5, GlobalHInst, NULL);
            ditherYes = CreateWindow(TEXT("button"), TEXT("是"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON|WS_GROUP, 165, 70, 50, 30, hwnd, (HMENU)6, GlobalHInst, NULL);
            ditherNo = CreateWindow(TEXT("button"), TEXT("否"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON, 225, 70, 50, 30, hwnd, (HMENU)7, GlobalHInst, NULL);

            orderedDitherText = CreateWindow(TEXT("static"), TEXT("QMatrix："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT,10, 100, 150, 30, hwnd, (HMENU)8, GlobalHInst, NULL);
            orderedDitherYes = CreateWindow(TEXT("button"), TEXT("是"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON|WS_GROUP, 165, 100, 50, 30, hwnd, (HMENU)9, GlobalHInst, NULL);
            orderedDitherNo = CreateWindow(TEXT("button"), TEXT("否"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON, 225, 100, 50, 30, hwnd, (HMENU)10, GlobalHInst, NULL);

            bayerMatrixSizeText = CreateWindow(TEXT("static"), TEXT("QM DIM："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT, 10, 130, 150, 30, hwnd, (HMENU)11, GlobalHInst, NULL);
            bayerMatrixSizeEdit = CreateWindow(TEXT("edit"), TEXT("16"), WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 165, 130, 100, 30, hwnd, (HMENU)12, GlobalHInst, NULL);

            boundText = CreateWindow(TEXT("static"), TEXT("黑白阈值："), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT, 10, 160, 150, 30, hwnd, (HMENU)13, GlobalHInst, NULL);
            boundEdit = CreateWindow(TEXT("edit"), TEXT("128"), WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 165, 160, 100, 30, hwnd, (HMENU)14, GlobalHInst, NULL);

            orderedDitherText = CreateWindow(TEXT("static"), TEXT("Use SIMD: "), WS_CHILD|WS_VISIBLE|SS_CENTERIMAGE|SS_RIGHT,10, 190, 150, 30, hwnd, (HMENU)15, GlobalHInst, NULL);
            orderedDitherYes = CreateWindow(TEXT("button"), TEXT("是"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON|WS_GROUP, 165, 190, 50, 30, hwnd, (HMENU)16, GlobalHInst, NULL);
            orderedDitherNo = CreateWindow(TEXT("button"), TEXT("否"), WS_CHILD|WS_VISIBLE|BS_LEFT|BS_AUTORADIOBUTTON, 225, 190, 50, 30, hwnd, (HMENU)17, GlobalHInst, NULL);

            SendMessage(ditherText, WM_SETFONT, (WPARAM)hFont, NULL);
            SendMessage(ditherYes, WM_SETFONT, (WPARAM)hFont, NULL);
            SendMessage(ditherNo, WM_SETFONT, (WPARAM)hFont, NULL);
            SendMessage(boundText, WM_SETFONT, (WPARAM)hFont, NULL);
            SendMessage(boundEdit, WM_SETFONT, (WPARAM)hFont, NULL);

            break;
        }
        case WM_COMMAND:
        {
            wmId    = LOWORD(wParam);
            wmEvent = HIWORD(wParam);
            switch (wmId){
                case 2:
                {
                    int strlen = GetWindowText(screenWidthEdit, screenWidth, 5);
                    if(strlen != 0) MYSCREEN_WIDTH = CHARToInt(screenWidth, strlen);
                    break;
                }
                case 4:
                {
                    int strlen = GetWindowText(screenHeightEdit, screenHeight, 5);
                    if(strlen != 0) MYSCREEN_HEIGHT = CHARToInt(screenHeight, strlen);
                    break;
                }
                case 6:
                    if(wmEvent == BN_CLICKED) DitherArg = true;
                    break;
                case 7:
                    if(wmEvent == BN_CLICKED) DitherArg = false;
                    break;
                case 9:
                    if(wmEvent == BN_CLICKED) OrderedDitherArg = true;
                    break;
                case 10:
                    if(wmEvent == BN_CLICKED) OrderedDitherArg = false;
                    break;
                case 12:
                {
                    int strlen = GetWindowText(bayerMatrixSizeEdit, bayerMatrixSizeInput, 5);
                    if(strlen != 0) BayerMatrixSizeArg = CHARToBYTE(bayerMatrixSizeInput, strlen);
                    break;
                }
                case 14:
                {
                    int strlen = GetWindowText(boundEdit, bwBoundInput, 5);
                    if(strlen != 0) BWBoundArg = CHARToBYTE(bwBoundInput, strlen);
                    break;
                }
                case 16:
                    if(wmEvent == BN_CLICKED) UseSIMD = true;
                    break;
                case 17:
                    if(wmEvent == BN_CLICKED) UseSIMD = false;
                    break;
                default:
                    return DefWindowProc(hwnd, message, wParam, lParam);
            }
            break;
        }
        case WM_PAINT:
        {
            processScreenShootPixels();
            break;
        }
        case WM_DESTROY:
        {
            DeleteObject(hFont);
            PostQuitMessage( 0 ) ;
            break;
        }
        default:
            return DefWindowProc( hwnd, message, wParam, lParam ) ;
    }

    return 0;
}
