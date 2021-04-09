//홍다예

#include <iostream> //입출력 헤더파일

#include <opencv2/opencv.hpp> //opencv 헤더파일
#include <opencv2/face.hpp> 
//얼굴분석 헤더파일

#include <Windows.h> //윈도우 c 및 c++ 헤더파일

#include <io.h> //저수준 입출력 함수와 파일처리 함수 선언 헤더파일

using namespace std; //std라는 클래스를 사용하는 것을 명시하기 위해 설정
using namespace cv; //cv라는 클래스를 사용하는 것을 명시하기 위해 설정
using namespace cv::dnn; //이미지 데이터 분석하는 dnn을 사용하기 위해 설정

int main()
{

	_finddata_t fd;
	string strFolderPath = "F:\\last4_2\\grad\\firetruck_data\\video"; //영상을 가져올 폴더 위치
	string strFindFolderPath = strFolderPath + "*.avi"; 
	intptr_t handle = _findfirst(strFindFolderPath.c_str(), &fd);  //현재 폴더 내 모든 파일을 찾는다.

	string strSaveFolderPath = "F:\\last4_2\\grad\\firetruck_data\\save_capture"; //캡처본을 저장할 폴더 위치

	int result = 0;
	do
	{

		string strFileName(fd.name); //파일 이름

		string strFileFullPath = strFolderPath + strFileName; //파일 위치

		int nCnt = 0;

		Mat orgImg, img_color; 

		VideoCapture cap(strFileFullPath.c_str());

		if (!cap.isOpened()) {
			cerr << "에러 - 카메라를 열 수 없습니다.\n";
			return -1;
		}

		int nCntFrame = 0;

		// Net net = readNet("hand_recognition_cnn.pb");


		while (1)
		{
			// 카메라로부터 캡쳐한 영상을 frame에 저장
			cap.read(orgImg);
			if (orgImg.empty()) {
				cerr << "빈 영상이 캡쳐되었습니다.\n";
				break;
			}

			nCntFrame++;

			resize(orgImg, img_color, Size(480, 270), 0, 0, INTER_LINEAR); //윈도우 크기 설정 

			Mat ycrcb; //ycrcb 영상을 저장할 matrix(행렬 구조체) 객체 선언
			cvtColor(img_color, ycrcb, COLOR_RGB2YCrCb); //RGB 컬러 영상을 YCbCr모델로 변환
			Mat img_cpy = img_color.clone(); //img_color를 복제하여 cpy에 넣음
			Mat ycrcb_split[3]; //객체 ycrcb를 3개로 쪼개서 저장할 객체 선언

			split(ycrcb, ycrcb_split); //ycrcb를 쪼개서 ycrcb_split에 저장

			// 영상을 화면에 보여줌
			imshow("Color", img_color); //컬러 영상 화면 출력
			imshow("ycrcb", ycrcb); //ycrcb 영상 화면 출력

			imshow("Y", ycrcb_split[0]); //ycrcb를 쪼개서 나온 첫번째 결과 y 영상을 화면에 출력
			imshow("Cr", ycrcb_split[1]); //ycrcb를 쪼개서 나온 두번째 결과 Cr 영상을 화면에 출력
			imshow("Cb", ycrcb_split[2]); //ycrcb를 쪼개서 나온 세번째 결과 Cb 영상을 화면에 출력

			Mat gray_tmp; //그레이 영상을 저장할 matrix (즉 행렬) 구조체 선언
			cvtColor(img_color, gray_tmp, COLOR_RGB2GRAY); //RGB 컬러 영상을 그레이 영상으로 바꿈

			for (int y = 0; y < ycrcb.cols; y++) //ycrcb의 열
			{
				for (int x = 0; x < ycrcb.rows; x++) //ycrcb의 행
				{
					unsigned char cr = ycrcb_split[1].data[y * ycrcb.rows + x];//Cr값의 데이터
					unsigned char cb = ycrcb_split[2].data[y * ycrcb.rows + x];//

					if (100 < cr && cr < 140 &&
						133 < cb && cb < 180)
					{
						gray_tmp.data[y * ycrcb.rows + x] = 255;
					}
					else
					{
						img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 0] = 0;
						img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 1] = 0;
						img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 2] = 0;

						gray_tmp.data[y * ycrcb.rows + x] = 0;
					}
				}
			}

			imshow("Color 2", img_color);
			Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
			Mat eroded, dilated, opened, closed;

			erode(gray_tmp, gray_tmp, mask, cv::Point(-1, -1), 3);
			dilate(gray_tmp, gray_tmp, mask, cv::Point(-1, -1), 3);

			cv::imshow("gray_tmp", gray_tmp); 

			Mat labels, stats, centroids;
			int numOfLables = connectedComponentsWithStats(gray_tmp, labels, stats, centroids, 8, CV_32S);

			int nMaxIdx = -1;
			int nMaxCnt = 0;

			int max_left;
			int max_top;
			int max_width;
			int max_height;

			for (int j = 1; j < numOfLables; j++)
			{
				int area = stats.at<int>(j, CC_STAT_AREA);
				int left = stats.at<int>(j, CC_STAT_LEFT);
				int top = stats.at<int>(j, CC_STAT_TOP);
				int width = stats.at<int>(j, CC_STAT_WIDTH);
				int height = stats.at<int>(j, CC_STAT_HEIGHT);

				if (area > 500)
				{

					if (nMaxCnt < area)
					{
						nMaxCnt = area;
						nMaxIdx = j;

						max_left = left;
						max_top = top;
						max_width = width;
						max_height = height;
					}
				}
			}

			for (int y = 0; y < labels.rows; y++)
			{
				for (int x = 0; x < labels.cols; ++x)
				{
					int nIdx = labels.data[(labels.cols * y * 4) + (x * 4)];

					if (nIdx == nMaxIdx)
					{
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 0] = 255;
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 1] = 255;
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 2] = 255;
					}
					else
					{
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 0] = 0;
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 1] = 0;
						img_cpy.data[(y * labels.cols * 3) + (x * 3) + 2] = 0;
					}
				}
			}

			dilate(img_cpy, img_cpy, mask, cv::Point(-1, -1), 3);
			erode(img_cpy, img_cpy, mask, cv::Point(-1, -1), 3);

			Mat gray_result;
			cvtColor(img_cpy, gray_result, COLOR_RGB2GRAY);


			/* contour

			*/
			std::vector<std::vector<cv::Point>> contours;
			std::vector<std::vector<cv::Point>> contours_filterd;
			cv::findContours(gray_result.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			for (int i = 0; i < contours.size(); i++)
			{
				if (contours[i].size() > 50)
				{
					contours_filterd.push_back(contours[i]);
					break;
				}
			}

			int nMinHigh = 1920;
			Point ptHigh;
			for (int i = 0; i < contours_filterd.size(); i++)
			{
				for (int j = 0; j < contours_filterd[i].size(); j++)
				{
					Point pt = contours_filterd[i][j];

					if (pt.y < nMinHigh)
					{
						nMinHigh = pt.y;

						ptHigh = pt;
						ptHigh.x = ptHigh.x + 5;
					}
				}
			}

			for (int i = 0; i < contours_filterd.size(); i++)
				cv::drawContours(img_color, contours_filterd, i, cv::Scalar(0, 0, 255), 2, 8);

			rectangle(img_color, Rect(ptHigh.x - 1, ptHigh.y - 1, 2, 2), Scalar(255, 255, 255), 3);
			rectangle(img_color, Rect(160, 30, 160, 80), Scalar(255, 0, 0), 3);

			if (160 < ptHigh.x && ptHigh.x < 320)
			{
				if (30 < ptHigh.y && ptHigh.y < 110)
				{
					float fX = 2560.0F / 160.0F;
					float fY = 1080.0F / 80.0F;
					//SetCursorPos((ptHigh.x - 160) * fX, (ptHigh.y - 30) * fY);
				}
			}

			cv::imshow("img_cpy", img_cpy);
			cv::imshow("result 11", img_color);

			if (nMaxIdx != -1)
			{
				if (max_height > 200)
					max_height = 200;
				Rect rect(max_left, max_top, max_width, max_height);
				Mat subImage = gray_result(rect);

				cv::resize(subImage, subImage, Size(32, 32), 0, 0, INTER_LINEAR);
				cv::imshow("subImage", subImage);

				Mat blob = blobFromImage(subImage, 1 / 255.f, Size(32, 32));


				////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
				string strSaveFilePath = strSaveFolderPath + strFileName + "_" + to_string(nCntFrame) + ".bmp"; //bmp파일 지정 위치에 저장
				imwrite(strSaveFilePath.c_str(), subImage); //파일 쓰기

				////////////////////////////////////////////////////////////////////////////////////////////////////////////

			}
			waitKey(1);
		}

	} while (_findnext(handle, &fd) == 0); //filespec에 대한 이전 호출에 지정된 _findfirst 인수와 일치하는 다음 이름을 찾음

	_findclose(handle); //지정된 검색 핸들을 닫고 연결된 리소스를 해제 

	return 0;
}
