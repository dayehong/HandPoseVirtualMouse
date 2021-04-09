//홍다예
#include <iostream> // 입출력 관련 헤더파일 포함

#include <opencv2/opencv.hpp> // OpenCV 라이브러리 포함
#include <opencv2/face.hpp> // OpenCV의 얼굴 인식 모델 관련 헤더파일 포함

#include <Windows.h> // 윈도우 api의 함수들을 위한 정의를 포함하는 헤더파일 포함

using namespace std; // std 생략하려고 사용
using namespace cv; //cv 생략하려고 사용(opencv의 모든 함수와 클래스는 cv namespace 안에 있어서 써야하기 때문에)
using namespace cv::dnn; //opencv에서 딥러닝 중 dnn 관련 함수

int main()
{
	
	FILE* file_setting; // 파일 클래스 객체 생성
	file_setting = fopen("./setting.ini", "r"); // 설정 파일을 읽음

	int nCrMin, nCrMax, nCbMin, nCbMax; //Cr,Cb 값의 범위를 정해주기 위한 변수 선언
	int nVideo = 0; //video 인지 camera 인지 설정하는 변수 선언
	int nflip = 0; //화면 반전 할 건지  설정하는 변수 선언
	int nMouseCtrl = 0; //마우스 컨트롤 변수 선언
	char FileName[1024]; //동영상 open할 때 사용  
	char desc[1024]; // file_setting에서 단어를 저장하기 위해 사용, 의미없음

	
	fscanf(file_setting, "%s %d", desc, &nVideo); // video 인지 camera 인지 확인
	fscanf(file_setting, "%s %s", desc, FileName);// 파일 이름 적용
	fscanf(file_setting, "%s %d", desc, &nCrMin); // Cr 최소값 지정
	fscanf(file_setting, "%s %d", desc, &nCrMax); // Cr 최대값 지정
	fscanf(file_setting, "%s %d", desc, &nCbMin); // Cb 최소값 지정
	fscanf(file_setting, "%s %d", desc, &nCbMax); // Cb 최대값 지정
	fscanf(file_setting, "%s %d", desc, &nflip); // 화면 반전 선택
	fscanf(file_setting, "%s %d", desc, &nMouseCtrl); // 마우스 이벤트 실행여부 확인

	Mat orgImg, img_color; //이미지 객체 선언
	VideoCapture cap; // 비디오 캡처를 위해 비디오 캡처 객체 cap 생성

	cout << FileName << endl; //파일이름 출력
	
	if (nVideo == 0) // 카메라를 사용하는 경우 parameter 를 0으로 넣음
	{
		cout << "camera running" << endl; // 카메라 실행 문구 출력
		cap = VideoCapture(0); // 비디오 캡처의 인자에 카메라(0)를 지정
	}
	else // 비디오를 open 하는 경우 
	{
		cap = VideoCapture(FileName);// 비디오 캡처의 인자에 동영상 파일명을 넣어 저장된 비디오를 불러옴
	}
	
	if (!cap.isOpened()) { // cap이 열리지 않았을 경우
		cerr << "에러 - 카메라를 열 수 없습니다.\n"; // 메세지 출력
		return -1;
	}

													// 매번 수행하지 않기 위해 while 밖에 구현
	Net net = readNet("hand_recognition_cnn.pb"); // 학습을 통해 생성 된 네트워크 모델 파일을 불러옴


	while (1)
	{
		cap.read(orgImg);  // 카메라로부터 캡쳐한 영상을 frame에 저장
		if (orgImg.empty()) { //orgImg가 비어있으면
			cerr << "빈 영상이 캡쳐되었습니다.\n"; //안내문구 출력
			continue; //다음으로 넘어감
		}
		
		// interpolation 은 linear 로 설정한다.
		resize(orgImg, img_color, Size(480, 270), 0, 0, INTER_LINEAR); // 화면의 크기를 480 X 270 으로 크기를 줄임

		
		if(nflip == 1) // flip 을 수행할 경우
			flip(img_color, img_color, 1); // 반전시킴

		img_color.convertTo(img_color, -1, 1, 30); //각 픽셀마다 30씩 밝기 올려주기

		// 손 영역을 찾기 위해 먼저 원본 이미지를 y cb cr color space 로 변경
		Mat ycrcb; //ycrcb 객체 생성
		cvtColor(img_color, ycrcb, COLOR_RGB2YCrCb); //rgb영상을 YCrCb 색상모델로 변환
		Mat img_cpy = img_color.clone(); //컬러영상 복제
		Mat ycrcb_split[3]; //ycrcb를 y,cr,cb 3채널로 쪼개기

		split(ycrcb, ycrcb_split); // cb 와 cr 영상을 나누기 위해 split opencv 함수를 이용

		namedWindow("Color"); // 윈도우 이름 지정
		//moveWindow("Color", 10, 740); // 윈도우 위치 지정
		imshow("Color", img_color); //  color image 윈도우 출력

		namedWindow("ycrcb"); // 윈도우 이름 지정
		//moveWindow("ycrcb", 490, 140); // 윈도우 위치 지정
		imshow("ycrcb", ycrcb); //  ycrcb image 윈도우 출력

		namedWindow("Cr"); // 윈도우 이름 지정
		//moveWindow("Cr", 10, 440); // 윈도우 위치 지정
		imshow("Cr", ycrcb_split[1]); // cr image 윈도우 출력
		
		namedWindow("Cb"); // 윈도우 이름 지정
		//moveWindow("Cb", 490, 440); // 윈도우 위치 지정
		imshow("Cb", ycrcb_split[2]); // cb image 영상 윈도우 출력

		Mat gray_tmp; // 이진화 이미지를 저장할 객체 생성
		cvtColor(img_color, gray_tmp, COLOR_RGB2GRAY); // color 이미지를 gray 이미지로 변환

		// cr, cb 의 각 픽셀 값을 이용하여 원하는 영역의 색상 값만을 가지고 있는 곳만 남기고 나머지 영역은 삭제
		for (int y = 0; y < ycrcb.cols; y++) //0부터 ycrcb의 열까지
		{
			for (int x = 0; x < ycrcb.rows; x++) //0부터 ycrcb의 행까지
			{
				unsigned char cr = ycrcb_split[1].data[y * ycrcb.rows + x]; // cr 값 가져오기
				unsigned char cb = ycrcb_split[2].data[y * ycrcb.rows + x]; // cb 값 가져오기

				if (nCrMin < cr && cr < nCrMax &&nCbMin < cb && cb < nCbMax)// 범위 안에 포함되어 있으면 
				{
					gray_tmp.data[y * ycrcb.rows + x] = 255; // 255값(흰색)을 gray_tmp 변수에 입력
				}
				else // 영역 안에 포함되어 있지 않으면 
				{
					// 손을 뺀 나머지 영역을 검정색으로 저장
					img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 0] = 0; //y=0
					img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 1] = 0; //cr=0
					img_color.data[y * (ycrcb.rows * 3) + (x * 3) + 2] = 0; //cb=0

					gray_tmp.data[y * ycrcb.rows + x] = 0; // 0값(검정색)을 gray_tmp 변수에 입력
				}
			}
		}

		Mat mask = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1)); // 모폴로지 트랜스포메이션을 하기 위한 커널 생성
		Mat eroded, dilated, opened, closed; // erode와 dilate 연산을 하기 위한 변수를 생성

		// 노이즈 제거
		erode(gray_tmp, gray_tmp, mask, cv::Point(-1, -1), 3); // erode 를 3회 수행 // 침식 (작은 흰 점이 사라진다.)
		dilate(gray_tmp, gray_tmp, mask, cv::Point(-1, -1), 3); // dilate 를 3회 수행 // 팽창 (다시 원래 사이즈로 늘어난다.)

		
		namedWindow("gray_tmp");// 윈도우 이름 설정
		//moveWindow("gray_tmp", 490, 740); //gray_tmp 윈도우 위치 설정
		imshow("gray_tmp", gray_tmp); //현재까지 gray_tmp 윈도우 출력

		Mat labels, stats, centroids; //라벨링을 하기 위한 객체 생성; 
	    //labels:출력 레이블 맵 행렬, stats:각각의 레이블 영역에 대한 통계 정보를 담은 행렬, centroids:각각의 레이블 영역의 무게중심좌표 정보를 담은 행렬
		
		// 가장 큰 영역을 제외한 나머지 영역을 삭제하기 위해서 아래와 같은 함수를 사용한다.
		int numOfLables = connectedComponentsWithStats(gray_tmp, 
			labels, stats, centroids, 8, CV_32S); // labeling 을 수행하는 함수, 각 영역의 위치 정보와 크기를 가져옴
		// connectedComponentsWithStats() : 레이블 맵과 각 객체 영역의 통계정보를 반환

		int nMaxIdx = -1; //최대 인덱스 값을 저장할 정수형 변수 선언
		int nMaxCnt = 0; //최대값

		int max_left; //최대 left 값
		int max_top; //최대 top 값
		int max_width; //최대 width 값
		int max_height; //최대 height 값

		for (int j = 1; j < numOfLables; j++)  //라벨링된 영역의 개수
		{
			// 영역의 크기와 위치를 가져오기 위해 값을 저장
			int area = stats.at<int>(j, CC_STAT_AREA); //영역
			int left = stats.at<int>(j, CC_STAT_LEFT); //왼쪽
			int top = stats.at<int>(j, CC_STAT_TOP); //위
			int width = stats.at<int>(j, CC_STAT_WIDTH); //가로
			int height = stats.at<int>(j, CC_STAT_HEIGHT); //높이

			// 기본 조건으로 영역의 크기가 500 이하가 되면 객체를 저장하지 않음
			if (area > 500) //영역의 크기가 500보다 클 때
			{
				if (nMaxCnt < area) //영역의 크기가 최대 값보다 클 경우
				{
					nMaxCnt = area; //현재 영역의 크기를 저장한다.
					nMaxIdx = j; //현재 인덱스를 저장한다.

					max_left = left; //현재 위치 저장(왼쪽)
					max_top = top; //현재 위치 저장(위)
					max_width = width; //현재 위치 저장(가로)
					max_height = height; //현재 위치 저장(세로)
				}
			}
		}

		// 가장 큰 영역을 갖는 인덱스를 찾는다. -> 그 인덱스를 갖는 픽셀에 모두 255로 저장
		for (int y = 0; y < labels.rows; y++) //labels의 세로(열)
		{
			for (int x = 0; x < labels.cols; ++x) //labels의 가로(행)
			{
				int nIdx = labels.data[(labels.cols * y * 4) + (x * 4)]; //컨투어 레이블, 4채널이라서 4를 곱함, 즉, 최대값의 인덱스 번호를 가져오겠다는 의미

				// 위 labeling 을 통해 확인 된 피부 영역의 label 에 해당하는 영역을 255 값으로 채움
				if (nIdx == nMaxIdx) //인덱스 값이 최대 인덱스 값일 때
				{
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 0] = 255; //3채널이라서 3을 곱함, 흰색 처리 b
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 1] = 255; //3채널이라서 3을 곱함, 흰색 처리 g
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 2] = 255; //3채널이라서 3을 곱함, 흰색 처리 r
				}
				else //nIdx가 최대가 아닐 때
				{
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 0] = 0; //검정색 처리 , b
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 1] = 0; //검정색 처리, g
					img_cpy.data[(y * labels.cols * 3) + (x * 3) + 2] = 0; //검정색 처리, r
				}
			}
		}

		// dilate (팽창) -> erode (침식) 순서로 이미지 처리 수행
		// 손 사이 빈 공간을 채워주는 역할
		dilate(img_cpy, img_cpy, mask, Point(-1, -1), 3); // 필터 내부의 가장 높은(밝은) 값으로 변환 ;팽창연산
		erode(img_cpy, img_cpy, mask, Point(-1, -1), 3); // 필터 내부의 가장 낮은(어두운) 값으로 변환 ; 침식연산

		Mat gray_result; //그레이 영상 저장할 객체 선언
		cvtColor(img_cpy, gray_result, COLOR_RGB2GRAY); //그레이 영상으로 변환

		vector<vector<Point>> contours; //외곽선의 배열
		vector<vector<Point>> contours_filterd; //필터가 된 외곽선의 배열(contours_filterd == 빨간색으로 표시한 덩어리)
		findContours(gray_result.clone(), contours, RETR_LIST, CHAIN_APPROX_SIMPLE);// 손가락의 상단 끝 점을 찾기 위해 외곽선을 추출

		
		//가장 큰 선을 저장(큰 영역인 손을 제외한 나머지를 삭제)
		for (int i = 0; i < contours.size(); i++) //i가 0부터 컨투어 크기까지
		{
			if (contours[i].size() > 50) //외곽선의 길이가 50 이상일 때
			{
				contours_filterd.push_back(contours[i]); //값을 저장
				break; //for문을 빠져나옴
			}
		}

		
		// mouse 의 x, y position 을 정하기
		int nMinHigh = 1080; //화면 최소 높이(1080이 화면의 크기이므로 가장 아래 있음) =가장 높이 있는 것=검지 손가락
		Point ptHigh; //마우스의 높이값
		for (int i = 0; i < contours_filterd.size(); i++) //i가 0부터 contours_filterd의 크기까지
		{
			for (int j = 0; j < contours_filterd[i].size(); j++) //j가 0부터 contours_filterd의 i번째 값까지
			{
				Point pt = contours_filterd[i][j]; //배열 위치값을 포인터 변수 pt에 대입

				if (pt.y < nMinHigh) //화면 높이가 pt의 y값보다 작을 때
				{
					nMinHigh = pt.y;//현재 위치 y값을 nMinHigh에 대입

					ptHigh = pt; // 외곽선에서 가장 높은 위치를 검출
					ptHigh.x = ptHigh.x + 5; //mouse의 x 위치값 크기 지정
				}
			}
		}

		for (int i = 0; i < contours_filterd.size(); i++) //0부터 contours_filterd의 크기까지
			drawContours(img_color, contours_filterd, i, Scalar(0, 0, 255), 2, 8); //컬러 영상의 손에 빨간색 외곽선 그리기

		//line(img_color, Point(160, 70), Point(320, 70), Scalar(0, 255, 255), 1); //노란색 선을 그림
		//rectangle(img_color, Rect(ptHigh.x - 1, ptHigh.y - 1, 2, 2), Scalar(255, 255, 255), 3); //검지 상단의 위치를 표시할 흰색 사각형을 그림
		//rectangle(img_color, Rect(140, 30, 200, 100), Scalar(255, 0, 0), 3); //마우스 이동 범위를 제한하기 위한 파란색 사각형을 그림

		if (nMaxIdx != -1)	// -1 인경우 큰 영역이 없다는 것. 즉 손 영역 검출 실패로 간주한다.
		{
			Rect rect(max_left, max_top, max_width, max_height); // 손만 출력 된 영역을 crop 
			Mat subImage = gray_result(rect);// 손만 출력된 영역을 그레이 영상으로 변환하여 저장

			// 32 x32 는 너무 작아서 100 X 100 으로 일단 resizing 후 출력
			resize(subImage, subImage, Size(100, 100), 0, 0, INTER_LINEAR); // 손 영역만 crop 된 영상을 100 X 100 으로 resizing하여 subImage에 저장
			
			namedWindow("crop image"); // 윈도우 이름 지정
			//moveWindow("crop image", 10, 0); // 윈도우 위치 지정
			imshow("crop image", subImage); // resizing 영상 저장

			Mat blob = blobFromImage(subImage, 1 / 255.f, Size(32, 32)); // resizing 영상을 255 로 나눠주고, 32 X 32 로 resizing
			net.setInput(blob); // network model 에 이미지를 입력하고 prediction 을 수행
			Mat prob = net.forward(); // 최종 softmax 결과를 출력

			double maxVal; // softmax의 값 중 최대값을 의미하는 변수 선언
			Point maxLoc; // softmax의 값 중 최대값을 지정할 포인터 변수 선언
			
			minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); // softmax의 값 중 최대값을 가져옴
			int digit = maxLoc.x; // softmax의 값 중 최대값을 저장할 변수 digit 선언 

			cout << digit << " (" << maxVal * 100 << "%)" << endl; // 결과 출력

			/*
			int thickness = 3; // 글씨 굵기 변수 선언
			int font = 2; // 폰트 글꼴 변수 선언
			double fontScale = 1.3; // 글자 크기 확대 비율 변수 선언
			Point ptLoc = Point(10, 50); // 문자열을 작성할 포인터 변수 선언

			// 화면에 결과를 출력
			if (digit == 0) // digit 값이 0일 때
				putText(img_color, "None", ptLoc, font, fontScale, Scalar::all(255), thickness); // "None" 문자열 출력
			if (digit == 1) // digit 값이 1일 때
				putText(img_color, "Move", ptLoc, font, fontScale, Scalar::all(255), thickness); // "Move" 문자열 출력
			if (digit == 2) // digit 값이 2일 때
				putText(img_color, "Left click", ptLoc, font, fontScale, Scalar::all(255), thickness); // "Left click" 문자열 출력
			if (digit == 3) // digit 값이 3일 때
				putText(img_color, "Right click", ptLoc, font, fontScale, Scalar::all(255), thickness);// "Right click" 문자열 출력
			if (digit == 4) // digit 값이 4일 때
				putText(img_color, "Scroll", ptLoc, font, fontScale, Scalar::all(255), thickness); // "Scroll" 문자열 출력
				*/

			namedWindow("Hand Recognition"); // 윈도우 이름 지정
			//moveWindow("Hand Recognition", 10, 140); //윈도우 위치 지정
			imshow("Hand Recognition", img_color); //컬러 image 영상 윈도우 출력
			/*
			if (140 < ptHigh.x && ptHigh.x < 340) // 화면의 가로 이동 최소,최대값을 지정
			{
				if (30 < ptHigh.y && ptHigh.y < 130) // 화면의 세로 이동 최소,최대값을 지정
				{
					// 화면 비율을 계산
					// 사이 공간 크기와 화면의 크기를 비례하여 계산
					float fX = 1920.0F / 200.0F; // X축 비율
					float fY = 1080.0F / 100.0F; // Y축 비율
					int nXPos = ((ptHigh.x - 140) * fX); // (손 위치 - 박스 x 값)*(X축 비율)
					int nYPos = ((ptHigh.y - 30) * fY); // (손 위치 - 박스 y 값)*(Y축 비율)
					//    [    |               |    ]
					//float fX = 1920.0F / 480.0F;
					//float fY = 1080.0F / 270.0F;
					//int nXPos = (ptHigh.x * fX);
					//int nYPos = (ptHigh.y * fY);

					if (digit == 0) // prediction 결과가 0일 때 실행
					{
						//
					}
					
					if (digit == 1) // prediction 결과가 1일 때 실행
					{
						if (nMouseCtrl == true) //마우스 컨트롤
						{
							SetCursorPos(nXPos, nYPos); // 마우스위 위치를 이동
						}
					}
					
					if (digit == 2) // prediction 결과가 2일 때 실행
					{
						if (nMouseCtrl == true ) //마우스 이벤트와 마우스 클릭 이벤트가 활성화되었을 때
						{
							mouse_event(MOUSEEVENTF_LEFTDOWN, nXPos, nYPos, 0, GetMessageExtraInfo()); // 마우스 왼쪽 down 
							Sleep(10);
							mouse_event(MOUSEEVENTF_LEFTUP, nXPos, nYPos, 0, GetMessageExtraInfo()); // 마우스 왼쪽을 up
						}
					}

					// prediction 결과가 3일때 실행
					if (digit == 3)
					{
						if (nMouseCtrl == true ) //마우스 이벤트와 마우스 클릭 이벤트가 활성화되었을 때
						{
							mouse_event(MOUSEEVENTF_RIGHTDOWN, nXPos, nYPos, 0, GetMessageExtraInfo()); // 마우스 오른쪽을 down
							Sleep(10); //딜레이 0.01s
							mouse_event(MOUSEEVENTF_RIGHTUP, nXPos, nYPos, 0, GetMessageExtraInfo()); // 마우스 오른쪽을 up
						}
					}

					if (digit == 4) // prediction 결과가 4인경우
					{
						bool bUp = true; // 선보다 위인지 아래인지 알기 위한 bool 타입의 변수 선언

						if (ptHigh.y > 70) // 사각 영역의 중앙 선보다 높을 때 위로 스크롤
							bUp = true;
						else
							bUp = false; // 낮을때는 아래로 스크롤

						if (nMouseCtrl == true ) //마우스 이벤트와 마우스 클릭 이벤트가 활성화되었을 때
						{
							if (bUp) // bup ==true일 때
							{
								mouse_event(MOUSEEVENTF_WHEEL, nXPos, nYPos, -100, GetMessageExtraInfo()); // 스크롤 Up
								Sleep(200); //딜레이 0.02s
							}
							else // bup == false일 때
							{
								mouse_event(MOUSEEVENTF_WHEEL, nXPos, nYPos, +100, GetMessageExtraInfo()); // 스크롤 Down
								Sleep(200); //딜레이 0.02s 
							}
						}						
					}
				}
			}
			*/

			subImage.release();
			blob.release();
		}

		// 메모리 해제
		img_color.release();
		ycrcb.release();
		img_cpy.release();

		ycrcb_split[0].release();
		ycrcb_split[1].release();
		ycrcb_split[2].release();

		mask.release();

		gray_result.release();

		gray_tmp.release();

		if (waitKey(25) >= 0)
			break;
	}

	return 0;
}
