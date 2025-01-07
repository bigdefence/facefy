# 얼굴 분석 및 운세 애플리케이션

이 프로젝트는 **Flask 기반의 웹 애플리케이션**으로, **얼굴 분석**, **관상학적 통찰**, **운세 제공** 기능을 제공합니다. OpenAI의 GPT 및 Gemini의 생성 모델과 같은 최신 AI 기술을 활용하여 사용자에게 의미 있고 개인화된 인사이트를 제공합니다.

---

## 🚀 주요 기능

- **얼굴 분석**  
  업로드된 이미지를 분석하여 다음과 같은 상세 피드백을 제공합니다:
  - 얼굴형, 눈, 코, 입, 턱 등 다양한 얼굴 특징.
  - 1-100 점수의 외모 평가 및 건설적인 개선 제안.
  - 독특한 특징과 이를 돋보이게 하는 방법 강조.

- **관상학적 분석**  
  전통 관상학 원칙을 기반으로 다음과 같은 분석 결과 제공:
  - 얼굴 특징에 따른 성격 특성.
  - 건강 관련 통찰.
  - 운세 및 미래 전망.

- **운세 제공**  
  얼굴 분석 결과를 바탕으로 다음 분야의 운세를 제공합니다:
  - **연애운**: 성격이 연애 성향 및 대인 관계 스타일에 미치는 영향.
  - **금전운**: 재정 관리 스타일과 미래 재정 전망.
  - **건강운**: 현재 건강 상태 및 미래 건강 예측.
  - **성공운**: 직업적 성공 가능성과 경력 전망.
  - **총괄 운세**: 전반적인 인생 전망.

- **사용자 친화적 인터페이스**  
  - 이미지를 업로드하고 결과를 손쉽게 확인.
  - 캐시 기반 세션 관리로 동일 이미지에 대해 재분석 불필요.

---

## 🔧 기술 스택

- **백엔드**: Flask, Python  
- **AI 모델**:  
  - OpenAI ChatGPT (GPT-4o-mini)  
  - Gemini AI (`gemini-1.5-flash` 생성 모델)  
- **이미지 처리**: PIL (Pillow)  
- **세션 관리**: Flask Session  
- **환경 관리**: dotenv  

---

## 📄 작동 방식

1. **이미지 업로드**  
   사용자는 분석할 JPEG 이미지를 업로드합니다.

2. **이미지 인코딩**  
   이미지는 base64로 인코딩되어 안전하게 처리됩니다.

3. **얼굴 분석**  
   AI 모델이 사전 정의된 지침에 따라 이미지를 분석하며, 결과는 JSON 형식으로 반환됩니다:
   - 얼굴 특징 분석.
   - 외모 평가.
   - 개선 제안.

4. **관상학 및 운세 통찰 제공**  
   분석 결과를 기반으로 심화된 관상학적 분석 및 운세 결과를 제공합니다.

5. **결과 캐싱**  
   세션 기반 캐싱으로 동일 이미지에 대해 중복 분석을 방지합니다.

---

## 🔧 설치 및 실행 방법

1. **저장소 클론**  
   ```bash
   git clone https://github.com/bigdefence/facefy.git
   cd facefy
   ```

2. **환경 설정**  
   필요한 의존성을 설치하고 환경 변수를 설정합니다:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # .env 파일에 OPENAI_API_KEY와 GEMINI_API_KEY를 추가하세요.
   ```

3. **애플리케이션 실행**  
   ```bash
   python app.py
   ```

4. **애플리케이션 접근**  
   브라우저에서 `http://127.0.0.1:5000`으로 이동합니다.

---

## 🖼️ 스크린샷

- **홈 화면**: 이미지를 업로드하여 분석 요청.
![main](https://github.com/user-attachments/assets/263e7276-ab2f-437a-85d1-9a6123942851)

- **결과 화면**: 분석 결과 및 추천 확인.
![result](https://github.com/user-attachments/assets/32f17c21-f279-47a7-88e5-4749c2a1d421)

---

## 🔒 API 통합

- **OpenAI GPT API**: 얼굴 분석 및 대화형 응답 생성에 사용.
- **Gemini AI API**: 관상학 및 운세 생성에 활용.

---


