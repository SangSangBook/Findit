# Findit

Findit은 이미지와 비디오에서 텍스트를 추출하고 분석하는 서비스입니다.

## 주요 기능

- 이미지에서 텍스트 추출 (OCR)
- 비디오에서 자막/텍스트 추출
- 추출된 텍스트의 의미 분석
- 다국어 지원 (한국어, 영어)

## 기술 스택

- Frontend: React, TypeScript
- Backend: Python, Flask
- OCR: EasyOCR, PaddleOCR, Tesseract
- AI: OpenAI GPT

## 설치 및 실행 방법

1. 저장소 클론
```bash
git clone https://github.com/SangSangBook/Findit.git
```

2. 의존성 설치
```bash
# Backend
pip install -r requirements.txt

# Frontend
npm install
```

3. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:
```
OPENAI_API_KEY=your_api_key_here
```

4. 서버 실행
```bash
# Backend
python app.py

# Frontend
npm start
```

## 라이센스

MIT License
