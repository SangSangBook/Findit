import os
import cv2
import pytesseract
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
from flask_cors import CORS
from datetime import datetime
import openai
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import re
import time
import requests
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
else:
    print(f"OpenAI API 키가 설정되었습니다. (길이: {len(OPENAI_API_KEY)}자)")
    openai.api_key = OPENAI_API_KEY

# Google Cloud Vision API 설정
API_KEY = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
VISION_API_URL = f'https://vision.googleapis.com/v1/images:annotate?key={API_KEY}'

# 연관어 캐시
related_words_cache = {}

def get_related_words(query, ocr_texts):
    """OCR에서 추출된 텍스트 중에서 연관어를 찾습니다."""
    if query in related_words_cache:
        print(f"캐시에서 연관어를 가져옵니다: {query}")
        return related_words_cache[query]
    
    max_retries = 2
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            prompt = f"""
            다음은 OCR로 추출된 텍스트 목록입니다:
            {ocr_texts}
            
            이 텍스트들 중에서 '{query}'와 관련된 단어들을 찾아주세요.
            예를 들어:
            - '음식'으로 검색했을 때: '닭', '갈비', '고기' 등
            - '사람'으로 검색했을 때: '김우진', '시우리' 등
            
            결과는 쉼표로 구분된 단어 목록으로 반환해주세요.
            OCR에서 추출된 텍스트만 사용해주세요.
            """
            
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 전문적인 언어 분석가입니다. 주어진 텍스트 목록에서만 연관어를 찾아야 합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            
            related_words = response.choices[0].message.content.strip().split(',')
            related_words = [word.strip() for word in related_words]
            related_words.append(query)  # 원래 검색어도 포함
            
            # 캐시에 저장
            related_words_cache[query] = related_words
            print(f"새로운 연관어를 캐시에 저장했습니다: {query}")
            print(f"찾은 연관어 목록: {related_words}")
            return related_words
            
        except Exception as e:
            print(f"GPT API 호출 중 오류 발생 (시도 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"{retry_delay}초 후 재시도합니다...")
                time.sleep(retry_delay)
            else:
                print("최대 재시도 횟수 초과로 빈 리스트를 반환합니다.")
                return [query]  # 실패 시 원래 검색어만 반환

# Flask 앱 초기화
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5001"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# 업로드 파일이 저장될 경로 설정
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tesseract 경로 설정 (macOS에서 Homebrew로 설치한 경우)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# 파일 확장자 체크
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_subtitles(video_path):
    try:
        subtitles = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # 비디오의 FPS와 총 프레임 수 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            raise Exception("Invalid video format")
        
        # 1초마다 프레임 추출
        frame_interval = int(fps)
        
        for frame_number in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # 프레임에서 텍스트 추출
            text = extract_text_from_frame(frame)
            
            if text:
                time_in_seconds = frame_number / fps
                subtitles.append({
                    'text': text,
                    'startTime': time_in_seconds,
                    'endTime': time_in_seconds + 1  # 1초 간격으로 가정
                })
        
        cap.release()
        return subtitles
    except Exception as e:
        print(f"Error in extract_subtitles: {e}")
        raise

def extract_text_from_frame(frame):
    try:
        # 이미지 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 이미지를 base64로 인코딩
        _, img_encoded = cv2.imencode('.jpg', thresh)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # API 요청 데이터 준비
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": img_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        # API 호출
        response = requests.post(VISION_API_URL, json=request_data)
        response.raise_for_status()
        result = response.json()
        
        if 'responses' in result and result['responses']:
            if 'textAnnotations' in result['responses'][0]:
                return result['responses'][0]['textAnnotations'][0]['description'].strip()
        return ""
        
    except Exception as e:
        print(f"Error in extract_text_from_frame: {e}")
        return ""

def process_image(image_path):
    try:
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not read image file")
        
        # 이미지 전처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 이미지를 base64로 인코딩
        _, img_encoded = cv2.imencode('.jpg', thresh)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # API 요청 데이터 준비
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": img_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        # API 호출
        response = requests.post(VISION_API_URL, json=request_data)
        response.raise_for_status()
        result = response.json()
        
        text_boxes = []
        if 'responses' in result and result['responses']:
            if 'textAnnotations' in result['responses'][0]:
                for text in result['responses'][0]['textAnnotations'][1:]:  # 첫 번째는 전체 텍스트이므로 건너뛰기
                    vertices = text['boundingPoly']['vertices']
                    x_coords = [vertex['x'] for vertex in vertices]
                    y_coords = [vertex['y'] for vertex in vertices]
                    
                    text_boxes.append({
                        'text': text['description'],
                        'confidence': 1.0,  # API 키 방식에서는 신뢰도를 제공하지 않음
                        'bbox': {
                            'x1': min(x_coords),
                            'y1': min(y_coords),
                            'x2': max(x_coords),
                            'y2': max(y_coords)
                        }
                    })
        
        return text_boxes
            
    except Exception as e:
        print(f"Error in process_image: {e}")
        raise

def analyze_image(image_path, query, mode='normal'):
    try:
        print(f"이미지 분석 시작: {image_path}")
        print(f"검색어: '{query}', 모드: {mode}")
        
        # OCR로 텍스트 추출
        ocr_text, coordinates = extract_text_with_vision(image_path)
        detected_objects = []
        
        print(f"검색 가능한 텍스트 목록: {list(coordinates.keys())}")
        
        if mode == 'smart':
            # OCR에서 추출된 텍스트를 기반으로 연관어를 찾음
            ocr_texts = list(coordinates.keys())
            related_words = get_related_words(query, ocr_texts)
            print(f"연관어 목록: {related_words}")
            
            # 연관어를 단어별로 분리하여 저장
            related_words_list = []
            for word in related_words:
                # 쉼표로 구분된 단어들을 분리
                words = word.split(',')
                for w in words:
                    # 숫자와 특수문자 제거
                    w = re.sub(r'[0-9\W]+', '', w.strip())
                    if w and len(w) > 1:  # 한 글자 이상인 단어만 포함
                        related_words_list.append(w)
            
            print(f"처리된 연관어 목록: {related_words_list}")
        
        # 인접한 텍스트를 조합하여 새로운 텍스트 생성
        combined_texts = {}
        text_items = list(coordinates.items())
        for i in range(len(text_items) - 1):
            current_text, current_data = text_items[i]
            next_text, next_data = text_items[i + 1]
            
            # 현재 텍스트와 다음 텍스트의 바운딩 박스가 가까운지 확인
            current_bbox = current_data['bbox']
            next_bbox = next_data['bbox']
            
            # x 좌표가 가까운지 확인 (같은 줄에 있는지)
            if abs(current_bbox['x2'] - next_bbox['x1']) < 50:  # 50픽셀 이내
                combined_text = current_text + next_text
                combined_bbox = {
                    'x1': min(current_bbox['x1'], next_bbox['x1']),
                    'y1': min(current_bbox['y1'], next_bbox['y1']),
                    'x2': max(current_bbox['x2'], next_bbox['x2']),
                    'y2': max(current_bbox['y2'], next_bbox['y2'])
                }
                combined_texts[combined_text] = {
                    'bbox': combined_bbox,
                    'confidence': min(current_data['confidence'], next_data['confidence'])
                }
        
        # 원본 텍스트와 조합된 텍스트를 모두 사용
        all_texts = {**coordinates, **combined_texts}
        
        if mode == 'smart':
            # OCR에서 추출한 텍스트 중 연관어와 매칭되는 것 찾기
            for text, data in all_texts.items():
                text_lower = text.lower().strip()
                # 특수문자와 공백 제거
                text_clean = re.sub(r'[^\w\s]', '', text_lower)
                
                # 연관어 목록과 비교
                for word in related_words_list:
                    word_lower = word.lower().strip()
                    # 완전한 단어 매칭만 수행
                    if text_clean == word_lower:
                        print(f"매칭 발견: '{text}' (연관어: '{word}')")
                        detected_objects.append({
                            'text': text,
                            'bbox': data['bbox'],
                            'color': 'yellow'  # 연관어는 노란색
                        })
                        break  # 한 번 매칭되면 다음 텍스트로 넘어감
        else:
            # 일반 검색
            query_lower = query.lower().strip()
            for text, data in all_texts.items():
                text_lower = text.lower().strip()
                # 완전한 단어 매칭만 수행
                if text_lower == query_lower:
                    print(f"매칭 발견: '{text}' (검색어: '{query}')")
                    detected_objects.append({
                        'text': text,
                        'bbox': data['bbox'],
                        'color': 'red'  # 일반 검색은 빨간색
                    })
        
        print(f"검색 결과: {len(detected_objects)}개의 매칭된 텍스트 발견")
        for obj in detected_objects:
            print(f"매칭된 텍스트: {obj['text']}")
        
        return detected_objects
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return []

def parse_gpt_response(response):
    boxes = []
    try:
        # GPT 응답을 파싱하여 bounding box 리스트로 변환
        # 예시: "얼룩말: (100, 200, 300, 400), 사자: (500, 600, 700, 800)"
        parts = response.split(',')
        for part in parts:
            if ':' in part:
                label, coords = part.split(':')
                coords = coords.strip('()').split(',')
                if len(coords) == 4:
                    boxes.append({
                        'label': label.strip(),
                        'bbox': {
                            'x1': int(coords[0]),
                            'y1': int(coords[1]),
                            'x2': int(coords[2]),
                            'y2': int(coords[3])
                        }
                    })
        print(f"Parsed boxes: {boxes}")  # 디버깅용
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
    return boxes

def extract_text_with_vision(image_path):
    """Google Cloud Vision API를 사용하여 텍스트 추출"""
    try:
        print(f"OCR 시작: {image_path}")
        
        # 이미지 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"이미지 파일이 존재하지 않습니다: {image_path}")
            return "", {}
            
        # 이미지 읽기
        try:
            with open(image_path, 'rb') as image_file:
                content = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"이미지 파일 읽기 오류: {str(e)}")
            return "", {}
        
        # API 요청 데이터 준비
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        # API 호출
        try:
            response = requests.post(VISION_API_URL, json=request_data)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"Google Vision API 호출 오류: {str(e)}")
            return "", {}
        
        coordinates = {}
        if 'responses' in result and result['responses']:
            if 'textAnnotations' in result['responses'][0]:
                for text in result['responses'][0]['textAnnotations'][1:]:  # 첫 번째는 전체 텍스트이므로 건너뛰기
                    try:
                        vertices = text['boundingPoly']['vertices']
                        x_coords = [vertex['x'] for vertex in vertices]
                        y_coords = [vertex['y'] for vertex in vertices]
                        
                        # 전체 텍스트의 바운딩 박스 사용
                        bbox = {
                            'x1': min(x_coords),
                            'y1': min(y_coords),
                            'x2': max(x_coords),
                            'y2': max(y_coords)
                        }
                        
                        text_content = text['description'].strip()
                        print(f"OCR 인식된 텍스트: '{text_content}'")
                        
                        # 텍스트가 비어있지 않은 경우에만 저장
                        if text_content:
                            coordinates[text_content] = {
                                'bbox': bbox,
                                'confidence': 1.0  # API 키 방식에서는 신뢰도를 제공하지 않음
                            }
                    except Exception as e:
                        print(f"텍스트 처리 중 오류: {str(e)}")
                        continue
        
        # 세로 텍스트 처리 비활성화
        combined_coordinates = coordinates
        
        recognized_text = ' '.join(combined_coordinates.keys())
        print(f"OCR 결과: {len(combined_coordinates)}개의 텍스트 발견")
        print(f"전체 텍스트: {recognized_text}")
        print(f"저장된 텍스트 목록: {list(combined_coordinates.keys())}")
        
        return recognized_text, combined_coordinates
        
    except Exception as e:
        print(f"OCR 오류: {str(e)}")
        return "", {}

def combine_vertical_texts(coordinates):
    """세로로 정렬된 텍스트들을 하나의 텍스트로 결합"""
    combined_coordinates = coordinates.copy()
    processed = set()
    
    texts = list(coordinates.items())
    
    def is_korean_char(text):
        """한글 문자인지 확인"""
        return any('\u3131' <= char <= '\u318F' or '\uAC00' <= char <= '\uD7A3' for char in text)
    
    def can_be_combined(text1, bbox1, text2, bbox2):
        """두 텍스트를 결합할 수 있는지 확인"""
        # 둘 다 한글 문자이고 한 글자인지 확인
        if not (is_korean_char(text1) and is_korean_char(text2) and 
                len(text1.strip()) == 1 and len(text2.strip()) == 1):
            return False
        
        # x 좌표 중심이 비슷한지 확인
        center_x1 = (bbox1['x1'] + bbox1['x2']) / 2
        center_x2 = (bbox2['x1'] + bbox2['x2']) / 2
        if abs(center_x1 - center_x2) > 20:  # x 좌표 차이가 20픽셀 이상이면 제외
            return False
        
        # y 좌표 차이가 적절한지 확인
        y_diff = abs(bbox1['y2'] - bbox2['y1'])
        height1 = bbox1['y2'] - bbox1['y1']
        height2 = bbox2['y2'] - bbox2['y1']
        avg_height = (height1 + height2) / 2
        
        if y_diff > avg_height * 1.5:  # 세로 간격이 평균 높이의 1.5배 이상이면 제외
            return False
        
        return True
    
    while texts:
        try:
            text1, data1 = texts[0]
            if text1 in processed:
                texts.pop(0)
                continue
            
            vertical_group = [(text1, data1)]
            processed.add(text1)
            
            # 현재 그룹과 결합 가능한 다른 텍스트들 찾기
            changed = True
            while changed:
                changed = False
                for text2, data2 in texts[1:]:
                    if text2 in processed:
                        continue
                    
                    # 현재 그룹의 모든 텍스트와 결합 가능한지 확인
                    can_combine = True
                    for grouped_text, grouped_data in vertical_group:
                        if not can_be_combined(grouped_text, grouped_data['bbox'], 
                                             text2, data2['bbox']):
                            can_combine = False
                            break
                    
                    if can_combine:
                        vertical_group.append((text2, data2))
                        processed.add(text2)
                        changed = True
            
            if len(vertical_group) > 1:
                # y 좌표로 정렬
                vertical_group.sort(key=lambda x: x[1]['bbox']['y1'])
                
                # 텍스트 결합
                combined_text = ''.join(text for text, _ in vertical_group)
                
                # 바운딩 박스 계산
                combined_bbox = {
                    'x1': min(data['bbox']['x1'] for _, data in vertical_group),
                    'y1': min(data['bbox']['y1'] for _, data in vertical_group),
                    'x2': max(data['bbox']['x2'] for _, data in vertical_group),
                    'y2': max(data['bbox']['y2'] for _, data in vertical_group)
                }
                
                # 원본 텍스트 삭제
                for text, _ in vertical_group:
                    combined_coordinates.pop(text, None)
                
                # 결합된 텍스트 추가
                combined_coordinates[combined_text] = {
                    'bbox': combined_bbox,
                    'confidence': min(data['confidence'] for _, data in vertical_group),
                    'is_vertical': True
                }
                
                print(f"세로 텍스트 결합: {[text for text, _ in vertical_group]} -> {combined_text}")
            
            texts.pop(0)
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {str(e)}, 현재 텍스트: {text1 if 'text1' in locals() else 'unknown'}")
            texts.pop(0)
            continue
    
    return combined_coordinates

def semantic_similarity(text1, text2):
    """두 텍스트 간의 의미적 유사도를 계산"""
    # 간단한 구현: 공통 단어 비율
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def extract_frames_from_video(video_path, interval=1.0):
    """비디오에서 일정 간격으로 프레임 추출"""
    frames = []
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append(frame)
            timestamps.append(timestamp)
            
        frame_count += 1
    
    cap.release()
    return frames, timestamps

def process_video(video_path, query, mode='normal'):
    """비디오 처리 및 타임라인 생성"""
    print(f"비디오 처리 시작: {video_path}")
    frames, timestamps = extract_frames_from_video(video_path)
    
    timeline_results = []
    
    if mode == 'smart':
        # 연관어 검색을 위한 GPT 프롬프트
        prompt = f"""
        '{query}'와 관련된 동의어, 상위어, 하위어, 연관어를 찾아주세요.
        예시:
        - 동의어: 같은 의미의 단어
        - 상위어: 더 넓은 범주의 단어
        - 하위어: 더 구체적인 단어
        - 연관어: 관련이 있는 단어
        
        결과는 쉼표로 구분된 단어 목록으로 반환해주세요.
        """
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 전문적인 언어 분석가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        related_words = response.choices[0].message.content.strip().split(',')
        related_words = [word.strip() for word in related_words]
        related_words.append(query)  # 원래 검색어도 포함
        
        print(f"연관어 목록: {related_words}")
    
    for frame, timestamp in zip(frames, timestamps):
        # 프레임을 임시 이미지 파일로 저장
        temp_frame_path = os.path.join(UPLOAD_FOLDER, f'temp_frame_{timestamp}.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # OCR 실행 (Google Cloud Vision API 사용)
            ocr_text, coordinates = extract_text_with_vision(temp_frame_path)
            
            if coordinates:
                detected_texts = []
                for text, data in coordinates.items():
                    if mode == 'smart':
                        # 연관어 검색
                        for word in related_words:
                            if word.lower() in text.lower():
                                detected_texts.append({
                                    'text': text,
                                    'confidence': data['confidence'],
                                    'bbox': data['bbox'],
                                    'color': 'yellow'  # 연관어는 노란색
                                })
                                break
                    else:
                        # 일반 검색
                        if query.lower() in text.lower():
                            detected_texts.append({
                                'text': text,
                                'confidence': data['confidence'],
                                'bbox': data['bbox'],
                                'color': 'red'  # 일반 검색은 빨간색
                            })
                
                if detected_texts:
                    timeline_results.append({
                        'timestamp': timestamp,
                        'texts': detected_texts
                    })
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
    
    return timeline_results

# 비디오 파일 업로드 처리
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        print("Received upload request")
        if 'video' not in request.files:
            print("No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            print(f"Invalid file format: {file.filename}")
            return jsonify({'error': 'Invalid file format. Allowed formats: mp4, avi, mov, mkv'}), 400
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Saving file to: {filepath}")
        
        file.save(filepath)
        print("File saved successfully")
        
        # 자막 추출은 별도의 엔드포인트로 분리
        return jsonify({
            'message': 'Video uploaded successfully',
            'file_url': f'/uploads/{filename}'
        })
    except Exception as e:
        print(f"Error in upload_video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/extract-subtitles/<filename>', methods=['POST'])
def extract_subtitles_endpoint(filename):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        print("Starting subtitle extraction")
        subtitles = extract_subtitles(filepath)
        print(f"Extracted {len(subtitles)} subtitles")
        
        return jsonify({
            'message': 'Subtitles extracted successfully',
            'subtitles': subtitles
        })
    except Exception as e:
        print(f"Error in extract_subtitles_endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 이미지 파일 업로드 처리
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Allowed formats: png, jpg, jpeg'}), 400
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 이미지는 즉시 URL만 반환
        return jsonify({
            'message': 'Image uploaded successfully',
            'file_url': f'/uploads/{filename}'
        })
    except Exception as e:
        print(f"Error in upload_image: {e}")
        return jsonify({'error': str(e)}), 500

# 이미지 분석 엔드포인트 추가
@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    try:
        if 'image' not in request.files or 'query' not in request.form:
            return jsonify({'error': '이미지와 쿼리가 필요합니다'}), 400

        file = request.files['image']
        query = request.form['query']
        mode = request.form.get('mode', 'normal')  # 'normal' 또는 'smart'

        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400

        allowed_extensions = ('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov', '.wmv')
        if not file or not file.filename.lower().endswith(allowed_extensions):
            return jsonify({'error': '지원하지 않는 파일 형식입니다'}), 400

        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # 이미지 파일 처리
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                detected_objects = analyze_image(filepath, query, mode)
                
                return jsonify({
                    'objects': detected_objects,
                    'type': 'image',
                    'mode': mode,
                    'file_url': f'/uploads/{filename}'
                })
            
            # 비디오 파일 처리
            else:
                timeline_results = process_video(filepath, query, mode)
                return jsonify({
                    'type': 'video',
                    'file_url': f'/uploads/{filename}',
                    'timeline': timeline_results
                })

        finally:
            # 이미지 파일인 경우에만 삭제 (비디오는 클라이언트에서 재생해야 하므로 유지)
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        # 에러 발생 시 파일 정리
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
