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
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account
import googlemaps
import yt_dlp

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

# OpenAI 클라이언트 초기화
client = OpenAI()  # 환경 변수에서 자동으로 API 키를 가져옴

# Google Maps 클라이언트 초기화
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_CLOUD_VISION_API_KEY'))

# Google Cloud Vision 클라이언트 초기화
try:
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        vision_client = vision.ImageAnnotatorClient()
    print("Google Cloud Vision 클라이언트가 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"Google Cloud Vision 클라이언트 초기화 오류: {str(e)}")
    vision_client = None

# OCR 결과를 저장할 전역 변수
ocr_results_cache = {}

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
            
            # 이미 설정된 API 키 사용
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
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
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
        
        # coordinates가 비어있으면 빈 배열 반환
        if not coordinates:
            print("OCR 결과가 비어있습니다.")
            return []
            
        print(f"검색 가능한 텍스트 목록: {list(coordinates.keys())}")
        
        # 일반 검색
        query_lower = query.lower().strip()
        for text, data in coordinates.items():
            text_lower = text.lower().strip()
            # 특수문자와 공백 제거
            text_clean = re.sub(r'[^\w\s]', '', text_lower)
            
            # 단어의 시작이나 끝에서 매칭
            if (text_clean == query_lower or  # 완전한 단어 매칭
                text_clean.startswith(query_lower) or  # 단어로 시작
                text_clean.endswith(query_lower) or  # 단어가 포함됨
                query_lower in text_clean):  # 단어가 포함됨
                print(f"매칭 발견: '{text}' (검색어: '{query}')")
                detected_objects.append({
                    'text': text,
                    'bbox': data['bbox'],
                    'confidence': data['confidence']
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
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Vision API 클라이언트를 직접 사용하는 대신 HTTP 요청 사용
        with open(image_path, 'rb') as image_file:
            content = base64.b64encode(image_file.read()).decode('utf-8')
        
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
        response = requests.post(VISION_API_URL, json=request_data)
        response.raise_for_status()
        result = response.json()
        
        if 'responses' not in result or not result['responses'] or 'textAnnotations' not in result['responses'][0]:
            return []
        
        texts = result['responses'][0]['textAnnotations']
        
        # 전체 텍스트와 좌표 정보 저장
        text_blocks = []
        img = Image.open(image_path)
        width, height = img.size
        
        # 첫 번째 텍스트는 전체 텍스트이므로 건너뜀
        for text in texts[1:]:
            vertices = text['boundingPoly']['vertices']
            x_coords = [vertex.get('x', 0) for vertex in vertices]
            y_coords = [vertex.get('y', 0) for vertex in vertices]
            
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            
            normalized_bbox = {
                'x1': x1 / width,
                'y1': y1 / height,
                'x2': x2 / width,
                'y2': y2 / height
            }
            
            text_blocks.append({
                'text': text['description'],
                'bbox': normalized_bbox,
                'abs_x1': x1,
                'abs_x2': x2,
                'abs_y1': y1,
                'abs_y2': y2
            })

        # y좌표로 정렬해서 같은 줄에 있는 단어들을 모음
        # 매우 가까운 y좌표는 동일한 줄로 간주
        y_threshold = 0.02  # 이미지 높이의 2%
        
        # y좌표 기준으로 먼저 그룹화
        y_groups = {}
        for block in text_blocks:
            y_val = round(block['bbox']['y1'] / y_threshold) * y_threshold
            if y_val not in y_groups:
                y_groups[y_val] = []
            y_groups[y_val].append(block)
        
        # 각 y_group 내에서 x좌표 순서대로 정렬
        combined_blocks = []
        for y_val, blocks in sorted(y_groups.items()):
            # x좌표 기준 정렬
            blocks.sort(key=lambda b: b['bbox']['x1'])
            
            # 모든 텍스트를 정규화된 띄어쓰기로 결합
            # 각 단어 사이의 실제 간격과 상관없이 한 칸 띄우기
            line_text = " ".join([b['text'] for b in blocks])
            
            # 단어들의 전체 bounding box 계산
            min_x1 = min(b['bbox']['x1'] for b in blocks)
            min_y1 = min(b['bbox']['y1'] for b in blocks)
            max_x2 = max(b['bbox']['x2'] for b in blocks)
            max_y2 = max(b['bbox']['y2'] for b in blocks)
            
            combined_blocks.append({
                'text': line_text,
                'bbox': {
                    'x1': min_x1,
                    'y1': min_y1,
                    'x2': max_x2,
                    'y2': max_y2
                }
            })
        
        return combined_blocks
    except Exception as e:
        print(f"Error in extract_text_with_vision: {str(e)}")
        return []

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

def extract_frames_from_video(video_path, interval=1.0, max_frames=None):
    """비디오에서 일정 간격으로 프레임 추출"""
    frames = []
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)
    
    # 최대 프레임 수 계산
    if max_frames and total_frames > max_frames:
        frame_interval = total_frames // max_frames
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append(frame)
            timestamps.append(timestamp)
            
            # 최대 프레임 수에 도달하면 중단
            if max_frames and len(frames) >= max_frames:
                break
            
        frame_count += 1
    
    cap.release()
    return frames, timestamps

def process_video(video_path, query, mode='normal'):
    """비디오 처리 및 타임라인 생성"""
    print(f"비디오 처리 시작: {video_path}")
    
    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    
    # 비디오 길이에 따라 최대 프레임 수 조정
    max_frames = min(100, int(duration * 2))  # 최대 100프레임 또는 2초당 1프레임
    frames, timestamps = extract_frames_from_video(video_path, interval=1.0, max_frames=max_frames)
    
    timeline_results = []
    all_ocr_text = []  # 모든 OCR 텍스트를 저장할 리스트
    
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
    
    # 배치 처리할 프레임 수
    batch_size = 5
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_timestamps = timestamps[i:i+batch_size]
        
        # 배치로 OCR 처리
        batch_results = []
        for frame, timestamp in zip(batch_frames, batch_timestamps):
            # 프레임을 임시 이미지 파일로 저장
            temp_frame_path = os.path.join(UPLOAD_FOLDER, f'temp_frame_{timestamp}.jpg')
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # OCR 실행 (수정된 함수 사용)
                text_blocks = extract_text_with_vision(temp_frame_path)
                
                if text_blocks:
                    # 현재 프레임의 OCR 텍스트 저장
                    frame_text = '\n'.join([block['text'] for block in text_blocks])
                    all_ocr_text.append(f"=== {timestamp}초 ===\n{frame_text}")
                    
                    detected_texts = []
                    for block in text_blocks:
                        text = block['text']
                        bbox = block['bbox']
                        
                        if mode == 'smart':
                            # 연관어 검색
                            for word in related_words:
                                if word.lower() in text.lower():
                                    detected_texts.append({
                                        'text': text,
                                        'bbox': bbox,
                                        'confidence': 1.0,  # 기본값 설정
                                        'color': 'yellow'  # 연관어는 노란색
                                    })
                                    break
                        else:
                            # 일반 검색
                            if query.lower() in text.lower():
                                detected_texts.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'confidence': 1.0,  # 기본값 설정
                                    'color': 'red'  # 일반 검색은 빨간색
                                })
                    
                    if detected_texts:
                        batch_results.append({
                            'timestamp': timestamp,
                            'texts': detected_texts
                        })
            except Exception as e:
                print(f"프레임 처리 중 오류 발생: {str(e)}")
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
        
        timeline_results.extend(batch_results)
    
    # 전체 OCR 텍스트를 하나의 문자열로 결합
    ocr_text = '\n'.join(all_ocr_text)
    print(f"=== 전체 OCR 텍스트 ===\n{ocr_text}")
    
    return timeline_results

# 비디오 파일 업로드 처리
@app.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': '비디오 파일이 필요합니다'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'지원하지 않는 파일 형식입니다: {file.filename}'}), 400
        
        # 세션 ID 생성 또는 기존 세션 ID 사용
        session_id = request.form.get('session_id')
        if not session_id:
            session_id = str(int(time.time()))
        
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 비디오 처리
        query = request.form.get('query', '')
        mode = request.form.get('mode', 'normal')
        timeline_results = process_video(filepath, query, mode)
        
        # OCR 텍스트 가져오기
        ocr_text = '\n'.join([f"=== {item['timestamp']}초 ===\n" + '\n'.join([text['text'] for text in item['texts']]) for item in timeline_results])
        
        # 세션에 비디오 정보 저장
        if session_id not in ocr_results_cache:
            ocr_results_cache[session_id] = {
                'text': '',
                'coordinates': {},
                'videos': []
            }
        
        ocr_results_cache[session_id]['text'] = ocr_text
        ocr_results_cache[session_id]['videos'].append({
            'filename': filename,
            'file_url': f'/uploads/{filename}',
            'timeline': timeline_results
        })
        
        return jsonify({
            'message': '비디오 업로드 성공',
            'session_id': session_id,
            'file': {
                'filename': filename,
                'file_url': f'/uploads/{filename}',
                'timeline': timeline_results
            },
            'text': ocr_text
        })
                
    except Exception as e:
        print(f"비디오 업로드 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'images[]' not in request.files:
            return jsonify({'error': '이미지 파일이 필요합니다'}), 400
        
        files = request.files.getlist('images[]')
        if not files:
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
        
        uploaded_files = []
        combined_ocr_text = ""
        combined_coordinates = {}
        
        # 세션 ID 생성 또는 기존 세션 ID 사용
        session_id = request.form.get('session_id')
        if not session_id:
            session_id = str(int(time.time()))
        
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                return jsonify({'error': f'지원하지 않는 파일 형식입니다: {file.filename}'}), 400
            
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # OCR로 텍스트 추출
                text_blocks = extract_text_with_vision(filepath)
                
                if text_blocks:
                    # 텍스트 블록에서 전체 텍스트와 좌표 정보 추출
                    ocr_text = '\n'.join([block['text'] for block in text_blocks])
                    coordinates = {block['text']: {'bbox': block['bbox'], 'confidence': 1.0} for block in text_blocks}
                    
                    # 이미지 타입 감지
                    image_type = detect_image_type(ocr_text)
                    print(f"감지된 이미지 타입: {image_type}")
                    
                    # 각 이미지의 OCR 결과를 저장
                    if session_id not in ocr_results_cache:
                        ocr_results_cache[session_id] = {
                            'text': '',
                            'coordinates': {},
                            'images': []
                        }
                    
                    ocr_results_cache[session_id]['text'] += f"\n--- 이미지 {len(uploaded_files) + 1} ---\n{ocr_text}"
                    ocr_results_cache[session_id]['coordinates'].update(coordinates)
                    
                    # 전체 OCR 텍스트와 좌표 정보 결합
                    combined_ocr_text += f"\n--- 이미지 {len(uploaded_files) + 1} ---\n{ocr_text}"
                    combined_coordinates.update(coordinates)
                
                uploaded_files.append({
                    'filename': filename,
                    'file_url': f'/uploads/{filename}',
                    'image_type': image_type if text_blocks else 'OTHER'
                })
                
                # 세션에 이미지 정보 추가
                ocr_results_cache[session_id]['images'].append({
                    'filename': filename,
                    'file_url': f'/uploads/{filename}',
                    'image_type': image_type if text_blocks else 'OTHER'
                })
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        if not uploaded_files:
            return jsonify({'error': '처리된 파일이 없습니다'}), 400
        
        print(f"=== OCR 결과 ===")
        print(f"텍스트: {combined_ocr_text}")
        print(f"좌표 정보: {combined_coordinates}")
        
        return jsonify({
            'message': '이미지 업로드 성공',
            'session_id': session_id,
            'files': uploaded_files,
            'text': combined_ocr_text,
            'image_type': uploaded_files[0]['image_type'] if uploaded_files else 'OTHER'
        })
        
    except Exception as e:
        print(f"이미지 업로드 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

def detect_image_type(text):
    """OCR 텍스트를 기반으로 이미지 유형을 감지합니다."""
    text = text.lower()
    
    # 각 유형별 특징적인 문구나 패턴 (우선순위 순)
    type_patterns = {
        'CONTRACT': [
            '계약서', '계약', '계약기간', '당사자', '서명', '계약조건',
            '계약서명', '계약일자', '계약금', '계약자', '피계약자',
            '계약내용', '계약조항', '계약서류', '계약서 작성',
            '계약서 확인', '계약서 검토', '계약서 승인',
            '계약서명', '계약서 작성일', '계약서 번호',
            '계약서 보관', '계약서 관리', '계약서 보관기간',
            '계약서 보관장소', '계약서 보관자', '계약서 보관방법',
            '계약서 보관기간', '계약서 보관장소', '계약서 보관자',
            '계약서 보관방법', '계약서 보관기간', '계약서 보관장소',
            '계약서 보관자', '계약서 보관방법', '계약서 보관기간',
            '계약서 보관장소', '계약서 보관자', '계약서 보관방법'
        ],
        'PAYMENT': [
            '영수증', '거래명세서', '결제', '금액', '지출', '수입',
            '정산', '지급', '수령', '지점', '매장', '결제일',
            '결제금액', '결제방법', '결제내역', '결제확인',
            '영수증 확인', '영수증 발급', '영수증 출력',
            '카드', '현금', '할인', '부가세', '합계',
            '일시불', '할부', '포인트', '적립', '승인',
            '매출', '매입', '거래', '판매', '구매',
            '가격', '단가', '수량', '금액', '총액',
            '세금', '부가세', '공급가액', '공급가',
            '신용카드', '체크카드', '현금영수증',
            '사업자', '사업자번호', '사업자등록번호',
            '주소', '전화번호', '대표자', '상호',
            '품목', '수량', '단가', '금액', '합계',
            '부가세', '공급가액', '공급가', '세액',
            '신용카드', '체크카드', '현금영수증',
            '승인번호', '승인일시', '승인금액',
            '할인', '포인트', '적립', '사용',
            '거래일시', '거래일자', '거래시간',
            '매장명', '지점명', '사업자명',
            '주소', '전화번호', '대표자',
            '품목', '수량', '단가', '금액',
            '합계', '부가세', '공급가액',
            '세액', '신용카드', '체크카드',
            '현금영수증', '승인번호', '승인일시',
            '승인금액', '할인', '포인트', '적립',
            '사용', '거래일시', '거래일자', '거래시간'
        ],
        'DOCUMENT': [
            '논문', '문서', '보고서', '작성자', '작성일', '결론',
            '요약', '목차', '서론', '본론', '참고문헌',
            '문서번호', '문서제목', '문서작성', '문서검토',
            '문서승인', '문서보관', '문서관리'
        ],
        'PRODUCT': [
            '제품', '모델', '기능', '제조사', '사양', '사용설명서',
            '제품명', '제품번호', '제품가격', '제품특징',
            '제품사양', '제품설명', '제품이미지', '제품카탈로그',
            '제품소개', '제품안내', '제품정보'
        ]
    }
    
    # 각 유형별 점수 계산
    type_scores = defaultdict(int)
    
    # 텍스트를 단어 단위로 분리
    words = set(text.split())
    
    # 각 유형별로 점수 계산
    for doc_type, patterns in type_patterns.items():
        for pattern in patterns:
            # 패턴이 텍스트에 포함되어 있는지 확인
            if pattern in text:
                # 패턴의 길이에 비례하여 점수 부여
                score = len(pattern) * 2
                
                # 특정 패턴에 가중치 부여
                if doc_type == 'CONTRACT' and ('계약서' in pattern or '계약' in pattern):
                    score *= 3
                elif doc_type == 'PAYMENT' and ('영수증' in pattern or '거래명세서' in pattern):
                    score *= 3
                
                type_scores[doc_type] += score
    
    # 가장 높은 점수를 가진 유형 선택
    if not type_scores:
        return 'OTHER'
    
    # 점수가 가장 높은 유형 찾기
    best_type = 'OTHER'
    best_score = 0
    
    for doc_type, score in type_scores.items():
        if score > best_score:
            best_score = score
            best_type = doc_type
    
    # 계약서와 영수증 구분을 위한 추가 검증
    if best_type == 'PAYMENT' and any(pattern in text for pattern in ['계약서', '계약']):
        # 계약서 관련 단어가 있으면 계약서로 판단
        best_type = 'CONTRACT'
    
    print(f"이미지 타입 감지 결과: {best_type} (점수: {best_score})")
    print(f"감지된 패턴: {[pattern for pattern in type_patterns[best_type] if pattern in text]}")
    
    return best_type

def get_prompt_for_image_type(image_type, text):
    """이미지 유형에 따른 프롬프트를 생성합니다."""
    prompts = {
        'CONTRACT': f"""이미지는 계약서입니다. 다음 정보를 추출해주세요:
- 계약 당사자
- 계약 기간
- 주요 계약 조건
- 특이사항

추출된 텍스트:
{text}""",

        'PAYMENT': f"""이미지는 정산/지출 관련 문서입니다. 다음 정보를 추출해주세요:
- 항목
- 금액
- 일자
- 장소/지점명
- 지급/수령인

추출된 텍스트:
{text}""",

        'DOCUMENT': f"""이미지는 논문/문서입니다. 다음 정보를 추출해주세요:
- 작성자
- 작성일
- 주요 내용
- 결론/요약

추출된 텍스트:
{text}""",

        'PRODUCT': f"""이미지는 제품 설명서입니다. 다음 정보를 추출해주세요:
- 제품명
- 모델명
- 주요 기능
- 제조사

추출된 텍스트:
{text}""",

        'OTHER': f"""이미지의 주요 정보를 추출해주세요:
- 주요 내용
- 중요 정보

추출된 텍스트:
{text}"""
    }
    
    return prompts.get(image_type, prompts['OTHER'])

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    temp_path = None
    try:
        print("=== analyze-image 요청 시작 ===")
        
        if 'images[]' not in request.files:
            return jsonify({'error': '이미지가 없습니다.'}), 400
            
        file = request.files['images[]']
        query = request.form.get('query', '')
        
        if not file or not query:
            return jsonify({'error': '이미지와 검색어가 필요합니다.'}), 400
            
        # 임시 파일로 저장
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        file.save(temp_path)
        
        # OCR 수행 (이제 리스트 반환)
        text_blocks = extract_text_with_vision(temp_path)
        
        if not text_blocks:
            return jsonify({'error': '텍스트를 인식할 수 없습니다.'}), 400
        
        # 검색어가 포함된 블록만 필터링
        query_clean = query.strip().lower()
        matches = [block for block in text_blocks if query_clean in block['text'].lower()]
        
        print(f"검색 결과: {len(matches)}개의 매칭된 텍스트 발견")
        
        return jsonify({
            'matches': matches,
            'text': '\n'.join([block['text'] for block in text_blocks])
        })
    except Exception as e:
        print(f"이미지 분석 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    try:
        # OpenAI API를 사용하여 메시지 처리
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 FindIt Assistant입니다. 사용자의 질문에 친절하게 답변하고, 간판이나 장소에 대한 질문이 있으면 위치 정보를 제공해주세요."},
                {"role": "user", "content": user_message}
            ]
        )
        
        bot_response = response.choices[0].message.content
        
        # 간판이나 장소에 대한 질문인지 확인
        if any(keyword in user_message.lower() for keyword in ['위치', '어디', '찾아', '간판', '장소']):
            # Google Maps API를 사용하여 위치 검색
            try:
                places_result = gmaps.places(bot_response)
                if places_result['results']:
                    place = places_result['results'][0]
                    location = place['geometry']['location']
                    return jsonify({
                        'type': 'map',
                        'content': f"{place['name']}의 위치를 찾았습니다.",
                        'location': location
                    })
            except Exception as e:
                print(f"Error searching location: {e}")
        
        return jsonify({
            'type': 'text',
            'content': bot_response
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'type': 'text',
            'content': '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.'
        }), 500

def is_valid_youtube_url(url):
    """YouTube URL이 유효한지 확인"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return bool(re.match(youtube_regex, url))

def download_youtube_video(url):
    """YouTube 영상 다운로드"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"\n=== YouTube 다운로드 시도 {attempt + 1}/{max_retries} ===")
            print(f"URL: {url}")
            
            # 업로드 폴더 확인
            if not os.path.exists(UPLOAD_FOLDER):
                print(f"업로드 폴더 생성: {UPLOAD_FOLDER}")
                os.makedirs(UPLOAD_FOLDER)
            
            # 파일명 생성
            timestamp = int(time.time())
            filename = f"youtube_{timestamp}.mp4"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(f"저장 경로: {filepath}")
            
            # yt-dlp 옵션 설정
            ydl_opts = {
                'format': 'best[height<=720]',  # 720p 이하의 최상의 품질
                'outtmpl': filepath,
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'noplaylist': True,
                'verbose': True
            }
            
            print("영상 정보 다운로드 중...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 영상 정보 가져오기
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '')
                duration = info.get('duration', 0)
                
                print(f"영상 제목: {title}")
                print(f"영상 길이: {duration}초")
                
                # 영상 다운로드
                print("다운로드 시작...")
                ydl.download([url])
            
            # 파일 존재 확인
            if not os.path.exists(filepath):
                raise Exception("다운로드된 파일을 찾을 수 없습니다.")
            
            print("다운로드 완료")
            return {
                'filepath': filepath,
                'title': title,
                'duration': duration
            }
            
        except Exception as e:
            print(f"시도 {attempt + 1} 실패: {str(e)}")
            if attempt < max_retries - 1:
                print(f"{retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
            else:
                print("최대 재시도 횟수 초과")
                raise Exception(f"YouTube 다운로드 실패: {str(e)}")

@app.route('/process-youtube', methods=['POST'])
def process_youtube():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON 데이터가 필요합니다'}), 400
            
        url = data.get('url')
        query = data.get('query', '')
        mode = data.get('mode', 'normal')
        video_id = data.get('video_id', '')
        
        if not url:
            return jsonify({'error': 'URL이 필요합니다'}), 400
            
        print(f"=== YouTube 처리 시작 ===")
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print(f"검색어: {query}")
        print(f"모드: {mode}")
        
        try:
            # YouTube 영상 다운로드
            video_info = download_youtube_video(url)
            print(f"다운로드 완료: {video_info['filepath']}")
            
            try:
                print("비디오 처리 시작")
                # 비디오 처리
                timeline_results = process_video(video_info['filepath'], query, mode)
                print(f"처리된 타임라인 결과: {len(timeline_results)}개 항목")
                
                # OCR 텍스트 가져오기
                ocr_text = '\n'.join([f"=== {item['timestamp']}초 ===\n" + '\n'.join([text['text'] for text in item['texts']]) for item in timeline_results])
                
                # 세션 ID 생성
                session_id = str(int(time.time()))
                
                # 세션에 비디오 정보 저장
                if session_id not in ocr_results_cache:
                    ocr_results_cache[session_id] = {
                        'text': '',
                        'coordinates': {},
                        'videos': []
                    }
                
                ocr_results_cache[session_id]['text'] = ocr_text
                ocr_results_cache[session_id]['videos'].append({
                    'filename': os.path.basename(video_info['filepath']),
                    'file_url': f'/uploads/{os.path.basename(video_info["filepath"])}',
                    'timeline': timeline_results
                })
                
                response_data = {
                    'type': 'video',
                    'file_url': f'/uploads/{os.path.basename(video_info["filepath"])}',
                    'timeline': timeline_results,
                    'title': video_info['title'],
                    'duration': video_info['duration'],
                    'ocr_text': ocr_text,
                    'session_id': session_id
                }
                print("응답 데이터 준비 완료")
                return jsonify(response_data)
                
            except Exception as e:
                print(f"비디오 처리 중 오류 발생: {str(e)}")
                import traceback
                print("상세 오류 정보:")
                print(traceback.format_exc())
                return jsonify({'error': f'비디오 처리 중 오류가 발생했습니다: {str(e)}'}), 500
                    
        except Exception as e:
            print(f"전체 처리 중 오류 발생: {str(e)}")
            import traceback
            print("상세 오류 정보:")
            print(traceback.format_exc())
            return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500
                
    except Exception as e:
        print(f"전체 처리 중 오류 발생: {str(e)}")
        import traceback
        print("상세 오류 정보:")
        print(traceback.format_exc())
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

def getInfoFromTextWithOpenAI(text: str) -> str | None:
    """OpenAI API를 사용하여 주어진 텍스트를 요약합니다.
    @param text 요약할 텍스트입니다.
    @returns 요약된 텍스트 또는 오류 메시지를 반환하는 Promise 객체입니다.
    """
    if not text.strip():
        return '정보를 추출할 텍스트가 제공되지 않았습니다.'
    try:
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                { 
                    'role': 'system', 
                    'content': '''당신은 이미지 OCR 처리 Q&A 및 정보 추출 어시스턴트입니다. 사용자가 사진을 업로드 하면 구글 클라우드 비전 API가 OCR 텍스트를 추출하여 제공된 텍스트를 분석하여 질문에 답하거나, 텍스트에서 중요한 정보를 추출해 사용자에게 유용하고 흥미로운 방식으로 전달하세요.
                    
당신의 임무는 다음과 같습니다:
1. 사진을 통해 추출된 텍스트를 주의 깊게 읽고 이해합니다.
2. 사진을 통해 추출된 텍스트의 끝부분에 특정 질문이 포함되어 있는지 확인합니다.
3. 특정 질문이 있는 경우:
    a. 텍스트의 이전 부분에 있는 정보만을 사용하여 질문에 답변합니다.
    b. 답변을 간결하면서도 창의적으로 표현합니다.
    c. 질문에 대한 답변을 텍스트에서 찾을 수 없는 경우, "제공된 사진에서 해당 질문에 대한 답변을 찾을 수 없습니다."라고 응답하되, 추가적인 통찰이나 관련 정보를 제공하려고 노력합니다.
4. 특정 질문이 없는 경우:
    a. 사진(텍스트)에서 핵심 정보, 사실, 주요 개체(예: 이름, 날짜, 장소, 회사, 특정 항목 및 해당 값) 및 중요한 세부 정보를 추출합니다.
    b. 이 정보를 명확하고 구조화된 방식으로 제시하되, 사용자에게 흥미롭고 창의적인 방식으로 전달합니다. (예: "주요 항목: 값" 또는 "흥미로운 사실: ...").

예시 1 (질문 포함):
텍스트: "문서 내용입니다. 프로젝트명: 오로라, 책임자: 이지혜, 시작일: 2024-03-01. 질문: 이 프로젝트의 책임자는 누구인가요?"
당신의 응답: "이 프로젝트의 책임자는 이지혜입니다. 그녀는 프로젝트의 성공을 이끌 중요한 역할을 맡고 있습니다."

예시 2 (질문 없음):
텍스트: "회의록 요약: 안건 - 신규 마케팅 전략 논의. 참석자: 김민준, 박서연, 최현우. 결정사항: 1분기 내 소셜 미디어 캠페인 실행."
당신의 응답: "회의의 주요 내용은 다음과 같습니다:\n- 안건: 신규 마케팅 전략 논의\n- 참석자: 김민준, 박서연, 최현우\n- 결정사항: 1분기 내 소셜 미디어 캠페인 실행"

예시 3 (질문은 있으나 답변을 찾을 수 없음):
텍스트: "제품 설명서: 스마트 워치 모델 X. 주요 기능: 심박수 측정, GPS, 방수. 질문: 배터리 지속 시간은 얼마나 되나요?"
당신의 응답: "제공된 사진에서 배터리 지속 시간에 대한 정보는 찾을 수 없습니다. 하지만 이 스마트 워치의 주요 기능은 심박수 측정, GPS, 방수입니다. 배터리 지속 시간에 대한 정보는 제조사에 문의해보세요!"

예시 4 (장소를 나타내는 사진):
텍스트: "사진 설명: 서울의 경복궁. 역사적 건축물로 유명하며, 매년 많은 관광객이 방문합니다.", "여기가 어디인가요?"
당신의 응답: "이곳은 서울의 경복궁입니다. 한국의 역사적 건축물로, 조선 왕조의 중심지였으며 지금도 많은 관광객이 찾는 명소입니다."

제공된 텍스트(사진)를 기반으로 정확하고 창의적으로 응답해주세요.'''
                },
                { 'role': 'user', 'content': text },
            ],
            max_tokens=500,  # 더 풍부한 응답을 위해 토큰 수 증가
            temperature=0.7,  # 창의성을 높이기 위해 temperature 값 증가
        )

        information = completion.choices[0].message.content
        return information or '텍스트에서 정보를 가져올 수 없습니다.'

    except Exception as error:
        print('정보 추출을 위해 OpenAI API 호출 중 오류:', error)
        return f'OpenAI API 오류: {str(error)}'

@app.route('/summarize', methods=['POST'])
def summarize_document():
    try:
        session_id = request.form.get('session_id')
        message = request.form.get('message', '')
        
        if not session_id or session_id not in ocr_results_cache:
            return jsonify({'error': '유효하지 않은 세션 ID입니다'}), 400
        
        ocr_data = ocr_results_cache[session_id]
        combined_text = ocr_data['text']
        
        if not combined_text:
            return jsonify({'error': '텍스트를 추출할 수 없습니다'}), 400
        
        # 사용자의 질문이 있는 경우, 질문과 함께 텍스트를 전달
        if message:
            prompt = f"다음은 이미지에서 추출된 텍스트입니다:\n\n{combined_text}\n\n사용자의 질문: {message}\n\n위 텍스트를 바탕으로 질문에 답변해주세요."
        else:
            prompt = combined_text
        
        # OpenAI를 사용하여 텍스트 요약 또는 질문 답변
        summary = getInfoFromTextWithOpenAI(prompt)
        
        return jsonify({
            'summary': summary,
            'original_text': combined_text,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"요약 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)