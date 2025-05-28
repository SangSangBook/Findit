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
import json
import sys
from multiprocessing import Pool

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file
load_dotenv()

# 환경 변수 디버깅
# print("\n=== 환경 변수 확인 ===")
# print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
# print(f"GOOGLE_CLOUD_VISION_API_KEY: {os.getenv('GOOGLE_CLOUD_VISION_API_KEY')}")
# print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
# print("=====================\n")

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
# else:
#     print(f"OpenAI API 키가 설정되었습니다. (길이: {len(OPENAI_API_KEY)}자)")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)  # API 키를 명시적으로 전달

# Google Cloud Vision API 설정
API_KEY = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
if not API_KEY:
    print("경고: GOOGLE_CLOUD_VISION_API_KEY가 설정되지 않았습니다.")
# else:
#     print(f"Google Vision API 키가 설정되었습니다. (길이: {len(API_KEY)}자)")
VISION_API_URL = f'https://vision.googleapis.com/v1/images:annotate?key={API_KEY}'

# 연관어 캐시
related_words_cache = {}

# Google Maps 클라이언트 초기화
gmaps = googlemaps.Client(key=API_KEY)  # Vision API와 동일한 키 사용

# Google Cloud Vision 클라이언트 초기화
try:
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        vision_client = vision.ImageAnnotatorClient()
    # print("Google Cloud Vision 클라이언트가 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"Google Cloud Vision 클라이언트 초기화 오류: {str(e)}")
    vision_client = None

# OCR 결과를 저장할 전역 변수
ocr_results_cache = {}

def get_related_words(query, ocr_texts):
    """OCR에서 추출된 텍스트 중에서 연관어를 찾습니다."""
    if query in related_words_cache:
        # print(f"캐시에서 연관어를 가져옵니다: {query}")
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
            # print(f"새로운 연관어를 캐시에 저장했습니다: {query}")
            # print(f"찾은 연관어 목록: {related_words}")
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
        
        # OCR로 텍스트 추출 및 객체 인식
        text_blocks, detected_objects = extract_text_with_vision(image_path)
        
        # 검색 결과 초기화
        matches = []
        all_detected_objects = []
        
        # 객체 인식 결과 처리
        if detected_objects:
            query_lower = query.lower()
            for obj in detected_objects:
                obj_name = obj['text'].lower()
                if query_lower in obj_name or obj_name in query_lower:
                    all_detected_objects.append({
                        'name': obj['text'],
                        'bbox': obj['bbox'],
                        'confidence': obj['confidence'],
                        'match_type': 'object'
                    })
        
        # 텍스트 검색
        if text_blocks:
            query_lower = query.lower()
            for block in text_blocks:
                text_lower = block['text'].lower()
                if query_lower in text_lower or text_lower in query_lower:
                    matches.append({
                        'text': block['text'],
                        'bbox': block['bbox'],
                        'confidence': block['confidence'],
                        'match_type': 'text'
                    })
        
        print(f"검색 결과: {len(matches)}개의 텍스트 매칭, {len(all_detected_objects)}개의 객체 매칭")
        return matches + all_detected_objects
        
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

        # 이미지 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # 원본 이미지 크기 저장
        original_height, original_width = image.shape[:2]
        
        # 이미지 크기 조정 (너무 작거나 큰 경우)
        height, width = image.shape[:2]
        scale = 1.0
        
        if width > 2000 or height > 2000:
            scale = min(2000/width, 2000/height)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # 이미지를 base64로 인코딩
        _, img_encoded = cv2.imencode('.jpg', image)
        content = base64.b64encode(img_encoded).decode('utf-8')
        
        # API 요청 데이터 준비
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": content
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 50
                        },
                        {
                            "type": "OBJECT_LOCALIZATION",
                            "maxResults": 10
                        }
                    ],
                    "imageContext": {
                        "languageHints": ["ko", "en"],
                        "textDetectionParams": {
                            "enableTextDetectionConfidenceScore": True
                        }
                    }
                }
            ]
        }
        
        # API 호출 - vision_client 사용
        if vision_client:
            image = vision.Image(content=content)
            response = vision_client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.TEXT_DETECTION, 'max_results': 50},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION, 'max_results': 10}
                ],
                'image_context': {
                    'language_hints': ['ko', 'en'],
                    'text_detection_params': {
                        'enable_text_detection_confidence_score': True
                    }
                }
            })
            result = response.to_dict()
        else:
            current_api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
            if not current_api_key:
                raise ValueError("Google Cloud Vision API 키가 설정되지 않았습니다.")
            
            vision_api_url = f'https://vision.googleapis.com/v1/images:annotate?key={current_api_key}'
            response = requests.post(vision_api_url, json=request_data)
            response.raise_for_status()
            result = response.json()

        # 텍스트 검출 결과 처리
        text_blocks = []
        detected_objects = []
        
        if 'responses' in result and result['responses']:
            response = result['responses'][0]
            
            # OCR 텍스트 처리
            if 'textAnnotations' in response:
                # 전체 텍스트를 하나의 블록으로 처리
                full_text = response['textAnnotations'][0]['description'] if response['textAnnotations'] else ""
                
                # 각 텍스트 블록의 좌표 정보 처리
                for text_annotation in response['textAnnotations'][1:]:  # 첫 번째는 전체 텍스트이므로 건너뛰기
                    vertices = text_annotation['boundingPoly']['vertices']
                    x_coords = [vertex.get('x', 0) for vertex in vertices]
                    y_coords = [vertex.get('y', 0) for vertex in vertices]
                    
                    # 정규화된 좌표를 원본 이미지 크기로 변환하고 scale 적용
                    x1 = min(x_coords) / scale
                    y1 = min(y_coords) / scale
                    x2 = max(x_coords) / scale
                    y2 = max(y_coords) / scale
                    
                    text_blocks.append({
                        'text': text_annotation['description'],
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'confidence': text_annotation.get('confidence', 1.0)
                    })
            
            # 객체 검출 결과 처리
            if 'localizedObjectAnnotations' in response:
                objects = response['localizedObjectAnnotations']
                
                for obj in objects:
                    vertices = obj['boundingPoly']['normalizedVertices']
                    x_coords = [vertex.get('x', 0) for vertex in vertices]
                    y_coords = [vertex.get('y', 0) for vertex in vertices]
                    
                    # 정규화된 좌표를 원본 이미지 크기로 변환하고 scale 적용
                    x1 = min(x_coords) * original_width / scale
                    y1 = min(y_coords) * original_height / scale
                    x2 = max(x_coords) * original_width / scale
                    y2 = max(y_coords) * original_height / scale
                    
                    detected_objects.append({
                        'text': obj['name'],
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'confidence': obj['score'],
                        'match_type': 'object'
                    })
        
        return text_blocks, detected_objects
    except Exception as e:
        print(f"Error in extract_text_with_vision: {e}")
        return [], []

def combine_vertical_texts(coordinates):
    """세로로 정렬된 텍스트들을 하나의 텍스트로 결합"""
    # coordinates가 리스트인 경우 딕셔너리로 변환
    if isinstance(coordinates, list):
        coordinates_dict = {}
        for i, coord in enumerate(coordinates):
            coordinates_dict[f'block_{i}'] = coord
        coordinates = coordinates_dict
    
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

def extract_frames_from_video(video_path, interval=2.0, max_frames=None):
    """비디오에서 일정 간격으로 프레임 추출"""
    frames = []
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)  # 2초마다 프레임 추출
    
    # 최대 프레임 수 계산
    if max_frames and total_frames > max_frames:
        frame_interval = total_frames // max_frames
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # 타임스탬프를 정수 초로 계산
            timestamp = int(frame_count / fps)
            frames.append(frame)
            timestamps.append(timestamp)
            
            # 최대 프레임 수에 도달하면 중단
            if max_frames and len(frames) >= max_frames:
                break
            
        frame_count += 1
    
    cap.release()
    return frames, timestamps

def process_frame_chunk(chunk_data):
    chunk_results = []
    for frame, timestamp in chunk_data:
        # 프레임을 임시 이미지 파일로 저장
        temp_frame_path = os.path.join(UPLOAD_FOLDER, f'temp_frame_{timestamp}.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # OCR 실행
            text_blocks, detected_objects = extract_text_with_vision(temp_frame_path)
            
            if text_blocks:
                # 현재 프레임의 OCR 텍스트 저장
                frame_text = '\n'.join([block['text'] for block in text_blocks])
                
                # 한 글자씩 분리된 텍스트를 문장으로 결합
                combined_text = ''
                current_sentence = ''
                for char in frame_text:
                    if char in ['\n', ' ']:
                        if current_sentence:
                            combined_text += current_sentence + char
                            current_sentence = ''
                    else:
                        current_sentence += char
                if current_sentence:
                    combined_text += current_sentence
                
                chunk_results.append((timestamp, combined_text, text_blocks))
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
    
    return chunk_results

def process_video(video_path, query, mode='normal', session_id=None):
    """비디오 처리 및 타임라인 생성"""
    print(f"비디오 처리 시작: {video_path}")
    print(f"세션 ID: {session_id}")
    print(f"검색 모드: {mode}")
    print(f"검색어: {query}")
    
    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(total_frames / fps)  # 정수로 변환
    cap.release()
    
    # 비디오 길이에 따라 최대 프레임 수 조정
    max_frames = min(30, duration)  # 최대 30프레임으로 제한
    frames, timestamps = extract_frames_from_video(video_path, interval=3.0, max_frames=max_frames)  # 3초 간격으로 변경
    
    timeline_results = []
    all_ocr_text = []  # 모든 OCR 텍스트를 저장할 리스트
    related_words = [query]  # 기본 검색어 추가
    
    if mode == 'smart':
        try:
            # 연관어 검색을 위한 GPT 프롬프트
            prompt = f"""
            '{query}'와 관련된 동의어, 상위어, 하위어, 연관어를 찾아주세요.
            예시:
            - 동의어: 같은 의미의 단어
            - 상위어: 더 넓은 범주의 단어 (예: '앵무새'의 상위어는 '조류', '동물')
            - 하위어: 더 구체적인 단어 (예: '조류'의 하위어는 '앵무새', '참새', '독수리')
            - 연관어: 관련이 있는 단어
            
            특히 동물이나 생물 관련 검색어의 경우, 같은 종류의 다른 동물들도 포함해주세요.
            예를 들어 '앵무새'로 검색하면 '조류', '동물'과 함께 '참새', '독수리', '까치' 등도 포함해주세요.
            
            결과는 쉼표로 구분된 단어 목록으로 반환해주세요.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 전문적인 언어 분석가입니다. 특히 동물이나 생물 관련 검색어의 경우, 같은 종류의 다른 동물들도 포함하여 연관어를 찾아주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            
            # 연관어 목록 파싱 및 정리
            words = response.choices[0].message.content.strip().split(',')
            related_words = [word.strip() for word in words if word.strip()]
            related_words.append(query)  # 원래 검색어도 포함
            
            print(f"연관어 목록: {related_words}")
        except Exception as e:
            print(f"연관어 검색 중 오류 발생: {str(e)}")
            related_words = [query]  # 오류 발생 시 원래 검색어만 사용
    
    # 병렬 처리를 위한 멀티프로세싱 설정
    num_processes = min(4, len(frames))  # 최대 4개의 프로세스 사용
    chunk_size = max(1, len(frames) // num_processes)
    
    # 프레임을 청크로 나누기
    frame_chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk = list(zip(frames[i:i+chunk_size], timestamps[i:i+chunk_size]))
        frame_chunks.append(chunk)
    
    # 멀티프로세싱으로 프레임 처리
    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_frame_chunk, frame_chunks)
    
    # 결과 처리
    for chunk_result in chunk_results:
        for timestamp, combined_text, text_blocks in chunk_result:
            all_ocr_text.append(f"=== {timestamp}초 ===\n{combined_text}")
            
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
                                'confidence': 1.0,
                                'color': '#000000',
                                'match_type': 'smart'
                            })
                            break
                else:
                    # 일반 검색
                    if query.lower() in text.lower():
                        detected_texts.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': 1.0,
                            'color': 'red'
                        })
            
            if detected_texts:
                timeline_results.append({
                    'timestamp': timestamp,
                    'texts': detected_texts
                })
    
    # 전체 OCR 텍스트를 하나의 문자열로 결합
    ocr_text = '\n'.join(all_ocr_text)
    print(f"=== 전체 OCR 텍스트 ===\n{ocr_text}")
    
    # 세션에 타임라인 결과 저장
    if session_id and session_id in ocr_results_cache:
        if 'videos' not in ocr_results_cache[session_id]:
            ocr_results_cache[session_id]['videos'] = []
        
        # 현재 비디오의 타임라인 결과 저장
        video_filename = os.path.basename(video_path)
        video_info = {
            'filename': video_filename,
            'file_url': f'/uploads/{video_filename}',
            'timeline': timeline_results
        }
        
        # 기존 비디오 정보 업데이트 또는 새로 추가
        video_exists = False
        for i, video in enumerate(ocr_results_cache[session_id]['videos']):
            if video['filename'] == video_filename:
                ocr_results_cache[session_id]['videos'][i] = video_info
                video_exists = True
                break
        
        if not video_exists:
            ocr_results_cache[session_id]['videos'].append(video_info)
    
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
        
        # 세션 초기화
        if session_id not in ocr_results_cache:
            ocr_results_cache[session_id] = {
                'text': '',
                'coordinates': {},
                'images': []
            }
        
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                return jsonify({'error': f'지원하지 않는 파일 형식입니다: {file.filename}'}), 400
            
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # OCR 및 객체 인식 실행
                text_blocks, detected_objects = extract_text_with_vision(filepath)
                
                if text_blocks or detected_objects:
                    # 텍스트 블록에서 전체 텍스트와 좌표 정보 추출
                    ocr_text = '\n'.join([block['text'] for block in text_blocks])
                    
                    # 텍스트 블록 좌표 정보를 리스트로 저장
                    coordinates = []
                    for block in text_blocks:
                        coordinates.append({
                            'text': block['text'],
                            'bbox': block['bbox'],
                            'confidence': block['confidence'],
                            'match_type': 'text'
                        })
                    
                    # 객체 인식 결과도 리스트에 추가
                    for obj in detected_objects:
                        coordinates.append({
                            'text': obj['text'],
                            'bbox': obj['bbox'],
                            'confidence': obj['confidence'],
                            'match_type': 'object'
                        })
                    
                    # 이미지 타입 감지
                    image_type = detect_image_type(ocr_text)
                    print(f"감지된 이미지 타입: {image_type}")
                    
                    # 각 이미지의 OCR 결과를 저장
                    ocr_results_cache[session_id]['text'] += f"\n--- 이미지 {len(uploaded_files) + 1} ---\n{ocr_text}"
                    ocr_results_cache[session_id]['coordinates'] = coordinates
                    
                    # 전체 OCR 텍스트와 좌표 정보 결합
                    combined_ocr_text += f"\n--- 이미지 {len(uploaded_files) + 1} ---\n{ocr_text}"
                    
                    # coordinates가 리스트인 경우 딕셔너리로 변환
                    if isinstance(coordinates, list):
                        coordinates_dict = {}
                        for i, coord in enumerate(coordinates):
                            coordinates_dict[f'image_{len(uploaded_files) + 1}_block_{i}'] = coord
                        combined_coordinates.update(coordinates_dict)
                    else:
                        combined_coordinates.update(coordinates)
                
                uploaded_files.append({
                    'filename': filename,
                    'file_url': f'/uploads/{filename}',
                    'image_type': image_type if text_blocks or detected_objects else 'OTHER',
                    'text_blocks': text_blocks,
                    'detected_objects': detected_objects
                })
                
                # 세션에 이미지 정보 추가
                ocr_results_cache[session_id]['images'].append({
                    'filename': filename,
                    'file_url': f'/uploads/{filename}',
                    'image_type': image_type if text_blocks or detected_objects else 'OTHER',
                    'text_blocks': text_blocks,
                    'detected_objects': detected_objects
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
        import traceback
        print(traceback.format_exc())
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

def get_smart_search_predictions(query: str, context: str) -> dict:
    """GPT를 사용하여 다음 검색어와 행동을 예측합니다."""
    print(f"스마트 검색 시작: 검색어 '{query}'와 문맥 '{context}'를 기반으로 예측합니다.")
    prompt = f"""
    다음 검색어와 문맥을 바탕으로 예측을 해주세요:
    
    검색어: {query}
    문맥: {context}
    
    1. 다음에 검색할 만한 키워드 3-5개를 예측해주세요.
    2. 현재 상황에 대한 행동 제안을 해주세요.
    
    JSON 형식으로 응답해주세요:
    {{
        "predicted_keywords": ["키워드1", "키워드2", "키워드3"],
        "action_recommendations": [
            {{
                "message": "행동 제안 메시지",
                "action": "실행할 수 있는 행동 (선택사항)"
            }}
        ]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that predicts next search terms and suggests actions based on the current context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        print(f"스마트 검색 결과: {result}")
        return result
    except Exception as e:
        print(f"Error in smart search prediction: {e}")
        # 임시 더미 데이터 반환
        return {
            "predicted_keywords": ["예시1", "예시2", "예시3"],
            "action_recommendations": [
                {"message": "이런 행동을 해보세요!", "action": "실행"}
            ]
        }

def get_task_suggestions(text: str, detected_objects: list = None) -> list:
    """OCR 텍스트와 감지된 객체를 기반으로 태스크 제안을 생성합니다."""
    try:
        # 텍스트가 너무 길 경우 잘라내기
        max_text_length = 4000  # GPT-4의 컨텍스트 제한을 고려
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        # 감지된 객체 정보 추가
        objects_info = ""
        if detected_objects:
            objects_info = "\n감지된 객체:\n" + "\n".join([
                f"- {obj['text']} (신뢰도: {obj.get('confidence', 1.0):.2f})"
                for obj in detected_objects
            ])
            print(f"감지된 객체 정보: {objects_info}")  # 디버깅용
        
        prompt = f"""
        다음은 OCR로 추출된 텍스트와 감지된 객체 정보입니다:
        
        OCR 텍스트:
        {text}
        
        {objects_info}
        
        이 정보들을 바탕으로 수행해야 할 태스크들을 한국어로 제안해주세요.
        반드시 다음 JSON 형식으로만 응답해주세요:
        {{
            "suggestions": [
                {{
                    "task": "태스크 제목",
                    "description": "태스크 설명",
                    "priority": "high/medium/low"
                }}
            ]
        }}
        
        주의사항:
        1. 반드시 위의 JSON 형식으로만 응답해주세요.
        2. 감지된 객체가 있다면, 반드시 그 객체와 관련된 구체적인 태스크를 제안하세요.
           예시:
           - 사과가 감지된 경우: "사과의 품질 검사", "사과의 신선도 확인", "사과의 가격 책정" 등
           - 자동차가 감지된 경우: "자동차 모델 식별", "자동차 상태 점검", "자동차 가격 조사" 등
        3. 각 태스크는 구체적이고 실행 가능한 형태로 제안해주세요.
        4. 일반적인 "객체 분석" 같은 모호한 태스크는 피해주세요.
        5. 태스크 제목과 설명은 반드시 한국어로 작성해주세요.
        6. 감지된 객체가 있다면, 그 객체에 대한 구체적인 분석 태스크를 우선적으로 제안하세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 한국어로 구체적이고 실행 가능한 태스크를 제안하는 전문가입니다. 감지된 객체가 있다면 반드시 그 객체와 관련된 구체적인 태스크를 제안해야 합니다. 반드시 지정된 JSON 형식으로만 응답해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # 응답 텍스트에서 JSON 부분만 추출
        response_text = response.choices[0].message.content.strip()
        try:
            # JSON 파싱 시도
            result = json.loads(response_text)
            suggestions = result.get('suggestions', [])
            print(f"태스크 제안 결과: {suggestions}")
            
            # 감지된 객체가 있는데 일반적인 태스크만 제안된 경우
            if detected_objects and not any(obj['text'].lower() in suggestion['task'].lower() or 
                                          obj['text'].lower() in suggestion['description'].lower() 
                                          for obj in detected_objects for suggestion in suggestions):
                # 객체별 구체적인 태스크 생성
                specific_suggestions = []
                for obj in detected_objects:
                    obj_name = obj['text']
                    specific_suggestions.extend([
                        {
                            "task": f"{obj_name} 품질 검사",
                            "description": f"{obj_name}의 상태와 품질을 자세히 검사하세요.",
                            "priority": "high"
                        },
                        {
                            "task": f"{obj_name} 특성 분석",
                            "description": f"{obj_name}의 특징과 특성을 분석하세요.",
                            "priority": "medium"
                        }
                    ])
                return specific_suggestions
            
            return suggestions
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"원본 응답: {response_text}")
            # 감지된 객체가 있는 경우 객체별 구체적인 태스크 생성
            if detected_objects:
                specific_suggestions = []
                for obj in detected_objects:
                    obj_name = obj['text']
                    specific_suggestions.extend([
                        {
                            "task": f"{obj_name} 품질 검사",
                            "description": f"{obj_name}의 상태와 품질을 자세히 검사하세요.",
                            "priority": "high"
                        },
                        {
                            "task": f"{obj_name} 특성 분석",
                            "description": f"{obj_name}의 특징과 특성을 분석하세요.",
                            "priority": "medium"
                        }
                    ])
                return specific_suggestions
            
            # 기본 태스크 제안 반환 (한국어)
            return [{
                "task": "객체 분석",
                "description": "감지된 객체의 특성과 상태를 자세히 분석하세요.",
                "priority": "high"
            }]
            
    except Exception as e:
        print(f"Error in task suggestion: {e}")
        # 감지된 객체가 있는 경우 객체별 구체적인 태스크 생성
        if detected_objects:
            specific_suggestions = []
            for obj in detected_objects:
                obj_name = obj['text']
                specific_suggestions.extend([
                    {
                        "task": f"{obj_name} 품질 검사",
                        "description": f"{obj_name}의 상태와 품질을 자세히 검사하세요.",
                        "priority": "high"
                    },
                    {
                        "task": f"{obj_name} 특성 분석",
                        "description": f"{obj_name}의 특징과 특성을 분석하세요.",
                        "priority": "medium"
                    }
                ])
            return specific_suggestions
        
        # 오류 발생 시 기본 태스크 제안 반환 (한국어)
        return [{
            "task": "객체 분석",
            "description": "감지된 객체의 특성과 상태를 자세히 분석하세요.",
            "priority": "high"
        }]

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        print("\n=== analyze-image 요청 시작 ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Files: {request.files}")
        print(f"Form data: {request.form}")
        
        # JSON 데이터 처리
        if request.is_json:
            print("JSON 요청 처리")
            data = request.get_json()
            text = data.get('text')
            type = data.get('type')
            session_id = data.get('session_id')
            
            if type == 'task_suggestion' and text:
                print(f"태스크 제안 요청 - 세션 ID: {session_id}")
                # 세션에 저장된 이전 OCR 결과와 새로운 텍스트 결합
                if session_id and session_id in ocr_results_cache:
                    previous_text = ocr_results_cache[session_id].get('text', '')
                    combined_text = f"{previous_text}\n=== 새 이미지 ===\n{text}" if previous_text else text
                    
                    # 감지된 객체 정보 가져오기
                    detected_objects = []
                    coordinates = ocr_results_cache[session_id].get('coordinates', {})
                    for text, data in coordinates.items():
                        if data.get('match_type') == 'object':
                            detected_objects.append({
                                'name': text,
                                'confidence': data.get('confidence', 1.0)
                            })
                else:
                    combined_text = text
                    detected_objects = []
                
                # 새로운 태스크 제안 생성 (감지된 객체 정보 포함)
                new_suggestions = get_task_suggestions(combined_text, detected_objects)
                
                # 세션에 태스크 제안 저장
                if session_id not in ocr_results_cache:
                    ocr_results_cache[session_id] = {
                        'text': '',
                        'coordinates': {},
                        'images': [],
                        'task_suggestions': []
                    }
                ocr_results_cache[session_id]['task_suggestions'] = new_suggestions
                
                return jsonify({
                    'suggestions': new_suggestions,
                    'session_id': session_id,
                    'total_images': len(ocr_results_cache[session_id]['images']) if session_id in ocr_results_cache else 0
                })
        
        # 기존 FormData 처리
        session_id = request.form.get('session_id')
        query = request.form.get('query')
        mode = request.form.get('mode', 'normal')
        files = request.files.getlist('images[]')
        
        print(f"세션 ID: {session_id}")
        print(f"검색어: {query}")
        print(f"모드: {mode}")
        print(f"파일 수: {len(files)}")

        if not session_id:
            return jsonify({'error': '세션 ID가 필요합니다'}), 400

        if session_id not in ocr_results_cache:
            return jsonify({'error': '유효하지 않은 세션 ID입니다'}), 400

        # OCR 결과와 감지된 객체 가져오기
        ocr_text = ocr_results_cache[session_id].get('text', '')
        coordinates = ocr_results_cache[session_id].get('coordinates', {})
        
        print(f"OCR 텍스트: {ocr_text}")
        print(f"좌표 정보: {coordinates}")

        # 검색 결과 초기화
        matches = []
        all_detected_objects = []

        # 텍스트 검색
        if query:
            query_lower = query.lower()
            is_korean_query = any('\uAC00' <= char <= '\uD7A3' for char in query)
            
            # 스마트 검색 모드일 때 연관어 검색
            related_words = [query]
            if mode == 'smart':
                try:
                    # 연관어 검색을 위한 GPT 프롬프트
                    prompt = f"""
                    '{query}'와 관련된 동의어, 상위어, 하위어, 연관어를 찾아주세요.
                    특히 동물이나 생물 관련 검색어의 경우, 같은 종류의 다른 동물들도 포함해주세요.
                    
                    예시:
                    - '동물'로 검색할 경우: '개', '고양이', '앵무새', '물고기', '새', '포유류', '조류', '어류' 등
                    - '조류'로 검색할 경우: '앵무새', '참새', '독수리', '까치', '새' 등
                    - '포유류'로 검색할 경우: '개', '고양이', '사자', '호랑이', '곰' 등
                    - '어류'로 검색할 경우: '물고기', '상어', '고등어', '참치' 등
                    
                    결과는 쉼표로 구분된 단어 목록으로 반환해주세요.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 전문적인 언어 분석가입니다. 특히 동물이나 생물 관련 검색어의 경우, 같은 종류의 다른 동물들도 포함하여 연관어를 찾아주세요. 검색어가 '동물', '조류', '포유류', '어류' 등 상위 개념인 경우, 그에 속하는 모든 하위 동물들도 포함해주세요."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200
                    )
                    
                    # 연관어 목록 파싱 및 정리
                    words = response.choices[0].message.content.strip().split(',')
                    related_words = [word.strip() for word in words if word.strip()]
                    related_words.append(query)  # 원래 검색어도 포함
                    
                    print(f"연관어 목록: {related_words}")
                except Exception as e:
                    print(f"연관어 검색 중 오류 발생: {str(e)}")
                    related_words = [query]  # 오류 발생 시 원래 검색어만 사용
            
            # 한글 검색어인 경우 영어 매핑 가져오기
            korean_matches = []
            if is_korean_query:
                from src.utils.languageMapping import koreanToEnglish
                korean_matches = koreanToEnglish.get(query_lower, [])
            
            # coordinates가 리스트인 경우 처리
            if isinstance(coordinates, list):
                for item in coordinates:
                    text = item.get('text', '')
                    text_lower = text.lower()
                    
                    # 직접 매칭 확인
                    direct_match = query_lower in text_lower
                    
                    # 연관어 매칭 확인 (스마트 검색 모드)
                    related_match = False
                    if mode == 'smart':
                        # 연관어 목록의 각 단어가 텍스트에 포함되어 있는지 확인
                        for word in related_words:
                            word_lower = word.lower()
                            if word_lower in text_lower or text_lower in word_lower:
                                related_match = True
                                print(f"연관어 매칭 성공: '{word_lower}' in '{text_lower}'")
                                break
                    
                    # 한글-영어 매핑 확인
                    mapped_match = False
                    if is_korean_query:
                        mapped_match = any(english.lower() in text_lower for english in korean_matches)
                    
                    if direct_match or related_match or mapped_match:
                        print(f"매칭된 텍스트: {text} (직접 매칭: {direct_match}, 연관어 매칭: {related_match}, 매핑 매칭: {mapped_match})")
                        if item.get('match_type') == 'object':
                            all_detected_objects.append({
                                'name': text,
                                'bbox': item['bbox'],
                                'confidence': item.get('confidence', 1.0),
                                'match_type': 'object',
                                'style': {
                                    'position': 'absolute',
                                    'border': '2px solid #00ff00',
                                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                                    'zIndex': 999999999999999,
                                    'pointerEvents': 'none',
                                    'padding': '4px',
                                    'fontSize': '14px',
                                    'color': '#00ff00',
                                    'fontWeight': 'bold',
                                    'textShadow': '1px 1px 2px rgba(0,0,0,0.5)',
                                    'whiteSpace': 'nowrap',
                                    'overflow': 'visible'
                                },
                                'label': text
                            })
                        else:
                            matches.append({
                                'text': text,
                                'bbox': item['bbox'],
                                'confidence': item.get('confidence', 1.0),
                                'match_type': 'text',
                                'style': {
                                    'position': 'absolute',
                                    'border': '2px solid #ff0000',
                                    'borderRadius': '50%',
                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                    'zIndex': 999999999999999,
                                    'pointerEvents': 'none'
                                }
                            })
            else:
                # 기존 딕셔너리 처리 로직
                for text, data in coordinates.items():
                    text_lower = text.lower()
                    
                    # 직접 매칭 확인
                    direct_match = query_lower in text_lower
                    
                    # 연관어 매칭 확인 (스마트 검색 모드)
                    related_match = False
                    if mode == 'smart':
                        # 연관어 목록의 각 단어가 텍스트에 포함되어 있는지 확인
                        for word in related_words:
                            word_lower = word.lower()
                            if word_lower in text_lower or text_lower in word_lower:
                                related_match = True
                                print(f"연관어 매칭 성공: '{word_lower}' in '{text_lower}'")
                                break
                    
                    # 한글-영어 매핑 확인
                    mapped_match = False
                    if is_korean_query:
                        mapped_match = any(english.lower() in text_lower for english in korean_matches)
                    
                    if direct_match or related_match or mapped_match:
                        print(f"매칭된 텍스트: {text} (직접 매칭: {direct_match}, 연관어 매칭: {related_match}, 매핑 매칭: {mapped_match})")
                        if data.get('match_type') == 'object':
                            all_detected_objects.append({
                                'name': text,
                                'bbox': data['bbox'],
                                'confidence': data.get('confidence', 1.0),
                                'match_type': 'object',
                                'style': {
                                    'position': 'absolute',
                                    'border': '2px solid #00ff00',
                                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                                    'zIndex': 999999999999999,
                                    'pointerEvents': 'none',
                                    'padding': '4px',
                                    'fontSize': '14px',
                                    'color': '#00ff00',
                                    'fontWeight': 'bold',
                                    'textShadow': '1px 1px 2px rgba(0,0,0,0.5)',
                                    'whiteSpace': 'nowrap',
                                    'overflow': 'visible'
                                },
                                'label': text
                            })
                        else:
                            matches.append({
                                'text': text,
                                'bbox': data['bbox'],
                                'confidence': data.get('confidence', 1.0),
                                'match_type': 'text',
                                'style': {
                                    'position': 'absolute',
                                    'border': '2px solid #ff0000',
                                    'borderRadius': '50%',
                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                    'zIndex': 999999999999999,
                                    'pointerEvents': 'none'
                                }
                            })

        response_data = {
            'matches': matches,
            'detected_objects': all_detected_objects,
            'ocr_text': ocr_text,
            'total_matches': len(matches),
            'total_objects': len(all_detected_objects),
            'show_modal': len(matches) > 0 or len(all_detected_objects) > 0,  # 검색 결과가 있으면 모달 표시
            'styles': {
                'overlay': {
                    'position': 'absolute',
                    'top': 0,
                    'left': 0,
                    'width': '100%',
                    'height': '100%',
                    'pointerEvents': 'none',
                    'zIndex': 999999999999999
                },
                'text': {
                    'position': 'absolute',
                    'color': '#ff0000',
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'borderRadius': '50%',
                    'border': '2px solid #ff0000',
                    'padding': '2px',
                    'fontSize': '12px',
                    'zIndex': 999999999999999,
                    'pointerEvents': 'none'
                },
                'object': {
                    'position': 'absolute',
                    'color': '#00ff00',
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                    'border': '2px solid #00ff00',
                    'padding': '2px',
                    'fontSize': '12px',
                    'zIndex': 999999999999999,
                    'pointerEvents': 'none'
                }
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_similar_terms(query: str) -> list:
    """주어진 검색어와 관련된 유사어를 찾습니다."""
    try:
        similar_terms_map = {
            '음식': ['닭갈비', '치킨', '한식', '요리', '식사', '음식점', '레스토랑', '맛집'],
            '사람': ['학생', '교사', '직원', '고객', '손님', '관리자', '대표'],
            '문서': ['계약서', '보고서', '논문', '서류', '자료', '문서'],
            '제품': ['상품', '물건', '아이템', '제품명', '모델명', '브랜드'],
            '커피': ['아메리카노', '라떼', '에스프레소', '카페', '카푸치노', '모카', '드립', '원두'],
            '음료': ['물', '주스', '차', '탄산음료', '커피', '우유', '맥주', '소주'],
            '과일': ['사과', '바나나', '오렌지', '포도', '딸기', '키위', '망고', '파인애플'],
            '채소': ['상추', '양파', '당근', '오이', '토마토', '고추', '마늘', '파'],
            '고기': ['소고기', '돼지고기', '닭고기', '양고기', '오리고기', '햄', '소시지', '베이컨'],
            '해산물': ['생선', '새우', '오징어', '문어', '게', '조개', '굴', '전복']
        }
        
        # 검색어를 소문자로 변환하여 매칭
        query_lower = query.lower()
        similar_terms = []
        
        # 정확한 매칭
        if query_lower in similar_terms_map:
            similar_terms.extend(similar_terms_map[query_lower])
        
        # 부분 매칭 (예: '커피'로 검색했을 때 '음료' 카테고리의 '커피'도 찾기)
        for category, terms in similar_terms_map.items():
            if query_lower in terms:
                similar_terms.extend(terms)
        
        # 중복 제거
        similar_terms = list(set(similar_terms))
        
        # 원래 검색어 추가
        similar_terms.append(query)
        
        print(f"찾은 유사어 목록: {similar_terms}")
        return similar_terms
        
    except Exception as e:
        print(f"유사어 검색 중 오류 발생: {str(e)}")
        return [query]

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    session_id = data.get('session_id', '')
    current_image_index = data.get('current_image_index', 0)  # 현재 이미지 인덱스 추가
    
    try:
        # 세션에서 객체 인식 결과 가져오기
        detected_objects = []
        if session_id and session_id in ocr_results_cache:
            # 현재 이미지의 객체 인식 결과만 가져오기
            images = ocr_results_cache[session_id].get('images', [])
            if 0 <= current_image_index < len(images):
                current_image = images[current_image_index]
                detected_objects = current_image.get('detected_objects', [])
                print(f"현재 이미지({current_image_index})의 감지된 객체: {detected_objects}")

        # 객체 정보를 포함한 프롬프트 생성
        objects_info = ""
        if detected_objects:
            objects_info = "\n감지된 객체:\n" + "\n".join([f"- {obj['text']} (신뢰도: {obj['confidence']:.2f})" for obj in detected_objects])

        # 전역 client 객체 사용
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""당신은 FindIt Assistant입니다. 이미지에서 감지된 객체 정보를 바탕으로 사용자의 질문에 답변해주세요.

{objects_info}

답변 형식:
1. 감지된 객체가 있다면, 반드시 다음 형식으로 시작하세요:
   "이미지에서 [객체명]이(가) 감지되었습니다. 신뢰도가 [신뢰도]%로 [높음/중간/낮음]습니다."

2. 그 다음에 객체에 대한 설명을 추가하세요:
   "이는 [객체의 특징]입니다. [객체의 설명]"

3. 마지막으로 사용자의 질문에 대한 답변을 제공하세요.

예시 답변:
"이미지에서 사과(Apple)가 감지되었습니다. 신뢰도가 89.7%로 매우 높습니다. 이는 빨간색 사과로 보입니다. 사과는 과일의 일종으로, 달콤하고 신맛이 나는 특징이 있습니다."

주의사항:
- 반드시 감지된 객체 정보를 포함하여 답변해주세요.
- 객체가 감지되지 않았다면, 그 사실을 먼저 언급해주세요.
- 사용자의 질문에 직접적으로 답변해주세요.
- 신뢰도가 80% 이상이면 "매우 높음", 50% 이상이면 "중간", 그 미만이면 "낮음"으로 표현해주세요."""},
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
                        'location': location,
                        'detected_objects': detected_objects
                    })
            except Exception as e:
                print(f"Error searching location: {e}")
        
        return jsonify({
            'type': 'text',
            'content': bot_response,
            'detected_objects': detected_objects
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'type': 'text',
            'content': '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.',
            'detected_objects': []
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
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',  # 더 유연한 포맷 선택
                'outtmpl': filepath,
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'noplaylist': True,
                'verbose': True,
                'merge_output_format': 'mp4',  # 출력 포맷을 mp4로 강제
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }]
            }
            
            print("영상 정보 다운로드 중...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 영상 정보 가져오기
                print("영상 메타데이터 추출 중...")
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '')
                duration = info.get('duration', 0)
                
                print(f"영상 제목: {title}")
                print(f"영상 길이: {duration}초")
                print(f"다운로드 가능한 포맷: {info.get('formats', [])}")
                
                # 영상 다운로드
                print("다운로드 시작...")
                ydl.download([url])
                print("다운로드 완료")
            
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
        session_id = data.get('session_id', '')  # 세션 ID 가져오기
        
        if not url:
            return jsonify({'error': 'URL이 필요합니다'}), 400
            
        print(f"=== YouTube 처리 시작 ===")
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print(f"검색어: {query}")
        print(f"모드: {mode}")
        print(f"세션 ID: {session_id}")
        
        try:
            # YouTube 영상 다운로드
            video_info = download_youtube_video(url)
            print(f"다운로드 완료: {video_info['filepath']}")
            
            try:
                print("비디오 처리 시작")
                # 비디오 처리 (세션 ID 전달)
                timeline_results = process_video(video_info['filepath'], query, mode, session_id)
                print(f"처리된 타임라인 결과: {len(timeline_results)}개 항목")
                
                # OCR 텍스트 가져오기
                ocr_text = '\n'.join([f"=== {item['timestamp']}초 ===\n" + '\n'.join([text['text'] for text in item['texts']]) for item in timeline_results])
                
                # 세션 ID가 없는 경우 새로 생성
                if not session_id:
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
    """OpenAI API를 사용하여 주어진 텍스트를 요약합니다."""
    if not text.strip():
        return '정보를 추출할 텍스트가 제공되지 않았습니다.'
    try:
        # 전역 client 객체 사용
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                { 
                    'role': 'system', 
                    'content': '''당신은 이미지 분석 및 Q&A 어시스턴트입니다. 이미지에서 감지된 객체 정보를 바탕으로 사용자의 질문에 답변해주세요.

답변 형식:
1. 감지된 객체가 있다면, 반드시 다음 형식으로 시작하세요:
   "이미지에서 [객체명]이(가) 감지되었습니다. 신뢰도가 [신뢰도]%로 [높음/중간/낮음]습니다."

2. 그 다음에 객체에 대한 설명을 추가하세요:
   "이는 [객체의 특징]입니다. [객체의 설명]"

3. 마지막으로 사용자의 질문에 대한 답변을 제공하세요.

예시 답변:
"이미지에서 사과(Apple)가 감지되었습니다. 신뢰도가 89.7%로 매우 높습니다. 이는 빨간색 사과로 보입니다. 사과는 과일의 일종으로, 달콤하고 신맛이 나는 특징이 있습니다."

주의사항:
- 반드시 감지된 객체 정보를 포함하여 답변해주세요.
- 객체가 감지되지 않았다면, 그 사실을 먼저 언급해주세요.
- 사용자의 질문에 직접적으로 답변해주세요.
- 신뢰도가 80% 이상이면 "매우 높음", 50% 이상이면 "중간", 그 미만이면 "낮음"으로 표현해주세요.'''
                },
                { 'role': 'user', 'content': text },
            ],
            max_tokens=500,
            temperature=0.7,
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
        ocr_text = request.form.get('ocr_text', '')  # OCR 텍스트 가져오기
        
        if not session_id or session_id not in ocr_results_cache:
            return jsonify({'error': '유효하지 않은 세션 ID입니다'}), 400
        
        ocr_data = ocr_results_cache[session_id]
        combined_text = ocr_text if ocr_text else ocr_data['text']  # 전달받은 OCR 텍스트 우선 사용
        
        if not combined_text:
            return jsonify({'error': '텍스트를 추출할 수 없습니다'}), 400
        
        # 감지된 객체 정보 가져오기
        detected_objects = []
        coordinates = ocr_data.get('coordinates', {})
        
        # coordinates가 리스트인 경우 처리
        if isinstance(coordinates, list):
            for item in coordinates:
                if item.get('match_type') == 'object':
                    detected_objects.append({
                        'name': item.get('text', ''),
                        'confidence': item.get('confidence', 1.0)
                    })
        else:
            # 기존 딕셔너리 처리
            for text, data in coordinates.items():
                if data.get('match_type') == 'object':
                    detected_objects.append({
                        'name': text,
                        'confidence': data.get('confidence', 1.0)
                    })
        
        # 객체 정보를 포함한 프롬프트 생성
        objects_info = ""
        if detected_objects:
            objects_info = "\n감지된 객체:\n" + "\n".join([
                f"- {obj['name']} (신뢰도: {obj['confidence']:.2f})"
                for obj in detected_objects
            ])
        
        # 사용자의 질문이 있는 경우, 질문과 함께 텍스트를 전달
        if message:
            prompt = f"""다음은 이미지에서 추출된 텍스트와 감지된 객체 정보입니다:

OCR 텍스트:
{combined_text}

{objects_info}

사용자의 질문: {message}

위 정보를 바탕으로 질문에 답변해주세요. OCR 텍스트를 우선적으로 사용하여 답변해주세요."""
        else:
            prompt = f"""다음은 이미지에서 추출된 텍스트와 감지된 객체 정보입니다:

OCR 텍스트:
{combined_text}

{objects_info}

위 정보를 바탕으로 이미지를 분석해주세요. OCR 텍스트를 우선적으로 사용하여 분석해주세요."""
        
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

@app.route('/seek-timestamp', methods=['POST'])
def seek_timestamp():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON 데이터가 필요합니다'}), 400
            
        timestamp = data.get('timestamp')
        video_id = data.get('video_id')
        session_id = data.get('session_id')
        
        if timestamp is None:
            return jsonify({'error': '타임스탬프가 필요합니다'}), 400
            
        print(f"타임스탬프 이동 요청: {timestamp}초, 비디오 ID: {video_id}, 세션 ID: {session_id}")
        
        # 타임스탬프를 초 단위로 변환
        if isinstance(timestamp, str):
            # "HH:MM:SS" 또는 "MM:SS" 형식의 문자열을 초로 변환
            parts = timestamp.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(float, parts)
                timestamp = int(hours * 3600 + minutes * 60 + seconds)
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(float, parts)
                timestamp = int(minutes * 60 + seconds)
        
        # 소수점이 있는 경우 정수로 변환
        if isinstance(timestamp, float):
            timestamp = int(timestamp)
        
        # 타임스탬프를 HH:MM:SS 형식으로 변환
        hours = timestamp // 3600
        minutes = (timestamp % 3600) // 60
        seconds = timestamp % 60
        formatted_timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # 세션에서 비디오 정보 가져오기
        if session_id and session_id in ocr_results_cache:
            videos = ocr_results_cache[session_id].get('videos', [])
            for video in videos:
                if video.get('filename') == video_id:
                    # 해당 타임스탬프 근처의 텍스트 찾기
                    timeline = video.get('timeline', [])
                    nearest_text = None
                    min_diff = float('inf')
                    
                    for item in timeline:
                        item_timestamp = item.get('timestamp', 0)
                        diff = abs(item_timestamp - timestamp)
                        if diff < min_diff:
                            min_diff = diff
                            nearest_text = item.get('texts', [])
                    
                    return jsonify({
                        'success': True,
                        'timestamp': timestamp,
                        'formatted_timestamp': formatted_timestamp,
                        'video_id': video_id,
                        'nearest_text': nearest_text,
                        'session_id': session_id
                    })
        
        return jsonify({
            'success': True,
            'timestamp': timestamp,
            'formatted_timestamp': formatted_timestamp,
            'video_id': video_id,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"타임스탬프 이동 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)