import os
import cv2
import pytesseract
import easyocr
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
from paddleocr import PaddleOCR
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 origin 허용

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# 업로드 파일이 저장될 경로 설정
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tesseract 경로 설정 (macOS에서 Homebrew로 설치한 경우)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# EasyOCR 리더 초기화
try:
    reader = easyocr.Reader(['ko', 'en'])  # 한국어, 영어 모델
    print("EasyOCR initialized successfully")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None

# PaddleOCR 초기화
ocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)

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
        
        # OCR 수행
        if reader:
            # EasyOCR 사용
            result = reader.readtext(thresh)
            text = ' '.join([detection[1] for detection in result])
        else:
            # Tesseract 사용
            text = pytesseract.image_to_string(thresh, lang='eng+kor')
        
        return text.strip()
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
        
        # OCR 수행 및 텍스트 위치 추출
        if reader:
            # EasyOCR 사용
            results = reader.readtext(thresh)
            text_boxes = []
            for detection in results:
                points = detection[0]
                text = detection[1]
                confidence = detection[2]
                
                # 좌표 추출
                x_coords = [int(p[0]) for p in points]
                y_coords = [int(p[1]) for p in points]
                
                text_boxes.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': {
                        'x1': min(x_coords),
                        'y1': min(y_coords),
                        'x2': max(x_coords),
                        'y2': max(y_coords)
                    }
                })
            
            return text_boxes
        else:
            # Tesseract 사용
            data = pytesseract.image_to_data(thresh, lang='eng+kor', output_type=pytesseract.Output.DICT)
            text_boxes = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 60:  # 신뢰도가 60% 이상인 경우만
                    text_boxes.append({
                        'text': data['text'][i],
                        'confidence': float(data['conf'][i]),
                        'bbox': {
                            'x1': data['left'][i],
                            'y1': data['top'][i],
                            'x2': data['left'][i] + data['width'][i],
                            'y2': data['top'][i] + data['height'][i]
                        }
                    })
            
            return text_boxes
            
    except Exception as e:
        print(f"Error in process_image: {e}")
        raise

def analyze_image_with_gpt(image_path, query):
    try:
        # 이미지를 base64로 인코딩
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image size: {width}x{height}")
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # GPT-4 Vision API 호출
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"이미지에서 '{query}'와 관련된 객체들을 찾아주세요. 이미지 크기는 {width}x{height}입니다. 각 객체의 위치를 (x1, y1, x2, y2) 형식으로 알려주세요. 예시: '얼룩말: (100, 200, 300, 400)'. 좌표는 이미지 크기 내에 있어야 합니다."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        # 응답 파싱
        result = response.choices[0].message.content
        print(f"GPT Response: {result}")
        boxes = parse_gpt_response(result)
        
        # 좌표가 이미지 크기 내에 있는지 확인
        valid_boxes = []
        for box in boxes:
            if (0 <= box['bbox']['x1'] <= width and 
                0 <= box['bbox']['y1'] <= height and 
                0 <= box['bbox']['x2'] <= width and 
                0 <= box['bbox']['y2'] <= height):
                valid_boxes.append(box)
            else:
                print(f"Invalid coordinates: {box}")
        
        print(f"Valid boxes: {valid_boxes}")
        return valid_boxes
    except Exception as e:
        print(f"Error in analyze_image_with_gpt: {e}")
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

def extract_text_with_paddleocr(image_path):
    """PaddleOCR을 사용하여 텍스트 추출"""
    try:
        print(f"OCR 시작: {image_path}")
        
        # OCR 실행
        result = ocr.ocr(image_path, cls=True)
        
        coordinates = {}
        for idx, line in enumerate(result[0]):
            bbox = line[0]
            text = line[1][0]  # 인식된 텍스트
            confidence = line[1][1]  # 신뢰도
            
            if not text.strip():  # 빈 텍스트 건너뛰기
                continue
            
            # 바운딩 박스 좌표 변환
            x_coords = [int(point[0]) for point in bbox]
            y_coords = [int(point[1]) for point in bbox]
            
            bbox_dict = {
                'x1': min(x_coords),
                'y1': min(y_coords),
                'x2': max(x_coords),
                'y2': max(y_coords)
            }
            
            print(f"인식된 텍스트: '{text}', 신뢰도: {confidence}")
            
            coordinates[text] = {
                'bbox': bbox_dict,
                'confidence': confidence
            }
        
        # 세로 텍스트 처리
        combined_coordinates = combine_vertical_texts(coordinates)
        
        recognized_text = ' '.join(combined_coordinates.keys())
        print(f"OCR 결과: {len(combined_coordinates)}개의 텍스트 발견")
        print(f"전체 텍스트: {recognized_text}")
        
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
    
    # 모든 텍스트 쌍을 검사하여 세로로 정렬된 텍스트 그룹 찾기
    while texts:
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

def process_video(video_path, query):
    """비디오 처리 및 타임라인 생성"""
    print(f"비디오 처리 시작: {video_path}")
    frames, timestamps = extract_frames_from_video(video_path)
    
    timeline_results = []
    
    for frame, timestamp in zip(frames, timestamps):
        # 프레임을 임시 이미지 파일로 저장
        temp_frame_path = os.path.join(UPLOAD_FOLDER, f'temp_frame_{timestamp}.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # OCR 실행
            result = ocr.ocr(temp_frame_path, cls=True)
            
            if result and result[0]:
                detected_texts = []
                for line in result[0]:
                    text = line[1][0]  # 인식된 텍스트
                    confidence = line[1][1]  # 신뢰도
                    bbox = line[0]  # 바운딩 박스
                    
                    # 쿼리와 매칭되는 텍스트 찾기
                    if query.lower() in text.lower():
                        detected_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': {
                                'x1': int(min(point[0] for point in bbox)),
                                'y1': int(min(point[1] for point in bbox)),
                                'x2': int(max(point[0] for point in bbox)),
                                'y2': int(max(point[1] for point in bbox))
                            }
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
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Allowed formats: mp4, avi, mov, mkv'}), 400
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 비디오 처리 및 자막 추출
        subtitles = extract_subtitles(filepath)
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'subtitles': subtitles
        })
    except Exception as e:
        print(f"Error in upload_video: {e}")
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
        
        # 이미지 처리 및 텍스트 추출
        text_boxes = process_image(filepath)
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'imageUrl': f'/uploads/{filename}',
            'textBoxes': text_boxes
        })
    except Exception as e:
        print(f"Error in upload_image: {e}")
        return jsonify({'error': str(e)}), 500

# 이미지 분석 엔드포인트 추가
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files or 'query' not in request.form:
            return jsonify({'error': '이미지와 쿼리가 필요합니다'}), 400

        file = request.files['image']
        query = request.form['query']
        mode = request.form.get('mode', 'normal')

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
                ocr_text, coordinates = extract_text_with_paddleocr(filepath)
                detected_objects = []
                query_lower = query.lower()

                if mode == 'normal':
                    for text, data in coordinates.items():
                        text_lower = text.lower()
                        if query_lower in text_lower:
                            detected_objects.append({
                                'text': text,
                                'bbox': data['bbox']
                            })
                else:
                    for text, data in coordinates.items():
                        if semantic_similarity(query, text) > 0.5:
                            detected_objects.append({
                                'text': text,
                                'bbox': data['bbox']
                            })

                return jsonify({
                    'objects': detected_objects,
                    'ocr_text': ocr_text,
                    'type': 'image'
                })
            
            # 비디오 파일 처리
            else:
                timeline_results = process_video(filepath, query)
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
