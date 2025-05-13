// 이미지 유형 정의
export type ImageType = 'CONTRACT' | 'PAYMENT' | 'DOCUMENT' | 'PRODUCT' | 'OTHER';

// 이미지 유형별 프롬프트 템플릿
export const IMAGE_TYPE_PROMPTS: Record<ImageType, string> = {
  CONTRACT: `이미지는 계약서입니다. 다음 정보를 추출해주세요:
- 계약 당사자
- 계약 기간
- 주요 계약 조건
- 특이사항`,

  PAYMENT: `이미지는 정산/지출 관련 문서입니다. 다음 정보를 추출해주세요:
- 항목
- 금액
- 일자
- 장소/지점명
- 지급/수령인`,

  DOCUMENT: `이미지는 논문/문서입니다. 다음 정보를 추출해주세요:
- 작성자
- 작성일
- 주요 내용
- 결론/요약`,

  PRODUCT: `이미지는 제품 설명서입니다. 다음 정보를 추출해주세요:
- 제품명
- 모델명
- 주요 기능
- 제조사`,

  OTHER: `이미지의 주요 정보를 추출해주세요:
- 주요 내용
- 중요 정보`
};

// 이미지 유형별 키워드 (자동 감지용)
export const IMAGE_TYPE_KEYWORDS: Record<ImageType, string[]> = {
  CONTRACT: [
    '계약서', '계약', '계약기간', '당사자', '서명', '계약조건',
    '계약서명', '계약일자', '계약금', '계약자', '피계약자',
    '계약내용', '계약조항', '계약서류', '계약서 작성',
    '계약서 확인', '계약서 검토', '계약서 승인'
  ],
  PAYMENT: [
    '영수증', '거래명세서', '결제', '금액', '지출', '수입',
    '정산', '지급', '수령', '지점', '매장', '결제일',
    '결제금액', '결제방법', '결제내역', '결제확인',
    '영수증 확인', '영수증 발급', '영수증 출력'
  ],
  DOCUMENT: [
    '논문', '문서', '보고서', '작성자', '작성일', '결론',
    '요약', '목차', '서론', '본론', '참고문헌',
    '문서번호', '문서제목', '문서작성', '문서검토',
    '문서승인', '문서보관', '문서관리'
  ],
  PRODUCT: [
    '제품', '모델', '기능', '제조사', '사양', '사용설명서',
    '제품명', '제품번호', '제품가격', '제품특징',
    '제품사양', '제품설명', '제품이미지', '제품카탈로그',
    '제품소개', '제품안내', '제품정보'
  ],
  OTHER: []
};

// 이미지 유형별 아이콘 (Font Awesome)
export const IMAGE_TYPE_ICONS: Record<ImageType, string> = {
  CONTRACT: 'fa-file-contract',
  PAYMENT: 'fa-receipt',
  DOCUMENT: 'fa-file-alt',
  PRODUCT: 'fa-box',
  OTHER: 'fa-file'
};

// 이미지 유형별 색상
export const IMAGE_TYPE_COLORS: Record<ImageType, string> = {
  CONTRACT: '#4CAF50',
  PAYMENT: '#2196F3',
  DOCUMENT: '#9C27B0',
  PRODUCT: '#FF9800',
  OTHER: '#607D8B'
}; 