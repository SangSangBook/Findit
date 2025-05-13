export type ImageType = 'CONTRACT' | 'PAYMENT' | 'DOCUMENT' | 'PRODUCT' | 'OTHER';

export const IMAGE_TYPE_ICONS: Record<ImageType, string> = {
  CONTRACT: 'fas fa-file-contract',
  PAYMENT: 'fas fa-receipt',
  DOCUMENT: 'fas fa-file-alt',
  PRODUCT: 'fas fa-box',
  OTHER: 'fas fa-question'
};

export const IMAGE_TYPE_LABELS: Record<ImageType, string> = {
  CONTRACT: '계약서',
  PAYMENT: '결제 문서',
  DOCUMENT: '일반 문서',
  PRODUCT: '제품 이미지',
  OTHER: '기타'
};

export const IMAGE_TYPE_COLORS: Record<ImageType, string> = {
  CONTRACT: '#4CAF50',  // 녹색
  PAYMENT: '#2196F3',   // 파란색
  DOCUMENT: '#FF9800',  // 주황색
  PRODUCT: '#9C27B0',   // 보라색
  OTHER: '#757575'      // 회색
};

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

export const IMAGE_TYPE_KEYWORDS: Record<ImageType, string[]> = {
  CONTRACT: ['계약서', '계약', '계약기간', '당사자', '서명', '계약조건'],
  PAYMENT: ['영수증', '거래명세서', '결제', '금액', '지출', '수입', '정산', '지급', '수령', '지점', '매장'],
  DOCUMENT: ['논문', '문서', '보고서', '작성자', '작성일', '결론', '요약'],
  PRODUCT: ['제품', '모델', '기능', '제조사', '사양', '사용설명서'],
  OTHER: []
}; 