# 한글-영어 매핑
koreanToEnglish = {
    '자동차': ['car', 'automobile', 'vehicle'],
    '차': ['car', 'automobile', 'vehicle'],
    '사람': ['person', 'human', 'people', 'customer'],
    '신발': ['shoe', 'footwear', 'sneaker'],
    '가방': ['bag', 'backpack', 'purse'],
    '의자': ['chair', 'seat'],
    '테이블': ['table', 'desk'],
    '컴퓨터': ['computer', 'laptop', 'pc'],
    '전화': ['phone', 'mobile', 'cellphone'],
    '책': ['book'],
    '시계': ['watch', 'clock'],
    '안경': ['glasses', 'eyeglasses'],
    '모자': ['hat', 'cap'],
    '옷': ['clothes', 'clothing', 'dress'],
    '바지': ['pants', 'trousers'],
    '상의': ['shirt', 'top', 't-shirt'],
    '가구': ['furniture'],
    '식물': ['plant', 'flower', 'tree'],
    '동물': ['animal', 'pet'],
    '음식': ['food', 'meal'],
    '음료': ['drink', 'beverage'],
    '건물': ['building', 'house'],
    '자전거': ['bicycle', 'bike'],
    '오토바이': ['motorcycle', 'bike'],
    '버스': ['bus'],
    '기차': ['train'],
    '비행기': ['airplane', 'plane'],
    '배': ['ship', 'boat'],
    '트럭': ['truck'],
    '영수증': ['receipt', 'invoice', 'bill'],
    '고객': ['customer', 'client', 'patron'],
    '사과': ['apple']
}

# 영어-한글 매핑
englishToKorean = {
    'car': ['자동차', '차'],
    'automobile': ['자동차', '차'],
    'vehicle': ['자동차', '차'],
    'person': ['사람'],
    'human': ['사람'],
    'people': ['사람'],
    'customer': ['사람', '고객'],
    'shoe': ['신발'],
    'footwear': ['신발'],
    'sneaker': ['신발'],
    'bag': ['가방'],
    'backpack': ['가방'],
    'purse': ['가방'],
    'chair': ['의자'],
    'seat': ['의자'],
    'table': ['테이블'],
    'desk': ['테이블'],
    'computer': ['컴퓨터'],
    'laptop': ['컴퓨터'],
    'pc': ['컴퓨터'],
    'phone': ['전화'],
    'mobile': ['전화'],
    'cellphone': ['전화'],
    'book': ['책'],
    'watch': ['시계'],
    'clock': ['시계'],
    'glasses': ['안경'],
    'eyeglasses': ['안경'],
    'hat': ['모자'],
    'cap': ['모자'],
    'clothes': ['옷'],
    'clothing': ['옷'],
    'dress': ['옷'],
    'pants': ['바지'],
    'trousers': ['바지'],
    'shirt': ['상의'],
    'top': ['상의'],
    't-shirt': ['상의'],
    'furniture': ['가구'],
    'plant': ['식물'],
    'flower': ['식물'],
    'tree': ['식물'],
    'animal': ['동물'],
    'pet': ['동물'],
    'food': ['음식'],
    'meal': ['음식'],
    'drink': ['음료'],
    'beverage': ['음료'],
    'building': ['건물'],
    'house': ['건물'],
    'bicycle': ['자전거'],
    'bike': ['자전거'],
    'motorcycle': ['오토바이'],
    'bus': ['버스'],
    'train': ['기차'],
    'airplane': ['비행기'],
    'plane': ['비행기'],
    'ship': ['배'],
    'boat': ['배'],
    'truck': ['트럭'],
    'receipt': ['영수증'],
    'invoice': ['영수증'],
    'bill': ['영수증'],
    'apple': ['사과']
}

def find_matches(search_term):
    """대소문자 구분 없이 검색할 수 있는 헬퍼 함수"""
    normalized_search_term = search_term.lower()
    matches = set()

    # 한글 검색어인 경우
    if any('\uAC00' <= char <= '\uD7A3' for char in search_term):
        if search_term in koreanToEnglish:
            for eng in koreanToEnglish[search_term]:
                matches.add(eng.lower())
    # 영어 검색어인 경우
    else:
        if normalized_search_term in englishToKorean:
            for kor in englishToKorean[normalized_search_term]:
                matches.add(kor)

    return list(matches) 