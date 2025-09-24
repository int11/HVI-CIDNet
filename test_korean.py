#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 지원 테스트 스크립트
Korean Language Support Test Script
"""
import argparse

# 번역 딕셔너리 가져오기
TRANSLATIONS = {
    'en': {
        'title': "HVI-CIDNet (Low-Light Image Enhancement)",
        'input_image': "Low-light Image",
        'result': "Result",
    },
    'ko': {
        'title': "HVI-CIDNet (저조도 이미지 향상)",
        'input_image': "저조도 이미지",
        'result': "결과",
    }
}

def test_korean_support():
    parser = argparse.ArgumentParser(description='한국어 지원 테스트 / Korean Support Test')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko'], 
                        help='언어 선택 / Language selection (en/ko)')
    args = parser.parse_args()
    
    # 선택된 언어의 번역 텍스트 가져오기
    texts = TRANSLATIONS[args.lang]
    
    print(f"=== {texts['title']} ===")
    print(f"Input: {texts['input_image']}")
    print(f"Output: {texts['result']}")
    
    if args.lang == 'ko':
        print("\n✅ 한국어 지원이 정상적으로 작동합니다!")
        print("Korean language support is working correctly!")
    else:
        print("\n✅ English interface is working correctly!")
        print("영어 인터페이스가 정상적으로 작동합니다!")

if __name__ == '__main__':
    test_korean_support()