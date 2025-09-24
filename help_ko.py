#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVI-CIDNet 한국어 도움말 / Korean Help
"""

def show_korean_help():
    print("=" * 60)
    print("🇰🇷 HVI-CIDNet 한국어 사용 가이드")
    print("🇺🇸 HVI-CIDNet Korean Usage Guide")
    print("=" * 60)
    print()
    
    print("📱 1. 한국어 웹 인터페이스 실행")
    print("   python app.py --lang ko")
    print("   # 한국어로 Gradio 웹 인터페이스 실행")
    print()
    
    print("📱 2. 영어 웹 인터페이스 실행 (기본값)")
    print("   python app.py --lang en")
    print("   # English Gradio web interface (default)")
    print()
    
    print("🔧 3. 한국어 평가 도구")
    print("   python eval_ko.py --help")
    print("   # 한국어 도움말 보기")
    print()
    print("   python eval_ko.py --lol")
    print("   # LOLv1 데이터셋 평가")
    print()
    print("   python eval_ko.py --unpaired --DICM --unpaired_weights ./weights/LOLv2_syn/w_perc.pth --alpha 0.9")
    print("   # DICM 언페어드 데이터셋 평가")
    print()
    
    print("📊 4. 지원되는 기능")
    print("   ✅ 한국어 웹 UI (Gradio)")
    print("   ✅ 한국어 명령행 도구")
    print("   ✅ 한국어 도움말 및 설명")
    print("   ✅ 언어 전환 (ko/en)")
    print()
    
    print("📖 5. 데이터셋 지원")
    print("   • LOLv1 (저조도 데이터셋 v1)")
    print("   • LOLv2-real/syn (저조도 데이터셋 v2)")
    print("   • SICE-grad/mix (SICE 데이터셋)")
    print("   • 언페어드 데이터셋: DICM, LIME, MEF, NPE, VV")
    print()
    
    print("❓ 도움이 더 필요하시면:")
    print("   📧 이슈 제출: https://github.com/int11/HVI-CIDNet/issues")
    print("   📚 README: ./Readme.md 참고")
    print()

if __name__ == '__main__':
    show_korean_help()