import re # 정규식으로 특수기호 및 char 처리
from nltk.tokenize import TreebankWordTokenizer

class CustomTokenizer:
    """
    다양한 설정이 가능한 커스텀 토크나이저
    """
    
    def __init__(self, 
                 preserve_contractions=True,
                 preserve_possessives=True,
                 preserve_numbers=True,
                 preserve_urls=True,
                 preserve_emails=True,
                 filter_tokens=True):
        
        self.treebank_tokenizer = TreebankWordTokenizer()
        self.preserve_contractions = preserve_contractions
        self.preserve_possessives = preserve_possessives
        self.preserve_numbers = preserve_numbers
        self.preserve_urls = preserve_urls
        self.preserve_emails = preserve_emails
        self.filter_tokens = filter_tokens
        
        # 허용할 문자 패턴 (영어 대소문자, 마침표, 콤마, 물음표, 느낌표)
        self.allowed_pattern = re.compile(r"^[a-zA-Z.,'!?]+$")
        
        # 보호할 패턴들
        self.protection_patterns = []
        
        if preserve_contractions:
            self.protection_patterns.extend([
                # be 동사 축약형
                r"\bI'm\b", r"\bI'M\b", r"\bi'm\b",
                r"\b(?:he|she|it)'s\b", r"\b(?:HE|SHE|IT)'S\b", r"\b(?:he|she|it)'s\b",
                r"\b(?:we|you|they)'re\b", r"\b(?:WE|YOU|THEY)'RE\b", r"\b(?:we|you|they)'re\b",
                
                # have 동사 축약형
                r"\b(?:I|you|we|they)'ve\b", r"\b(?:I|YOU|WE|THEY)'VE\b", r"\b(?:I|you|we|they)'ve\b",
                
                # will 축약형
                r"\b(?:I|you|he|she|it|we|they)'ll\b", r"\b(?:I|YOU|HE|SHE|IT|WE|THEY)'LL\b", r"\b(?:I|you|he|she|it|we|they)'ll\b",
                
                # would 축약형
                r"\b(?:I|you|he|she|it|we|they)'d\b", r"\b(?:I|YOU|HE|SHE|IT|WE|THEY)'D\b", r"\b(?:I|you|he|she|it|we|they)'d\b",
                
                # 부정 축약형
                r"\b(?:do|does|did|can|could|will|would|should|must|is|are|was|were|has|have|had)n't\b",
                r"\b(?:DO|DOES|DID|CAN|COULD|WILL|WOULD|SHOULD|MUST|IS|ARE|WAS|WERE|HAS|HAVE|HAD)N'T\b",
                r"\b(?:do|does|did|can|could|will|would|should|must|is|are|was|were|has|have|had)n't\b",
            ])
        
        if preserve_possessives:
            self.protection_patterns.append(r"\b\w+'s\b")
        
        if preserve_numbers:
            self.protection_patterns.append(r"\b\d+(?:\.\d+)?\b")
        
        if preserve_urls:
            self.protection_patterns.append(r"https?://\S+")
        
        if preserve_emails:
            self.protection_patterns.append(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        
        # 정규식 컴파일
        if self.protection_patterns:
            self.protection_regex = re.compile('|'.join(self.protection_patterns))
        else:
            self.protection_regex = None
    
    def is_allowed_token(self, token):
        """
        토큰이 허용된 패턴인지 확인
        """
        # 빈 토큰은 허용하지 않음
        if not token:
            return False
        
        # 허용된 패턴에 맞는지 확인
        return bool(self.allowed_pattern.match(token))
    
    def tokenize(self, text):
        """
        설정에 따라 텍스트를 토큰화
        """

        # 특수기호 필터링
        text = text.replace("//", " ")

        if not self.protection_regex:
            # 보호할 패턴이 없으면 Treebank Tokenizer만 사용
            tokens = self.treebank_tokenizer.tokenize(text)
        else:
            # 1. 보호할 패턴들을 임시 토큰으로 대체
            protected_text = text
            protected_matches = []
            
            for match in self.protection_regex.finditer(text):
                match_text = match.group()
                protected_matches.append(match_text)
                protected_text = protected_text.replace(match_text, f"__PROTECTED_{len(protected_matches)-1}__")
            
            # 2. Treebank Tokenizer로 토큰화
            tokens = self.treebank_tokenizer.tokenize(protected_text)
            
            # 3. 임시 토큰을 원래 형태로 복원
            final_tokens = []
            for token in tokens:
                if token.startswith("__PROTECTED_") and token.endswith("__"):
                    # 정규식을 사용하여 인덱스 추출
                    match = re.search(r"__PROTECTED_(\d+)__", token)
                    if match:
                        idx = int(match.group(1))
                        if idx < len(protected_matches):
                            final_tokens.append(protected_matches[idx])
                        else:
                            final_tokens.append(token)  # 인덱스가 범위를 벗어나면 원본 토큰 유지
                    else:
                        final_tokens.append(token)  # 패턴이 맞지 않으면 원본 토큰 유지
                else:
                    final_tokens.append(token)
            
            tokens = final_tokens
        
        # 4. 온점을 명시적으로 분리 (토큰 중간에 있는 온점도 포함)
        processed_tokens = []
        for token in tokens:
            # 온점이 포함된 토큰인지 확인
            if '.' in token:
                # 온점을 기준으로 분리
                parts = token.split('.')
                for i, part in enumerate(parts):
                    if part:  # 빈 문자열이 아닌 경우만 추가
                        processed_tokens.append(part)
                    if i < len(parts) - 1:  # 마지막 부분이 아니면 온점 추가
                        processed_tokens.append('.')
            else:
                processed_tokens.append(token)
        
        # 5. 토큰 필터링 (허용되지 않은 토큰 제거)
        if self.filter_tokens:
            filtered_tokens = []
            for token in processed_tokens:
                if self.is_allowed_token(token):
                    filtered_tokens.append(token)
            return filtered_tokens
        
        return processed_tokens