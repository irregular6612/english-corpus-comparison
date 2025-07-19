import nltk # Natural Language Toolkit
from nltk import word_tokenize, pos_tag, sent_tokenize # Natural Language Toolkit
from nltk.tokenize import WhitespaceTokenizer
from customTokenizer import CustomTokenizer
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import Any, Dict, List
from pathlib import Path
import os


class Preprocessor:
    def __init__(self, type:str, tokenizer_type:str = 'custom'):
        # corpus 종류에 따라 구분 / 수능 or 교과서
        if type not in ['test', 'textbook']:
            raise ValueError('type must be either "test" or "textbook"')     # 둘 다 아니라면,,
        self.type = type
        
        if tokenizer_type not in ['word', 'whitespace', 'custom']:
            raise ValueError('tokenizer must be either "word" or "whitespace" or "custom"')
        self.tokenizer_type = tokenizer_type
        
        # nltk에서 제공하는 tokenizer와 tagger를 다운로드: 
        try:
            nltk.download('punkt')  
            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger_eng')
        except:
            pass
        
    def split_sentences(self, text: str) -> list: # 여러 문장을 각각의 문장으로 분리
        return sent_tokenize(text)
    
    def tokenize_sentence(self, sentence: str) -> list: # str 형태의 문장은 단어별 품사 태깅(single sentence)
        """
        사용할 tokenizer는 크게 두 가지로
            1. word: nltk의 word_tokenize, penn treebank tagger 기반의 최근 많이 사용되고 있는 정밀 tokenizer
                비교적 최근에 나온 SUBTLEX 같은 corpus에서 사용된 것으로 추정.
            2. whitespace: nltk의 WhitespaceTokenizer, 가장 단순한 tokenizer로, 공백을 기준으로 토큰화.
                비교적 예전인 80-90년대 사용되던 tokenizer. HAL이 이렇게 단순하게 tokenized 된 것으로 추정.
            -> 위 두 개의 tokenizer를 구분한 것은 EDA 중 SUBTLEX에서는 축약형이 검색되지 않은 것을 기반으로 한다.
        두 가지 모두 품사 태깅을 위해 사용되며, 품사 태깅 결과는 동일하게 나온다.
        
        """
            
        if self.tokenizer_type == 'word':
            tokens = word_tokenize(sentence) # tokenize sentence -> word level
            tagged = pos_tag(tokens, lang='eng') # tagging the word -> pos(품사) level
            return tagged
        elif self.tokenizer_type == 'whitespace':
            tokens = WhitespaceTokenizer().tokenize(sentence)
            tagged = pos_tag(tokens, lang='eng') # tagging the word -> pos(품사) level
            return tagged
        elif self.tokenizer_type == 'custom':
            tokenizer = CustomTokenizer(preserve_contractions=True, 
                                        preserve_possessives=True, 
                                        preserve_numbers=False, 
                                        preserve_urls=False, 
                                        preserve_emails=False)
            tokens = tokenizer.tokenize(sentence)
            tagged = pos_tag(tokens, lang='eng') # tagging the word -> pos(품사) level
            return tagged
        else:
            raise ValueError('option must be either "word" or "whitespace" or "custom".')
    
    def split_word_pos(self, tagged: list) -> list: # 위의 정보를 단어와 품사로 분리
        word_list, pos_list = [], [] # 각각 단어와 품사를 저장할 리스트
        for word, pos in tagged:    # token과 품사 정보를 분리하여 저장.
            if self.tokenizer_type == 'whitespace' and any(char in word for char in ['.', '!', '?']): # whitespace tokenizer는 온점을 제거하지 않음.
                word = word.replace('.', '')
                word = word.replace('!', '')
                word = word.replace('?', '')
            word_list.append(word)
            pos_list.append(pos)
            
        return word_list, pos_list
    
    def change_data_type(self, corpus: pd.DataFrame): # 열 별 데이터 타입 수정.
        
        if self.type == 'test': # 수능 읽기 혹은 듣기 지문
            if '년도' in corpus.columns:    # 수능 읽기, 듣기 지문 파일에만 적용!!
                corpus['년도'] = corpus['년도'].astype('int')
                corpus['월'] = corpus['월'].astype('int')
                corpus['번호'] = corpus['번호'].astype('int')

                corpus = corpus.astype({
                    '년도': 'str',
                    '월': 'str',
                    '번호': 'str',
                    '출처': 'str',
                    '비고': 'str',
                    '본문': 'str'
                })
        elif self.type == 'textbook': # 영어 교과서
            corpus = corpus.astype({
                '출처': 'str',
                '비고': 'str',
                '본문': 'str'})
        

        return corpus
        
    def fillter_values(self, df: pd.DataFrame): # 일부 열 형식 정리
        
        # 숫자만 해당하는 정규식
        number_expression = r'^\d+'

        if self.type == 'test': # 수능 관련 지문들만 처리,,
            if '년도' not in df.columns:    # 해당 파일들은 '년도' column이 있을 테니,,
                raise ValueError('년도 열이 없습니다.')

            for idx, row in tqdm(df.iterrows(), desc='데이터 형식 통일 중..', total=len(df)):
                # NaN, null 값을 채운다.
                if row['본문'] is None: # 혹시라도 본문이 없는 row면 pass
                    continue

                # 모든 정보가 다 있는 경우, 
                if not np.isnan(row['년도']):   # 년도 정보가 없다면 바로 이전 row에서 참고
                    if type(row['년도']) == str:
                        df.loc[idx, '년도'] = re.search(number_expression, row['년도']).group()
                    if type(row['월']) == str:
                        df.loc[idx, '월'] = re.search(number_expression, row['월']).group()
                    if type(row['번호']) == str:
                        df.loc[idx, '번호'] = re.search(number_expression, row['번호']).group()
                    if type(row['출처']) == str:
                        if '수능' in row['출처']:
                            df.loc[idx, '출처'] = '수능'
                        elif '모의' in row['출처']:
                            df.loc[idx, '출처'] = '모의'
                        else:
                            pass
                
                # 일부 정보가 없는 경우
                else:   # 년도 정보가 없다면 바로 이전 row에서 참고
                    df.loc[idx, '년도'] = df.loc[idx-1, '년도']
                    df.loc[idx, '월'] = df.loc[idx-1, '월']
                    df.loc[idx, '출처'] = df.loc[idx-1, '출처']
                    if df.loc[idx, '비고'] is None:
                        df.loc[idx, '비고'] = "."
        
        elif self.type == 'textbook':
            if '저자' not in df.columns:    # 해당 파일들은 '저자' column이 있을 테니,,
                raise ValueError('저자 열이 없습니다.')
            
            remove_row_idxs = []    # 본문이 없는 열들 번호를 저장할 리스트
            sources = [] # 출처 정보를 통합하여 저장할 리스트
            notes = [] # 비고 정보 저장 열
            
            for idx, row in tqdm(df.iterrows(), desc='데이터 형식 통일 중..', total=len(df)):
                
                if row['본문'] is None: # 혹시라도 본문이 없는 row면 collect
                    remove_row_idxs.append(idx)
                
                if row['출판사'] is None:   # 메타 정보 중 일부가 None이라면 str _ 값으로 대체
                    row['출판사'] = '_'
                if row['저자'] is None:
                    row['저자'] = '_'
                if row['과정'] is None:
                    row['과정'] = '_'
                if row['교과서'] is None:
                    row['교과서'] = '_'
                if row['단원'] is None:
                    row['단원'] = '_'
                elif type(row['단원']) != str:
                    row['단원'] = str(row['단원'])
                if row['단원명'] is None:
                    row['단원명'] = '_'
                if row['본문제목'] is None:
                    row['본문제목'] = '_'
                
                source = f"{row['출판사']} {row['저자']} {row['과정']} {row['교과서']}"
                sources.append(source)

                if row['비고'] is None:
                    row['비고'] = '.'
                note = f"{row['단원']} {row['단원명']} {row['본문제목']} / {row['비고']}"
                notes.append(note)
            
            
            # 출처 정보 추가 및 불필요 열 제거
            df['출처'] = sources
            df['비고'] = notes
            df = df.drop(index=remove_row_idxs)
            df = df.drop(columns=['출판사', '저자', '과정', '교과서', '단원', '단원명', '본문제목'])
        
        else:
            raise ValueError('type must be either "test" or "textbook"')
        
        return df

    def generate_json_data(self, id: int, text:str, tokens:list, pos_tags:list, metadata:dict) -> dict:
        """
        주어진 행의 데이터를 JSON 형식으로 변환하는 함수
        
        Args:
            text (str): 원본 텍스트
            tokens (list): 토큰 리스트
            pos_tags (list): 품사 태깅 리스트
            metadata (dict): 메타데이터
        """ 
        
        return {
            "id": id,
            "text": text,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "metadata": metadata
        }

    def get_gender(self, text: str) -> str: # 문장 내에서 화자 성별 탐지
        if 'M:' in text or 'M ' in text:
            return 'M'
        elif 'W:' in text or 'W ' in text:
            return 'W'
        else:
            return 'N'

class TargetCorpusPreprocessor:
    """
    Change the data type:
        Word: object(keep)

        Freq_CSAT : int (keep)
        LogE_Freq_CSAT : float(keep) (log base: e)
        Log10_Freq_CSAT : float(keep) (log base: 10 with bias 1) : log_10_(value+1)

        Length : int64(keep)

        Freq_HAL : object -> int (remove comma)
        Log_Freq_HAL : object -> float (log base: e)

        SUBTLWF : object -> float (remove comma, freq per million)
        LgSUBTLWF : object -> float / log10(number of times the word appears in the corpus + 1)
        
        Ortho_N : int64 -> int64 (keep)
        OLD : object -> float
        OLDF : object -> float
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.col_name = df.columns
    
    
    def remove_hash(self, col_name: str) -> pd.DataFrame:
        for idx, value in enumerate(self.df[col_name]):
            if '#' in value:
                self.df.loc[idx, col_name] = None
        return self.df

    def remove_comma(self, col_name: str) -> pd.DataFrame:
        self.df[col_name] = self.df[col_name].str.replace(',', '')
        return self.df
    
    def change_col_type(self, col_name: str, dtype: str) -> pd.DataFrame:
        self.df[col_name] = self.df[col_name].astype(dtype)
        return self.df

    def preprocess(self) -> pd.DataFrame:

        for col_name in tqdm(self.col_name, desc='Cleaning information col by col'):
            if self.df[col_name].dtype == 'object' and not col_name in ['Word']:
                print(col_name)
                self.df = self.remove_hash(col_name)
                self.df = self.remove_comma(col_name)
        
        #self.df.dropna(inplace=True)
        
        # change column data type
        if 'Freq_HAL' in self.df.columns:
            self.df = self.change_col_type('Freq_HAL', 'int64')
        #self.df = self.change_col_type('Log_Freq_HAL', 'float')
        if 'SUBTLWF' in self.df.columns:
            self.df = self.change_col_type('SUBTLWF', 'float')
        if 'LgSUBTLWF' in self.df.columns:
            self.df = self.change_col_type('LgSUBTLWF', 'float')
        if 'OLD' in self.df.columns:
            self.df = self.change_col_type('OLD', 'float')
        if 'OLDF' in self.df.columns:
            self.df = self.change_col_type('OLDF', 'float')
            
        return self.df
